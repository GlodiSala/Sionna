"""
diag_csi_imperfect_seedfix.py -- version corrigee de
diag_csi_imperfect_signed_attn_sweep.py (panneau CSI imparfait de
fig:sumrate_umi) :
  1) sionna_config.seed = SEED (vrai fix -- config.tf_rng non seede par
     tf.random.set_seed).
  2) Calcul du debit par les DEUX metriques : Sionna (_sum_rate, comme
     l'original) ET SINR manuel direct (sans lmmse_matrix), sur le MEME
     h_true/g pour chaque methode -- deja partage par construction dans le
     script original (h_true tire une fois par draw, reutilise pour les 5
     methodes).

Meme mecanisme que l'original : precodeur calcule sur h_est (bruite,
pilote 20dB), debit rapporte score sur h_true (vrai canal). Confirme :
diag_csi_imperfect_signed_attn_sweep.py:126,131,137 appellent tous
system._sum_rate(h_true, g_xxx, no) -- meme chaine LMMSEPostEqualization
SINR(interference_whitening=True) que Table 3.9, donc meme biais
structurel.

Ne modifie PAS diag_csi_imperfect_signed_attn_sweep.py (fichier separe).
"""
import os, sys, json, pickle, hashlib, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 2026   # meme seed que l'original (diag_csi_imperfect_signed_attn_sweep.py: SEED=2026)
tf.random.set_seed(SEED)
np.random.seed(SEED)

from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED   # <-- LE FIX

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, set_locked_topology
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_DRAWS = 20
BATCH = 32

RESULT_JSONS = {
    'SingleSC': 'results/diag_front_a_signed_attn.json',
    'IntraRB':  'results/diag_front_a_signed_attn_intra_rb.json',
    'TA_RB_residual': 'results/diag_front_a_signed_attn_ta_rb_residual.json',
}
NAME_MAP = {'SingleSC': 'SC', 'IntraRB': 'IB', 'TA_RB_residual': 'TA-RB'}


def resolve_ckpt(name):
    with open(RESULT_JSONS[name]) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
    return ckpt


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=128, num_heads=4, num_layers=4)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    raise ValueError(kind)


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


def manual_sinr_rate(stream_management, h_true, g, no):
    """SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no) -- meme
    formule que precoders_w.py:103-110, evaluee sur le VRAI canal h_true
    (pas h_est), comme _sum_rate le fait deja pour la metrique Sionna."""
    if len(g.shape) == 5:
        g = tf.expand_dims(g, axis=1)
    h_pc = _get_desired_channels(h_true, stream_management)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    Yb = tf.reduce_mean(rate, axis=[1, 2])
    return float(tf.reduce_sum(tf.reduce_mean(Yb, axis=0)))


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    system = LockedSystem(M, K)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models = {}
    ckpts = {}
    for name in RESULT_JSONS:
        ckpt = resolve_ckpt(name)
        ckpts[name] = ckpt
        no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        m = build_neural(name)
        load_weights(m, ckpt, dummy_h, no0)
        neural_models[name] = m
        print(f'OK {name} charge depuis {ckpt}', flush=True)

    rng = np.random.RandomState(SEED)
    methods = ['RZF', 'WMMSE'] + list(RESULT_JSONS.keys())
    sionna_res = {m: {} for m in methods}
    manual_res = {m: {} for m in methods}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_sionna = {m: [] for m in methods}
        r_manual = {m: [] for m in methods}
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_rzf_n = rzf_precoder(h_est, stream_management=system.sm, no=no)
            r_sionna['RZF'].append(float(system._sum_rate(h_true, g_rzf_n, no)))
            r_manual['RZF'].append(manual_sinr_rate(system.sm, h_true, g_rzf_n, no))

            g_wmmse_n = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            r_sionna['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse_n, no)))
            r_manual['WMMSE'].append(manual_sinr_rate(system.sm, h_true, g_wmmse_n, no))

            for name, model in neural_models.items():
                g_n = model(h_est, no=no, training=False)
                r_sionna[name].append(float(system._sum_rate(h_true, g_n, no)))
                r_manual[name].append(manual_sinr_rate(system.sm, h_true, g_n, no))

        for m in methods:
            sionna_res[m][str(data_snr)] = float(np.mean(r_sionna[m]))
            manual_res[m][str(data_snr)] = float(np.mean(r_manual[m]))
        print(f"SNR={data_snr} done | " +
              " ".join(f"{m}:sionna={sionna_res[m][str(data_snr)]:.3f}/manuel={manual_res[m][str(data_snr)]:.3f}"
                        for m in methods), flush=True)

    def to_table(res):
        return {'snr': DATA_SNRS_DB,
                **{NAME_MAP.get(m, m): [res[m][str(s)] for s in DATA_SNRS_DB] for m in methods}}

    meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'h_true tire une fois par draw (deja le cas dans le script original), reutilise pour les 5 methodes; precodeur calcule sur h_est bruite (pilote 20dB), debit score sur h_true',
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders_w.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts,
        'batch_size': BATCH,
        'num_draws': NUM_DRAWS,
        'pilot_snr_db': PILOT_SNR_DB,
        'config': STANDARD_CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }

    date_tag = time.strftime('%Y%m%d_%H%M%S')
    out_sionna = {'metric': 'sionna_LMMSEPostEqualizationSINR (_sum_rate, comme original)',
                  'table': to_table(sionna_res), 'metadata': meta, 'raw': sionna_res}
    out_manual = {'metric': 'manual_sinr (precoders_w.py:103-110, evalue sur h_true)',
                  'table': to_table(manual_res), 'metadata': meta, 'raw': manual_res}

    p1 = f'results/diag_csi_imperfect_seedfix_sionna_{date_tag}.json'
    p2 = f'results/diag_csi_imperfect_seedfix_manual_sinr_{date_tag}.json'
    with open(p1, 'w') as f: json.dump(out_sionna, f, indent=2)
    with open(p2, 'w') as f: json.dump(out_manual, f, indent=2)
    # copie "stable" (nom sans date) pour le script de figure
    with open('results/diag_csi_imperfect_seedfix_manual_sinr.json', 'w') as f: json.dump(out_manual, f, indent=2)

    print("\n=== TABLEAU AVANT/APRES (SNR=0/5/10/15/20 dB) ===")
    print("--- Sionna ---")
    print(to_table(sionna_res))
    print("--- Manuel ---")
    print(to_table(manual_res))
    print(f"\nSauvegarde -> {p1}\nSauvegarde -> {p2}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
