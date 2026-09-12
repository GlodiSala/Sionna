"""
diag_csi_imperfect_massive_capD384L4_seedfix.py -- version corrigee (POINT 3,
regime UMi-massive M64K8 D=384 CSI imparfait) de
diag_csi_imperfect_massive_capD384L4.py.

Corrections (memes que classical_comparison_seedfix_v2.py) :
1) sionna_config.seed = SEED (le vrai fix -- config.tf_rng n'est PAS seede
   par tf.random.set_seed).
2) Calcul du debit par DEUX metriques sur les MEMES tirages : la metrique
   Sionna (_sum_rate/LMMSEPostEqualizationSINR, comme l'original) ET un
   SINR manuel direct SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no), sans
   lmmse_matrix.

Le canal etait DEJA partage dans le script original (h_true tire une fois
par draw, reutilise pour RZF/WMMSE/3 reseaux, perfect ET pilot) -- rien a
changer sur ce point pour ce regime.

Version allegee (NUM_DRAWS/BATCH reduits) pour rester rapide -- objectif :
confirmer le sens/l'ampleur de l'ecart metrique Sionna vs SINR manuel sur
ce regime, pas produire une table finale publication-ready.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_csi_imperfect_massive_capD384L4_seedfix.py
"""
import os, sys, json, pickle, time, hashlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED   # <-- LE FIX

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
D, L = 384, 4
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0          # un seul pilote (convention Table 3.9 / point 2), pas les 2 de l'original
NUM_DRAWS = 4                # allege (original: 12)
BATCH = 16
RNG_SEED = 2026

RESULT_JSONS = {
    'SingleSC': 'results/diag_massive_gap_fullbudget_single_sc_cap_d384l4.json',
    'IntraRB':  'results/diag_massive_gap_fullbudget_intra_rb_cap_d384l4.json',
    'TA_RB_residual': 'results/diag_massive_gap_fullbudget_ta_rb_residual_cap_d384l4.json',
}
METHODS = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']


def resolve_ckpt(name):
    with open(RESULT_JSONS[name]) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
    return ckpt


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=D, num_heads=4, num_layers=L)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    raise ValueError(kind)


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=MASSIVE_TRUE_CONFIG['FORCE_LOS'],
                             indoor_probability=self.indoor_probability)


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


def manual_sinr_rate(system, h_true, g, no):
    """SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no), sur le
    canal EFFECTIF REEL h_true (meme convention que la metrique Sionna:
    le precodeur peut avoir ete calcule sur h_est, mais le score utilise
    toujours h_true)."""
    h_pc = _get_desired_channels(h_true, system.sm)          # [B,1,ofdm,fft,K,M]
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
    Y = tf.reduce_mean(rate, axis=[1, 2])
    return float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))


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
        print(f'OK {name} (D={D}) charge depuis {ckpt}', flush=True)

    rng = np.random.RandomState(RNG_SEED)
    sionna_out = {m: {} for m in METHODS}
    manual_out = {m: {} for m in METHODS}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_sionna = {m: [] for m in METHODS}
        r_manual = {m: [] for m in METHODS}
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_rzf = rzf_precoder(h_est, stream_management=system.sm, no=no)
            r_sionna['RZF'].append(float(system._sum_rate(h_true, g_rzf, no)))
            r_manual['RZF'].append(manual_sinr_rate(system, h_true, g_rzf, no))

            g_wmmse = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            r_sionna['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse, no)))
            r_manual['WMMSE'].append(manual_sinr_rate(system, h_true, g_wmmse, no))

            for name, model in neural_models.items():
                g_n = model(h_est, no=no, training=False)
                r_sionna[name].append(float(system._sum_rate(h_true, g_n, no)))
                r_manual[name].append(manual_sinr_rate(system, h_true, g_n, no))

        for m in METHODS:
            sionna_out[m][data_snr] = float(np.mean(r_sionna[m]))
            manual_out[m][data_snr] = float(np.mean(r_manual[m]))
        print(f'SNR={data_snr:5.1f} done | ' +
              ' '.join(f'{m}:sionna={sionna_out[m][data_snr]:.3f}/manuel={manual_out[m][data_snr]:.3f}'
                        for m in METHODS), flush=True)

    meta = {
        'seed': SEED, 'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'h_true tire une fois par draw (deja le cas dans le script original), reutilise pour les 5 methodes ET pour le calcul du SINR manuel',
        'regime': 'umi_massive_capD384L4 (M64K8, D=384, L=4, CSI imparfait pilote 20dB)',
        'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders_w.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts, 'batch_size': BATCH, 'num_draws': NUM_DRAWS,
        'pilot_snr_db': PILOT_SNR_DB, 'config': MASSIVE_TRUE_CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'note': 'Version allegee (NUM_DRAWS=4 vs 12 original, 1 seul pilote au lieu de 2) -- confirmation representative, pas table finale',
    }
    date_tag = time.strftime('%Y%m%d_%H%M%S')
    for tag, data in [('sionna', sionna_out), ('manual_sinr', manual_out)]:
        out = {'metric': tag, 'table': data, 'metadata': meta}
        path = f'results/diag_csi_imperfect_massive_capD384L4_seedfix_{tag}_{date_tag}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Sauvegarde -> {path}')

    print("\n=== METRIQUE SIONNA ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>16}" for m in METHODS))
    for s in DATA_SNRS_DB:
        print(f"{s:6.1f} " + " ".join(f"{sionna_out[m][s]:16.3f}" for m in METHODS))
    print("\n=== SINR MANUEL ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>16}" for m in METHODS))
    for s in DATA_SNRS_DB:
        print(f"{s:6.1f} " + " ".join(f"{manual_out[m][s]:16.3f}" for m in METHODS))
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
