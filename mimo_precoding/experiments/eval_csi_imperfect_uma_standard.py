"""
experiments/eval_csi_imperfect_uma_standard.py — panneau CSI imparfait du
régime **UMa standard (M=8, K=4)**, seul trou de la matrice régime × condition
CSI : tous les autres régimes (UMi standard, UMi/UMa massive) avaient leurs
deux conditions, UMa standard n'avait que le CSI parfait.

Réplique exacte de experiments/eval_csi_imperfect_umi_standard.py — même
protocole, même graine, mêmes deux métriques — avec trois substitutions et
rien d'autre :
  1. canal UMa au lieu d'UMi : `UMaLockedClusterSystem`, copie conforme de
     celui de experiments/eval_uma_standard.py (UMa + gen_topology_clustered
     avec scenario='uma', los=False) ;
  2. checkpoints UMa (`results/diag_uma_signed_attn_*.json`) au lieu des
     checkpoints UMi ;
  3. **TA-RB à T=4** — c'est le point de fonctionnement du checkpoint UMa
     standard (`UMa_TA_RB_residual_signed_attn_4tok_4L_128d`), alors que le
     CSI imparfait UMi utilise T=6. Les deux régimes ne comparent donc pas le
     même TA-RB ; c'est déjà le cas pour le CSI parfait (§6.4 du README).

Protocole : le précodeur voit h_est (canal bruité, pilote 20 dB, bruit LS
gaussien), le débit est calculé sur h_true (vrai canal). Un seul tirage de
canal par draw, partagé par les 5 méthodes. Débit rapporté par les deux
métriques : Sionna (`_sum_rate`) et SINR manuel direct.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/eval_csi_imperfect_uma_standard.py
"""
import os, sys, json, pickle, hashlib, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 2026   # meme graine que le pendant UMi (eval_csi_imperfect_umi_standard.py)
tf.random.set_seed(SEED)
np.random.seed(SEED)

from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from sionna.phy.channel.tr38901 import UMa
from sionna.phy.utils import ebnodb2no
from eval_system import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, gen_topology_clustered
from precoders.classical import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_DRAWS = 20
BATCH = 32
TOKENS_PER_RB = 4          # checkpoint UMa standard (le pendant UMi est a T=6)

RESULT_JSONS = {
    'SingleSC':       'results/diag_uma_signed_attn_single_sc.json',
    'IntraRB':        'results/diag_uma_signed_attn_intra_rb.json',
    'TA_RB_residual': 'results/diag_uma_signed_attn_ta_rb_residual.json',
}
NAME_MAP = {'SingleSC': 'SC', 'IntraRB': 'IB', 'TA_RB_residual': 'TA-RB'}


class UMaLockedClusterSystem(ConfigurableMIMOSystem):
    """Copie conforme de experiments/eval_uma_standard.py::UMaLockedClusterSystem."""
    cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

    def __init__(self, num_tx, num_rx):
        super().__init__(num_tx, num_rx)
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        topology = gen_topology_clustered(
            batch_size, self.num_rx, 'uma', self.cluster_radius_m,
            indoor_probability=self.indoor_probability)
        self.channel_model.set_topology(*topology, los=False)


def resolve_ckpt(name):
    with open(RESULT_JSONS[name]) as f:
        ckpt = json.load(f)['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
    return ckpt


def build_neural(kind, num_ofdm, fft_size):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=num_ofdm, fft_size=fft_size,
                  embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=TOKENS_PER_RB, **kwargs)
    raise ValueError(kind)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    path = os.path.join(ckpt_dir, 'weights.pkl') if os.path.isdir(ckpt_dir) else ckpt_dir
    with open(path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    sigma2 = 1.0 / 10.0 ** (pilot_snr_db / 10.0)
    nre = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    nim = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(nre, nim), h_freq.dtype)


def manual_sinr_rate(stream_management, h_true, g, no):
    """SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no), evalue sur h_true."""
    if len(g.shape) == 5:
        g = tf.expand_dims(g, axis=1)
    h_pc = _get_desired_channels(h_true, stream_management)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal_pwr = tf.abs(tf.linalg.diag_part(HW)) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    sinr = signal_pwr / (interference_pwr + tf.cast(tf.reshape(no, []), tf.float64))
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    return float(tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(rate, axis=[1, 2]), axis=0)))


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    system = UMaLockedClusterSystem(M, K)
    num_ofdm, fft_size = system.rg.num_ofdm_symbols, system.rg.fft_size
    dummy_h = tf.zeros([1, K, 1, 1, M, num_ofdm, fft_size], dtype=tf.complex64)

    neural_models, ckpts = {}, {}
    no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    for name in RESULT_JSONS:
        ckpts[name] = resolve_ckpt(name)
        neural_models[name] = load_weights(build_neural(name, num_ofdm, fft_size),
                                           ckpts[name], dummy_h, no0)
        print(f'OK {name} charge depuis {ckpts[name]}', flush=True)

    rng = np.random.RandomState(SEED)
    methods = ['RZF', 'WMMSE'] + list(RESULT_JSONS)
    sionna_res = {m: {} for m in methods}
    manual_res = {m: {} for m in methods}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_s = {m: [] for m in methods}
        r_m = {m: [] for m in methods}
        for _ in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32),
                                              tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g = rzf_precoder(h_est, stream_management=system.sm, no=no)
            r_s['RZF'].append(float(system._sum_rate(h_true, g, no)))
            r_m['RZF'].append(manual_sinr_rate(system.sm, h_true, g, no))

            g = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            r_s['WMMSE'].append(float(system._sum_rate(h_true, g, no)))
            r_m['WMMSE'].append(manual_sinr_rate(system.sm, h_true, g, no))

            for name, model in neural_models.items():
                g = model(h_est, no=no, training=False)
                r_s[name].append(float(system._sum_rate(h_true, g, no)))
                r_m[name].append(manual_sinr_rate(system.sm, h_true, g, no))

        for m in methods:
            sionna_res[m][str(data_snr)] = float(np.mean(r_s[m]))
            manual_res[m][str(data_snr)] = float(np.mean(r_m[m]))
        print(f"SNR={data_snr} | " + " ".join(
            f"{m}:{sionna_res[m][str(data_snr)]:.3f}/{manual_res[m][str(data_snr)]:.3f}"
            for m in methods), flush=True)

    def to_table(res):
        return {'snr': DATA_SNRS_DB,
                **{NAME_MAP.get(m, m): [res[m][str(s)] for s in DATA_SNRS_DB] for m in methods}}

    meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'h_true tire une fois par draw, reutilise pour les 5 methodes ; '
                                'precodeur calcule sur h_est (pilote 20dB), debit score sur h_true',
        'scenario': 'UMa', 'regime': 'uma_standard_M8K4',
        'tokens_per_rb_ta_rb': TOKENS_PER_RB,
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders/classical.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts, 'batch_size': BATCH, 'num_draws': NUM_DRAWS,
        'pilot_snr_db': PILOT_SNR_DB, 'config': STANDARD_CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    tag = time.strftime('%Y%m%d_%H%M%S')
    for res, metric, suffix in [
            (sionna_res, 'sionna_LMMSEPostEqualizationSINR (_sum_rate)', 'sionna'),
            (manual_res, 'manual_sinr (precoders/classical.py, evalue sur h_true)', 'manual_sinr')]:
        p = f'results/diag_csi_imperfect_uma_standard_seedfix_{suffix}_{tag}.json'
        with open(p, 'w') as f:
            json.dump({'metric': metric, 'table': to_table(res), 'metadata': meta, 'raw': res}, f, indent=2)
        print(f'Sauve -> {p}')
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
