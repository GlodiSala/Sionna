"""
classical_comparison_massive_seedfix_v2.py -- tab:results_massive (M=64, K=8,
MASSIVE_TRUE_CONFIG, canal UMi STANDARD sans clustering force, CSI parfait),
D=128 vs D=384, meme methodologie que classical_comparison_seedfix_v2.py :

1) sionna_config.seed = SEED (fix reel de config.tf_rng)
2) UN SEUL tirage h_freq/no par batch, PARTAGE par les 6 methodes
3) double metrique : Sionna (_sum_rate/LMMSEPostEqualizationSINR) vs SINR
   manuel direct SINR_k=|h_k^Hw_k|^2/(sum_{j!=k}|h_k^Hw_j|^2+no)

Source des donnees classiques : diag_classical_comparison_massive_true.py
(RZF-Full, RZF-RB12, WMMSE, LockedClusterSystemMassive/MASSIVE_TRUE_CONFIG).
Source des checkpoints neuraux : results/diag_massive_true_{single_sc,intra_rb,
ta_rb_residual}.json (D=128) et results/diag_massive_gap_fullbudget_*_cap_d384l4.json
(D=384). TA-RB = T=6 (tokens_per_rb=6) a cette echelle, pas T=4.

Ne modifie aucun fichier existant.
"""
import os, sys, json, time, hashlib, argparse, pickle
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
sionna_config.seed = SEED

from wmmse_convergence_check import ConfigurableMIMOSystem
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from main_finall import MU_MIMO_System
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
BATCH_SIZE = 16     # comme diag_classical_comparison_massive_true.py (M=64 plus lourd)
RB_SIZE = 12
WMMSE_ITERS = 10
METHODS = ['rzf_sc', 'rzf_rb12', 'wmmse', 'SC', 'IB', 'TA-RB']

CKPTS = {
    128: {
        'SC':    'results/diag_massive_true_single_sc_extbudget.json',
        'IB':    'results/diag_massive_true_intra_rb_extbudget.json',
        'TA-RB': 'results/diag_massive_true_ta_rb_residual_extbudget.json',
    },
    384: {
        'SC':    'results/diag_massive_gap_fullbudget_single_sc_cap_d384l4.json',
        'IB':    'results/diag_massive_gap_fullbudget_intra_rb_cap_d384l4.json',
        'TA-RB': 'results/diag_massive_gap_fullbudget_ta_rb_residual_cap_d384l4.json',
    },
}


class LockedClusterSystemMassive(ConfigurableMIMOSystem):
    cluster_radius_m   = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def build_neural_precoder(name, embed_dim):
    ckpt_json = CKPTS[embed_dim][name]
    with open(ckpt_json) as f:
        ckpt = json.load(f)['best_ckpt']
    dummy_system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='single_sc',
                                   embed_dim=embed_dim, num_heads=4, num_layers=4,
                                   tokens_per_rb=6)
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=dummy_system.rg.num_ofdm_symbols,
                  fft_size=dummy_system.rg.fft_size, embed_dim=embed_dim, num_heads=4,
                  num_layers=4, snr_aware=True)
    if name == 'SC':
        precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif name == 'IB':
        precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, dummy_system.rg.num_ofdm_symbols, dummy_system.rg.fft_size],
                        dtype=tf.complex64)
    _ = precoder(dummy_h, no=tf.constant(0.01), training=False)
    weights_path = os.path.join(ckpt, 'weights.pkl') if os.path.isdir(ckpt) else ckpt
    with open(weights_path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(precoder.trainable_variables, ws):
        v.assign(w)
    print(f"OK poids charges pour {name} (D={embed_dim}) depuis {ckpt}", flush=True)
    return precoder, ckpt


def normalize_g(g):
    if len(g.shape) == 5:
        return tf.expand_dims(g, axis=1)
    return g


def manual_sinr_rate(stream_management, h_freq, g, no):
    g = normalize_g(g)
    h_pc = _get_desired_channels(h_freq, stream_management)
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


def run(embed_dim, num_batches):
    t0 = time.time()
    system = LockedClusterSystemMassive(M, K)
    precoders, ckpts = {}, {}
    for name in ['SC', 'IB', 'TA-RB']:
        precoders[name], ckpts[name] = build_neural_precoder(name, embed_dim)

    sionna_out = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}
    manual_out = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}

    for snr in REPORT_SNRS:
        snr_t = tf.constant(snr, tf.float32)
        for b in range(num_batches):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)

            g = {}
            g['rzf_sc']   = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            g['rzf_rb12'] = normalize_g(rzf_precoder_with_rb_grouping(
                                h_freq, stream_management=system.sm, rb_size=RB_SIZE))
            g['wmmse']    = wmmse_precoder(h_freq, no, system.sm, num_iterations=WMMSE_ITERS)
            for name in ['SC', 'IB', 'TA-RB']:
                g[name] = precoders[name](h_freq, no=no, training=False)

            for m in METHODS:
                sionna_out[m][snr].append(float(system._sum_rate(h_freq, g[m], no)))
                manual_out[m][snr].append(manual_sinr_rate(system.sm, h_freq, g[m], no))

        print(f"[D={embed_dim}] SNR={snr:5.1f} done | " +
              " ".join(f"{m}:{np.mean(sionna_out[m][snr]):.2f}/{np.mean(manual_out[m][snr]):.2f}"
                        for m in METHODS), flush=True)

    def build_table(out):
        t = {'snr': REPORT_SNRS}
        for m in METHODS:
            t[m] = [float(np.mean(out[m][s])) for s in REPORT_SNRS]
        return t

    sionna_table = build_table(sionna_out)
    manual_table = build_table(manual_out)

    base_meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'h_freq/no tires UNE fois par batch (LockedClusterSystemMassive), reutilises identiques pour les 6 methodes',
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders_w.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts,
        'batch_size': BATCH_SIZE,
        'num_batches': num_batches,
        'rb_size': RB_SIZE,
        'wmmse_iters': WMMSE_ITERS,
        'embed_dim': embed_dim,
        'tokens_per_rb_ta_rb': 6,
        'config': MASSIVE_TRUE_CONFIG,
        'M': M, 'K': K,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }

    date_tag = time.strftime('%Y%m%d_%H%M%S')
    out_sionna = {'metric': 'sionna_LMMSEPostEqualizationSINR (_sum_rate)',
                  'table': sionna_table, 'metadata': base_meta, 'raw_per_batch': sionna_out}
    out_manual = {'metric': 'manual_sinr (SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no))',
                  'table': manual_table, 'metadata': base_meta, 'raw_per_batch': manual_out}

    path_sionna = f'results/classical_comparison_M64K8_true_D{embed_dim}_seedfix_sionna_{date_tag}.json'
    path_manual = f'results/classical_comparison_M64K8_true_D{embed_dim}_seedfix_manual_sinr_{date_tag}.json'
    with open(path_sionna, 'w') as f:
        json.dump(out_sionna, f, indent=2)
    with open(path_manual, 'w') as f:
        json.dump(out_manual, f, indent=2)

    print(f"\n=== D={embed_dim} -- METRIQUE SIONNA ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{sionna_table[m][i]:10.3f}" for m in METHODS))
    print(f"\n=== D={embed_dim} -- SINR MANUEL ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{manual_table[m][i]:10.3f}" for m in METHODS))
    print(f"\nD={embed_dim} sauvegarde -> {path_sionna}\n              -> {path_manual}")
    print(f"D={embed_dim} time: {time.time()-t0:.0f}s")
    return path_sionna, path_manual


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_batches', type=int, default=5)
    p.add_argument('--dims', type=int, nargs='+', default=[128, 384])
    args = p.parse_args()
    for d in args.dims:
        run(d, args.num_batches)
