"""
classical_comparison_uma_seedfix_v2.py -- POINT 1 (Table 3.10, UMa, M8K4,
CSI parfait) : meme methodologie que classical_comparison_seedfix_v2.py
(UMi), adaptee au canal UMa.

Corrections combinees :
1) sionna_config.seed = SEED -- fix reel de config.tf_rng (utilise par
   gen_topology_clustered dans channel_config.py, non seede par
   tf.random.set_seed).
2) UN SEUL tirage h_freq/no par batch, partage par les 6 methodes
   (rzf_sc, rzf_rb12, wmmse, SC, IB, TA-RB) -- systeme UMaLockedClusterSystem
   (mirroir exact de diag_classical_comparison_uma.py::UMaLockedClusterSystem).
3) Debit calcule par DEUX metriques sur les MEMES canaux :
   (a) metrique Sionna standard (_sum_rate/LMMSEPostEqualizationSINR)
   (b) SINR manuel direct SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2+no)
       (meme formule que le critere interne de WMMSE, precoders_w.py:103-110)

num_rx_ant=1 confirme (ut_array = AntennaArray(num_rows=1,num_cols=1,...),
wmmse_convergence_check.py:130-133, herite tel quel par UMaLockedClusterSystem)
-- meme mecanisme structurel de double-comptage bruit+interference que Table
3.9 (UMi) s'applique donc ici de la meme facon.

Checkpoints UMa (confirmes via figA_sumrate_csi_perfect.py::ARCH_KEY,
results/figures_final/uma_standard/) :
  SC    -> weights/UMa_SingleSC_signed_attn_4L_128d/best_20260808_231408
  IB    -> weights/UMa_IntraRB_signed_attn_4L_128d/best_20260809_005458
  TA-RB -> weights/UMa_TA_RB_residual_signed_attn_4tok_4L_128d/best_20260809_001031
  (T=4, meme que UMi -- confirme, pas suppose)

Ne modifie ni classical_comparison.py ni diag_classical_comparison_uma.py.

Usage: python3 classical_comparison_uma_seedfix_v2.py [--num_batches N]
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
sionna_config.seed = SEED   # fix reel de config.tf_rng

from sionna.phy.channel.tr38901 import UMa
from wmmse_convergence_check import ConfigurableMIMOSystem
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from channel_config import STANDARD_CONFIG, gen_topology_clustered
from main_finall import MU_MIMO_System
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
BATCH_SIZE = 32
RB_SIZE = 12
WMMSE_ITERS = 10
METHODS = ['rzf_sc', 'rzf_rb12', 'wmmse', 'SC', 'IB', 'TA-RB']

CKPTS = {
    'SC':    ('single_sc',      'results/diag_uma_signed_attn_single_sc.json'),
    'IB':    ('intra_rb',       'results/diag_uma_signed_attn_intra_rb.json'),
    'TA-RB': ('ta_rb_residual', 'results/diag_uma_signed_attn_ta_rb_residual.json'),
}


class UMaLockedClusterSystem(ConfigurableMIMOSystem):
    """Mirroir exact de diag_classical_comparison_uma.py::UMaLockedClusterSystem."""
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


def build_neural_precoder(name):
    arch, json_path = CKPTS[name]
    with open(json_path) as f:
        ckpt = json.load(f)['best_ckpt']
    dummy_system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch, embed_dim=128,
                                   num_heads=4, num_layers=4, tokens_per_rb=4)
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=dummy_system.rg.num_ofdm_symbols,
                  fft_size=dummy_system.rg.fft_size, embed_dim=128, num_heads=4,
                  num_layers=4, snr_aware=True)
    if name == 'SC':
        precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif name == 'IB':
        precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=4, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, dummy_system.rg.num_ofdm_symbols, dummy_system.rg.fft_size],
                        dtype=tf.complex64)
    _ = precoder(dummy_h, no=tf.constant(0.01), training=False)
    weights_path = os.path.join(ckpt, 'weights.pkl') if os.path.isdir(ckpt) else ckpt
    with open(weights_path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(precoder.trainable_variables, ws):
        v.assign(w)
    print(f"OK poids charges pour {name} depuis {ckpt}", flush=True)
    return precoder, ckpt


def normalize_g(g):
    if len(g.shape) == 5:
        return tf.expand_dims(g, axis=1)
    return g


def sionna_rate(system, h_freq, g, no):
    return float(system._sum_rate(h_freq, g, no))


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
    val = tf.reduce_sum(tf.reduce_mean(Y, axis=0))
    return float(val)


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num_batches', type=int, default=10)
    args = p.parse_args()
    nb = args.num_batches
    t0 = time.time()

    system = UMaLockedClusterSystem(M, K)

    precoders = {}
    ckpts = {}
    for name in ['SC', 'IB', 'TA-RB']:
        precoders[name], ckpts[name] = build_neural_precoder(name)

    sionna_out = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}
    manual_out = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}

    for snr in REPORT_SNRS:
        snr_t = tf.constant(snr, tf.float32)
        for b in range(nb):
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
                sionna_out[m][snr].append(sionna_rate(system, h_freq, g[m], no))
                manual_out[m][snr].append(manual_sinr_rate(system.sm, h_freq, g[m], no))

        print(f"SNR={snr:5.1f} done | " +
              " ".join(f"{m}:sionna={np.mean(sionna_out[m][snr]):.3f}/manuel={np.mean(manual_out[m][snr]):.3f}"
                        for m in METHODS), flush=True)

    def build_table(out):
        table = {'snr': REPORT_SNRS}
        for m in METHODS:
            table[m] = [float(np.mean(out[m][s])) for s in REPORT_SNRS]
        return table

    sionna_table = build_table(sionna_out)
    manual_table = build_table(manual_out)

    base_meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'h_freq/no tires UNE fois par batch (UMaLockedClusterSystem), reutilises identiques pour les 6 methodes',
        'scenario': 'UMa',
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders_w.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts,
        'batch_size': BATCH_SIZE,
        'num_batches': nb,
        'rb_size': RB_SIZE,
        'wmmse_iters': WMMSE_ITERS,
        'config': STANDARD_CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }

    date_tag = time.strftime('%Y%m%d_%H%M%S')

    out_sionna = {'metric': 'sionna_LMMSEPostEqualizationSINR (_sum_rate, comme Table 3.10)',
                  'table': sionna_table, 'metadata': base_meta, 'raw_per_batch': sionna_out}
    path_sionna = f'results/classical_comparison_M8K4_uma_seedfix_sionna_{date_tag}.json'
    with open(path_sionna, 'w') as f:
        json.dump(out_sionna, f, indent=2)

    out_manual = {'metric': 'manual_sinr (SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no), precoders_w.py:103-110)',
                  'table': manual_table, 'metadata': base_meta, 'raw_per_batch': manual_out}
    path_manual = f'results/classical_comparison_M8K4_uma_seedfix_manual_sinr_{date_tag}.json'
    with open(path_manual, 'w') as f:
        json.dump(out_manual, f, indent=2)

    print("\n=== METRIQUE SIONNA (_sum_rate / LMMSEPostEqualizationSINR) -- UMa ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{sionna_table[m][i]:10.3f}" for m in METHODS))

    print("\n=== SINR MANUEL -- UMa ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{manual_table[m][i]:10.3f}" for m in METHODS))

    print(f"\nSauvegarde -> {path_sionna}")
    print(f"Sauvegarde -> {path_manual}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
