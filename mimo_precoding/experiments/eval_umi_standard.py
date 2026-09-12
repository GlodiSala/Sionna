"""
experiments/eval_umi_standard.py -- combine deux corrections en un seul run :

1) Meme graine + meme mecanisme que classical_comparison_seedfix.py
   (sionna_config.seed = SEED, en plus de tf.random.set_seed(SEED)).
2) UN SEUL tirage de canal (h_freq, no) par batch, PARTAGE par les 6
   methodes (RZF-SC, RZF-RB12, WMMSE, SC, IB, TA-RB) -- contrairement a
   classical_comparison_seedfix.py (v1) ou les 3 methodes neurales
   evaluaient sur un tirage separe (topologies MU_MIMO_System distinctes).

Calcule le debit de CHAQUE methode de DEUX facons, sur les MEMES canaux :
  (a) metrique Sionna standard : _sum_rate() / LMMSEPostEqualizationSINR
      (comme Table 3.9 actuelle)
  (b) SINR manuel direct : SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no)
      -- meme formule que le critere interne de WMMSE (precoders/classical.py:103-110),
      appliquee uniformement aux 6 methodes a partir de h_freq et du
      precodeur g (pas de lmmse_matrix, pas de blanchiment/regularisation).

Sauvegarde DEUX fichiers separes (meme run, memes donnees brutes, deux
metriques) :
  results/classical_comparison_M8K4_seedfix_sionna_<date>.json
  results/classical_comparison_M8K4_seedfix_manual_sinr_<date>.json

Ne modifie ni classical_reference.py ni classical_comparison_seedfix.py.

Usage: python3 experiments/eval_umi_standard.py [--num_batches N]
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED   # fix reel de config.tf_rng

from classical_reference import LockedClusterSystem, BATCH_SIZE, RB_SIZE, WMMSE_ITERS
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from precoders.classical import rzf_precoder, wmmse_precoder, _get_desired_channels
from channel_config import STANDARD_CONFIG
from system import MU_MIMO_System
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
METHODS = ['rzf_sc', 'rzf_rb12', 'wmmse', 'SC', 'IB', 'TA-RB']

CKPTS = {
    'SC':    ('single_sc',       'results/diag_front_a_signed_attn.json'),
    'IB':    ('intra_rb',        'results/diag_front_a_signed_attn_intra_rb.json'),
    'TA-RB': ('ta_rb_residual',  'results/diag_tarb_residual_signed_attn_T4_train.json'),
}


def build_neural_precoder(name):
    """Construit UNIQUEMENT le precodeur (poids charges), sans systeme
    complet -- on reutilise le canal du LockedClusterSystem partage."""
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
    import pickle
    weights_path = os.path.join(ckpt, 'weights.pkl') if os.path.isdir(ckpt) else ckpt
    with open(weights_path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(precoder.trainable_variables, ws):
        v.assign(w)
    print(f"OK poids charges pour {name} depuis {ckpt}", flush=True)
    return precoder, ckpt


def normalize_g(g):
    """Uniformise toutes les sorties de precodeur a [B,1,ofdm,fft,M,K] --
    rzf_precoder_with_rb_grouping retourne [B,ofdm,fft,M,K] (sans l'axe
    num_tx=1), les autres l'ont deja."""
    if len(g.shape) == 5:
        return tf.expand_dims(g, axis=1)
    return g


def sionna_rate(system, h_freq, g, no):
    """Metrique Table 3.9 actuelle -- ConfigurableMIMOSystem._sum_rate
    (eval_system.py:174-181)."""
    return float(system._sum_rate(h_freq, g, no))


def manual_sinr_rate(stream_management, h_freq, g, no):
    """SINR manuel direct : SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no)
    -- meme formule que precoders/classical.py:103-110, sans lmmse_matrix ni blanchiment."""
    g = normalize_g(g)
    h_pc = _get_desired_channels(h_freq, stream_management)      # [B,1,ofdm,fft,K,M]
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)          # [B,ofdm,fft,K,M]
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)             # [B,ofdm,fft,M,K]
    HW = tf.matmul(H, W)                                           # [B,ofdm,fft,K,K] ; HW[...,k,j]=h_k^H w_j
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))  # [B,ofdm,fft,K]
    Y = tf.reduce_mean(rate, axis=[1, 2])            # [B,K] -- moyenne ofdm,fft
    val = tf.reduce_sum(tf.reduce_mean(Y, axis=0))   # moyenne batch, somme users
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

    system = LockedClusterSystem(M, K)
    system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

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
        'shared_channel_note': 'h_freq/no tires UNE fois par batch (LockedClusterSystem), reutilises identiques pour les 6 methodes (rzf_sc, rzf_rb12, wmmse, SC, IB, TA-RB)',
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders/classical.py'),
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

    out_sionna = {
        'metric': 'sionna_LMMSEPostEqualizationSINR (_sum_rate, comme Table 3.9)',
        'table': sionna_table,
        'metadata': base_meta,
        'raw_per_batch': sionna_out,
    }
    path_sionna = f'results/classical_comparison_M8K4_seedfix_sionna_{date_tag}.json'
    with open(path_sionna, 'w') as f:
        json.dump(out_sionna, f, indent=2)

    out_manual = {
        'metric': 'manual_sinr (SINR_k = |h_k^H w_k|^2 / (sum_j!=k |h_k^H w_j|^2 + no), precoders/classical.py:103-110)',
        'table': manual_table,
        'metadata': base_meta,
        'raw_per_batch': manual_out,
    }
    path_manual = f'results/classical_comparison_M8K4_seedfix_manual_sinr_{date_tag}.json'
    with open(path_manual, 'w') as f:
        json.dump(out_manual, f, indent=2)

    print("\n=== METRIQUE SIONNA (_sum_rate / LMMSEPostEqualizationSINR) ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{sionna_table[m][i]:10.3f}" for m in METHODS))

    print("\n=== SINR MANUEL (formule WMMSE interne, sans lmmse_matrix) ===")
    print(f"{'SNR':>6} " + " ".join(f"{m:>10}" for m in METHODS))
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} " + " ".join(f"{manual_table[m][i]:10.3f}" for m in METHODS))

    print("\n=== ECART WMMSE - RZF-SC, cote a cote ===")
    print(f"{'SNR':>6} {'sionna':>12} {'manuel':>12}")
    for i, s in enumerate(REPORT_SNRS):
        d_sionna = sionna_table['wmmse'][i] - sionna_table['rzf_sc'][i]
        d_manual = manual_table['wmmse'][i] - manual_table['rzf_sc'][i]
        print(f"{s:6.1f} {d_sionna:+12.4f} {d_manual:+12.4f}")

    print(f"\nSauvegarde -> {path_sionna}")
    print(f"Sauvegarde -> {path_manual}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
