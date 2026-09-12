"""
check_massive_d128_repro.py -- reproduction exacte de l'evaluation
d'origine pour SC/IB/TA-RB (D=128, massif M64K8), en reutilisant le MEME
jeu de donnees cache (CachedSionnaDataset, meme cache_file, meme seed=42
passe au constructeur -- charge le fichier .npz existant, ne regenere
rien) et le MEME protocole d'evaluation (eval_at_snr, EVAL_BATCHES=20,
batch_size=64) que diag_massive_true_train.py -- teste directement les
deux hypotheses (nombre de batches + tirage de canal) en une fois.

Ne modifie aucun fichier existant.
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main_finall import MU_MIMO_System
from datasets import CachedSionnaDataset
from channel_config import MASSIVE_TRUE_CONFIG
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no
import pickle

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
DATASET_SIZE = 4000
CACHE_FILE = f'/export/tmp/sala/sionna_joint_massive_true_{DATASET_SIZE//1000}k_{M}x{K}.npz'
EVAL_SNRS = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES = 20   # == diag_massive_true_train.py

CKPT_EXT = {
    'SC':    ('results/diag_massive_true_single_sc_extbudget.json', 'single_sc'),
    'IB':    ('results/diag_massive_true_intra_rb_extbudget.json', 'intra_rb'),
    'TA-RB': ('results/diag_massive_true_ta_rb_residual_extbudget.json', 'ta_rb_residual'),
}
PUBLISHED_SC = {10.0: 72.2, 15.0: 80.9, 20.0: 86.0}


def eval_at_snr(system, dataset, snr_db, num_batches, batch_size=64):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    rates = []
    for _ in range(num_batches):
        h = dataset.get_batch(batch_size)
        g = system._call_precoder(h, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))
        rates.append(float(rate))
    return float(np.mean(rates))


def build_precoder(arch, ckpt, embed_dim=128):
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')  # dummy, juste pour rg
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols,
                  fft_size=system.rg.fft_size, embed_dim=embed_dim, num_heads=4,
                  num_layers=4, snr_aware=True)
    if arch == 'single_sc':
        precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif arch == 'intra_rb':
        precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
    _ = precoder(dummy_h, no=tf.constant(0.01), training=False)
    weights_path = os.path.join(ckpt, 'weights.pkl') if os.path.isdir(ckpt) else ckpt
    with open(weights_path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(precoder.trainable_variables, ws):
        v.assign(w)
    system.precoder = precoder
    return system


t0 = time.time()
print(f"Chargement du dataset cache : {CACHE_FILE}")
print(f"Existe deja : {os.path.exists(CACHE_FILE)}")
dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
dataset = CachedSionnaDataset(
    dummy, dataset_size=DATASET_SIZE, batch_size=32, cache_file=CACHE_FILE,
    cluster_radius_m=MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M'],
    indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'], seed=42)

results = {}
for name, (json_path, arch) in CKPT_EXT.items():
    with open(json_path) as f:
        ckpt = json.load(f)['best_ckpt']
    system = build_precoder(arch, ckpt)
    print(f"\n=== {name} (arch={arch}) depuis {ckpt} ===", flush=True)
    results[name] = {}
    for snr in EVAL_SNRS:
        r = eval_at_snr(system, dataset, snr, num_batches=EVAL_BATCHES, batch_size=64)
        results[name][snr] = r
        print(f"  SNR={snr:5.1f} -> {r:.4f}", flush=True)

print("\n=== COMPARAISON A LA REPRODUCTION EXACTE (meme dataset cache, meme EVAL_BATCHES=20, batch=64) ===")
print(f"{'SNR':>6} {'SC_repro':>10} {'SC_publie':>10} {'IB_repro':>10} {'TA-RB_repro':>12}")
for snr in EVAL_SNRS:
    pub = PUBLISHED_SC.get(snr, float('nan'))
    print(f"{snr:6.1f} {results['SC'][snr]:10.4f} {pub:10.4f} {results['IB'][snr]:10.4f} {results['TA-RB'][snr]:12.4f}")

out = {'results': results, 'published_SC': PUBLISHED_SC, 'eval_batches': EVAL_BATCHES,
       'batch_size': 64, 'cache_file': CACHE_FILE, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z')}
with open('results/check_massive_d128_repro_result.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegarde -> results/check_massive_d128_repro_result.json")
print(f"Total time: {time.time()-t0:.0f}s")
