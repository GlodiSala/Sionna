"""
diag_v2_smoke.py — Validation smoke de TransformerPrecoderClean (fix OFDM +
décodeur v4.2 sans gate) : correction du forward, mémoire à batch=256,
PUIS ablation individuelle mean/slope/var/covariance (400 steps chacune,
sum-rate direct, même protocole qu'hier pour comparabilité directe).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_v2_smoke.py
"""
import os, sys, time, json, gc, itertools
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from datasets import CachedSionnaDataset
from precoders_v2 import TransformerPrecoderClean
from main_finall import MU_MIMO_System, SNR_MIN_TRAIN, SNR_MAX_TRAIN
from sionna.phy.utils import ebnodb2no

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

NUM_TX, NUM_RX = 8, 4
N_STEPS = 400
BATCH = 32   # même protocole qu'hier pour comparer aux chiffres v4.0-v4.3
EVAL_SNRS = [0.0, 10.0, 20.0]

_dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    _dummy_sys, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)

# ── 1. Validation shape + mémoire à batch=256 ───────────────────────────
print(f'\n{"="*70}\nVALIDATION FORWARD + MÉMOIRE (batch=256)\n{"="*70}')
m = TransformerPrecoderClean(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=96,
                              rb_size=12, tokens_per_rb=3, embed_dim=128,
                              num_heads=4, num_layers=4)
h = dataset.get_batch(256)
tf.config.experimental.reset_memory_stats('GPU:0')
with tf.GradientTape() as tape:
    w_re, w_im = m(h, no=tf.constant(1e-3), training=True, return_real_imag=True)
    loss = tf.reduce_mean(w_re**2 + w_im**2)
grads = tape.gradient(loss, m.trainable_variables)
_ = [tf.reduce_sum(g) for g in grads if g is not None]
info = tf.config.experimental.get_memory_info('GPU:0')
n_params = sum(int(tf.size(v)) for v in m.trainable_variables)
print(f'shape w_re: {w_re.shape}  (attendu [256,1,14,96,8,4])')
print(f'params: {n_params:,}')
print(f'peak mémoire à batch=256: {info["peak"]/1e6:.1f} MB')
del m
tf.keras.backend.clear_session()
gc.collect()

# ── 2. Ablation features (mean/slope/var/cov), 400 steps, batch=32 ─────
print(f'\n{"="*70}\nABLATION FEATURES (400 steps, batch=32, sum-rate direct)\n{"="*70}')

configs = {
    'full (mean+slope+var+cov)': dict(mean=True, slope=True, var=True, cov=True),
    'no_mean':  dict(mean=False, slope=True,  var=True,  cov=True),
    'no_slope': dict(mean=True,  slope=False, var=True,  cov=True),
    'no_var':   dict(mean=True,  slope=True,  var=False, cov=True),
    'no_cov':   dict(mean=True,  slope=True,  var=True,  cov=False),
    'mean_only':dict(mean=True,  slope=False, var=False, cov=False),
}

_dummy_sys2 = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')

all_results = {}
for name, flags in configs.items():
    print(f'\n--- {name} ---')
    tf.random.set_seed(42)
    np.random.seed(42)

    system = _dummy_sys2  # réutilise rg/sm/ch_helper/lmmse_sinr (pas de poids dedans)
    precoder = TransformerPrecoderClean(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=96,
                                         rb_size=12, tokens_per_rb=3, embed_dim=128,
                                         num_heads=4, num_layers=4, feature_flags=flags)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
    precoder(dummy_h, no=tf.constant(1e-3), training=False)
    vars_ = precoder.trainable_variables
    n_params = sum(int(tf.size(v)) for v in vars_)

    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, N_STEPS, alpha=0.02)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0)
    rate_norm = float(NUM_RX) * 9.0

    @tf.function
    def train_step(h_freq):
        snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
        no = ebnodb2no(snr_db, system.num_bits_per_symbol, 0.5, system.rg)
        with tf.GradientTape() as tape:
            g = precoder(h_freq, no=no, training=True, return_real_imag=True)
            g = tf.complex(g[0], g[1])
            h_eff = system.ch_helper.compute_effective_channel(h_freq, g)
            sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate = tf.reduce_sum(tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
                axis=[0, 1, 2, 4]))
            loss = -tf.where(tf.math.is_finite(rate), rate / rate_norm, tf.constant(0.0))
        grads = tape.gradient(loss, vars_)
        grads = [tf.where(tf.math.is_finite(g_), g_, tf.zeros_like(g_)) for g_ in grads]
        gnorm = tf.linalg.global_norm(grads)
        grads_c, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads_c, vars_))
        return loss, gnorm

    @tf.function
    def eval_step(h_freq, snr_db):
        no = ebnodb2no(snr_db, system.num_bits_per_symbol, 0.5, system.rg)
        g = precoder(h_freq, no=no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h_freq, g)
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        return tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0), axis=[0, 1, 2, 4]))

    gnorms, t0 = [], time.time()
    for step in range(N_STEPS):
        hb = dataset.get_batch(BATCH)
        loss, gnorm = train_step(hb)
        gnorms.append(float(gnorm))

    eval_res = {}
    for snr in EVAL_SNRS:
        rr = [float(eval_step(dataset.get_batch(64), tf.constant(snr, tf.float32))) for _ in range(10)]
        eval_res[str(snr)] = float(np.mean(rr))

    gnorms = np.array(gnorms)
    all_results[name] = {
        'feat_dim': precoder.feat_dim, 'n_params': n_params,
        'gnorm_mean_last40': float(gnorms[-40:].mean()), 'gnorm_max': float(gnorms.max()),
        'eval_sum_rate': eval_res, 'wall_time_s': time.time() - t0,
    }
    print(f'  feat_dim={precoder.feat_dim} params={n_params:,} '
          f'gn_mean={all_results[name]["gnorm_mean_last40"]:.3f} gn_max={all_results[name]["gnorm_max"]:.3f} '
          f'eval@0/10/20={eval_res["0.0"]:.2f}/{eval_res["10.0"]:.2f}/{eval_res["20.0"]:.2f} '
          f'({time.time()-t0:.0f}s)')

    del precoder
    tf.keras.backend.clear_session()
    gc.collect()

print(f'\n\n=== RÉSUMÉ ABLATION FEATURES ===')
print(f'{"config":<28} {"feat_dim":>8} {"params":>9} {"gn_mean":>8} {"gn_max":>8} | eval@0/10/20dB')
for name, r in all_results.items():
    ev = r['eval_sum_rate']
    print(f'{name:<28} {r["feat_dim"]:>8} {r["n_params"]:>9,} {r["gnorm_mean_last40"]:>8.3f} '
          f'{r["gnorm_max"]:>8.3f} | {ev["0.0"]:.2f} / {ev["10.0"]:.2f} / {ev["20.0"]:.2f}')

with open('results/diag_v2_feature_ablation.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSauvé -> results/diag_v2_feature_ablation.json')
