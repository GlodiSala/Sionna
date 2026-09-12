"""
diag_intrarb_window.py — Volet 1 / IntraRB : compare tailles de fenêtre
fréquentielle (RB=4/6/12/24 SC, non chevauchant, en réutilisant
IntraRBTransformerPrecoder tel quel -- rb_size est déjà paramétrable)
sur le canal verrouillé actuel (mesuré quasi-plat en fréquence, cf.
diag précédent).

Attente documentée : le canal verrouillé (FORCE_LOS, angle 7.5°) ne
décorrèle quasiment pas sur toute la bande (|rho|>0.80 partout) --
donc les différentes tailles de fenêtre devraient donner un sum-rate
proche ; le signal utile ici est surtout le COÛT (params/FLOPs), pas
un gain de performance.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_intrarb_window.py
"""
import os, sys, time, json, gc
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from datasets import CachedSionnaDataset
from precoder_intra_rb import IntraRBTransformerPrecoder
from main_finall import MU_MIMO_System, SNR_MIN_TRAIN, SNR_MAX_TRAIN
from sionna.phy.utils import ebnodb2no

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

NUM_TX, NUM_RX = 8, 4
N_STEPS = 400
BATCH = 64
EVAL_SNRS = [0.0, 10.0, 20.0]
RB_SIZES = [4, 6, 12, 24, 48]   # 96 doit être divisible -> tous ok (96/4=24,/6=16,/12=8,/24=4,/48=2)

_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    _sys, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)

all_results = {}
for rb_size in RB_SIZES:
    print(f'\n{"="*70}\nRB_SIZE = {rb_size} (N_RB={96//rb_size})\n{"="*70}')
    tf.random.set_seed(42)
    np.random.seed(42)

    precoder = IntraRBTransformerPrecoder(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=96,
        rb_size=rb_size, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
    precoder(dummy_h, no=tf.constant(1e-3), training=False)
    vars_ = precoder.trainable_variables
    n_params = sum(int(tf.size(v)) for v in vars_)
    flops, weights, acts = precoder.complexity(num_ofdm=14)

    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, N_STEPS, alpha=0.02)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0)
    rate_norm = float(NUM_RX) * 9.0

    @tf.function
    def train_step(h_freq):
        snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
        no = ebnodb2no(snr_db, _sys.num_bits_per_symbol, 0.5, _sys.rg)
        with tf.GradientTape() as tape:
            g = precoder(h_freq, no=no, training=True, return_real_imag=True)
            g = tf.complex(g[0], g[1])
            h_eff = _sys.ch_helper.compute_effective_channel(h_freq, g)
            sinr = _sys.lmmse_sinr(h_eff, no=no, interference_whitening=True)
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
        no = ebnodb2no(snr_db, _sys.num_bits_per_symbol, 0.5, _sys.rg)
        g = precoder(h_freq, no=no, training=False)
        h_eff = _sys.ch_helper.compute_effective_channel(h_freq, g)
        sinr = _sys.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        return tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0), axis=[0, 1, 2, 4]))

    gnorms, t0 = [], time.time()
    for step in range(N_STEPS):
        hb = dataset.get_batch(BATCH)
        loss, gnorm = train_step(hb)
        gnorms.append(float(gnorm))
        if (step + 1) % 200 == 0:
            print(f'  step {step+1}/{N_STEPS} loss={float(loss):+.4f} gn={float(gnorm):.3f} ({time.time()-t0:.0f}s)')

    eval_res = {}
    for snr in EVAL_SNRS:
        rr = [float(eval_step(dataset.get_batch(64), tf.constant(snr, tf.float32))) for _ in range(15)]
        eval_res[str(snr)] = float(np.mean(rr))

    gnorms = np.array(gnorms)
    all_results[str(rb_size)] = {
        'n_rb': 96 // rb_size, 'n_params': n_params,
        'flops_M': flops / 1e6, 'weights_K': weights / 1e3,
        'gnorm_mean_last40': float(gnorms[-40:].mean()),
        'eval_sum_rate': eval_res, 'wall_time_s': time.time() - t0,
    }
    print(f'  -> params={n_params:,} FLOPs={flops/1e6:.1f}M gn={all_results[str(rb_size)]["gnorm_mean_last40"]:.3f} '
          f'eval@0/10/20={eval_res["0.0"]:.2f}/{eval_res["10.0"]:.2f}/{eval_res["20.0"]:.2f}')

    del precoder
    tf.keras.backend.clear_session()
    gc.collect()

print(f'\n\n=== RÉSUMÉ FENÊTRAGE INTRARB ===')
print(f'{"rb_size":>8} {"N_RB":>5} {"params":>9} {"FLOPs(M)":>9} {"gn_mean":>8} | eval@0/10/20dB')
for rb, r in all_results.items():
    ev = r['eval_sum_rate']
    print(f'{rb:>8} {r["n_rb"]:>5} {r["n_params"]:>9,} {r["flops_M"]:>9.1f} {r["gnorm_mean_last40"]:>8.3f} | '
          f'{ev["0.0"]:.2f} / {ev["10.0"]:.2f} / {ev["20.0"]:.2f}')

with open('results/diag_intrarb_window.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSauvé -> results/diag_intrarb_window.json')
