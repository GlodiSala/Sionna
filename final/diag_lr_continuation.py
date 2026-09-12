"""
diag_lr_continuation.py — Bloc B.4 : le run IntraRB_fullres s'est arrêté
avec LR ~2e-5 (quasi-plancher du cosine decay sur seulement 6 époques).
Teste si reprendre l'entraînement à un LR modéré et CONSTANT (pas de
décroissance agressive) permet au sum-rate à haut SNR de continuer à
progresser -- ce qui indiquerait un plateau "budget d'entraînement /
scheduler", pas un mur structurel de l'objectif sum-rate direct.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_lr_continuation.py
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from datasets import CachedSionnaDataset
from main_finall import MU_MIMO_System, SNR_MIN_TRAIN, SNR_MAX_TRAIN
from sionna.phy.utils import ebnodb2no

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

NUM_TX, NUM_RX = 8, 4
N_STEPS = 600
BATCH = 256
EVAL_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
CONST_LR = 3e-4   # LR modéré, constant -- vs. ~2e-5 (plancher) à la fin du run réel

system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='intra_rb',
                         embed_dim=128, num_heads=4, num_layers=4)
dataset = CachedSionnaDataset(
    system, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=7)

dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
system.precoder(dummy_h, no=tf.constant(1e-3), training=False)
vars_ = system.precoder.trainable_variables
ok = system.load_weights_from('weights/IntraRB_fullres/weights.pkl')
assert ok
rate_norm = float(system.num_users) * 9.0

opt = tf.keras.optimizers.Adam(CONST_LR, clipnorm=5.0)


@tf.function
def train_step(h_freq):
    snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
    no = ebnodb2no(snr_db, system.num_bits_per_symbol, 0.5, system.rg)
    with tf.GradientTape() as tape:
        g = system._call_precoder(h_freq, no, training=True, return_real_imag=True)
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
    g = system._call_precoder(h_freq, no, training=False)
    h_eff = system.ch_helper.compute_effective_channel(h_freq, g)
    sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
    return tf.reduce_sum(tf.reduce_mean(
        tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
        axis=[0, 1, 2, 4]))


def eval_all():
    res = {}
    for snr in EVAL_SNRS:
        rr = []
        for _ in range(15):
            h = dataset.get_batch(64)
            rr.append(float(eval_step(h, tf.constant(snr, tf.float32))))
        res[str(snr)] = float(np.mean(rr))
    return res


print('=== EVAL AVANT reprise (checkpoint tel quel, fin du run réel epoch 6) ===')
eval_before = eval_all()
for k, v in eval_before.items():
    print(f'  SNR={k:>5} : {v:.3f}')

print(f'\n=== Reprise entraînement : {N_STEPS} steps, LR CONSTANT={CONST_LR:.1e}, batch={BATCH} ===')
t0 = time.time()
for step in range(N_STEPS):
    h = dataset.get_batch(BATCH)
    loss, gnorm = train_step(h)
    if (step + 1) % 100 == 0:
        print(f'  step {step+1:4d}/{N_STEPS} | loss={float(loss):+.4f} | gn={float(gnorm):.3f} | '
              f'{time.time()-t0:.0f}s')

print('\n=== EVAL APRÈS reprise ===')
eval_after = eval_all()
for k, v in eval_after.items():
    print(f'  SNR={k:>5} : {v:.3f}')

print('\n=== DELTA (après - avant) ===')
for k in eval_before:
    d = eval_after[k] - eval_before[k]
    print(f'  SNR={k:>5} : {eval_before[k]:.3f} -> {eval_after[k]:.3f}  ({d:+.3f}, {100*d/eval_before[k]:+.1f}%)')

with open('results/diag_lr_continuation.json', 'w') as f:
    json.dump({'before': eval_before, 'after': eval_after, 'n_steps': N_STEPS,
                'const_lr': CONST_LR, 'batch': BATCH}, f, indent=2)
print('\nSauvé -> results/diag_lr_continuation.json')
