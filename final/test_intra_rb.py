"""
test_intra_rb.py — Validation rapide IntraRBTransformerPrecoder
Lance avec : sbatch -J test_intra_rb sionna_copy.sh test_intra_rb.py
"""
import matplotlib as plt
import numpy as np
import tensorflow as tf
from sionna.phy.utils import ebnodb2no

from precoder_intra_rb import IntraRBTransformerPrecoder
from main_finall import MU_MIMO_System, SupervisedTrainer, rzf_precoder
from datasets import CachedSionnaDataset

NUM_TX, NUM_RX = 8, 4
SEED = 42
tf.random.set_seed(SEED)

print("\n" + "="*60)
print("  TEST INTRA-RB TRANSFORMER PRECODER")
print("="*60)

# ── 1. Test shapes ────────────────────────────────────────────────────────────
print("\n[1/4] Test shapes...")

model = IntraRBTransformerPrecoder(
    num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
    rb_size=12, embed_dim=128, num_heads=4, num_layers=4,
    snr_aware=True)

B = 4
h_dummy = tf.complex(
    tf.random.normal([B, 4, 1, 1, 8, 14, 72]),
    tf.random.normal([B, 4, 1, 1, 8, 14, 72]))
no_dummy = tf.constant(1e-2, tf.float32)

g = model(h_dummy, no=no_dummy, training=False)
print(f"  Input  h_freq : {h_dummy.shape}")
print(f"  Output g      : {g.shape}")
assert g.shape == (B, 1, 14, 72, 8, 4), f"❌ Shape inattendu : {g.shape}"
print("  ✅ Shape correct : [B, 1, 14, 72, 8, 4]")

# ── 2. Test norme de puissance ────────────────────────────────────────────────
print("\n[2/4] Test power normalization...")
g_sq = tf.squeeze(g, axis=1)   # [B, ofdm, fft, M, K]
# Pour chaque (batch, ofdm, fft, user) → norme sur M doit être ~1
norms = tf.abs(tf.reduce_sum(tf.abs(g_sq)**2, axis=3))  # [B, ofdm, fft, K]
mean_norm = float(tf.reduce_mean(norms))
print(f"  Norme moyenne par user/SC : {mean_norm:.4f}  (attendu ≈ 1.0)")
assert abs(mean_norm - 1.0) < 0.01, f"❌ Norme incorrecte : {mean_norm}"
print("  ✅ Power norm correcte")

# ── 3. Complexité ─────────────────────────────────────────────────────────────
print("\n[3/4] Complexité...")
flops, weights, acts = model.complexity(num_ofdm=14)
print(f"  FLOPs      : {flops/1e9:.2f} G")
print(f"  Paramètres : {weights/1e6:.2f} M")
print(f"  Activations: {acts/1e6:.2f} M")
n_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"  Params réels (tf.size) : {n_params:,}")

# ── 4. Mini-entraînement (5 iters warmup + 5 iters finetune) ─────────────────
print("\n[4/4] Mini-entraînement (sanity check)...")

system = MU_MIMO_System(
    num_tx=NUM_TX, num_rx=NUM_RX,
    precoder_type='intra_rb',
    embed_dim=64, num_heads=4, num_layers=2)  # petit pour aller vite

# Dataset minimal en mémoire (pas de cache fichier)
dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    dummy_sys,
    dataset_size=200,
    batch_size=64,
    cache_file=f'/export/tmp/sala/sionna_test_200_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=5,
    seed=SEED)

# SNR test
no_test = ebnodb2no(tf.constant(15.0), system.num_bits_per_symbol,
                    0.5, system.rg)

# Warmup : 5 iters MSE
print("  Warmup MSE vs RZF...")
opt = tf.keras.optimizers.Adam(1e-3)
vars_ = system.precoder.trainable_variables

for i in range(5):
    h = dataset.get_batch(32)
    no = ebnodb2no(
        tf.random.uniform([], 5.0, 25.0),
        system.num_bits_per_symbol, 0.5, system.rg)
    g_rzf = tf.stop_gradient(rzf_precoder(h, system.sm, no=no))
    with tf.GradientTape() as tape:
        g_pred = system._call_precoder(h, no, training=True,
                                        return_real_imag=True)
        loss = tf.reduce_mean(tf.abs(g_pred - g_rzf)**2)
    grads = tape.gradient(loss, vars_)
    grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
             for g in grads]
    opt.apply_gradients(zip(grads, vars_))
    print(f"    iter {i+1}/5 | MSE loss = {float(loss):.4f}")

# Finetune : 5 iters sum-rate
print("  Finetune sum-rate...")
for i in range(5):
    h = dataset.get_batch(32)
    no = ebnodb2no(
        tf.random.uniform([], 5.0, 25.0),
        system.num_bits_per_symbol, 0.5, system.rg)
    with tf.GradientTape() as tape:
        g = system._call_precoder(h, no, training=True, return_real_imag=True)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr  = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate  = tf.reduce_sum(
            tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
                / tf.math.log(2.0),
                axis=[0, 1, 2, 4]))
        loss = -tf.where(tf.math.is_finite(rate), rate / 36.0,
                         tf.constant(0.0))
    grads = tape.gradient(loss, vars_)
    grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
             for g in grads]
    opt.apply_gradients(zip(grads, vars_))
    print(f"    iter {i+1}/5 | sum-rate = {float(rate):.3f} bps/Hz")

print("\n" + "="*60)
print("  ✅ TOUS LES TESTS PASSÉS")
print("="*60 + "\n")