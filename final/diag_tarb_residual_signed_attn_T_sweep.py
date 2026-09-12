"""
diag_tarb_residual_signed_attn_T_sweep.py — T-sweep (demande utilisateur)
pour TA-RB-résiduel-signed_attn (l'architecture qui a produit le résultat
final P2, PAS l'ancienne TransformerPrecoderClean). Réplique la
méthodologie de diag_tarb_T_sweep.py (500 pas sum-rate direct, pas de
warmup MSE séparé -- sweep directionnel/comparatif, pas le protocole de
production complet) MAIS corrige un problème trouvé en le reprenant :
l'original utilisait encore l'ANCIEN dataset mono-user+recombinaison
(`sionna_base_5k_8x4.npz`, `augmentation_multiplier=50`) -- exactement le
mécanisme cassé par le bug §0 (SESSION_NUIT_RESUME.md), silencieusement
détruit la corrélation de cluster. Corrigé ici : cache de production
joint-cluster actuel (`sionna_joint_20k_8x4.npz`, `CachedSionnaDataset`
sans augmentation_multiplier).

T doit diviser rb_size=12 -- T=[1,2,3,4,6,12] tous valides.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_tarb_residual_signed_attn_T_sweep.py
"""
import os, sys, time, json, gc
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from datasets import CachedSionnaDataset
from precoder_experimental import TransformerPrecoderCleanResidualSignedAttn
from main_finall import MU_MIMO_System, CHOSEN_CONFIG, DATASET_SIZE, NUM_TX, NUM_RX, SNR_MIN_TRAIN, SNR_MAX_TRAIN
from sionna.phy.utils import ebnodb2no

N_STEPS = 500
BATCH = 128
EVAL_SNRS = [0.0, 10.0, 20.0]
T_VALUES = [1, 2, 3, 4, 6, 12]

_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    _sys, dataset_size=DATASET_SIZE, batch_size=128,
    cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
    cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
    indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

all_results = {}
for T in T_VALUES:
    print(f'\n{"="*70}\nT = {T} tokens/RB (sc_per_token={12//T})\n{"="*70}')
    tf.random.set_seed(42)
    np.random.seed(42)

    precoder = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=96,
        rb_size=12, tokens_per_rb=T, embed_dim=128, num_heads=4, num_layers=4)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
    precoder(dummy_h, no=tf.constant(1e-3), training=False)
    vars_ = precoder.trainable_variables
    n_params_real = sum(int(tf.size(v)) for v in vars_)
    flops_conv, w_, a_ = precoder.complexity(num_ofdm=14, convention_x_ofdm=True)
    flops_real, _, _ = precoder.complexity(num_ofdm=14, convention_x_ofdm=False)
    assert int(w_) == n_params_real, f"T={T}: poids complexity()={int(w_)} != réel={n_params_real}"

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

    eval_res = {}
    for snr in EVAL_SNRS:
        rr = [float(eval_step(dataset.get_batch(64), tf.constant(snr, tf.float32))) for _ in range(10)]
        eval_res[str(snr)] = float(np.mean(rr))

    gnorms = np.array(gnorms)
    all_results[str(T)] = {
        'n_params': n_params_real, 'flops_conv_M': flops_conv/1e6, 'flops_real_M': flops_real/1e6,
        'gnorm_mean_last40': float(gnorms[-40:].mean()), 'gnorm_max': float(gnorms.max()),
        'eval_sum_rate': eval_res, 'wall_time_s': time.time() - t0,
    }
    print(f'  -> params={n_params_real:,} FLOPs_reel={flops_real/1e6:.1f}M gn={all_results[str(T)]["gnorm_mean_last40"]:.3f} '
          f'eval@0/10/20={eval_res["0.0"]:.2f}/{eval_res["10.0"]:.2f}/{eval_res["20.0"]:.2f}')

    with open('results/diag_tarb_residual_signed_attn_T_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2)   # sauvé après CHAQUE T -- reprenable

    del precoder
    tf.keras.backend.clear_session()
    gc.collect()

print(f'\n\n=== RÉSUMÉ SWEEP T (TA-RB-résiduel-signed_attn, batch=128, {N_STEPS} pas) ===')
print(f'{"T":>3} {"params":>9} {"FLOPs_reel(M)":>13} {"gn_mean":>8} | eval@0/10/20dB')
for T, r in all_results.items():
    ev = r['eval_sum_rate']
    print(f'{T:>3} {r["n_params"]:>9,} {r["flops_real_M"]:>13.1f} {r["gnorm_mean_last40"]:>8.3f} | '
          f'{ev["0.0"]:.2f} / {ev["10.0"]:.2f} / {ev["20.0"]:.2f}')

print('\nSauvé -> results/diag_tarb_residual_signed_attn_T_sweep.json')
