"""
diag_ta_rb_ablation.py — Bloc A.3 : ablation smoke-scale des versions
v4.0/v4.1/v4.2/v4.3 (T=3) — même protocole que le run TA_RB_T3 réel
(sum-rate direct, batch=32) sur 400 steps, pour comparer stabilité
(gnorm) et vitesse d'apprentissage (loss/eval) entre versions.

v4.0 : upsample=repeat, refine=Conv1D(k=3) sur [w_up,h_re,h_im]        (le plus simple)
v4.1 : upsample=Conv1DTranspose, refine=Conv1D(k=RB) sur [w_up] seul   (décodeur minimal)
v4.2 : + features riches (h_re,h_im,|h|,angle(h))                     (sans gate, sans SNR-aware)
v4.3 : + SNR-aware + gate sigmoid                                     (baseline actuelle)

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_ta_rb_ablation.py
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
N_STEPS = 400
BATCH = 32
EVAL_SNRS = [0.0, 10.0, 20.0]

# dataset partagé entre versions (même seed => mêmes batches à chaque run,
# comparaison equitable si on réinitialise np.random avant chaque run)
_dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    _dummy_sys, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)

all_results = {}

for version in ['v4.0', 'v4.1', 'v4.2', 'v4.3']:
    print(f'\n{"="*70}\nVERSION {version}\n{"="*70}')
    tf.random.set_seed(42)
    np.random.seed(42)

    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='transformer_rb',
                             tokens_per_rb=3, version=version,
                             embed_dim=128, num_heads=4, num_layers=4)
    precoder = system.precoder
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
    precoder(dummy_h, no=tf.constant(1e-3), training=False)
    vars_ = precoder.trainable_variables
    n_params = sum(int(tf.size(v)) for v in vars_)
    print(f'  params={n_params:,}')

    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, N_STEPS, alpha=0.02)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0)
    rate_norm = float(system.num_users) * 9.0

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
        return loss, gnorm, rate

    @tf.function
    def eval_step(h_freq, snr_db):
        no = ebnodb2no(snr_db, system.num_bits_per_symbol, 0.5, system.rg)
        g = system._call_precoder(h_freq, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h_freq, g)
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        return tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))

    log = []
    t0 = time.time()
    for step in range(N_STEPS):
        h = dataset.get_batch(BATCH)
        loss, gnorm, rate = train_step(h)
        log.append({'step': step, 'loss': float(loss), 'gnorm': float(gnorm),
                    'train_rate': float(rate)})
        if (step + 1) % 100 == 0:
            print(f'  step {step+1:4d}/{N_STEPS} | loss={float(loss):+.4f} | '
                  f'gn={float(gnorm):.3f} | {time.time()-t0:.0f}s')

    # eval finale sur quelques SNR (petit nombre de batches, juste indicatif)
    eval_res = {}
    for snr in EVAL_SNRS:
        rates = []
        for _ in range(10):
            h = dataset.get_batch(64)
            rates.append(float(eval_step(h, tf.constant(snr, tf.float32))))
        eval_res[str(snr)] = float(np.mean(rates))

    gnorms = np.array([l['gnorm'] for l in log])
    all_results[version] = {
        'n_params': n_params,
        'gnorm_mean_last40': float(gnorms[-40:].mean()),
        'gnorm_max': float(gnorms.max()),
        'gnorm_p95': float(np.percentile(gnorms, 95)),
        'loss_first40_mean': float(np.mean([l['loss'] for l in log[:40]])),
        'loss_last40_mean': float(np.mean([l['loss'] for l in log[-40:]])),
        'eval_sum_rate': eval_res,
        'wall_time_s': time.time() - t0,
    }
    print(f'  -> eval @ SNR {EVAL_SNRS}: {eval_res}')
    print(f'  -> gnorm mean(last40)={all_results[version]["gnorm_mean_last40"]:.3f} '
          f'max={all_results[version]["gnorm_max"]:.3f} p95={all_results[version]["gnorm_p95"]:.3f}')

    del system, precoder, opt
    tf.keras.backend.clear_session()

print('\n\n=== RÉSUMÉ ABLATION (400 steps, batch=32, sum-rate direct) ===')
print(f'{"version":<8} {"params":>10} {"loss@40":>9} {"loss@400":>9} '
      f'{"gn_mean":>8} {"gn_max":>8} {"gn_p95":>8} | eval@0/10/20dB')
for v, r in all_results.items():
    ev = r['eval_sum_rate']
    print(f'{v:<8} {r["n_params"]:>10,} {r["loss_first40_mean"]:>9.4f} '
          f'{r["loss_last40_mean"]:>9.4f} {r["gnorm_mean_last40"]:>8.3f} '
          f'{r["gnorm_max"]:>8.3f} {r["gnorm_p95"]:>8.3f} | '
          f'{ev["0.0"]:.2f} / {ev["10.0"]:.2f} / {ev["20.0"]:.2f}')

with open('results/diag_ta_rb_ablation.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSauvé -> results/diag_ta_rb_ablation.json')
