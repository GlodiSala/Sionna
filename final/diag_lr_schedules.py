"""
diag_lr_schedules.py — Volet 2 : compare plusieurs profils de LR sur
IntraRB (architecture nettoyée, rb_size=12 -- le fenêtrage n'a pas
d'impact mesurable sur ce canal quasi-plat, cf. diag_intrarb_window.py),
sum-rate direct sans warmup, à budget de steps ÉGAL entre profils
(comparaison équitable).

Profils testés (esprit Bloc C : LR petit et soutenu > cosinus agressif court) :
  A. baseline_hier   : cosinus 1e-3 -> 2e-5 sur SEULEMENT ces N_STEPS (reproduit
                        le comportement d'hier, sert de référence "mauvais élève")
  B. constant_bas    : LR constant 1e-4 (esprit Trans/freq/main_freq.py, lr=1e-5,
                        mis à l'échelle -- notre optimiseur/archi diffèrent)
  C. cosinus_long    : cosinus 1e-3 -> 1e-4 mais étalé sur un horizon BEAUCOUP
                        plus long (alpha=0.1, décroissance douce)
  D. warm_restart    : cosinus 5e-4 -> 5e-5 répété 3x (SGDR-like)

Budget : N_STEPS identique pour les 4 (comparaison à budget de calcul égal).
Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_lr_schedules.py
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
N_STEPS = 900      # ~1 "époque" à batch=256 (250k éch. eff. / 256 ~ 977 steps/ép.)
                    # -- réduit de 2000 à 900 pour rester dans l'esprit smoke-test
                    # (4 profils x 900 steps ~ 45-50min total au lieu de ~2h)
BATCH = 256
EVAL_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
EVAL_EVERY = 300

_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    _sys, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)


def make_schedule(name):
    if name == 'baseline_hier':
        return tf.keras.optimizers.schedules.CosineDecay(1e-3, N_STEPS, alpha=0.02)
    if name == 'constant_bas':
        return 1e-4
    if name == 'cosinus_long':
        return tf.keras.optimizers.schedules.CosineDecay(1e-3, N_STEPS, alpha=0.1)
    if name == 'warm_restart':
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            5e-4, N_STEPS // 3, t_mul=1.0, m_mul=0.7, alpha=0.05)
    raise ValueError(name)


all_results = {}
for sched_name in ['baseline_hier', 'constant_bas', 'cosinus_long', 'warm_restart']:
    print(f'\n{"="*70}\nSCHEDULE = {sched_name}\n{"="*70}')
    tf.random.set_seed(42)
    np.random.seed(42)

    precoder = IntraRBTransformerPrecoder(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=96,
        rb_size=12, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
    precoder(dummy_h, no=tf.constant(1e-3), training=False)
    vars_ = precoder.trainable_variables

    opt = tf.keras.optimizers.Adam(make_schedule(sched_name), clipnorm=5.0)
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

    def eval_all():
        res = {}
        for snr in EVAL_SNRS:
            rr = [float(eval_step(dataset.get_batch(64), tf.constant(snr, tf.float32))) for _ in range(10)]
            res[str(snr)] = float(np.mean(rr))
        return res

    history = []
    t0 = time.time()
    for step in range(N_STEPS):
        hb = dataset.get_batch(BATCH)
        loss, gnorm = train_step(hb)
        if (step + 1) % EVAL_EVERY == 0 or step == N_STEPS - 1:
            ev = eval_all()
            history.append({'step': step + 1, 'loss': float(loss), 'gnorm': float(gnorm), 'eval': ev})
            print(f'  step {step+1:4d}/{N_STEPS} | loss={float(loss):+.4f} | gn={float(gnorm):.3f} | '
                  f'eval@15/20dB={ev["15.0"]:.2f}/{ev["20.0"]:.2f} | {time.time()-t0:.0f}s')

    all_results[sched_name] = {'history': history, 'wall_time_s': time.time() - t0}
    del precoder, opt
    tf.keras.backend.clear_session()
    gc.collect()

print(f'\n\n=== RÉSUMÉ SCHEDULES LR (WMMSE réf. ~34.9 à 20dB, ~28.2 à 15dB) ===')
print(f'{"schedule":<16} {"eval@0":>7} {"eval@5":>7} {"eval@10":>7} {"eval@15":>7} {"eval@20":>7}')
for name, r in all_results.items():
    ev = r['history'][-1]['eval']
    print(f'{name:<16} ' + ' '.join(f'{ev[s]:7.2f}' for s in ['0.0','5.0','10.0','15.0','20.0']))

with open('results/diag_lr_schedules.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSauvé -> results/diag_lr_schedules.json')
