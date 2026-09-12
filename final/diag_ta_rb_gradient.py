"""
diag_ta_rb_gradient.py — Bloc A.2 : gradient norm par composant pendant un
smoke-training réel (sum-rate direct, batch=32) sur TransformerPrecoderV4
v4.3 T=3, pour voir quel composant domine/dérive vers l'instabilité.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_ta_rb_gradient.py
"""
import os, sys, time
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

system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='transformer_rb',
                         tokens_per_rb=3, version='v4.3',
                         embed_dim=128, num_heads=4, num_layers=4)
dataset = CachedSionnaDataset(
    system, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)

precoder = system.precoder
dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
precoder(dummy_h, no=tf.constant(1e-3), training=False)
vars_ = list(precoder.trainable_variables)

# Groupement par sous-module (var.name ne contient pas le chemin complet
# en Keras3 -- on part directement des références de sous-modules).
submodule_groups = {
    'input_proj_norm': [precoder.input_proj, precoder.input_norm],
    'snr_proj': [precoder.snr_proj],
    'pos_embed': [precoder.pos_embed],
    'transformer_blocks': list(precoder.blocks),
    'joint_output_proj': [precoder.joint_output_proj],
    'upsample': [precoder.upsample],
    'final_proj': [precoder.final_proj],
    'sc_refine': [precoder.sc_refine],
    'gate_sigmoid': [precoder.sc_gate],
}
groups = {}
var_id_to_group = {}
for gname, mods in submodule_groups.items():
    gvars = []
    for mod in mods:
        gvars.extend(mod.trainable_variables)
    groups[gname] = gvars
    for v in gvars:
        var_id_to_group[id(v)] = gname
groups['alpha_residual'] = [precoder.alpha]
var_id_to_group[id(precoder.alpha)] = 'alpha_residual'

# Vérifie qu'on couvre bien toutes les variables
covered = set(id(v) for gvars in groups.values() for v in gvars)
missing = [v for v in vars_ if id(v) not in covered]
if missing:
    print(f'⚠️  {len(missing)} variables non groupées (ajoutées à "other")')
    groups['other'] = missing
    for v in missing:
        var_id_to_group[id(v)] = 'other'

print('\nGroupes de variables :')
for g, vs in groups.items():
    n_params = sum(int(tf.size(v)) for v in vs)
    print(f'  {g:<20} {len(vs):3d} vars   {n_params:>10,} params')

n_params_total = sum(int(tf.size(v)) for v in vars_)
print(f'\nTotal: {n_params_total:,} params\n')

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
    global_norm = tf.linalg.global_norm(grads)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    opt.apply_gradients(zip(grads_clipped, vars_))
    return loss, global_norm, grads


group_names = list(groups.keys())
var_to_group_idx = [group_names.index(var_id_to_group[id(v)]) for v in vars_]

log = []
t0 = time.time()
for step in range(N_STEPS):
    h = dataset.get_batch(BATCH)
    loss, gnorm, grads = train_step(h)

    group_norms = {gn: 0.0 for gn in group_names}
    for gi, g_ in zip(var_to_group_idx, grads):
        group_norms[group_names[gi]] += float(tf.reduce_sum(tf.square(g_)))
    group_norms = {k: v**0.5 for k, v in group_norms.items()}

    log.append({'step': step, 'loss': float(loss), 'global_gnorm': float(gnorm),
                **{f'gn_{k}': v for k, v in group_norms.items()}})

    if (step + 1) % 40 == 0:
        elapsed = time.time() - t0
        top = sorted(group_norms.items(), key=lambda x: -x[1])[:3]
        print(f'  step {step+1:4d}/{N_STEPS} | loss={float(loss):+.4f} | '
              f'global_gn={float(gnorm):.3f} | top3=' +
              ', '.join(f'{k}={v:.3f}' for k, v in top) +
              f' | {elapsed:.0f}s')

import json
with open('results/diag_ta_rb_gradient_log.json', 'w') as f:
    json.dump(log, f)
print('\nSauvé -> results/diag_ta_rb_gradient_log.json')

# Résumé : moyenne des 40 derniers steps par groupe (fin d'entraînement smoke)
print('\n=== Moyenne des normes de gradient par groupe (40 derniers steps) ===')
last = log[-40:]
avgs = {}
for gn in group_names:
    avgs[gn] = np.mean([l[f'gn_{gn}'] for l in last])
for gn, v in sorted(avgs.items(), key=lambda x: -x[1]):
    n_params = sum(int(tf.size(vv)) for vv in groups[gn])
    print(f'  {gn:<20} gnorm_moy={v:8.4f}   n_params={n_params:>10,}   '
          f'gnorm/sqrt(params)={v/np.sqrt(max(n_params,1)):.5f}')

print('\n=== Moyenne des normes de gradient par groupe (40 premiers steps) ===')
first = log[:40]
for gn in group_names:
    v = np.mean([l[f'gn_{gn}'] for l in first])
    print(f'  {gn:<20} gnorm_moy={v:8.4f}')
