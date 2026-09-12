"""
diag_lr_extended.py — ÉTAPE 3 suite : les 4 schedules de
diag_lr_schedule_search.py montraient tous une courbe encore en hausse à
l'époque 25 (aucun plateau) -- confirmation du diagnostic "LR/schedule
trop agressif pour trop peu d'époques". Ce script étend le meilleur
schedule trouvé (cosine_long) à un budget d'époques bien plus long pour
voir s'il rejoint/dépasse WMMSE à haut SNR, et détermine le budget
d'époques à utiliser pour les runs complets (ÉTAPE 4).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_lr_extended.py
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer, CHOSEN_CONFIG, DATASET_SIZE, NUM_TX, NUM_RX
from datasets import CachedSionnaDataset
from sionna.phy.utils import ebnodb2no

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='single_sc', choices=['single_sc', 'intra_rb', 'ta_rb'])
_p.add_argument('--finetune_epochs', type=int, default=60)
_p.add_argument('--tokens_per_rb', type=int, default=3)
_p.add_argument('--batch_size', type=int, default=128)
_args = _p.parse_args()

WARMUP_EPOCHS   = 3
FINETUNE_EPOCHS = _args.finetune_epochs
STEPS_PER_EPOCH = 150
BATCH_SIZE      = _args.batch_size
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
ARCH            = _args.arch

_ref = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}
RZF_REF   = {snr: rzf for snr, rzf in zip(_ref['snr'], _ref['rzf_full'])}


def eval_at_snr(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=BATCH_SIZE):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    rates = []
    for _ in range(num_batches):
        h = dataset.get_batch(batch_size)
        g = system._call_precoder(h, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr  = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate  = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))
        rates.append(float(rate))
    return float(np.mean(rates))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    sys_kwargs = dict(embed_dim=128, num_heads=4, num_layers=4)
    if ARCH == 'ta_rb':
        sys_kwargs['tokens_per_rb'] = _args.tokens_per_rb
    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ARCH, **sys_kwargs)
    tag = f'{ARCH}_T{_args.tokens_per_rb}' if ARCH == 'ta_rb' else ARCH
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'lrsearch_cosine_long_extended_{tag}',
        warmup_epochs=WARMUP_EPOCHS, finetune_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nRÉSULTAT {tag} cosine_long ÉTENDU ({FINETUNE_EPOCHS}ep finetune, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | '
              f'RZF={RZF_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    ft_hist = [e['sum_rate'] for e in trainer.history if e['mode'] == 'FINETUNE']
    plateaued = (len(ft_hist) >= 10 and
                 abs(np.mean(ft_hist[-5:]) - np.mean(ft_hist[-10:-5])) < 0.02 * np.mean(ft_hist[-5:]))
    print(f'\n  Courbe finetune (rate @15dB, dernières 10 époques) : {[round(v,1) for v in ft_hist[-10:]]}')
    print(f'  Plateau détecté (<2% de variation sur les 2 dernières fenêtres de 5ep) : {plateaued}')

    with open(f'results/diag_lr_extended_{tag}.json', 'w') as f:
        json.dump({'finetune_epochs': FINETUNE_EPOCHS, 'evals': evals, 'pct_of_wmmse': pct,
                   'train_time_min': train_time / 60, 'history': trainer.history,
                   'plateaued': bool(plateaued)}, f, indent=2)
    print(f'\nSauvé -> results/diag_lr_extended_{tag}.json')
