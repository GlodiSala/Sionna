"""
diag_lr_schedule_search.py — ÉTAPE 3 (SESSION_NUIT_RESUME.md), recherche de
protocole d'entraînement sur STANDARD (M8K4, R=20m, dataset §0 corrigé).

Compare 4 schedules LR pour la phase finetune (sum-rate), à budget de pas
IDENTIQUE (comparaison à compute constant) : baseline (cosinus court,
comportement actuel), constant_low, cosine_long (cosinus sur horizon 2x
plus long que le budget réel), warm_restart (SGDR).

Architecture : single_sc (la plus rapide à itérer), D=128 L=4 (taille de
production MODELS_TO_TRAIN) -- protocole gagnant à re-vérifier sur
intra_rb/ta_rb avant les runs complets (ÉTAPE 4), pas supposé transférable
automatiquement (idem MASSIVE, cf. tâche suivante).

Objectif : sum-rate à 15-20dB doit ÉGALER/DÉPASSER WMMSE (référence
classical_comparison_M8K4.npy, déjà régénérée avec le nouveau canal).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_lr_schedule_search.py
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import (MU_MIMO_System, SupervisedTrainer, CHOSEN_CONFIG,
                          DATASET_SIZE, NUM_TX, NUM_RX, EVALUATION_SNR_RANGE)
from datasets import CachedSionnaDataset
from sionna.phy.utils import ebnodb2no

SCHEDULES = ['baseline', 'constant_low', 'cosine_long', 'warm_restart']

WARMUP_EPOCHS   = 2
FINETUNE_EPOCHS = 25
STEPS_PER_EPOCH = 150
BATCH_SIZE      = 128
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]   # haut SNR -- objectif ÉTAPE 3
EVAL_BATCHES    = 20

# Référence WMMSE/RZF déjà validée (classical_comparison_M8K4.npy)
_ref = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}
RZF_REF   = {snr: rzf for snr, rzf in zip(_ref['snr'], _ref['rzf_full'])}


def eval_at_snr(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=BATCH_SIZE):
    no = ebnodb2no(tf.constant(snr_db, tf.float32),
                    system.num_bits_per_symbol, 0.5, system.rg)
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


def run_one(schedule, dataset):
    print(f'\n{"#"*70}\n# SCHEDULE = {schedule}\n{"#"*70}', flush=True)
    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='single_sc',
                             embed_dim=128, num_heads=4, num_layers=4)
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'lrsearch_{schedule}',
        warmup_epochs=WARMUP_EPOCHS, finetune_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule=schedule)

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct_of_wmmse = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n  --- Résultat [{schedule}] (train={train_time/60:.1f}min) ---')
    for snr in EVAL_SNRS:
        print(f'    SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | '
              f'WMMSE={WMMSE_REF[snr]:7.2f} | RZF={RZF_REF[snr]:7.2f} | '
              f'{pct_of_wmmse[snr]:5.1f}% WMMSE')

    tf.keras.backend.clear_session()
    return {'schedule': schedule, 'train_time_min': train_time / 60,
            'evals': evals, 'pct_of_wmmse': pct_of_wmmse,
            'best_train_rate': trainer.best,
            'history': trainer.history}


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    results = {}
    results_path = 'results/diag_lr_schedule_search.json'
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)   # reprise : ne re-tourne pas les schedules déjà finis
        print(f'Résultats déjà présents pour : {list(results.keys())}')

    for schedule in SCHEDULES:
        if schedule in results:
            continue
        results[schedule] = run_one(schedule, dataset)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)   # sauvé après CHAQUE schedule

    print(f'\n\n{"="*80}\nRÉSUMÉ RECHERCHE PROTOCOLE (STANDARD, single_sc D128L4)\n{"="*80}')
    print(f'{"Schedule":<15}', end='')
    for snr in EVAL_SNRS:
        print(f' | {snr}dB %WMMSE', end='')
    print()
    for schedule, r in results.items():
        print(f'{schedule:<15}', end='')
        for snr in EVAL_SNRS:
            print(f' | {r["pct_of_wmmse"][snr]:9.1f}%', end='')
        print(f'  (train={r["train_time_min"]:.1f}min)')

    best = max(results.values(), key=lambda r: np.mean(list(r['pct_of_wmmse'].values())))
    print(f'\n✅ Meilleur schedule (moyenne %WMMSE haut SNR) : {best["schedule"]}')
    print('Sauvé -> results/diag_lr_schedule_search.json')
