"""
diag_tarb_residual_variant.py — Partie 2 (demande utilisateur), pistes
de repli si le décodeur résiduel par défaut est mitigé/mauvais :
1. LR différent (le décodeur résiduel a ~40% moins de FLOPs/params)
2. alpha_init différent (amplitude de la correction résiduelle)
Budget réduit (30 époques par défaut) pour un signal rapide, pas un
run complet -- comparable au protocole des tests warmup+rampe.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_tarb_residual_variant.py \
       --learning_rate 2e-3 --alpha_init 0.1 --tag lr2e3
"""
import os, sys, json, time, argparse
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

_p = argparse.ArgumentParser()
_p.add_argument('--learning_rate', type=float, default=1e-3)
_p.add_argument('--alpha_init', type=float, default=0.1)
_p.add_argument('--finetune_epochs', type=int, default=30)
_p.add_argument('--warmup_epochs', type=int, default=3)
_p.add_argument('--tag', default='default')
_args = _p.parse_args()

STEPS_PER_EPOCH = 150
BATCH_SIZE      = 256
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20

_ref = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}


def eval_at_snr(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=128):
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

    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='ta_rb_residual',
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6,
                             alpha_init=_args.alpha_init)
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'TA_RB_residual_variant_{_args.tag}',
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=_args.learning_rate,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nRÉSULTAT variant [{_args.tag}] lr={_args.learning_rate} alpha_init={_args.alpha_init} '
          f'({_args.finetune_epochs}ep finetune, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    ft_hist = [e['sum_rate'] for e in trainer.history if e['mode'] == 'FINETUNE']
    print(f'\n  Courbe finetune (rate @15dB) : {[round(v,1) for v in ft_hist]}')

    with open(f'results/diag_tarb_residual_variant_{_args.tag}.json', 'w') as f:
        json.dump({'lr': _args.learning_rate, 'alpha_init': _args.alpha_init,
                   'evals': evals, 'pct_of_wmmse': pct, 'history': trainer.history}, f, indent=2)
    print(f'\nSauvé -> results/diag_tarb_residual_variant_{_args.tag}.json')
