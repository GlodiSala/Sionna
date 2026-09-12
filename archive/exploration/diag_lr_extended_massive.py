"""
diag_lr_extended_massive.py — ÉTAPE 3, vérification MASSIVE (M32K8, R=5m).

Applique le protocole gagnant trouvé sur STANDARD (cosine_long,
warmup court, finetune long) à MASSIVE, SANS supposer que ça transfère
automatiquement (consigne explicite). N'importe PAS CHOSEN_CONFIG de
main_finall.py (fixé à STANDARD_CONFIG au niveau module) -- construit
tout directement avec MASSIVE_CONFIG pour ne pas perturber l'état
partagé du pipeline STANDARD.

Budget réduit vs la recherche STANDARD (30 au lieu de 60 époques
finetune) : ~0.80s/pas mesuré à MASSIVE (vs ~0.24s/pas STANDARD, ~3.3x
plus lent -- feat_dim/M/K plus grands) -- un diagnostic directionnel à
budget réduit est plus rentable ici qu'une réplique exacte du budget
STANDARD avant de savoir si ça vaut le coup d'investir plus.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_lr_extended_massive.py [--arch intra_rb] [--finetune_epochs 30]
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer
from channel_config import MASSIVE_CONFIG
from datasets import CachedSionnaDataset
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='intra_rb', choices=['single_sc', 'intra_rb', 'ta_rb'])
_p.add_argument('--finetune_epochs', type=int, default=30)
_p.add_argument('--tokens_per_rb', type=int, default=6)
_args = _p.parse_args()

M, K            = MASSIVE_CONFIG['NUM_TX'], MASSIVE_CONFIG['NUM_RX']
DATASET_SIZE    = 8000
WARMUP_EPOCHS   = 3
FINETUNE_EPOCHS = _args.finetune_epochs
STEPS_PER_EPOCH = 150
BATCH_SIZE      = 128
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
ARCH            = _args.arch
TAG             = f'{ARCH}_T{_args.tokens_per_rb}' if ARCH == 'ta_rb' else ARCH

_ref = np.load('results/classical_comparison_M32K8.npy', allow_pickle=True).item()
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
    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{M}x{K}.npz',
        cluster_radius_m=MASSIVE_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=MASSIVE_CONFIG['INDOOR_PROBABILITY'], seed=42)

    sys_kwargs = dict(embed_dim=128, num_heads=4, num_layers=4)
    if ARCH == 'ta_rb':
        sys_kwargs['tokens_per_rb'] = _args.tokens_per_rb
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=ARCH, **sys_kwargs)
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'massive_{TAG}',
        warmup_epochs=WARMUP_EPOCHS, finetune_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nRÉSULTAT MASSIVE {TAG} cosine_long ({FINETUNE_EPOCHS}ep finetune, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | '
              f'RZF={RZF_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    ft_hist = [e['sum_rate'] for e in trainer.history if e['mode'] == 'FINETUNE']
    print(f'\n  Courbe finetune (rate @15dB, toutes époques) : {[round(v,1) for v in ft_hist]}')

    with open(f'results/diag_lr_extended_massive_{TAG}.json', 'w') as f:
        json.dump({'finetune_epochs': FINETUNE_EPOCHS, 'evals': evals, 'pct_of_wmmse': pct,
                   'train_time_min': train_time / 60, 'history': trainer.history}, f, indent=2)
    print(f'\nSauvé -> results/diag_lr_extended_massive_{TAG}.json')
