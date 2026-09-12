"""
diag_uma_quick_train.py — teste si le canal UMa (sélectivité fréquentielle
plus marquée, rho intra-RB=0.632 vs 0.797 pour UMi, cf. diag_uma_
selectivity.py) change le classement SingleSC vs IntraRB, en particulier
si IntraRB peut dépasser SingleSC sous CSI parfait -- ce que le canal UMi
actuel ne permet pas structurellement (corrélation cross-SC redondante
avec ce que RZF/WMMSE utilisent déjà, Partie 0/hypothèse #5).

Test CIBLÉ, budget réduit (30 époques, comme les autres tests rapides de
cette session) -- PAS une régénération complète du pipeline (dataset de
production, classical_comparison, MASSIVE). Cache UMa jetable, distinct
du cache UMi de production (aucun risque de collision/écrasement).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_uma_quick_train.py --arch single_sc
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
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, gen_topology_clustered
from datasets import CachedSionnaDataset
from precoders_w import rzf_precoder, wmmse_precoder
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb'])
_p.add_argument('--finetune_epochs', type=int, default=30)
_p.add_argument('--dataset_size', type=int, default=8000)
_args = _p.parse_args()

M, K, R = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX'], STANDARD_CONFIG['CLUSTER_RADIUS_M']
STEPS_PER_EPOCH = 150
BATCH_SIZE      = 256
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_uma_{_args.dataset_size//1000}k_{M}x{K}.npz'


class UMaSystem(ConfigurableMIMOSystem):
    """Pour calcul de la référence WMMSE/RZF sous UMa -- même RG que
    diag_uma_selectivity.py (comparabilité classical_comparison-style)."""
    cluster_radius_m = R
    def new_topology(self, batch_size):
        topology = gen_topology_clustered(batch_size, self.num_rx, 'uma', self.cluster_radius_m, indoor_probability=0.0)
        self.channel_model.set_topology(*topology, los=False)


def wmmse_ref_uma(snr_points, num_batches=10, batch=16):
    from precoders_w import rzf_precoder, wmmse_precoder
    sys_ref = UMaSystem(M, K)
    ref = {}
    for snr in snr_points:
        no = ebnodb2no(tf.constant(snr, tf.float32), 2, 0.5, sys_ref.rg)
        r_wmmse = []
        for _ in range(num_batches):
            sys_ref.new_topology(batch)
            h, _ = sys_ref.channel_and_no(tf.constant(batch, tf.int32), tf.constant(snr, tf.float32))
            g = wmmse_precoder(h, no=no, stream_management=sys_ref.sm, num_iterations=10)
            r_wmmse.append(float(sys_ref._sum_rate(h, g, no)))
        ref[snr] = float(np.mean(r_wmmse))
    return ref


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
    print(f'\n{"="*70}\nWMMSE réf. UMa (10 batchs, {EVAL_SNRS})\n{"="*70}', flush=True)
    wmmse_ref = wmmse_ref_uma(EVAL_SNRS)
    for snr, r in wmmse_ref.items():
        print(f'  SNR={snr}dB : WMMSE={r:.2f}')

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=_args.dataset_size, batch_size=128,
        cache_file=CACHE_FILE,
        cluster_radius_m=R, indoor_probability=0.0, scenario='uma', seed=42)

    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=_args.arch,
                             embed_dim=128, num_heads=4, num_layers=4)
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'uma_quick_{_args.arch}',
        warmup_epochs=3, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / wmmse_ref[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nRÉSULTAT UMa {_args.arch} ({_args.finetune_epochs}ep finetune, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE(UMa)={wmmse_ref[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    with open(f'results/diag_uma_quick_{_args.arch}.json', 'w') as f:
        json.dump({'evals': evals, 'wmmse_ref': wmmse_ref, 'pct_of_wmmse': pct,
                   'train_time_min': train_time / 60}, f, indent=2)
    print(f'\nSauvé -> results/diag_uma_quick_{_args.arch}.json')
