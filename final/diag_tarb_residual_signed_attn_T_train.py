"""
diag_tarb_residual_signed_attn_T_train.py — Étape 1 (demande utilisateur) :
réentraînement PROPRE du T-sweep pour Agrégation par RB (décodeur
résiduel, signed_attn) -- remplace le T-sweep précédent (500 pas,
~3.3 équiv-époques, jugé pas exploitable). Budget minimum 30 époques
(3 warmup + 27 finetune), protocole identique à P1/P2 (cosine_long,
steps_per_epoch=150), cache joint-cluster de production. Un T par
invocation -- lancé en parallèle sur plusieurs GPU par le runner.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_tarb_residual_signed_attn_T_train.py --T 3 --finetune_epochs 27
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import (MU_MIMO_System, SupervisedTrainer, CHOSEN_CONFIG,
                          DATASET_SIZE, NUM_TX, NUM_RX)
from datasets import CachedSionnaDataset
from precoder_experimental import TransformerPrecoderCleanResidualSignedAttn
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--T', type=int, required=True, choices=[1, 2, 3, 4, 6, 12])
_p.add_argument('--finetune_epochs', type=int, default=27)
_p.add_argument('--warmup_epochs', type=int, default=3)
_p.add_argument('--batch_size', type=int, default=256,
                 help='réduire si OOM sur GPU partagé, notamment T=12 (tenseurs plus gros)')
_args = _p.parse_args()

STEPS_PER_EPOCH = 150
LR              = 1e-3
BATCH_SIZE      = _args.batch_size   # 256 par défaut, cohérent avec ta_rb_residual (P1/P2)
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
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
    T = _args.T

    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='ta_rb_residual',
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=T)
    system.precoder = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
        fft_size=system.rg.fft_size, rb_size=12, tokens_per_rb=T,
        embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    run_name = f'TA_RB_residual_signed_attn_T{T}_4L_128d'

    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    flops, params, acts = system.precoder.complexity(system.rg.num_ofdm_symbols, convention_x_ofdm=False)
    real_params = int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables))
    assert int(params) == real_params, f"T={T}: poids {int(params)} != réel {real_params}"

    print(f'\n{"="*70}\nRÉSULTAT T={T} ({_args.finetune_epochs}ep finetune, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')
    print(f'  FLOPs_reel={flops/1e6:.1f}M  params={real_params:,}')

    out = {'T': T, 'evals': evals, 'wmmse_ref': WMMSE_REF, 'pct_of_wmmse': pct,
           'flops_real_M': flops / 1e6, 'params': real_params,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs}
    with open(f'results/diag_tarb_residual_signed_attn_T{T}_train.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> results/diag_tarb_residual_signed_attn_T{T}_train.json')
