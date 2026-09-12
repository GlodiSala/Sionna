"""
diag_standard_extbudget_train.py — Priorité 3 (nuit, demande utilisateur) :
test budget étendu (10 warmup + 160 finetune, même protocole que le
diagnostic MASSIVE) sur le régime STANDARD (UMi, M8K4) -- vérifie si 83ep
suffisait déjà pour STANDARD (attendu, vu le signal déjà en main : rate
plat + gradient stabilisé dès le milieu du budget 83ep pour la référence
M8K4 utilisée dans le diagnostic MASSIVE) ou si un gain significatif
apparaît comme à M=64.

SC-STANDARD d'abord (--arch single_sc) ; script généralisable aux 3
architectures (mêmes CLI que diag_massive_true_train.py / diag_uma_signed_
attn_train.py) si le gain se révèle significatif.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_standard_extbudget_train.py --arch single_sc --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget
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
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb', 'ta_rb_residual'])
_p.add_argument('--finetune_epochs', type=int, default=160)
_p.add_argument('--warmup_epochs', type=int, default=10)
_p.add_argument('--tag', type=str, default='_extbudget')
_args = _p.parse_args()

M, K = NUM_TX, NUM_RX
STEPS_PER_EPOCH = 150
LR              = 1e-3
BATCH_SIZE      = 256
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
TOKENS_PER_RB   = 4   # T=4 = référence (cf. Figure B)

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
    arch = _args.arch

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{M}x{K}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    if arch == 'single_sc':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='single_sc',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = SingleSCTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'STANDARD_SingleSC_signed_attn_4L_128d' + _args.tag
    elif arch == 'intra_rb':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='intra_rb',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = IntraRBTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'STANDARD_IntraRB_signed_attn_4L_128d' + _args.tag
    else:
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='ta_rb_residual',
                                 embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=TOKENS_PER_RB)
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, tokens_per_rb=TOKENS_PER_RB, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
        run_name = f'STANDARD_TA_RB_residual_signed_attn_{TOKENS_PER_RB}tok_4L_128d' + _args.tag

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

    print(f'\n{"="*70}\nRÉSULTAT STANDARD_EXTBUDGET [{arch}] ({_args.finetune_epochs}ep, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE', flush=True)

    out = {'arch': arch, 'evals': evals, 'pct_of_wmmse': pct, 'flops_real_M': flops / 1e6,
           'params': real_params, 'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs, 'M': M, 'K': K}
    with open(f'results/diag_standard_extbudget_{arch}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Sauvé -> results/diag_standard_extbudget_{arch}.json')
