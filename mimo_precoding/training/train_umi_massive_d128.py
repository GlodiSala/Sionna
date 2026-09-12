"""
training/train_umi_massive_d128.py — Priorité 3.2 (demande utilisateur) : entraîne
une des 3 architectures signed_attn (SC/IB/TA-RB T=6) à budget complet
(83ep) sur MASSIVE_TRUE_CONFIG (M=64, K=8, canal STANDARD R=20m, PAS de
clustering serré artificiel -- channel_config.py). Même protocole que
P1/P2 (cosine_long, warmup=3, finetune=80, steps_per_epoch=150).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 training/train_umi_massive_d128.py --arch single_sc
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from system import MU_MIMO_System, SupervisedTrainer
from datasets import CachedSionnaDataset
from channel_config import MASSIVE_TRUE_CONFIG
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb', 'ta_rb_residual'])
_p.add_argument('--finetune_epochs', type=int, default=80)
_p.add_argument('--warmup_epochs', type=int, default=3)
_p.add_argument('--tag', type=str, default='',
                 help='suffixe pour run_name/JSON de sortie -- évite d\'écraser un run '
                      'existant (ex. diagnostic budget étendu, cf. SESSION_LOG)')
_args = _p.parse_args()

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
STEPS_PER_EPOCH = 150
LR              = 1e-3
DATASET_SIZE    = 4000
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_massive_true_{DATASET_SIZE//1000}k_{M}x{K}.npz'
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
BATCH_SIZE      = {'single_sc': 16, 'intra_rb': 16, 'ta_rb_residual': 32}[_args.arch]
# réduit depuis 128/256 (STANDARD) -- OOM au smoke test à M=64 (tenseur SINR/
# effective-channel [B,K,1,ofdm,fft,M,K] beaucoup plus lourd, cf. diag_massive_
# true_smoke.py). Valeur à re-confirmer par le smoke test avant tout run long.


def eval_at_snr(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=64):
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
        dummy, dataset_size=DATASET_SIZE, batch_size=32,
        cache_file=CACHE_FILE,
        cluster_radius_m=MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'], seed=42)

    if arch == 'single_sc':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='single_sc',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = SingleSCTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'MASSIVE_TRUE_SingleSC_signed_attn_4L_128d' + _args.tag
    elif arch == 'intra_rb':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='intra_rb',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = IntraRBTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'MASSIVE_TRUE_IntraRB_signed_attn_4L_128d' + _args.tag
    else:
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='ta_rb_residual',
                                 embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6)
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, tokens_per_rb=6, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
        run_name = 'MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d' + _args.tag

    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}

    flops, params, acts = system.precoder.complexity(system.rg.num_ofdm_symbols, convention_x_ofdm=False)
    real_params = int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables))
    assert int(params) == real_params, f"{arch}: poids {int(params)} != réel {real_params}"

    print(f'\n{"="*70}\nRÉSULTAT MASSIVE_TRUE [{arch}] ({_args.finetune_epochs}ep, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f}')
    print(f'  FLOPs_reel={flops/1e6:.1f}M  params={real_params:,}')

    out = {'arch': arch, 'evals': evals, 'flops_real_M': flops / 1e6, 'params': real_params,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'M': M, 'K': K}
    with open(f'results/diag_massive_true_{arch}{_args.tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> results/diag_massive_true_{arch}{_args.tag}.json')
