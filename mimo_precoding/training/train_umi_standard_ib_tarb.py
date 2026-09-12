"""
training/train_umi_standard_ib_tarb.py — Priorité 2 (demande utilisateur) :
applique la piste gagnante de Front A (SignedGateMHA, cf. precoder_
experimental.py) à IntraRB et TA-RB résiduel, budget complet UMi (même
protocole que le run de production : 3 warmup + 80 finetune, cosine_long,
steps_per_epoch=150), pour la comparaison des 3 architectures avec le
mécanisme gagnant.

Batch size par architecture = celui du run de production original
(cohérence directe) : intra_rb=128 (BATCH_SIZE global system.py),
ta_rb_residual=256 (MODELS_TO_TRAIN['batch_size'], diag_tarb_residual_
test.py).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 training/train_umi_standard_ib_tarb.py --arch intra_rb
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

from system import (MU_MIMO_System, SupervisedTrainer, CHOSEN_CONFIG,
                          DATASET_SIZE, NUM_TX, NUM_RX)
from datasets import CachedSionnaDataset
from precoders.signed_attention import (IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['intra_rb', 'ta_rb_residual'])
_p.add_argument('--finetune_epochs', type=int, default=80)
_p.add_argument('--warmup_epochs', type=int, default=3)
_args = _p.parse_args()

STEPS_PER_EPOCH = 150
LR              = 1e-3
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
BATCH_SIZE      = {'intra_rb': 128, 'ta_rb_residual': 256}[_args.arch]

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

    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    if arch == 'intra_rb':
        system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='intra_rb',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = IntraRBTransformerPrecoderSignedAttn(
            num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
            fft_size=system.rg.fft_size, rb_size=12, embed_dim=128,
            num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'IntraRB_signed_attn_4L_128d'
    else:  # ta_rb_residual
        system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='ta_rb_residual',
                                 embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6)
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(
            num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
            fft_size=system.rg.fft_size, rb_size=12, tokens_per_rb=6,
            embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
        run_name = 'TA_RB_residual_signed_attn_6tok_4L_128d'

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

    print(f'\n{"="*70}\nRÉSULTAT signed_attn [{arch}] ({_args.finetune_epochs}ep finetune, '
          f'train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    out = {'arch': arch, 'evals': evals, 'wmmse_ref': WMMSE_REF, 'pct_of_wmmse': pct,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs}
    with open(f'results/diag_front_a_signed_attn_{arch}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> results/diag_front_a_signed_attn_{arch}.json')
