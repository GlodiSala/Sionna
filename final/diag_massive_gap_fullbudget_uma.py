"""
diag_massive_gap_fullbudget_uma.py — réplique de diag_massive_gap_
fullbudget.py (UMi, session du 12/08, cf. RAPPORT_MASSIVE_GAP_CLOSING_
20260812.md) pour le canal UMa MASSIVE (M=64,K=8). Diagnostic à coût
réduit déjà tranché sur UMi (D=384,L=4 = point de rendements
décroissants, effet générique indépendant de l'architecture) -- pas
refait ici sauf signal contraire. Run direct à budget complet.

Différences vs la version UMi :
  - `scenario='uma'` passé à CachedSionnaDataset (au lieu du défaut 'umi')
  - cache file `sionna_joint_uma_massive_4k_64x8.npz` (au lieu de
    `sionna_joint_massive_true_4k_64x8.npz`) -- même fichier que celui
    déjà utilisé par `diag_uma_massive_train.py` pour le run D=128 de
    référence (réutilisé tel quel, pas régénéré)
  - RUN_NAMES préfixés `UMa_MASSIVE_...` (au lieu de `MASSIVE_TRUE_...`)
    -- cohérent avec le nommage déjà utilisé par les checkpoints D=128
    UMa existants (`UMa_MASSIVE_SingleSC_signed_attn_4L_128d/...`)

Poids sauvés dans ./weights/ sous un nom de run DISTINCT des
checkpoints D=128 existants (`UMa_MASSIVE_*` sans suffixe, intacts) --
suffixe `_capD384L4`.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_massive_gap_fullbudget_uma.py \
       --arch single_sc --variant cap_d384l4
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer
from datasets import CachedSionnaDataset
from channel_config import MASSIVE_TRUE_CONFIG
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb', 'ta_rb_residual'])
_p.add_argument('--variant', required=True, choices=['cap_d384l4'])   # seule variante retenue, cf. RAPPORT du 12/08
_p.add_argument('--warmup_epochs', type=int, default=10)
_p.add_argument('--finetune_epochs', type=int, default=160)
_args = _p.parse_args()

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
STEPS_PER_EPOCH = 150
LR              = 1e-3
DATASET_SIZE    = 4000
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_uma_massive_{DATASET_SIZE//1000}k_{M}x{K}.npz'   # même cache que diag_uma_massive_train.py
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
BATCH_SIZE      = {'single_sc': 16, 'intra_rb': 16, 'ta_rb_residual': 32}[_args.arch]

D, L = {'cap_d384l4': (384, 4)}[_args.variant]
SUFFIX = {'cap_d384l4': '_capD384L4'}[_args.variant]

RUN_NAMES = {
    'single_sc':      'UMa_MASSIVE_SingleSC_signed_attn_4L_128d' + SUFFIX,
    'intra_rb':       'UMa_MASSIVE_IntraRB_signed_attn_4L_128d' + SUFFIX,
    'ta_rb_residual': 'UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d' + SUFFIX,
}


def build_precoder(arch, embed_dim, num_layers):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=embed_dim, num_heads=4, num_layers=num_layers, snr_aware=True)
    if arch == 'single_sc':
        return SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif arch == 'intra_rb':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)


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
    arch, variant = _args.arch, _args.variant

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=32,
        cache_file=CACHE_FILE,
        cluster_radius_m=MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'],
        scenario='uma', seed=42)

    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch,
                             embed_dim=D, num_heads=4, num_layers=L,
                             tokens_per_rb=(6 if arch == 'ta_rb_residual' else 1))
    system.precoder = build_precoder(arch, D, L)
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)

    run_name = RUN_NAMES[arch]
    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')
    # run_dir défaut = ./weights/<run_name> -- DISTINCT de UMa_MASSIVE_*
    # (D=128, production, non touché), déjà garanti par RUN_NAMES ci-dessus.

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}

    flops, params, acts = system.precoder.complexity(system.rg.num_ofdm_symbols, convention_x_ofdm=False)
    real_params = int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables))

    print(f'\n{"="*70}\nRÉSULTAT FULLBUDGET UMa [{arch}/{variant}] D={D} L={L} '
          f'({_args.finetune_epochs}ep, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f}')
    print(f'  FLOPs_reel={flops/1e6:.1f}M  params={real_params:,}')

    out = {'arch': arch, 'variant': variant, 'D': D, 'L': L, 'evals': evals,
           'flops_real_M': flops / 1e6, 'params': real_params,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs, 'M': M, 'K': K}
    out_path = f'results/diag_massive_gap_fullbudget_uma_{arch}_{variant}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> {out_path}')
