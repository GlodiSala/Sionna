"""
diag_massive_gap_diagnostic.py — investigation demandée (11/08, soir) :
le plafond CSI-parfait de MASSIVE_TRUE (M=64,K=8) est à 73-80% de RZF
(budget déjà étendu à 170ep, cf. results/diag_massive_true_*_extbudget.
json) -- comparaison RAPIDE (budget réduit, même recette pour tous les
variants) de plusieurs pistes pour fermer cet écart, AVANT de committer
un budget complet à la plus prometteuse :

  - baseline   : recette actuelle telle quelle (D=128,L=4, init aléatoire)
  - curriculum : blocs transformer (attn/FFN, D=128,L=4) TRANSPLANTÉS
                 depuis le checkpoint STANDARD (M=8,K=4) déjà convergé
                 -- seuls input_embed (dim dépend de feat_dim=3M+1) et
                 output_proj (dim dépend de 2M) sont réinitialisés
                 aléatoirement, tout le reste (attention/FFN/normes)
                 démarre du poids M=8 convergé. Vérifié : 75-111/78-123
                 variables selon l'archi sont shape-compatibles
                 (transplant direct), voir shapes exactes ci-dessous.
  - cap_d256l4 : capacité doublée (D=256, L=4, init aléatoire)
  - cap_d128l8 : profondeur doublée (D=128, L=8, init aléatoire)

Portée : MASSIVE_TRUE_CONFIG (M=64,K=8) UMi UNIQUEMENT -- STANDARD (M=8)
et UMa ne sont PAS touchés (checkpoints/scripts existants intacts).
Poids sauvés dans un dossier SÉPARÉ (./weights_massive_gap_diag/), les
checkpoints MASSIVE_TRUE_CONFIG de production restent intacts.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_massive_gap_diagnostic.py \
       --arch single_sc --variant curriculum
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
_p.add_argument('--variant', required=True,
                 choices=['baseline', 'curriculum', 'cap_d256l4', 'cap_d128l8'])
_p.add_argument('--warmup_epochs', type=int, default=2)
_p.add_argument('--finetune_epochs', type=int, default=18)
_p.add_argument('--tag', type=str, default='')
_args = _p.parse_args()

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
STEPS_PER_EPOCH = 150
LR              = 1e-3
DATASET_SIZE    = 4000
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_massive_true_{DATASET_SIZE//1000}k_{M}x{K}.npz'
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
BATCH_SIZE      = {'single_sc': 16, 'intra_rb': 16, 'ta_rb_residual': 32}[_args.arch]
WEIGHTS_DIR     = './weights_massive_gap_diag'   # dossier SÉPARÉ, ne touche jamais
                                                  # ./weights/MASSIVE_TRUE_*  (production)

# checkpoints STANDARD (M=8,K=4) déjà convergés, source du transplant curriculum
STANDARD_CKPT = {
    'single_sc':      './weights/front_a_signed_attn/best_20260808_103631',
    'intra_rb':       './weights/IntraRB_signed_attn_4L_128d/best_20260808_113308',
    'ta_rb_residual': './weights/TA_RB_residual_signed_attn_T4_4L_128d/best_20260808_181436',
}

D, L = {'baseline': (128, 4), 'curriculum': (128, 4),
        'cap_d256l4': (256, 4), 'cap_d128l8': (128, 8)}[_args.variant]


def build_precoder(arch, embed_dim, num_layers):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=embed_dim, num_heads=4, num_layers=num_layers, snr_aware=True)
    if arch == 'single_sc':
        return SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif arch == 'intra_rb':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)


def transplant(dst_precoder, src_ckpt_dir):
    """Copie position-à-position les variables shape-compatibles depuis un
    checkpoint STANDARD (M=8) déjà entraîné dans le précoder M=64 flambant
    neuf. input_embed (dépend de feat_dim=3M+1) et output_proj (dépend de
    2M) diffèrent en shape -> restent en init aléatoire. Le reste (poids
    d'attention signée + FFN + LayerNorm, indépendants de M/K) est copié
    tel quel."""
    with open(os.path.join(src_ckpt_dir, 'weights.pkl'), 'rb') as f:
        src_weights = pickle.load(f)
    dst_vars = dst_precoder.trainable_variables
    assert len(src_weights) == len(dst_vars), \
        f"nb variables différent: src={len(src_weights)} dst={len(dst_vars)} -- vérifier D/L identiques (128/4)"
    n_copied, n_skipped = 0, 0
    for v, w in zip(dst_vars, src_weights):
        if tuple(v.shape) == tuple(w.shape):
            v.assign(w)
            n_copied += 1
        else:
            n_skipped += 1
    print(f"🔀 Transplant depuis {src_ckpt_dir} : {n_copied} copiées, {n_skipped} réinitialisées (shape M-dépendante)")
    return n_copied, n_skipped


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
        indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'], seed=42)

    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch,
                             embed_dim=D, num_heads=4, num_layers=L,
                             tokens_per_rb=(6 if arch == 'ta_rb_residual' else 1))
    system.precoder = build_precoder(arch, D, L)

    # force build (Keras lazy) avant transplant / avant de compter les variables
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)

    if variant == 'curriculum':
        transplant(system.precoder, STANDARD_CKPT[arch])

    run_name = f'{arch}_{variant}{_args.tag}'
    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')
    trainer.ckpt.run_dir = os.path.join(WEIGHTS_DIR, run_name)
    os.makedirs(trainer.ckpt.run_dir, exist_ok=True)

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nDIAGNOSTIC [{arch}/{variant}] '
          f'D={D} L={L} ({_args.finetune_epochs}ep, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f}')

    out = {'arch': arch, 'variant': variant, 'D': D, 'L': L, 'evals': evals,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs, 'M': M, 'K': K}
    out_path = f'results/diag_massive_gap_{arch}_{variant}{_args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> {out_path}')
