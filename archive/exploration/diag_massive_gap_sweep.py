"""
diag_massive_gap_sweep.py — version consolidée de diag_massive_gap_
diagnostic.py : charge le dataset MASSIVE_TRUE UNE SEULE FOIS (fichier
cache 1.6GB, décompression coûteuse -- machine partagée sous forte
contention CPU/RAM ce soir, cf. SESSION_LOG) puis enchaîne plusieurs
variants dans le MÊME process, au lieu de relancer un script par
variant (qui rechargerait le cache à chaque fois).

Portée : MASSIVE_TRUE_CONFIG (M=64,K=8) UMi uniquement. Poids dans
./weights_massive_gap_diag/ (dossier séparé de la production). Mêmes
variants que diag_massive_gap_diagnostic.py (baseline/curriculum/
cap_d256l4/cap_d128l8), même transplant/logique -- voir ce fichier pour
le détail commenté du raisonnement.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_massive_gap_sweep.py \
       --arch single_sc --variants baseline,curriculum,cap_d256l4,cap_d128l8
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
_p.add_argument('--variants', type=str, default='baseline,curriculum,cap_d256l4,cap_d128l8')
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
WEIGHTS_DIR     = './weights_massive_gap_diag'

STANDARD_CKPT = {
    'single_sc':      './weights/front_a_signed_attn/best_20260808_103631',
    'intra_rb':       './weights/IntraRB_signed_attn_4L_128d/best_20260808_113308',
    'ta_rb_residual': './weights/TA_RB_residual_signed_attn_T4_4L_128d/best_20260808_181436',
}

VARIANT_DL = {'baseline': (128, 4), 'curriculum': (128, 4),
              'cap_d256l4': (256, 4), 'cap_d128l8': (128, 8),
              'cap_d384l4': (384, 4), 'cap_d256l8': (256, 8),
              'cap_d512l4': (512, 4)}


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
    with open(os.path.join(src_ckpt_dir, 'weights.pkl'), 'rb') as f:
        src_weights = pickle.load(f)
    dst_vars = dst_precoder.trainable_variables
    assert len(src_weights) == len(dst_vars), \
        f"nb variables différent: src={len(src_weights)} dst={len(dst_vars)}"
    n_copied, n_skipped = 0, 0
    for v, w in zip(dst_vars, src_weights):
        if tuple(v.shape) == tuple(w.shape):
            v.assign(w); n_copied += 1
        else:
            n_skipped += 1
    print(f"🔀 Transplant depuis {src_ckpt_dir} : {n_copied} copiées, {n_skipped} réinitialisées")
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


def run_variant(arch, variant, dataset, tag):
    D, L = VARIANT_DL[variant]
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch,
                             embed_dim=D, num_heads=4, num_layers=L,
                             tokens_per_rb=(6 if arch == 'ta_rb_residual' else 1))
    system.precoder = build_precoder(arch, D, L)
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)

    if variant == 'curriculum':
        transplant(system.precoder, STANDARD_CKPT[arch])

    run_name = f'{arch}_{variant}{tag}'
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

    print(f'\n{"="*70}\nDIAGNOSTIC [{arch}/{variant}] D={D} L={L} '
          f'({_args.finetune_epochs}ep, train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f}')

    out = {'arch': arch, 'variant': variant, 'D': D, 'L': L, 'evals': evals,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs, 'M': M, 'K': K}
    out_path = f'results/diag_massive_gap_{arch}_{variant}{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Sauvé -> {out_path}', flush=True)
    del system, trainer
    return out


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    arch = _args.arch
    variants = _args.variants.split(',')

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    t0 = time.time()
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=32,
        cache_file=CACHE_FILE,
        cluster_radius_m=MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'], seed=42)
    print(f'⏱️ Dataset chargé en {(time.time()-t0)/60:.1f}min (une seule fois pour tous les variants)', flush=True)

    all_out = {}
    for v in variants:
        print(f'\n\n{"#"*70}\n# VARIANT: {v}\n{"#"*70}', flush=True)
        all_out[v] = run_variant(arch, v, dataset, _args.tag)

    print(f'\n\n{"="*70}\nRÉCAPITULATIF [{arch}]\n{"="*70}')
    for v, o in all_out.items():
        print(f'{v:15s} D={o["D"]:3d} L={o["L"]} | ' +
              ' | '.join(f'{s}dB={o["evals"][s]:.1f}' for s in EVAL_SNRS))
