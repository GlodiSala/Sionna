"""
diag_front_a_variant_train.py — Front A (demande utilisateur, exploration
libre du décrochage haut-SNR canal-indépendant) : entraîne UNE variante de
SingleSC (le plus rapide) à budget réduit (30ep finetune par défaut) sur
UMi, évalue sur [0,5,10,15,17.5,20]dB contre WMMSE, sauve un JSON
directement comparable entre variantes.

Variantes :
  baseline      SingleSCTransformerPrecoder inchangé (référence, même
                budget/protocole que les autres, pour comparaison honnête)
  gram          + features Gram matrix H^H.H (precoder_experimental.py)
  snr_weighted  loss finetune pondérée par le SNR du step (poids linéaire
                1x à SNR_MIN_TRAIN -> 3x à SNR_MAX_TRAIN)
  bilinear      + couche de mixing bilinéaire signé avant output_proj
                (precoder_experimental.py)

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_front_a_variant_train.py --variant gram
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
                          DATASET_SIZE, NUM_TX, NUM_RX,
                          SNR_MIN_TRAIN, SNR_MAX_TRAIN)
from datasets import CachedSionnaDataset
from precoder_experimental import (SingleSCTransformerPrecoderGram,
                                    SingleSCTransformerPrecoderBilinearOut,
                                    SingleSCTransformerPrecoderSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--variant', required=True,
                 choices=['baseline', 'gram', 'snr_weighted', 'bilinear', 'signed_attn'])
_p.add_argument('--finetune_epochs', type=int, default=30)
_p.add_argument('--warmup_epochs', type=int, default=3)
_p.add_argument('--snr_weight_max', type=float, default=3.0)
_args = _p.parse_args()

STEPS_PER_EPOCH = 150
BATCH_SIZE      = 128
LR              = 1e-3
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20

_ref = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}


class SNRWeightedTrainer(SupervisedTrainer):
    """Front A idée #2 -- pondère la loss finetune par le SNR du step
    (poids linéaire 1x à SNR_MIN_TRAIN -> W_MAX x à SNR_MAX_TRAIN). Seule
    _finetune_step change vs SupervisedTrainer ; warmup MSE-RZF identique."""

    def __init__(self, *args, w_max=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_max = w_max

    @tf.function
    def _finetune_step(self, h_freq):
        snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
        no = ebnodb2no(snr_db, self.system.num_bits_per_symbol, 0.5, self.system.rg)
        weight = 1.0 + (self.w_max - 1.0) * (snr_db - SNR_MIN_TRAIN) / (SNR_MAX_TRAIN - SNR_MIN_TRAIN)

        with tf.GradientTape() as tape:
            g = self.system._call_precoder(h_freq, no, training=True, return_real_imag=True)
            h_eff = self.system.ch_helper.compute_effective_channel(h_freq, g)
            sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate = tf.reduce_sum(tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
                axis=[0, 1, 2, 4]))
            loss = -tf.where(tf.math.is_finite(rate),
                              weight * rate / self.rate_norm,
                              tf.constant(0.0))

        grads = tape.gradient(loss, self.vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.opt.apply_gradients(zip(grads, self.vars))
        return loss, tf.linalg.global_norm(grads)


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

    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='single_sc',
                             embed_dim=128, num_heads=4, num_layers=4,
                             use_abs=True, use_cossin=False)

    variant = _args.variant
    if variant == 'gram':
        system.precoder = SingleSCTransformerPrecoderGram(
            num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
            fft_size=system.rg.fft_size, embed_dim=128, num_heads=4, num_layers=4,
            snr_aware=True, use_abs=True, use_cossin=False)
    elif variant == 'bilinear':
        system.precoder = SingleSCTransformerPrecoderBilinearOut(
            num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
            fft_size=system.rg.fft_size, embed_dim=128, num_heads=4, num_layers=4,
            snr_aware=True, use_abs=True, use_cossin=False)
    elif variant == 'signed_attn':
        system.precoder = SingleSCTransformerPrecoderSignedAttn(
            num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
            fft_size=system.rg.fft_size, embed_dim=128, num_heads=4, num_layers=4,
            snr_aware=True, use_abs=True, use_cossin=False)
    # baseline / snr_weighted : system.precoder reste le SingleSC standard

    TrainerCls = SNRWeightedTrainer if variant == 'snr_weighted' else SupervisedTrainer
    trainer_kwargs = dict(
        run_name=f'front_a_{variant}',
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')
    if variant == 'snr_weighted':
        trainer = TrainerCls(system, dataset, w_max=_args.snr_weight_max, **trainer_kwargs)
    else:
        trainer = TrainerCls(system, dataset, **trainer_kwargs)

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=999)
    train_time = time.time() - t0

    evals = {snr: eval_at_snr(system, dataset, snr) for snr in EVAL_SNRS}
    pct = {snr: 100.0 * evals[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nRÉSULTAT Front A [{variant}] ({_args.finetune_epochs}ep finetune, '
          f'train={train_time/60:.1f}min)\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | model={evals[snr]:7.2f} | WMMSE={WMMSE_REF[snr]:7.2f} | {pct[snr]:5.1f}% WMMSE')

    out = {'variant': variant, 'evals': evals, 'wmmse_ref': WMMSE_REF, 'pct_of_wmmse': pct,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs}
    with open(f'results/diag_front_a_{variant}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauvé -> results/diag_front_a_{variant}.json')
