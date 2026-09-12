"""
diag_grad_vs_snr.py — Bloc B.2 : magnitude du gradient de la loss sum-rate
en fonction du SNR, sur le checkpoint IntraRB_fullres déjà entraîné (état
du plateau observé), pour tester si log2(1+SINR) aplati à haut SINR
explique le ralentissement de convergence à haut SNR.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_grad_vs_snr.py
"""
import os, sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from datasets import CachedSionnaDataset
from main_finall import MU_MIMO_System
from sionna.phy.utils import ebnodb2no

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

NUM_TX, NUM_RX = 8, 4
SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
N_BATCHES = 20
BATCH = 64

system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='intra_rb',
                         embed_dim=128, num_heads=4, num_layers=4)
dataset = CachedSionnaDataset(
    system, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=123)

dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 96], dtype=tf.complex64)
system.precoder(dummy_h, no=tf.constant(1e-3), training=False)
vars_ = system.precoder.trainable_variables


def run_at(weights_label, load_path):
    if load_path:
        ok = system.load_weights_from(load_path)
        assert ok
    else:
        # ré-initialise (poids aléatoires) pour comparer à l'état "début d'entraînement"
        for v in vars_:
            v.assign(tf.keras.initializers.get(
                v.initializer if hasattr(v, 'initializer') else 'glorot_uniform'
            )(v.shape) if False else v)  # no-op safe fallback

    rate_norm = float(system.num_users) * 9.0
    print(f'\n=== {weights_label} ===')
    print(f'{"SNR":>6} {"SINR_moy":>10} {"rate/user":>10} {"|dL/dvars|":>12} {"gnorm/param":>12}')

    for snr in SNRS:
        gnorms, sinrs, rates = [], [], []
        for _ in range(N_BATCHES):
            h = dataset.get_batch(BATCH)
            no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol,
                            0.5, system.rg)
            with tf.GradientTape() as tape:
                g = system._call_precoder(h, no, training=True, return_real_imag=True)
                h_eff = system.ch_helper.compute_effective_channel(h, g)
                sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
                rate = tf.reduce_sum(tf.reduce_mean(
                    tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
                    axis=[0, 1, 2, 4]))
                loss = -rate / rate_norm
            grads = tape.gradient(loss, vars_)
            grads = [tf.where(tf.math.is_finite(gr), gr, tf.zeros_like(gr)) for gr in grads]
            gnorms.append(float(tf.linalg.global_norm(grads)))
            sinrs.append(float(tf.reduce_mean(sinr)))
            rates.append(float(rate) / system.num_users)

        n_params = sum(int(tf.size(v)) for v in vars_)
        gn_mean = np.mean(gnorms)
        print(f'{snr:6.1f} {np.mean(sinrs):10.3f} {np.mean(rates):10.3f} '
              f'{gn_mean:12.5f} {gn_mean/np.sqrt(n_params):12.7f}')
    return


print('\n########## CHECKPOINT ENTRAÎNÉ (état du plateau, fin epoch 6) ##########')
run_at('IntraRB_fullres (checkpoint entraîné)', 'weights/IntraRB_fullres/weights.pkl')
