"""
diag_short_lag_correlation.py — Investigation Partie 1, hypothèse #1 :
corrélation fréquentielle à lags COURTS (1-12 SC, intra-RB), pas
seulement à 95 SC comme mesuré précédemment (channel_config.py §1).

Si rho chute déjà fortement dans les 12 SC d'un RB, la SC-attention n'a
structurellement pas grand-chose à exploiter à CETTE granularité, même
si une corrélation à longue distance existe (rho@95sc~0.31-0.37).

Batch réduit (16) pour rester léger sur le GPU partagé avec le run
MASSIVE en cours.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_short_lag_correlation.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG, set_locked_topology

BATCH = 16
NUM_DRAWS = 20   # nb de tirages de topologie indépendants, moyennés
LAGS = list(range(1, 13)) + [24, 48, 95]   # 1..12 = intra-RB (RB=12), reste = référence


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = 20.0
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def freq_correlation(h_freq_np, lags):
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]   # [B,K,M,fft] (1 symbole)
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    return {lag: float(np.abs(np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))) / den)
            for lag in lags}


def measure(name, cfg):
    M, K, R = cfg['NUM_TX'], cfg['NUM_RX'], cfg['CLUSTER_RADIUS_M']
    print(f'\n{"="*70}\n{name} M={M} K={K} R={R}m\n{"="*70}', flush=True)
    system = LockedSystem(M, K)
    system.cluster_radius_m = R

    all_corrs = {lag: [] for lag in LAGS}
    for _ in range(NUM_DRAWS):
        system.new_topology(BATCH)
        h_freq, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(15.0, tf.float32))
        c = freq_correlation(h_freq.numpy(), LAGS)
        for lag in LAGS:
            all_corrs[lag].append(c[lag])

    result = {lag: float(np.mean(all_corrs[lag])) for lag in LAGS}
    result_std = {lag: float(np.std(all_corrs[lag])) for lag in LAGS}
    print(f'{"lag(SC)":>8} {"rho moy":>9} {"rho std":>9}')
    for lag in LAGS:
        tag = '  <-- intra-RB (RB=12)' if lag <= 12 else ''
        print(f'{lag:>8} {result[lag]:>9.4f} {result_std[lag]:>9.4f}{tag}')

    # Décroissance relative DANS le RB : rho(12)/rho(1)
    decay_in_rb = result[12] / result[1] if result[1] > 0 else float('nan')
    print(f'\n  Décroissance intra-RB : rho(lag=12)/rho(lag=1) = {decay_in_rb:.3f} '
          f'(1.0=pas de décroissance, 0=corrélation nulle en bord de RB)')
    return {'corr_mean': result, 'corr_std': result_std, 'decay_in_rb_12_over_1': decay_in_rb}


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    out = {}
    for name, cfg in [('STANDARD', STANDARD_CONFIG), ('MASSIVE', MASSIVE_CONFIG)]:
        out[name] = measure(name, cfg)
        tf.keras.backend.clear_session()

    with open('results/diag_short_lag_correlation.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_short_lag_correlation.json')
