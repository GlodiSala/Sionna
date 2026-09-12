"""
diag_rzf_grouped_for_tsweep.py — référence RZF-RB-groupé pour la figure 3
(T-sweep, demande utilisateur) : compare chaque T de TA-RB-résiduel-
signed_attn à un RZF groupé par SC avec une taille de groupe ÉQUIVALENTE
(group_size = rb_size/T -- même compression fréquentielle que le nombre de
tokens du modèle à ce T). T=1 (1 token/RB) <-> RZF-RB12 (tout le RB
partage un seul précodeur, compression max) ... T=12 (12 tokens/RB, pas
de compression) <-> RZF-Full (aucun groupement).

Méthodologie identique à classical_comparison.py (LockedClusterSystem,
10 batchs, batch=32) mais aux 3 SNR du T-sweep (0/10/20dB) pour
comparabilité directe.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_rzf_grouped_for_tsweep.py
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
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from channel_config import STANDARD_CONFIG, set_locked_topology

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
BATCH_SIZE = 32
NUM_BATCHES = 10
RB_SIZE = 12
SNR_POINTS = [0.0, 10.0, 20.0]
T_VALUES = [1, 2, 3, 4, 6, 12]


class LockedClusterSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    system = LockedClusterSystem(M, K)
    results = {}

    for T in T_VALUES:
        group_size = RB_SIZE // T
        print(f'\nT={T} (group_size={group_size}) ...', flush=True)
        results[str(T)] = {'group_size': group_size, 'rates': {}}
        for snr in SNR_POINTS:
            snr_t = tf.constant(snr, tf.float32)
            rates = []
            for _ in range(NUM_BATCHES):
                system.new_topology(BATCH_SIZE)
                h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
                g = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=group_size)
                rates.append(float(system._sum_rate(h_freq, g, no)))
            results[str(T)]['rates'][str(snr)] = float(np.mean(rates))
            print(f'  SNR={snr:5.1f}dB | RZF-group{group_size:>2d} = {results[str(T)]["rates"][str(snr)]:.2f}', flush=True)

    with open('results/diag_rzf_grouped_for_tsweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSauvé -> results/diag_rzf_grouped_for_tsweep.json')
