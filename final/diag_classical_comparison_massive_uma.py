"""
diag_classical_comparison_massive_true.py — référence classique (RZF-Full,
RZF-RB12, WMMSE) pour MASSIVE_TRUE_CONFIG (M=64, K=8, canal STANDARD
R=20m, PAS de clustering serré artificiel). Même méthodologie que
classical_comparison.py (LockedClusterSystem, 10 batchs, batch=32).

Le calcul WMMSE à M=64 est déjà géré (eigh forcé CPU pour cette taille,
cf. precoders_w.py -- "LAPACK CPU gère ce cas ~35-1000x plus vite" -- pas
un problème nouveau introduit ici).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_classical_comparison_massive_true.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from wmmse_convergence_check import ConfigurableMIMOSystem
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder, wmmse_precoder
from channel_config import MASSIVE_TRUE_CONFIG, SNR_RANGE_DB, set_locked_topology
from sionna.phy.channel.tr38901 import UMa

BATCH_SIZE = 16   # réduit vs M8K4 (32) -- M=64 alourdit chaque instance WMMSE/RZF
NUM_BATCHES = 10
RB_SIZE = 12
WMMSE_ITERS = 10


class LockedClusterSystemMassiveUMa(ConfigurableMIMOSystem):
    cluster_radius_m   = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

    def __init__(self, num_tx, num_rx):
        super().__init__(num_tx, num_rx)
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
    name = f"M{M}K{K}_true_uma"
    print(f"\n{'='*70}\n{name}  (M={M}, K={K}, R={MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']}m)\n{'='*70}", flush=True)

    system = LockedClusterSystemMassiveUMa(M, K)
    results = {'snr': [], 'rzf_full': [], 'rzf_rb12': [], 'wmmse': []}
    ckpt_path = f'./results/classical_comparison_{name}_ckpt.npy'

    for snr in SNR_RANGE_DB:
        snr_t = tf.constant(float(snr), dtype=tf.float32)
        rates_full, rates_rb, rates_wmmse = [], [], []
        for _ in range(NUM_BATCHES):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)

            g_full = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            rates_full.append(float(system._sum_rate(h_freq, g_full, no)))

            g_rb = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=RB_SIZE)
            rates_rb.append(float(system._sum_rate(h_freq, g_rb, no)))

            r_wmmse = float(system.eval_wmmse_from_h(h_freq, no, WMMSE_ITERS))
            rates_wmmse.append(r_wmmse)

        results['snr'].append(float(snr))
        results['rzf_full'].append(float(np.mean(rates_full)))
        results['rzf_rb12'].append(float(np.mean(rates_rb)))
        results['wmmse'].append(float(np.mean(rates_wmmse)))
        print(f"  SNR={snr:5.1f} dB | RZF-Full={np.mean(rates_full):7.2f} | "
              f"RZF-RB12={np.mean(rates_rb):7.2f} | WMMSE={np.mean(rates_wmmse):7.2f} bps/Hz", flush=True)
        np.save(ckpt_path, dict(results, done=False))

    np.save(ckpt_path, dict(results, done=True))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results['snr'], results['rzf_full'], 'o-', label='RZF (Full CSI)')
    ax.plot(results['snr'], results['rzf_rb12'], 's--', label=f'RZF (RB={RB_SIZE})')
    ax.plot(results['snr'], results['wmmse'], '^-', label=f'WMMSE ({WMMSE_ITERS} iters)')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title(f'{name}: RZF vs WMMSE (canal STANDARD, pas de clustering forcé)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'./results/classical_comparison_{name}.png', dpi=150)
    fig.savefig(f'./results/classical_comparison_{name}.pdf')
    plt.close(fig)
    np.save(f'./results/classical_comparison_{name}.npy', results)

    print(f"\n  Gap moyen RZF-WMMSE : {np.mean([100*(w-r)/r for w,r in zip(results['wmmse'], results['rzf_full'])]):+.2f}% "
          f"(positif = WMMSE > RZF ; proche de 0 = channel hardening, résultat honnête à M=64)")
    print(f"\nAll done. Sauvé -> results/classical_comparison_{name}.{{npy,png,pdf}}")
