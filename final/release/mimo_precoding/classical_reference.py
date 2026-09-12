"""
classical_reference.py — RZF (full CSI + per-RB-grouped) and WMMSE sum-
rate reference for a given channel x scale regime. Must be run once per
regime before train.py's %WMMSE normalization can be computed (train.py
looks for results/classical_reference_{channel}_{scale}.npy).

Usage:
    python3 classical_reference.py --channel umi --scale standard
    python3 classical_reference.py --channel uma --scale massive
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import UMa
from eval_system import ConfigurableMIMOSystem
from precoders.classical import rzf_precoder, wmmse_precoder
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from channel_config import STANDARD_CONFIG, MASSIVE_TRUE_CONFIG, SNR_RANGE_DB, set_locked_topology

SCALE_CONFIG = {'standard': STANDARD_CONFIG, 'massive': MASSIVE_TRUE_CONFIG}
BATCH_SIZE = {'standard': 32, 'massive': 16}   # M=64 needs a smaller batch (OOM otherwise)
NUM_BATCHES = 10
RB_SIZE = 12
WMMSE_ITERS = 10


class LockedSystem(ConfigurableMIMOSystem):
    def __init__(self, num_tx, num_rx, channel, cluster_radius_m, indoor_probability):
        super().__init__(num_tx, num_rx)
        self.cluster_radius_m = cluster_radius_m
        self.indoor_probability = indoor_probability
        if channel == 'uma':
            self.channel_model = UMa(
                carrier_frequency=2.6e9, o2i_model='low',
                ut_array=self.ut_array, bs_array=self.bs_array,
                direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--channel', required=True, choices=['umi', 'uma'])
    p.add_argument('--scale', required=True, choices=['standard', 'massive'])
    args = p.parse_args()

    os.makedirs('results', exist_ok=True)
    cfg = SCALE_CONFIG[args.scale]
    M, K, R = cfg['NUM_TX'], cfg['NUM_RX'], cfg['CLUSTER_RADIUS_M']
    name = f'{args.channel}_{args.scale}'
    print(f"\n{'='*70}\n{name}  (M={M}, K={K}, R={R}m)\n{'='*70}", flush=True)

    system = LockedSystem(M, K, args.channel, R, cfg['INDOOR_PROBABILITY'])
    results = {'snr': [], 'rzf_full': [], 'rzf_rb12': [], 'wmmse': []}
    batch = BATCH_SIZE[args.scale]

    for snr in SNR_RANGE_DB:
        snr_t = tf.constant(float(snr), dtype=tf.float32)
        rates_full, rates_rb, rates_wmmse = [], [], []
        for _ in range(NUM_BATCHES):
            system.new_topology(batch)
            h_freq, no = system.channel_and_no(tf.constant(batch, tf.int32), snr_t)
            g_full = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            rates_full.append(float(system._sum_rate(h_freq, g_full, no)))
            g_rb = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=RB_SIZE)
            rates_rb.append(float(system._sum_rate(h_freq, g_rb, no)))
            rates_wmmse.append(float(system.eval_wmmse_from_h(h_freq, no, WMMSE_ITERS)))

        results['snr'].append(float(snr))
        results['rzf_full'].append(float(np.mean(rates_full)))
        results['rzf_rb12'].append(float(np.mean(rates_rb)))
        results['wmmse'].append(float(np.mean(rates_wmmse)))
        print(f"  SNR={snr:5.1f} dB | RZF-Full={np.mean(rates_full):7.2f} | "
              f"RZF-RB12={np.mean(rates_rb):7.2f} | WMMSE={np.mean(rates_wmmse):7.2f} bps/Hz", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results['snr'], results['rzf_full'], 'o-', label='RZF (Full CSI)')
    ax.plot(results['snr'], results['rzf_rb12'], 's--', label=f'RZF (RB={RB_SIZE})')
    ax.plot(results['snr'], results['wmmse'], '^-', label=f'WMMSE ({WMMSE_ITERS} iters)')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title(f'{name}: RZF vs WMMSE')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'results/classical_reference_{name}.png', dpi=150)
    plt.close(fig)
    np.save(f'results/classical_reference_{name}.npy', results)

    gap = np.mean([100 * (w - r) / r for w, r in zip(results['wmmse'], results['rzf_full'])])
    print(f"\n  Gap moyen RZF-WMMSE : {gap:+.2f}% (proche de 0 = channel hardening, "
          f"attendu à M=64 -- Marzetta 2010)")
    print(f"\nSauvé -> results/classical_reference_{name}.{{npy,png}}")
