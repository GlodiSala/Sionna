"""
Final classical-methods comparison (RZF-Full, RZF-RB-grouped, WMMSE) over
the full EVALUATION_SNR_RANGE, at both locked channel configs
(STANDARD_CONFIG = M8K4, MASSIVE_CONFIG = M32K8). No transformer training
required -- this stands on its own as a complete, presentable result.

REVISION 2026-08-07: was still on the pre-clustering mechanism (HALF_ANGLE_
DEG/FORCE_LOS narrow-window+LOS), superseded by channel_config.py's
REVISION (b) (spatial user clustering, CLUSTER_RADIUS_M). Fixed to draw
topologies via channel_config.set_locked_topology like every other
Stage3/4 script (see SESSION_NUIT_RESUME.md §0/§1) -- this is the
"official" reference table the fixed datasets.py training pipeline is
validated against, so it has to use the exact same channel mechanism.

Reuses rzf_precoder_with_rb_grouping from precoders/rb_grouping.py (RB=12,
matching the architectures' default rb_size) and rzf_precoder/
wmmse_precoder from precoders/classical.py, on the locked topology mechanism from
channel_config.py.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from eval_system import ConfigurableMIMOSystem
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from precoders.classical import rzf_precoder, wmmse_precoder
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG, SNR_RANGE_DB, set_locked_topology

BATCH_SIZE = 32
NUM_BATCHES = 10
RB_SIZE = 12
WMMSE_ITERS = 10


class LockedClusterSystem(ConfigurableMIMOSystem):
    """ConfigurableMIMOSystem with new_topology overridden to draw from the
    locked spatial-clustering mechanism (channel_config.set_locked_topology)
    instead of the base class's narrow-angle-window gen_topology_custom."""
    cluster_radius_m   = 20.0
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False,
                             indoor_probability=self.indoor_probability)


def evaluate_config(name, cfg, checkpoint_path=None):
    """checkpoint_path: if given, results are saved to this .npy path after
    EVERY SNR point (not just at the end), so a long run (e.g. MASSIVE on
    CPU, ~1hr) survives an interruption with partial results intact.
    Prints are flushed immediately so progress is visible even when stdout
    is redirected to a file (block-buffered by default in that case)."""
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    print(f"\n{'='*70}\n{name}  (M={M}, K={K}, R={cfg['CLUSTER_RADIUS_M']}m)\n{'='*70}", flush=True)
    system = LockedClusterSystem(M, K)
    system.cluster_radius_m   = cfg['CLUSTER_RADIUS_M']
    system.indoor_probability = cfg['INDOOR_PROBABILITY']
    results = {'snr': [], 'rzf_full': [], 'rzf_rb12': [], 'wmmse': []}

    for snr in SNR_RANGE_DB:
        snr_t = tf.constant(float(snr), dtype=tf.float32)
        rates_full, rates_rb, rates_wmmse = [], [], []
        for _ in range(NUM_BATCHES):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(
                tf.constant(BATCH_SIZE, tf.int32), snr_t)

            g_full = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            rates_full.append(float(system._sum_rate(h_freq, g_full, no)))

            g_rb = rzf_precoder_with_rb_grouping(
                h_freq, stream_management=system.sm, rb_size=RB_SIZE)
            rates_rb.append(float(system._sum_rate(h_freq, g_rb, no)))

            r_wmmse = float(system.eval_wmmse_from_h(h_freq, no, WMMSE_ITERS))
            rates_wmmse.append(r_wmmse)

        results['snr'].append(float(snr))
        results['rzf_full'].append(float(np.mean(rates_full)))
        results['rzf_rb12'].append(float(np.mean(rates_rb)))
        results['wmmse'].append(float(np.mean(rates_wmmse)))
        print(f"  SNR={snr:5.1f} dB | RZF-Full={np.mean(rates_full):7.2f} | "
              f"RZF-RB12={np.mean(rates_rb):7.2f} | "
              f"WMMSE={np.mean(rates_wmmse):7.2f} bps/Hz", flush=True)

        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
            np.save(checkpoint_path, dict(results, done=False))

    if checkpoint_path is not None:
        np.save(checkpoint_path, dict(results, done=True))

    return results


def plot_and_save(name, results, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results['snr'], results['rzf_full'], 'o-', label='RZF (Full CSI)')
    ax.plot(results['snr'], results['rzf_rb12'], 's--', label=f'RZF (RB={RB_SIZE})')
    ax.plot(results['snr'], results['wmmse'], '^-', label=f'WMMSE ({WMMSE_ITERS} iters)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title(f'{name}: RZF vs WMMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(save_dir, f'classical_comparison_{name}.png')
    pdf_path = os.path.join(save_dir, f'classical_comparison_{name}.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    npy_path = os.path.join(save_dir, f'classical_comparison_{name}.npy')
    np.save(npy_path, results)

    print(f"  Saved: {png_path}, {pdf_path}, {npy_path}")


if __name__ == '__main__':
    for cfg in (STANDARD_CONFIG, MASSIVE_CONFIG):
        name = f"M{cfg['NUM_TX']}K{cfg['NUM_RX']}"
        results = evaluate_config(
            name, cfg,
            checkpoint_path=f'./results/classical_comparison_{name}_ckpt.npy')
        plot_and_save(name, results)

    print("\nAll done.")
