"""
diag_classical_comparison_uma.py — demande utilisateur : classical_comparison
(RZF-Full, RZF-RB12, WMMSE) n'existait que pour UMi (M8K4/M32K8, cf.
results/classical_comparison_M8K4.npy) -- jamais régénéré sous UMa, alors
que single_sc/intra_rb ONT un run complet 83ep sur UMa (diag_uma_quick_
train.py, weights/uma_quick_{single_sc,intra_rb}).

Mirroir exact de classical_comparison.py::LockedClusterSystem/evaluate_config,
seule différence : canal UMa au lieu de UMi (même pattern que diag_uma_
selectivity.py::UMaSystem), même config M8K4/R=20m (STANDARD_CONFIG) pour
être directement comparable aux runs neuronaux UMa déjà faits. Pas de
MASSIVE (aucun run neuronal UMa MASSIVE n'existe, pas demandé).

Aucun entraînement -- RZF/WMMSE seulement, ~même coût que classical_
comparison.py sur M8K4 (quelques minutes).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_classical_comparison_uma.py
"""
import os
import sys
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

from sionna.phy.channel.tr38901 import UMa
from wmmse_convergence_check import ConfigurableMIMOSystem
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder, wmmse_precoder
from channel_config import STANDARD_CONFIG, SNR_RANGE_DB, gen_topology_clustered

BATCH_SIZE = 32
NUM_BATCHES = 10
RB_SIZE = 12
WMMSE_ITERS = 10


class UMaLockedClusterSystem(ConfigurableMIMOSystem):
    """LockedClusterSystem (classical_comparison.py) avec UMa au lieu de
    UMi -- même ut_array/bs_array/rg (signature de constructeur partagée,
    cf. datasets.py commentaire), seul le channel_model et new_topology
    changent."""
    cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

    def __init__(self, num_tx, num_rx):
        super().__init__(num_tx, num_rx)
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        topology = gen_topology_clustered(
            batch_size, self.num_rx, 'uma', self.cluster_radius_m,
            indoor_probability=self.indoor_probability)
        self.channel_model.set_topology(*topology, los=False)


def evaluate_config(name, cfg, checkpoint_path=None):
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    print(f"\n{'='*70}\n{name} (UMa)  (M={M}, K={K}, R={cfg['CLUSTER_RADIUS_M']}m)\n{'='*70}", flush=True)
    system = UMaLockedClusterSystem(M, K)
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
    ax.set_title(f'{name} (UMa): RZF vs WMMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(save_dir, f'classical_comparison_{name}_uma.png')
    pdf_path = os.path.join(save_dir, f'classical_comparison_{name}_uma.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    npy_path = os.path.join(save_dir, f'classical_comparison_{name}_uma.npy')
    np.save(npy_path, results)

    print(f"  Saved: {png_path}, {pdf_path}, {npy_path}")


if __name__ == '__main__':
    name = f"M{STANDARD_CONFIG['NUM_TX']}K{STANDARD_CONFIG['NUM_RX']}"
    results = evaluate_config(
        name, STANDARD_CONFIG,
        checkpoint_path=f'./results/classical_comparison_{name}_uma_ckpt.npy')
    plot_and_save(name, results)
    print("\nAll done.")
