"""
Check whether the small RZF-ahead-of-WMMSE gap seen at SNR=15/20 in the
final classical comparison (M8K4: ~0.6%/0.2%, M64K32: ~0.09%/0.09%) is the
known WMMSE-vs-RZF metric-mismatch mechanism showing an extended SNR
footprint under the hardened/loaded config, or something new.

Method: run the SAME paired methodology on the wide-angle/NLOS REFERENCE
(low-loading-equivalent, near-orthogonal) at the same M, same SNR points
(15, 20 dB). If the reference stays clean (WMMSE >= RZF, gap <= ~0) while
the hard config shows the small positive gap, that confirms it's the known
mechanism extending further under loading. If the reference ALSO shows a
similar small positive gap, that's not loading-specific and needs more
investigation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG

ITER_LIST = [10]
BATCH_SIZE = 32
NUM_BATCHES = 10
SNR_POINTS = [15.0, 20.0]

CONFIGS = [('M8K4', STANDARD_CONFIG), ('M64K32', MASSIVE_CONFIG)]

if __name__ == '__main__':
    all_results = {}
    for name, cfg in CONFIGS:
        M, K = cfg['NUM_TX'], cfg['NUM_RX']
        for snr in SNR_POINTS:
            ref = run_config(f'REF_{name}_wide_nlos_SNR{snr:g}', M, K, snr,
                              60.0, False, 0.0, ITER_LIST, BATCH_SIZE, NUM_BATCHES)
            all_results[(name, snr)] = ref

    os.makedirs('./results', exist_ok=True)
    np.save('./results/check_highsnr_ref.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY: reference (wide/NLOS) gap at high SNR\n{'='*90}")
    for (name, snr), ref in all_results.items():
        gap = 100.0 * (ref['rzf'] - ref['wmmse'][10]) / ref['rzf']
        print(f"  {name:8s} SNR={snr:4.1f}dB  REF: RZF={ref['rzf']:8.3f} "
              f"WMMSE={ref['wmmse'][10]:8.3f}  gap={gap:+.3f}%")

    print("\nSaved to ./results/check_highsnr_ref.npy")
