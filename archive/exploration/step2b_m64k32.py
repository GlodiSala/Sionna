"""
Confirm M=64/K=32 (50% loading, genuine massive-MIMO scale) at SNR=5 and
SNR=10, same paired-reference methodology as Step 2, to check whether the
ref-adjusted gap at matched ~50% loading holds consistent across scale
against the already-validated M=8/K=4 (~3.99% @ SNR=5) and M=16/K=8
(~4.63% @ SNR=5).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config

ITER_LIST = [10]
BATCH_SIZE = 32
NUM_BATCHES = 10   # 320 samples/arm, matching Step 2

M, K = 64, 32

if __name__ == '__main__':
    all_results = {}
    for snr in [5.0, 10.0]:
        ref = run_config(f'REF_M{M}K{K}_wide_nlos_SNR{snr:g}', M, K, snr,
                          60.0, False, 0.0, ITER_LIST, BATCH_SIZE, NUM_BATCHES)
        hard = run_config(f'HARD_M{M}K{K}_ang15_los_SNR{snr:g}', M, K, snr,
                           7.5, True, 0.0, ITER_LIST, BATCH_SIZE, NUM_BATCHES)
        all_results[snr] = (ref, hard)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/step2b_m64k32.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY M={M} K={K} (50% loading), WMMSE iters=10, FFT_SIZE=96\n{'='*90}")
    for snr, (ref, hard) in all_results.items():
        ref_gap = 100.0 * (ref['rzf'] - ref['wmmse'][10]) / ref['rzf']
        hard_gap = 100.0 * (hard['rzf'] - hard['wmmse'][10]) / hard['rzf']
        adj = hard_gap - ref_gap
        print(f"  SNR={snr:4.1f}dB  REF: RZF={ref['rzf']:8.2f} WMMSE={ref['wmmse'][10]:8.2f} "
              f"gap={ref_gap:+6.2f}%  |  "
              f"HARD: RZF={hard['rzf']:8.2f} WMMSE={hard['wmmse'][10]:8.2f} "
              f"gap={hard_gap:+6.2f}%  ref-adjusted={adj:+6.2f}%")

    print("\nSaved to ./results/step2b_m64k32.npy")
