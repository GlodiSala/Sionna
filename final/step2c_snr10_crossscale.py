"""
Get SNR=10 paired-reference numbers for M=8/K=4 and M=16/K=8, matching
their existing SNR=5 runs (Step 2), to complete the cross-scale x cross-SNR
comparison table alongside M=64/K=32 (already done in step2b_m64k32.py).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config

ITER_LIST = [10]
BATCH_SIZE = 32
NUM_BATCHES = 10

CONFIGS = [(8, 4), (16, 8)]

if __name__ == '__main__':
    all_results = {}
    for M, K in CONFIGS:
        ref = run_config(f'REF_M{M}K{K}_wide_nlos_SNR10', M, K, 10.0,
                          60.0, False, 0.0, ITER_LIST, BATCH_SIZE, NUM_BATCHES)
        hard = run_config(f'HARD_M{M}K{K}_ang15_los_SNR10', M, K, 10.0,
                           7.5, True, 0.0, ITER_LIST, BATCH_SIZE, NUM_BATCHES)
        all_results[(M, K)] = (ref, hard)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/step2c_snr10_crossscale.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY SNR=10dB cross-scale\n{'='*90}")
    for (M, K), (ref, hard) in all_results.items():
        ref_gap = 100.0 * (ref['rzf'] - ref['wmmse'][10]) / ref['rzf']
        hard_gap = 100.0 * (hard['rzf'] - hard['wmmse'][10]) / hard['rzf']
        adj = hard_gap - ref_gap
        print(f"  M={M:>3} K={K:>2}  REF gap={ref_gap:+6.2f}%  HARD gap={hard_gap:+6.2f}%  "
              f"ref-adjusted={adj:+6.2f}%")

    print("\nSaved to ./results/step2c_snr10_crossscale.npy")
