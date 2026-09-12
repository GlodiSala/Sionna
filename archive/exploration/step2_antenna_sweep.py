"""
Step 2 -- Antenna sweep at fixed, realistic K.

M in {8,16,32,64,128}, K in {4,8} (separately), using the forced-LOS +
narrow-angle-window (15 deg) approach validated in Steps 1-3 (Step 1:
CDL/spatial-consistency investigated and ruled out as a cleaner
alternative -- spatial consistency is already unconditionally active in
UMi for LSPs but doesn't reach cluster/ray-level geometry, and CDL has no
native multi-user support; see conversation history / channel_config.py).

Correction for the record: CDL-D and CDL-E are the LOS-dominant CDL
profiles (not CDL-C/D as originally guessed) -- moot for this sweep since
CDL wasn't used, noted here only so it doesn't get misremembered later.

Each hard config (narrow window, LOS) is paired with a matched reference
(same M/K, wide window, NLOS) at the same SNR, so any gap seen is
reference-adjusted and trustworthy per the Steps 1-3 methodology.

FFT_SIZE=96 (Step 0), WMMSE @ 10 iterations (Step 1 finding: more doesn't
help under this reporting metric). SNR=5dB, the point where Step 3 found
the largest validated, reference-clean gap.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config

SNR_DB = 5.0
ITER_LIST = [10]
BATCH_SIZE = 32
NUM_BATCHES = 10   # 320 samples/arm

M_VALUES = [8, 16, 32, 64, 128]
K_VALUES = [4, 8]


def build_configs(K):
    configs = []
    for M in M_VALUES:
        configs.append(dict(
            name=f'REF_M{M}K{K}_wide_nlos', num_tx=M, num_rx=K,
            half_angle_deg=60.0, los=False, indoor_probability=0.0))
        configs.append(dict(
            name=f'HARD_M{M}K{K}_ang15_los', num_tx=M, num_rx=K,
            half_angle_deg=7.5, los=True, indoor_probability=0.0))
    return configs


if __name__ == '__main__':
    all_results = {}
    for K in K_VALUES:
        all_results[K] = []
        for cfg in build_configs(K):
            r = run_config(cfg['name'], cfg['num_tx'], cfg['num_rx'], SNR_DB,
                            cfg['half_angle_deg'], cfg['los'],
                            cfg['indoor_probability'], ITER_LIST,
                            BATCH_SIZE, NUM_BATCHES)
            all_results[K].append(r)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/step2_antenna_sweep.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY (SNR={SNR_DB} dB, WMMSE iters=10, FFT_SIZE=96)\n{'='*90}")
    for K in K_VALUES:
        print(f"\n--- K={K} ---")
        results = all_results[K]
        for i in range(0, len(results), 2):
            ref = results[i]
            hard = results[i + 1]
            ref_gap = 100.0 * (ref['rzf'] - ref['wmmse'][10]) / ref['rzf']
            hard_gap = 100.0 * (hard['rzf'] - hard['wmmse'][10]) / hard['rzf']
            adj = hard_gap - ref_gap
            M = ref['config'].split('_M')[1].split('K')[0]
            print(f"  M={M:>4}  REF: RZF={ref['rzf']:8.2f} WMMSE={ref['wmmse'][10]:8.2f} "
                  f"gap={ref_gap:+6.2f}%  |  "
                  f"HARD: RZF={hard['rzf']:8.2f} WMMSE={hard['wmmse'][10]:8.2f} "
                  f"gap={hard_gap:+6.2f}%  ref-adjusted={adj:+6.2f}%")

    print("\nSaved to ./results/step2_antenna_sweep.npy")
