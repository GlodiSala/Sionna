"""
Step 3 follow-up — three dials on top of the M64K36_ang15_los "hard" config
(the one showing a clean, reference-adjusted +2.53% gap at SNR=10, iters=10):

1. Narrower angular window: 8 deg and 5 deg (vs 15 deg tested before), same
   M=64/K=36/LOS/SNR=10, to see if correlation-driven gap grows.
2. Higher loading: confirmed impossible for this resource grid -- divisors
   of FFT_SIZE=72 are {1,2,3,4,6,8,9,12,18,24,36,72}; nothing between 36 and
   72, and K=72 > M=64 is not a valid config (more streams than antennas).
   36/64 (56%) is the practical ceiling here. No run needed for this dial.
3. SNR=5 with a MATCHED reference at SNR=5 (not reusing the SNR=10 ref,
   since the known metric artifact is SNR-dependent and may not vanish at
   SNR=5).

Same paired, reference-adjusted methodology as before.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config

ITER_LIST = [10]
BATCH_SIZE = 24
NUM_BATCHES = 16

# ---- Dial 1: narrower angle, same SNR=10 as the original C config ----
DIAL1_CONFIGS = [
    dict(name='D_M64K36_ang8_los_SNR10', num_tx=64, num_rx=36, snr_db=10.0,
         half_angle_deg=4.0, los=True, indoor_probability=0.0),
    dict(name='E_M64K36_ang5_los_SNR10', num_tx=64, num_rx=36, snr_db=10.0,
         half_angle_deg=2.5, los=True, indoor_probability=0.0),
]

# ---- Dial 3: SNR=5, hard config + matched reference at the SAME SNR ----
DIAL3_CONFIGS = [
    dict(name='REF_M64K8_wide_nlos_SNR5', num_tx=64, num_rx=8, snr_db=5.0,
         half_angle_deg=60.0, los=False, indoor_probability=0.0),
    dict(name='C_M64K36_ang15_los_SNR5', num_tx=64, num_rx=36, snr_db=5.0,
         half_angle_deg=7.5, los=True, indoor_probability=0.0),
]

ALL_CONFIGS = DIAL1_CONFIGS + DIAL3_CONFIGS


if __name__ == '__main__':
    all_results = []
    for cfg in ALL_CONFIGS:
        r = run_config(cfg['name'], cfg['num_tx'], cfg['num_rx'], cfg['snr_db'],
                        cfg['half_angle_deg'], cfg['los'],
                        cfg['indoor_probability'], ITER_LIST,
                        BATCH_SIZE, NUM_BATCHES)
        all_results.append(r)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/step3_followup.npy', all_results)

    print(f"\n{'='*70}\nFOLLOW-UP SUMMARY (WMMSE iters=10)\n{'='*70}")
    for r in all_results:
        gap_pct = 100.0 * (r['rzf'] - r['wmmse'][10]) / r['rzf']
        print(f"  {r['config']:30s}  RZF={r['rzf']:7.2f}  WMMSE={r['wmmse'][10]:7.2f}  "
              f"gap={gap_pct:+6.2f}%")

    print("\nSaved to ./results/step3_followup.npy")
