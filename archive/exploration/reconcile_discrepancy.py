"""
Reconcile the apparent discrepancy between:
- Paired-reference validation (step2b_m64k32.py): M64K32 @ SNR=5 raw HARD
  gap = +4.08% (RZF=249.54, WMMSE=239.36), num_batches=10.
- Final classical comparison (classical_comparison.py): M64K32 @ SNR=5
  gap = +1.90% (RZF=264.16, WMMSE=259.13), num_batches=10.

Methodology confirmed IDENTICAL in both scripts (paired topology, RZF
alpha=no/2, WMMSE 10 iters) -- so if the discrepancy is real and not a
bug, it must be run-to-run sampling variance exceeding what the reported
per-run std suggested. Test with substantially more samples to get a
tighter, more trustworthy estimate, and run it twice independently to
check reproducibility.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG

ITER_LIST = [10]
BATCH_SIZE = 32
NUM_BATCHES = 40   # 4x more than the original 10, for a tighter estimate
SNR = 5.0

CONFIGS = [('M8K4', STANDARD_CONFIG), ('M64K32', MASSIVE_CONFIG)]

if __name__ == '__main__':
    for run_idx in [1, 2]:
        print(f"\n{'#'*90}\nRUN {run_idx}\n{'#'*90}")
        for name, cfg in CONFIGS:
            M, K = cfg['NUM_TX'], cfg['NUM_RX']
            r = run_config(f'{name}_ang15_los_SNR5_run{run_idx}', M, K, SNR,
                            cfg['HALF_ANGLE_DEG'], cfg['FORCE_LOS'],
                            cfg['INDOOR_PROBABILITY'], ITER_LIST,
                            BATCH_SIZE, NUM_BATCHES)
            gap = 100.0 * (r['rzf'] - r['wmmse'][10]) / r['rzf']
            print(f"  >>> {name} run{run_idx}: RZF={r['rzf']:.3f} WMMSE={r['wmmse'][10]:.3f} gap={gap:+.3f}%")
