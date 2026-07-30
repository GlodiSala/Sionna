"""
Step 3 — Proposed massive-MIMO configs, with matched low-loading/near-
orthogonal reference points, tested with WMMSE at 10 iterations (per Step 1
finding: more iterations does not help under this reporting metric, and
running more would only inflate the reported gap without telling us
anything new about convergence).

Design: a 2x2-ish grid over the two real levers identified in Step 2
(narrow angular window + forced LOS, and K/M loading), all at M=64, holding
indoor_probability=0 constant (hygiene only, not itself an orthogonality
lever):

  REF  M64K8_wide_nlos    : K=8  (12.5% load), wide window (120 deg), NLOS
  A    M64K8_ang15_los    : K=8  (12.5% load), narrow window (15 deg), LOS
  B    M64K36_ang60_los   : K=36 (56%  load), wide window (120 deg), LOS
  C    M64K36_ang15_los   : K=36 (56%  load), narrow window (15 deg), LOS

REF isolates "neither lever active" (near-orthogonal baseline).
A isolates the angular/LOS lever alone (low loading).
B isolates the loading lever alone (wide angle).
C is the full combination (both levers stacked) -- the main "hard" candidate.

Any gap in REF beyond the known ~0.5-0.7% low-SNR-only baseline effect would
indicate the metric-mismatch artifact is not fully controlled for; gaps in
A/B/C beyond REF's gap are attributable to the respective lever(s).
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import run_config

SNR_DB = 10.0
ITER_LIST = [10]
BATCH_SIZE = 24
NUM_BATCHES = 16   # 384 samples per config

CONFIGS = [
    dict(name='REF_M64K8_wide_nlos', num_tx=64, num_rx=8,
         half_angle_deg=60.0, los=False, indoor_probability=0.0),
    dict(name='A_M64K8_ang15_los', num_tx=64, num_rx=8,
         half_angle_deg=7.5, los=True, indoor_probability=0.0),
    dict(name='B_M64K36_ang60_los', num_tx=64, num_rx=36,
         half_angle_deg=60.0, los=True, indoor_probability=0.0),
    dict(name='C_M64K36_ang15_los', num_tx=64, num_rx=36,
         half_angle_deg=7.5, los=True, indoor_probability=0.0),
]


if __name__ == '__main__':
    all_results = []
    for cfg in CONFIGS:
        r = run_config(cfg['name'], cfg['num_tx'], cfg['num_rx'], SNR_DB,
                        cfg['half_angle_deg'], cfg['los'],
                        cfg['indoor_probability'], ITER_LIST,
                        BATCH_SIZE, NUM_BATCHES)
        all_results.append(r)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/step3_massive_mimo_test.npy', all_results)

    ref_gap = all_results[0]['rzf']
    print(f"\n{'='*70}\nSUMMARY (SNR={SNR_DB} dB, WMMSE iters=10)\n{'='*70}")
    ref_gap_pct = 100.0 * (all_results[0]['rzf'] - all_results[0]['wmmse'][10]) / all_results[0]['rzf']
    for r in all_results:
        gap_pct = 100.0 * (r['rzf'] - r['wmmse'][10]) / r['rzf']
        adj = gap_pct - ref_gap_pct
        print(f"  {r['config']:24s}  RZF={r['rzf']:7.2f}  WMMSE={r['wmmse'][10]:7.2f}  "
              f"gap={gap_pct:+6.2f}%  ref-adjusted={adj:+6.2f}%")

    print("\nSaved to ./results/step3_massive_mimo_test.npy")
