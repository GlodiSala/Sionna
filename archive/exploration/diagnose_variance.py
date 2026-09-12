"""
Diagnose why M64K32's RZF-vs-WMMSE gap estimate is unstable even at 40
batches (run1: +1.03%, run2: +1.96%, vs step2b's original +4.08%), while
M8K4 is stable (~3.9-4.3% across runs). Print PER-BATCH gap values (not
just the aggregate mean/std) to see if a few outlier/pathological batches
(e.g. near-degenerate narrow-window topologies) are driving the variance,
which would explain a heavy-tailed distribution that a small batch count
badly underestimates.
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import MASSIVE_CONFIG

BATCH_SIZE = 32
NUM_BATCHES = 60
SNR = 5.0

if __name__ == '__main__':
    cfg = MASSIVE_CONFIG
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['HALF_ANGLE_DEG'],
                                     los=cfg['FORCE_LOS'],
                                     indoor_probability=cfg['INDOOR_PROBABILITY'])
    bs = tf.constant(BATCH_SIZE, tf.int32)
    snr_t = tf.constant(SNR, tf.float32)

    per_batch = []
    for i in range(NUM_BATCHES):
        system.new_topology(BATCH_SIZE)
        h_freq, no = system.channel_and_no(bs, snr_t)
        r_rzf = float(system.eval_rzf_from_h(h_freq, no))
        r_wmmse = float(system.eval_wmmse_from_h(h_freq, no, 10))
        gap = 100.0 * (r_rzf - r_wmmse) / r_rzf
        per_batch.append((r_rzf, r_wmmse, gap))
        print(f"  batch {i:3d}: RZF={r_rzf:8.3f}  WMMSE={r_wmmse:8.3f}  gap={gap:+7.3f}%")

    rzf_vals = np.array([p[0] for p in per_batch])
    wmmse_vals = np.array([p[1] for p in per_batch])
    gaps = np.array([p[2] for p in per_batch])

    print(f"\n{'='*70}")
    print(f"RZF:   mean={rzf_vals.mean():.3f}  std={rzf_vals.std():.3f}  "
          f"min={rzf_vals.min():.3f}  max={rzf_vals.max():.3f}")
    print(f"WMMSE: mean={wmmse_vals.mean():.3f}  std={wmmse_vals.std():.3f}  "
          f"min={wmmse_vals.min():.3f}  max={wmmse_vals.max():.3f}")
    print(f"Gap %: mean={gaps.mean():.3f}  std={gaps.std():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}  median={np.median(gaps):.3f}")
    print(f"Overall gap (aggregate mean RZF/WMMSE, matches run_config's method): "
          f"{100.0*(rzf_vals.mean()-wmmse_vals.mean())/rzf_vals.mean():.3f}%")

    np.save('./results/diagnose_variance_m64k32.npy',
            dict(rzf=rzf_vals, wmmse=wmmse_vals, gap=gaps))
