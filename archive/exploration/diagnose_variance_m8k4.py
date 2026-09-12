"""Same diagnostic as diagnose_variance.py but for M8K4, to check whether
the GPU-specific non-determinism found at M64K32 is scale-specific
(M=64's larger tf.linalg.solve) or a general issue."""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG

BATCH_SIZE = 32
NUM_BATCHES = 60
SNR = 5.0

if __name__ == '__main__':
    cfg = STANDARD_CONFIG
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['HALF_ANGLE_DEG'],
                                     los=cfg['FORCE_LOS'],
                                     indoor_probability=cfg['INDOOR_PROBABILITY'])
    bs = tf.constant(BATCH_SIZE, tf.int32)
    snr_t = tf.constant(SNR, tf.float32)

    gaps = []
    for i in range(NUM_BATCHES):
        system.new_topology(BATCH_SIZE)
        h_freq, no = system.channel_and_no(bs, snr_t)
        r_rzf = float(system.eval_rzf_from_h(h_freq, no))
        r_wmmse = float(system.eval_wmmse_from_h(h_freq, no, 10))
        gaps.append(100.0 * (r_rzf - r_wmmse) / r_rzf)

    gaps = np.array(gaps)
    print(f"Gap %: mean={gaps.mean():.3f}  std={gaps.std():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}")
