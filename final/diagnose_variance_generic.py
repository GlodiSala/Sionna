"""Generic version of diagnose_variance.py -- takes M, K via argv so it can
be reused for any scale (here: M16K8) without duplicating the script per
config. Same locked mechanism (15deg window, forced LOS, indoor_prob=0),
same SNR=5, same batch_size=32/num_batches=60."""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem

BATCH_SIZE = 32
NUM_BATCHES = 60
SNR = 5.0

if __name__ == '__main__':
    M, K = int(sys.argv[1]), int(sys.argv[2])
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=7.5, los=True,
                                     indoor_probability=0.0)
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
    print(f"M{M}K{K} Gap %: mean={gaps.mean():.3f}  std={gaps.std():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}")
