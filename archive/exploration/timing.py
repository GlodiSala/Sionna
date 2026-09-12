"""
timing_test.py — Compare le temps/iter V4.0 vs V4.1
"""
import os, logging, time
import matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from datasets import CachedSionnaDataset
from precoders_w import TransformerPrecoderV4
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.utils import ebnodb2no
from precoders_w import rzf_precoder

CONFIGS = [
    {'name': 'V4.0 (3tok)', 'tokens_per_rb': 3, 'use_learned_upsample': False},
    {'name': 'V4.1 (3tok)', 'tokens_per_rb': 3, 'use_learned_upsample': True},
    {'name': 'V4.0 (6tok)', 'tokens_per_rb': 6, 'use_learned_upsample': False},
    {'name': 'V4.1 (6tok)', 'tokens_per_rb': 6, 'use_learned_upsample': True},
]

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU: {gpus[0]}\n")

    # Dataset minimal pour le test
    from main_final import MU_MIMO_System, CachedSionnaDataset
    dummy = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="rzf")
    dataset = CachedSionnaDataset(
        dummy, dataset_size=5000, batch_size=512,
        cache_file='/export/tmp/sala/sionna_base_5k_8x4.npz',
        augmentation_multiplier=50, seed=42)

    print(f"{'Config':<20} {'s/iter':>8} {'min/epoch':>12} {'h total (15ep)':>16}")
    print("─" * 60)

    for cfg in CONFIGS:
        system = MU_MIMO_System(
            num_tx=8, num_rx=4,
            precoder_type='transformer_rb',
            rb_size=12,
            tokens_per_rb=cfg['tokens_per_rb'],
            embed_dim=128, num_heads=4,
            use_learned_upsample=cfg['use_learned_upsample'])

        # Variables pour le step
        trainable_vars = system.precoder.trainable_variables
        rg   = system.rg
        sm   = system.sm

        # Build le modèle
        dummy_h = tf.zeros([1, 4, 1, 1, 8, 14, 72], dtype=tf.complex64)
        _ = system.precoder(dummy_h, training=False)

        optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=5.0)

        @tf.function
        def step(h):
            no = ebnodb2no(tf.constant(15.0), 2, 0.5, rg)
            g_teacher = tf.stop_gradient(
                rzf_precoder(h, stream_management=sm, alpha=0.1))
            with tf.GradientTape() as tape:
                g_pred   = system.precoder(h, training=True)
                mse_loss = 2.0 * tf.reduce_mean(tf.abs(g_pred - g_teacher)**2)
            grads = tape.gradient(mse_loss, trainable_vars)
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                     for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            return mse_loss

        # Chauffe JIT
        h_test = dataset.get_batch(256)
        _ = step(h_test)

        # Mesure 10 iters
        t0 = time.time()
        for _ in range(10):
            h = dataset.get_batch(256)
            _ = step(h)
        secs = (time.time() - t0) / 10

        num_iters  = 250000 // 256  # 1953
        min_epoch  = secs * num_iters / 60
        h_total    = min_epoch * 15 / 60

        print(f"  {cfg['name']:<18} {secs:>8.2f}s {min_epoch:>11.1f}min "
              f"{h_total:>15.1f}h")

    print()

if __name__ == "__main__":
    main()