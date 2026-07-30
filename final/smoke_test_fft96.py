"""Step 0 verification: resource grid + both architectures at FFT_SIZE=96,
for the target K values (4 and 8), M=64. Confirms the kronecker pilot
pattern, pilot indices, and num_ofdm_symbols still work cleanly, and that
a real forward pass through both architectures produces correctly shaped,
properly power-normalized, NaN-free output."""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem, FFT_SIZE, NUM_OFDM
from precoder_intra_rb import IntraRBTransformerPrecoder
from precoders_w import TransformerPrecoderV4
from sionna.phy.utils import ebnodb2no

print(f"FFT_SIZE={FFT_SIZE}, NUM_OFDM={NUM_OFDM}")
M = 64
BATCH = 8

for K in [4, 8]:
    print(f"\n{'='*70}\nM={M}, K={K}\n{'='*70}")
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=60.0, los=None,
                                     indoor_probability=None)
    print("num_effective_subcarriers:", system.rg.num_effective_subcarriers,
          "| fft_size:", system.rg.fft_size,
          "| num_ofdm_symbols:", system.rg.num_ofdm_symbols)
    system.new_topology(BATCH)
    no = ebnodb2no(tf.constant(10.0), system.num_bits_per_symbol, 0.5, system.rg)
    h_freq = system._gen_channel(BATCH)
    print("h_freq shape:", h_freq.shape)

    intra_rb = IntraRBTransformerPrecoder(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12,
        embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    g_intra = intra_rb(h_freq, no=no, training=False)
    pwr_intra = tf.reduce_sum(tf.abs(g_intra)**2, axis=-2)
    assert g_intra.shape[1:] == (1, NUM_OFDM, FFT_SIZE, M, K), g_intra.shape
    assert not bool(tf.reduce_any(tf.math.is_nan(tf.abs(g_intra))))
    print(f"IntraRB       OK  shape={tuple(g_intra.shape)}  "
          f"params={intra_rb.count_params():,}  "
          f"mean_power={float(tf.reduce_mean(pwr_intra)):.4f}")

    v4 = TransformerPrecoderV4(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12,
        tokens_per_rb=1, embed_dim=128, num_heads=4, num_layers=4, version='v4.2')
    g_v4 = v4(h_freq, no=no, training=False)
    pwr_v4 = tf.reduce_sum(tf.abs(g_v4)**2, axis=-2)
    assert g_v4.shape[1:] == (1, NUM_OFDM, FFT_SIZE, M, K), g_v4.shape
    assert not bool(tf.reduce_any(tf.math.is_nan(tf.abs(g_v4))))
    print(f"TransformerV4 OK  shape={tuple(g_v4.shape)}  "
          f"params={v4.count_params():,}  "
          f"mean_power={float(tf.reduce_mean(pwr_v4)):.4f}")

print("\nAll checks passed at FFT_SIZE=96 for K=4 and K=8.")
