"""
Diagnostic tests for multi-user MIMO setup
Tests each component independently to find the issue
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import sys
tf.random.set_seed(42)
np.random.seed(42)

import sionna
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

print("="*80)
print("MULTI-USER MIMO DIAGNOSTIC TESTS")
print("="*80)

# System parameters
fft_size = 72
num_ofdm = 14
num_bs_ant = 8
num_users = 4
num_ut_ant = 1
rb_size = 12
carrier_freq = 2.6e9

print(f"\nSystem Config:")
print(f"  BS antennas: {num_bs_ant}")
print(f"  Users: {num_users}")
print(f"  RX antennas per user: {num_ut_ant}")
print(f"  FFT size: {fft_size}")
print(f"  RB size: {rb_size}")

# ============================================================================
# TEST 1: Channel Generation
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Multi-User Channel Generation")
print("="*80)

batch_size = 4

ut_array = AntennaArray(
    num_rows=1, num_cols=1, polarization="single",
    polarization_type="V", antenna_pattern="38.901",
    carrier_frequency=carrier_freq
)

bs_array = AntennaArray(
    num_rows=1, num_cols=num_bs_ant//2, polarization="dual",
    polarization_type="cross", antenna_pattern="38.901",
    carrier_frequency=carrier_freq
)

cdl = CDL(
    model="A", delay_spread=30e-9, carrier_frequency=carrier_freq,
    ut_array=ut_array, bs_array=bs_array, direction="downlink"
)

frequencies = subcarrier_frequencies(fft_size, 15e3)

# Generate channel for each user
h_list = []
for user_idx in range(num_users):
    cir = cdl(batch_size, num_ofdm, 1 / (num_ofdm * 15e3))
    h_user = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    print(f"  User {user_idx+1} channel shape: {h_user.shape}")
    h_list.append(h_user)

h_freq = tf.concat(h_list, axis=1)
print(f"✅ Combined channel shape: {h_freq.shape}")
print(f"   Expected: [batch={batch_size}, K={num_users}, 1, 1, M={num_bs_ant}, ofdm={num_ofdm}, fft={fft_size}]")

# Check channel properties
h_power = tf.reduce_mean(tf.abs(h_freq)**2)
print(f"   Channel power (should be ~1.0 due to normalization): {h_power.numpy():.4f}")

if h_freq.shape[1] != num_users:
    print(f"❌ ERROR: Expected {num_users} users, got {h_freq.shape[1]}")
else:
    print(f"✅ Channel generation passed!")

# ============================================================================
# TEST 2: MRT Precoder (Baseline)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Maximum Ratio Transmission (MRT) Precoder")
print("="*80)

# Clean channel format
h_clean = tf.squeeze(h_freq, axis=[2, 3])  # [B, K, M, ofdm, fft]
print(f"  Cleaned channel: {h_clean.shape}")

# Average over OFDM
h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, K, M, fft]
print(f"  Averaged channel: {h_avg.shape}")

# Reshape into RBs
num_rb = fft_size // rb_size
h_rb = tf.reshape(h_avg, [batch_size, num_users, num_bs_ant, num_rb, rb_size])
h_rb = tf.transpose(h_rb, perm=[0, 1, 3, 4, 2])  # [B, K, num_rb, rb_size, M]
print(f"  RB format: {h_rb.shape}")

# MRT: W_k = h_k^H (conjugate transpose)
W_mrt = tf.math.conj(h_rb)
print(f"  MRT precoder: {W_mrt.shape}")

# Normalize per RB
power_per_rb = tf.reduce_sum(tf.abs(W_mrt)**2, axis=[3, 4], keepdims=True)
W_mrt_norm = W_mrt / tf.cast(tf.sqrt(power_per_rb + 1e-8), tf.complex64)

# Total power normalization
total_power = tf.reduce_sum(tf.abs(W_mrt_norm)**2) / batch_size
target_power = float(num_users)
scale = tf.sqrt(target_power / total_power)
W_mrt_scaled = W_mrt_norm * tf.cast(scale, tf.complex64)

final_power = tf.reduce_mean(tf.abs(W_mrt_scaled)**2)
print(f"  MRT power: {final_power.numpy():.6f}")
print(f"  Target: {target_power / (num_users * num_rb * rb_size * num_bs_ant):.6f}")

# Compute effective channel (signal strength)
h_expanded = tf.expand_dims(h_rb, axis=2)  # [B, K_rx, 1, num_rb, rb_size, M]
W_expanded = tf.expand_dims(W_mrt_scaled, axis=1)  # [B, 1, K_tx, num_rb, rb_size, M]

h_eff_all = tf.reduce_sum(
    h_expanded * tf.math.conj(W_expanded),
    axis=5
)  # [B, K_rx, K_tx, num_rb, rb_size]

# Extract diagonal (signal for user k from beam k)
signal_power_mrt = []
for k in range(num_users):
    sig = tf.abs(h_eff_all[:, k, k, :, :])**2
    signal_power_mrt.append(tf.reduce_mean(sig))

print(f"\n  MRT Signal Powers:")
for k, pwr in enumerate(signal_power_mrt):
    print(f"    User {k+1}: {pwr.numpy():.4f}")

avg_signal_mrt = np.mean([p.numpy() for p in signal_power_mrt])
print(f"  Average signal power: {avg_signal_mrt:.4f}")

if avg_signal_mrt < 0.1:
    print(f"❌ ERROR: Signal power too low! MRT should give strong signal.")
else:
    print(f"✅ MRT precoder passed!")

# ============================================================================
# TEST 3: Your Transformer Precoder
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Transformer Precoder Output")
print("="*80)

from transformer_5d_sionna import SimplifiedTransformer_PerUser

transformer = SimplifiedTransformer_PerUser(
    num_tx_antennas=num_bs_ant,
    num_ofdm_symbols=num_ofdm,
    fft_size=fft_size,
    rb_size=rb_size
)

# Test on single user
h_user_0 = h_clean[:, 0, :, :, :]  # [B, M, ofdm, fft]
print(f"  Input for user 0: {h_user_0.shape}")

W_transformer = transformer(h_user_0, training=False)
print(f"  Transformer output: {W_transformer.shape}")

W_trans_power = tf.reduce_mean(tf.abs(W_transformer)**2)
print(f"  Transformer raw power: {W_trans_power.numpy():.6f}")

# Check if output correlates with channel
correlation = tf.reduce_mean(
    tf.abs(tf.reduce_sum(h_rb[:, 0, :, :, :] * tf.math.conj(W_transformer), axis=[2, 3]))**2
)
print(f"  Channel-Precoder correlation: {correlation.numpy():.6f}")

if W_trans_power < 1e-6:
    print(f"❌ ERROR: Transformer output is nearly zero!")
elif correlation < 0.01:
    print(f"⚠️  WARNING: Low correlation - transformer may not be using channel info")
else:
    print(f"✅ Transformer output looks reasonable!")

# ============================================================================
# TEST 4: Full Precoder Pipeline
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Full TransformerPrecoder5D_PerUser Pipeline")
print("="*80)

from sionna.phy.mimo import StreamManagement
from transformer_5d_sionna import TransformerPrecoder5D_PerUser

# Create resource grid
rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm,
    fft_size=fft_size,
    subcarrier_spacing=15e3,
    num_tx=num_users,
    num_streams_per_tx=1,
    cyclic_prefix_length=6,
    num_guard_carriers=[5, 6],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11]
)

sm = StreamManagement(np.array([[1], [1], [1], [1]]), 1)

precoder_full = TransformerPrecoder5D_PerUser(
    rg, sm,
    num_tx_antennas=num_bs_ant,
    num_users=num_users,
    num_rx_ant_per_user=num_ut_ant,
    rb_size=rb_size
)

# Create dummy data
from sionna.phy.mapping import Mapper, BinarySource

binary_source = BinarySource()
mapper = Mapper("qam", 2)  # QPSK
rg_mapper = ResourceGridMapper(rg)

# ✅ FIX: Use num_data_symbols directly
num_symbols_per_user = rg.num_data_symbols
num_bits_per_symbol = 2  # QPSK
num_bits_per_user = num_symbols_per_user * num_bits_per_symbol

print(f"  Data symbols per user: {num_symbols_per_user}")
print(f"  Bits per user: {num_bits_per_user}")

# Generate bits: [batch, num_users, num_streams=1, num_bits]
b = binary_source([batch_size, num_users, 1, num_bits_per_user])
print(f"  Bits shape: {b.shape}")

# Map to symbols: [batch, num_users, num_streams=1, num_symbols]
x = mapper(b)
print(f"  Symbol shape: {x.shape}")

# Map to resource grid: [batch, num_users, num_streams=1, num_ofdm_symbols, fft_size]
x_rg = rg_mapper(x)
print(f"  Resource grid shape: {x_rg.shape}")

print(f"  Data symbols: {x_rg.shape}")
print(f"  Channel: {h_freq.shape}")

# Apply precoder
x_precoded, h_eff, W_full = precoder_full(x_rg, h_freq, training=False)

print(f"  Precoded signal: {x_precoded.shape}")
print(f"  Effective channel: {h_eff.shape}")
print(f"  Precoding matrix: {W_full.shape}")

# Check powers
tx_power = tf.reduce_mean(tf.abs(x_precoded)**2)
W_power = tf.reduce_mean(tf.abs(W_full)**2)

print(f"\n  TX signal power: {tx_power.numpy():.6f}")
print(f"  Precoder power: {W_power.numpy():.6f}")

if tx_power < 1e-6:
    print(f"❌ ERROR: TX power is nearly zero!")
elif W_power < 1e-4:
    print(f"❌ ERROR: Precoder power too low!")
else:
    print(f"✅ Full pipeline passed!")

# ============================================================================
# TEST 5: Sum Rate Calculation
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Sum Rate Calculation with MRT vs Transformer")
print("="*80)

def compute_sum_rate_simple(W, h_rb, noise_power=1e-3):
    """Simple sum rate calculation"""
    eps = 1e-10
    
    # W: [B, K, num_rb, rb_size, M]
    # h_rb: [B, K, num_rb, rb_size, M]
    
    h_expanded = tf.expand_dims(h_rb, axis=2)  # [B, K_rx, 1, num_rb, rb_size, M]
    W_expanded = tf.expand_dims(W, axis=1)     # [B, 1, K_tx, num_rb, rb_size, M]
    
    # Effective channel
    h_eff_all = tf.reduce_sum(
        h_expanded * tf.math.conj(W_expanded),
        axis=5
    )  # [B, K_rx, K_tx, num_rb, rb_size]
    
    # Signal and interference
    signal_powers = []
    interference_powers = []
    
    for k in range(num_users):
        signal = tf.abs(h_eff_all[:, k, k, :, :])**2
        total = tf.reduce_sum(tf.abs(h_eff_all[:, k, :, :, :])**2, axis=1)
        interference = total - signal
        
        signal_powers.append(tf.reduce_mean(signal))
        interference_powers.append(tf.reduce_mean(interference))
    
    # SINR and rate
    total_rate = 0.0
    for k in range(num_users):
        sinr = signal_powers[k] / (interference_powers[k] + noise_power + eps)
        rate = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
        total_rate += rate
        print(f"    User {k+1}: Signal={signal_powers[k].numpy():.4f}, "
              f"Interf={interference_powers[k].numpy():.4f}, "
              f"SINR={10*tf.math.log(sinr)/tf.math.log(10.0).numpy():.2f} dB, "
              f"Rate={rate.numpy():.3f} bps/Hz")
    
    return total_rate

print("\n  MRT Precoder:")
rate_mrt = compute_sum_rate_simple(W_mrt_scaled, h_rb)
print(f"  MRT Sum Rate: {rate_mrt.numpy():.3f} bps/Hz")

print("\n  Transformer Precoder:")
# Need to get all users
W_trans_list = []
for k in range(num_users):
    h_k = h_clean[:, k, :, :, :]
    W_k = transformer(h_k, training=False)
    W_trans_list.append(W_k)

W_trans_all = tf.stack(W_trans_list, axis=1)

# Normalize
total_power = tf.reduce_sum(tf.abs(W_trans_all)**2) / batch_size
scale = tf.sqrt(float(num_users) / total_power)
W_trans_scaled = W_trans_all * tf.cast(scale, tf.complex64)

rate_trans = compute_sum_rate_simple(W_trans_scaled, h_rb)
print(f"  Transformer Sum Rate: {rate_trans.numpy():.3f} bps/Hz")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"  MRT Sum Rate:         {rate_mrt.numpy():.3f} bps/Hz")
print(f"  Transformer Sum Rate: {rate_trans.numpy():.3f} bps/Hz")
print(f"  Ratio (Trans/MRT):    {rate_trans.numpy()/rate_mrt.numpy():.3f}")

if rate_trans < 0.5 * rate_mrt:
    print(f"\n❌ PROBLEM FOUND: Transformer performs much worse than MRT!")
    print(f"   This suggests the transformer is not learning proper channel-matched precoding.")
elif rate_trans < 0.8 * rate_mrt:
    print(f"\n⚠️  WARNING: Transformer underperforms MRT (untrained is expected)")
else:
    print(f"\n✅ Results look reasonable!")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)