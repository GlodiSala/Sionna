# =============================================================================
# FREQUENCY CORRELATION TESTER FOR YOUR UMi DATASET
# =============================================================================
import os
import json
import pickle
import logging
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import layers

import numpy as np

# Set random seeds
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration failed: {e}")

import sionna
sionna.phy.config.seed = SEED

from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LMMSEEqualizer, 
                             LMMSEPostEqualizationSINR, PrecodedChannel)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.mimo import rzf_precoding_matrix

def test_umi_frequency_correlation(system, num_samples=1000, batch_size=64):
    """
    Quick test: Measure frequency correlation in your actual UMi channels.
    This tells us if linear interpolation is sufficient.
    """
    print(f"\n{'='*80}")
    print(f"  FREQUENCY CORRELATION TEST - UMi Channels")
    print(f"{'='*80}")
    print(f"  Samples: {num_samples}")
    print(f"  Channel: UMi (your training data)")
    print(f"  FFT: {system.rg.fft_size}, RB size: 12")
    print(f"{'='*80}\n")
    
    from scipy.stats import pearsonr
    
    # Collect channel samples
    all_h_freq = []
    num_batches = num_samples // batch_size
    
    print("1️⃣  Generating channel samples...")
    for i in range(num_batches):
        system.new_topology(batch_size)
        cir = system.channel_model(
            batch_size, 
            system.rg.num_ofdm_symbols, 
            1.0 / system.rg.ofdm_symbol_duration
        )
        h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
        all_h_freq.append(h_freq.numpy())
        
        if (i+1) % 5 == 0:
            print(f"   Progress: {i+1}/{num_batches} batches")
    
    h_all = np.concatenate(all_h_freq, axis=0)  # [N, K, 1, 1, M, OFDM, FFT]
    
    # Average over all dimensions except frequency
    h_avg = np.mean(np.abs(h_all), axis=(1, 2, 3, 4, 5))  # [N, FFT=72]
    
    print(f"   ✅ Collected {len(h_avg)} samples\n")
    
    # Compute correlations
    print("2️⃣  Computing frequency correlations...")
    fft_size = h_avg.shape[1]
    
    # Adjacent correlations
    distances = list(range(1, 21))
    correlations = []
    
    for d in distances:
        corr_values = []
        for i in range(fft_size - d):
            corr = pearsonr(h_avg[:, i], h_avg[:, i + d])[0]
            corr_values.append(corr)
        correlations.append(np.mean(corr_values))
    
    # Find coherence bandwidth
    coherence_bw = fft_size
    for i, corr in enumerate(correlations):
        if corr < 0.7:
            coherence_bw = distances[i]
            break
    
    print(f"   ✅ Done!\n")
    
    # Results
    print(f"{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  Correlation at distance 1:  {correlations[0]:.4f}")
    print(f"  Correlation at distance 6:  {correlations[5]:.4f}")
    print(f"  Correlation at distance 12: {correlations[11]:.4f}")
    print(f"  Coherence Bandwidth:        {coherence_bw} subcarriers")
    print(f"{'='*80}\n")
    
    # Decision
    print("📊 INTERPOLATION RECOMMENDATION:")
    if correlations[11] > 0.95:  # Very high correlation within RB
        print("   ✅ LINEAR interpolation is EXCELLENT")
        print("      → Correlation within RB (12 subcarriers) > 0.95")
        print("      → No need for learnable upsampling")
        recommendation = "linear"
    elif correlations[11] > 0.85:
        print("   ✅ LINEAR interpolation is GOOD")
        print("      → Correlation within RB = {correlations[11]:.3f}")
        print("      → Linear should work, but learnable might help slightly")
        recommendation = "linear"
    else:
        print("   ⚠️  Consider LEARNABLE upsampling")
        print(f"      → Correlation within RB = {correlations[11]:.3f}")
        print("      → Channel varies significantly within RBs")
        recommendation = "learnable"
    
    print(f"{'='*80}\n")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(distances, correlations, 'o-', linewidth=2, markersize=8)
    plt.axhline(0.7, color='r', linestyle='--', label='Coherence threshold (0.7)')
    plt.axvline(12, color='g', linestyle='--', label='RB size (12)')
    plt.xlabel('Subcarrier Distance')
    plt.ylabel('Correlation')
    plt.title('UMi Channel: Frequency Correlation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Show correlation within one RB
    rb_corr = correlations[:12]
    rb_positions = list(range(1, 13))
    plt.bar(rb_positions, rb_corr, color='skyblue', edgecolor='navy')
    plt.axhline(0.95, color='g', linestyle='--', label='Excellent threshold')
    plt.axhline(0.85, color='orange', linestyle='--', label='Good threshold')
    plt.xlabel('Position within RB')
    plt.ylabel('Correlation')
    plt.title('Correlation Within Resource Block')
    plt.ylim([0.7, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('umi_frequency_correlation_test.png', dpi=300, bbox_inches='tight')
    print("💾 Plot saved: umi_frequency_correlation_test.png\n")
    plt.show()
    
    return recommendation, correlations

class WMMSEPrecodedChannel(PrecodedChannel):
    """
    WMMSE precoder that properly integrates with Sionna's framework
    """
    def __init__(self, resource_grid, stream_management, num_iterations=10, **kwargs):
        super().__init__(resource_grid, stream_management, **kwargs)
        self.num_iterations = num_iterations
        self._num_streams = stream_management.num_streams_per_tx

    @tf.function
    def call(self, inputs):
        """
        Inputs:
            y: dummy (not used for precoding)
            h_freq: [batch, num_rx, rx_ant, num_tx, tx_ant, S, F]
            no: (optional) noise power
            
        Returns:
            g: [batch, num_tx, S, F, tx_ant, num_streams]
        """
        # Handle optional noise argument
        if isinstance(inputs, tuple) and len(inputs) == 3:
            y, h_freq, no = inputs
        else:
            y, h_freq = inputs
            no = None
        
        # ✅ Get channels in Sionna format
        h_pc_desired = self.get_desired_channels(h_freq)
        
        # ✅ Initialize with RZF (correct shape guaranteed)
        g_init = rzf_precoding_matrix(h_pc_desired, alpha=0.1)
        
        # Extract dimensions from h_freq for WMMSE iterations
        # h_freq: [batch, num_rx, rx_ant, num_tx, tx_ant, S, F]
        s_h = tf.shape(h_freq)
        batch_size = s_h[0]
        num_rx = s_h[1]
        rx_ant = s_h[2]
        num_tx = s_h[3]
        tx_ant = s_h[4]
        S = s_h[5]
        F = s_h[6]
        
        K = num_rx * rx_ant  # Total receivers (streams)
        M = num_tx * tx_ant  # Total transmitters (antennas)
        
        # Reshape h_freq to [Batch, S, F, K, M]
        h_perm = tf.transpose(h_freq, [0, 5, 6, 1, 2, 3, 4])
        # Now: [batch, S, F, num_rx, rx_ant, num_tx, tx_ant]
        H = tf.reshape(h_perm, [batch_size, S, F, K, M])
        
        complex_dtype = H.dtype
        real_dtype = complex_dtype.real_dtype
        
        # Initialize precoding matrix from RZF
        # g_init: [batch, num_tx, S, F, tx_ant, num_streams]
        # Squeeze and reshape to [Batch, S, F, M, K]
        V = tf.squeeze(g_init, axis=1)  # [Batch, S, F, M, K]
        
        # ✅ CORRECTED noise power handling (use broadcasting, don't replicate)
        if no is None:
            no_val = tf.constant(1e-2, dtype=real_dtype)
        else:
            no_val = tf.cast(no, real_dtype)
        
        # ✅ Just add dimensions for broadcasting, don't set batch size
        # Shape: [1, 1, 1] will broadcast to any [Batch, S, F] automatically
        no_val = tf.reshape(no_val, [1, 1, 1])
        no_complex = tf.cast(no_val, complex_dtype)
        
        # WMMSE iterations
        K_float = tf.cast(K, real_dtype)
        eye_M = tf.eye(M, dtype=complex_dtype)[None, None, None, :, :]
        
        for _ in range(self.num_iterations):
            # === Receiver Update ===
            HV = tf.matmul(H, V)  # [Batch, S, F, K, K]
            signal = tf.linalg.diag_part(HV)  # [Batch, S, F, K]
            signal_power = tf.abs(signal)**2
            
            total_power = tf.reduce_sum(tf.abs(HV)**2, axis=-1)
            interference_power = total_power - signal_power
            
            # SINR calculation with noise (broadcasts automatically)
            denom = signal_power + interference_power + no_val
            U = tf.math.conj(signal) / tf.cast(denom, complex_dtype)
            
            # === Weight Update ===
            mse = 1.0 - tf.math.real(U * signal)
            W = 1.0 / tf.maximum(mse, 1e-6)
            
            # === Transmitter Update ===
            weights = tf.cast(W * tf.abs(U)**2, complex_dtype)
            H_H = tf.transpose(H, [0, 1, 2, 4, 3], conjugate=True)
            
            # Weighted channel
            weighted_H = tf.expand_dims(weights, -1) * H
            A = tf.matmul(H_H, weighted_H)  # [B, S, F, M, M]
            
            # Regularization with noise power (broadcasts automatically)
            A_reg = A + no_complex[..., None, None] * eye_M
            
            # Target vector
            target = tf.cast(W, complex_dtype) * tf.math.conj(U)
            B = H_H * tf.expand_dims(target, -2)  # [B, S, F, M, K]
            
            # Solve for new precoder
            V = tf.linalg.solve(A_reg, B)
            
            # === Power Normalization ===
            power = tf.reduce_sum(tf.abs(V)**2, axis=[3, 4], keepdims=True)
            scale = tf.sqrt(K_float / (power + 1e-12))
            V = V * tf.cast(scale, complex_dtype)
        
        # ✅ Return in Sionna format: [batch, num_tx=1, S, F, M, K]
        g_out = tf.expand_dims(V, axis=1)
        
        return g_out# =============================================================================

class RZFPrecodedChannel(PrecodedChannel):
    def __init__(self, resource_grid, stream_management, **kwargs):
        super().__init__(resource_grid, stream_management, **kwargs)
    
    def call(self, inputs):
        y, h_freq = inputs
        h_pc_desired = self.get_desired_channels(h_freq)
        g = rzf_precoding_matrix(h_pc_desired, alpha=0.1)
        return g

class FactorizedTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_rx, num_rb, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_rx = num_rx
        self.num_rb = num_rb
        self.embed_dim = embed_dim
        
        # ✅ Frequency Attention (FIXED)
        self.freq_att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,  # ✅ Fixed!
            dropout=dropout  # ✅ Added!
        )
        self.norm_freq = layers.LayerNormalization(epsilon=1e-6)
        
        # ✅ Spatial Attention (FIXED)
        self.space_att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,  # ✅ Fixed!
            dropout=dropout  # ✅ Added!
        )
        self.norm_space = layers.LayerNormalization(epsilon=1e-6)

        # ✅ FFN with SwiGLU (FIXED)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 8),  # ✅ 2x for split
            SwiGLU(),                      # ✅ Using SwiGLU!
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
        self.norm_ffn = layers.LayerNormalization(epsilon=1e-6)
        
        # ✅ Residual scaling (NEW)
        self.alpha_freq = self.add_weight(
            name='alpha_freq', shape=(), 
            initializer=tf.keras.initializers.Zeros(), trainable=True
        )
        self.alpha_space = self.add_weight(
            name='alpha_space', shape=(), 
            initializer=tf.keras.initializers.Zeros(), trainable=True
        )
        self.alpha_ffn = self.add_weight(
            name='alpha_ffn', shape=(), 
            initializer=tf.keras.initializers.Zeros(), trainable=True
        )
  
    def call(self, x, training=False):
        B = tf.shape(x)[0]
        
        # --- Frequency Attention ---
        x_freq = tf.transpose(x, perm=[0, 2, 1, 3])
        x_freq = tf.reshape(x_freq, [B * self.num_rx, self.num_rb, self.embed_dim])
        
        x_norm = self.norm_freq(x_freq)
        att = self.freq_att(x_norm, x_norm, training=training)
        x_freq = x_freq + self.alpha_freq * att  # ✅ Scaled
        
        x_restored = tf.reshape(x_freq, [B, self.num_rx, self.num_rb, self.embed_dim])
        x = tf.transpose(x_restored, perm=[0, 2, 1, 3])

        # --- Spatial Attention ---
        x_space = tf.reshape(x, [B * self.num_rb, self.num_rx, self.embed_dim])
        
        x_norm = self.norm_space(x_space)
        att = self.space_att(x_norm, x_norm, training=training)
        x_space = x_space + self.alpha_space * att  # ✅ Scaled
        
        x = tf.reshape(x_space, [B, self.num_rb, self.num_rx, self.embed_dim])

        # --- FFN ---
        x_flat = tf.reshape(x, [B, self.num_rb * self.num_rx, self.embed_dim])
        
        x_norm = self.norm_ffn(x_flat)
        ffn_out = self.ffn(x_norm, training=training)
        x_flat = x_flat + self.alpha_ffn * ffn_out  # ✅ Scaled
        
        x = tf.reshape(x_flat, [B, self.num_rb, self.num_rx, self.embed_dim])
        
        return x

# TRANSFORMER PRECODER
# =============================================================================

class TransformerPrecoderCore(PrecodedChannel):
    def __init__(self, resource_grid, stream_management, num_tx=4, num_rx=4, 
                 rb_size=12, fft_size=72, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        
        super().__init__(resource_grid, stream_management, **kwargs)
        
        self.M = num_tx
        self.K = num_rx
        self.rb_size = rb_size
        self.fft_size = fft_size
        self.num_rb = fft_size // rb_size
        self.power_scale = tf.constant(np.sqrt(self.K), dtype=tf.complex64)
        self.num_ofdm = resource_grid.num_ofdm_symbols


        self.rb_aggregation = layers.Dense(1, use_bias=False)
        self.input_embedding = layers.Dense(embed_dim, kernel_initializer='glorot_uniform')
        self.feature_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # ✅ Trackable Block Registration
        self.blocks = []
        for i in range(num_layers):
            block = FactorizedTransformerBlock(
                num_rx=self.K, num_rb=self.num_rb, embed_dim=embed_dim, num_heads=num_heads
            )
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        
        self.output_projection = layers.Dense(2 * self.M, kernel_initializer='glorot_uniform')
        print(f"🔧 Building Transformer Precoder...")
        
        dummy_h_freq = tf.zeros([1, self.K, 1, 1, self.M, 14, self.fft_size], dtype=tf.complex64)
        dummy_y = tf.zeros([1, 1, 1, self.num_ofdm, self.fft_size], dtype=tf.complex64)
        _ = self.call((dummy_y, dummy_h_freq))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Parameters: {total_params:,}\n")

    @property
    def trainable_variables(self):
        """Aggregate all variables: Core components + All nested blocks."""
        v = []
        # Core input components
        v += self.rb_aggregation.trainable_variables
        v += self.input_embedding.trainable_variables
        v += self.feature_norm.trainable_variables
        
        # Collect from each block in the list
        for i in range(len(self.blocks)):
            # Access via getattr to ensure Keras tracking is maintained
            block = getattr(self, f"block_{i}")
            v += block.trainable_variables
            
        # Core output component
        v += self.output_projection.trainable_variables
        return v

    def call(self, inputs, training=False):
        _, h_freq = inputs 
        B = tf.shape(h_freq)[0]
        h_sq = tf.squeeze(h_freq, axis=[2, 3]) 
        
        h_rb = tf.reshape(h_sq, [B, self.K, self.M, 14, self.num_rb, self.rb_size])
        h_rb = tf.transpose(h_rb, [0, 3, 4, 5, 1, 2])
        
        # Domain Knowledge Stack
        h_math = tf.stack([
            tf.math.real(h_rb), tf.math.imag(h_rb)
        ], axis=-1)
        
        # Aggregate Size dim
        h_math_agg = self.rb_aggregation(tf.transpose(h_math, [0, 1, 2, 4, 5, 6, 3]))
        h_math_agg = tf.squeeze(h_math_agg, axis=-1)
        
        feat = tf.reshape(h_math_agg, [-1, self.num_rb, self.K, self.M * 2])
        x = self.feature_norm(self.input_embedding(feat))
        
        for block in self.blocks:
            x = block(x, training=training)
        
        out = self.output_projection(x)
        out = tf.reshape(out, [B, 14, self.num_rb, self.K, self.M, 2])
        w = tf.complex(out[..., 0], out[..., 1])
        w = tf.transpose(w, perm=[0, 1, 2, 4, 3])
        
        w_up = tf.tile(tf.expand_dims(w, axis=3), [1, 1, 1, self.rb_size, 1, 1])
        w_full = tf.reshape(w_up, [B, 14, self.fft_size, self.M, self.K])
        
        # Final Norm logic
        #power_tot = tf.reduce_sum(tf.abs(w_full)**2, axis=[3,4], keepdims=True)
        #w_normalized = (w_full / tf.cast(tf.math.sqrt(power_tot + 1e-12), tf.complex64)) * self.power_scale
        w_normalized = tf.math.l2_normalize(w_full, axis=[3, 4], epsilon=1e-12)* tf.cast(self.power_scale, w_full.dtype)
        return tf.expand_dims(w_normalized, axis=1)
    


# =============================================================================
# ADD THIS TO YOUR MAIN() FUNCTION
# =============================================================================
class MU_MIMO_System(tf.keras.Model):
    def __init__(self, num_tx=8, num_rx=4, precoder_type="transformer", rb_size=12):
        super().__init__()
        
        self.num_bs_antennas = num_tx
        self.num_users = num_rx
        self.precoder_type = precoder_type
        self.num_bits_per_symbol = 2

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_rx)
        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11]
        )
        
        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization="single",
            polarization_type="V", antenna_pattern="omni", carrier_frequency=2.6e9
        )
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx/2), polarization="dual",
            polarization_type="cross", antenna_pattern="38.901", carrier_frequency=2.6e9
        )
        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=False, enable_shadow_fading=False
        )
        
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(int(self.rg.num_data_symbols), int(self.rg.num_data_symbols * 2))
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(resource_grid=self.rg, stream_management=self.sm)
        
        if precoder_type == "transformer":
            print("✅ Using Transformer Precoder")
            self.precoder = TransformerPrecoderCore(
                self.rg, self.sm,
                num_tx_antennas=num_tx,
                num_rx_antennas=num_rx,
                rb_size=rb_size,
                embed_dim=128,
                num_heads=4,
                num_layers=4
            )
        elif precoder_type == "rzf":
            print("✅ Using RZF Precoder")
            self.precoder = RZFPrecodedChannel(self.rg, self.sm)
        elif precoder_type == "wmmse":
            print("✅ Using WMMSE Precoder")
            self.precoder = WMMSEPrecodedChannel(resource_grid=self.rg,stream_management=self.sm, num_tx_ant=self.num_bs_antennas,num_rx_ant=self.num_users,num_iterations=10 )
    
    def new_topology(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        topology = gen_topology(batch_size, self.num_users, "umi")
        self.channel_model.set_topology(*topology)
    
    @tf.function
    def call_with_cached_channel(self, batch_size, ebno_db, h_freq_cached, training=False):
        """Forward pass using cached channel"""
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        
        b = self.binary_source([batch_size, 1, self.num_users, int(self.rg.num_data_symbols)])
        c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)
        
        # ✅ Use cached channel instead of generating new one
        h_freq = h_freq_cached
        
        if self.precoder_type == "transformer":
            g = self.precoder((x_rg, h_freq), training=training)
        else:
            g = self.precoder((x_rg, h_freq))
        
        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded_struct = tf.matmul(W, x_vec)
        x_precoded_struct = tf.squeeze(x_precoded_struct, axis=-1)
        x_precoded = tf.transpose(x_precoded_struct, perm=[0, 3, 1, 2])
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        h_eff = self.precoder.compute_effective_channel(h_freq, g)
        
        y = self.apply_channel(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)
        
        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g
    
    @tf.function
    def call(self, batch_size, ebno_db, training=False):
        """Standard forward pass (for baselines)"""
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        
        b = self.binary_source([batch_size, 1, self.num_users, int(self.rg.num_data_symbols)])
        c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)
        
        cir = self.channel_model(batch_size, self.rg.num_ofdm_symbols, 1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        
        if self.precoder_type == "transformer":
            g = self.precoder((x_rg, h_freq), training=training)
        elif self.precoder_type == "wmmse":
            g = self.precoder((x_rg, h_freq, no))  # ✅ Pass noise power
        else:
            g = self.precoder((x_rg, h_freq))
        
        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded_struct = tf.matmul(W, x_vec)
        x_precoded_struct = tf.squeeze(x_precoded_struct, axis=-1)
        x_precoded = tf.transpose(x_precoded_struct, perm=[0, 3, 1, 2])
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        h_eff = self.precoder.compute_effective_channel(h_freq, g)
        
        y = self.apply_channel(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)
        
        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g


NUM_TX = 4
NUM_RX = 4

print("="*80)
print("STEP 0: FREQUENCY CORRELATION TEST")
print("="*80 + "\n")

# Create system for testing
test_system = MU_MIMO_System(
    num_tx=NUM_TX, num_rx=NUM_RX,
    precoder_type="rzf",  # Any precoder works for channel analysis
    rb_size=12
)

# Run correlation test
recommendation, correlations = test_umi_frequency_correlation(
    test_system, 
    num_samples=1000,
    batch_size=128
)

# Decision point
if correlations[11] < 0.85:
    print("⚠️  WARNING: Low correlation detected!")
    print("    Consider adding learnable upsampling (12 parameters)")
    user_input = input("\n    Continue with linear anyway? (y/n): ")
    if user_input.lower() != 'y':
        print("Exiting. Modify upsampling code and re-run.")
        

# Continue with training...
print("\n" + "="*80)
print("TRAINING WITH 50k CACHED CHANNELS")
print("="*80 + "\n")

system = MU_MIMO_System(
    num_tx=NUM_TX, num_rx=NUM_RX,
    precoder_type="transformer",
    rb_size=12
)

