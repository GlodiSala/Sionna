import os
import json
import pickle
import logging
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

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


# =============================================================================
# TRANSFORMER BLOCKS
# =============================================================================
from tensorflow.keras import layers
from tensorflow.keras.layers import RMSNormalization

DATASET_SIZE = 50000  
BATCH_SIZE =   256

class SwiGLU(layers.Layer):
    def call(self, x):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=-1)
        return (x_a * tf.nn.sigmoid(x_a)) * x_b

'''
class SimpleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads, 
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2), 
            SwiGLU(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
        
        self.norm1 = RMSNormalization(epsilon=1e-6)
        self.norm2 = RMSNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x_norm = self.norm1(x)
        attn_output = self.att(x_norm, x_norm, training=training)
        x = x + self.dropout1(attn_output, training=training)
        
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output
        return x
'''

# =============================================================================
# SIMPLE LEARNED UPSAMPLER
# =============================================================================
class LearnedBetweenRBUpsampler(layers.Layer):
    """
    Learned interpolation between RB centers.
    12 parameters - learns optimal blending per position.
    
    Why learn: Your correlation (0.37 @ 12 SC) suggests non-linear variation!
    """
    
    def __init__(self, rb_size=12, fft_size=72, **kwargs):
        super().__init__(**kwargs)
        self.rb_size = rb_size
        self.fft_size = fft_size
        self.num_rb = fft_size // rb_size
        
        # Learnable blending weights (start linear, then optimize)
        self.alpha_raw = self.add_weight(
            name='alpha_raw',
            shape=(rb_size,),
            initializer=tf.keras.initializers.Constant(
                # Initialize to linear: inverse sigmoid of [0, 1/12, ..., 11/12]
                # sigmoid^-1(x) = log(x / (1-x))
                np.log(np.linspace(0.01, 0.99, rb_size) / 
                       (1 - np.linspace(0.01, 0.99, rb_size)))
            ),
            trainable=True
        )
    
    def call(self, w_rb):
        """
        Args:
            w_rb: [B, OFDM, num_rb, M, K]
        Returns:
            w_full: [B, OFDM, FFT, M, K]
        """
        B = tf.shape(w_rb)[0]
        OFDM = tf.shape(w_rb)[1]
        
        # Constrain to [0, 1] via sigmoid
        alpha = tf.nn.sigmoid(self.alpha_raw)  # [12]
        
        # Pad boundaries
        w_padded = tf.concat([
            w_rb[:, :, 0:1, :, :],
            w_rb,
            w_rb[:, :, -1:, :, :]
        ], axis=2)  # [B, OFDM, num_rb+2, M, K]
        
        # Interpolate with learned weights
        w_list = []
        for rb_idx in range(self.num_rb):
            w_current = w_padded[:, :, rb_idx, :, :]
            w_next = w_padded[:, :, rb_idx + 1, :, :]
            
            for sc_idx in range(self.rb_size):
                # Learned blend
                alpha_val = tf.cast(alpha[sc_idx], w_current.dtype)
                w_sc = (1.0 - alpha_val) * w_current + alpha_val * w_next
                w_list.append(w_sc)
        
        w_full = tf.stack(w_list, axis=2)
        
        return w_full

# =============================================================================
# TRANSFORMER PRECODER WITH LEARNED UPSAMPLING
# =============================================================================
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

        # Input processing
        self.rb_aggregation = layers.Dense(1, use_bias=False)
        self.input_embedding = layers.Dense(embed_dim, kernel_initializer='glorot_uniform')
        self.feature_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            block = FactorizedTransformerBlock(
                num_rx=self.K, num_rb=self.num_rb, embed_dim=embed_dim, num_heads=num_heads
            )
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        
        # Output projection
        self.output_projection = layers.Dense(2 * self.M, kernel_initializer='glorot_uniform')
        
        # ✅ LEARNED UPSAMPLER (NEW!)
        self.upsampler = LearnedBetweenRBUpsampler(rb_size=rb_size, fft_size=fft_size)
        
        print(f"🔧 Building 5D Transformer Precoder with Learned Upsampling")
        print(f"   RB Size: {rb_size}, Num RBs: {self.num_rb}, FFT: {fft_size}")
        print(f"   OFDM Symbols: {self.num_ofdm}")
        print(f"   Upsampling: Learned position-based (24 parameters)")
        print(f"   Reason: Low freq correlation (0.37 @ 12 subcarriers)")
        
        # Build model
        dummy_h_freq = tf.zeros([1, self.K, 1, 1, self.M, self.num_ofdm, self.fft_size], dtype=tf.complex64)
        dummy_y = tf.zeros([1, 1, 1, self.num_ofdm, self.fft_size], dtype=tf.complex64)
        _ = self.call((dummy_y, dummy_h_freq))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Total Parameters: {total_params:,}")
        print(f"   (Upsampling adds 24 parameters)\n")

    @property
    def trainable_variables(self):
        """Aggregate all variables: Core components + All nested blocks + Upsampler."""
        v = []
        # Core input components
        v += self.rb_aggregation.trainable_variables
        v += self.input_embedding.trainable_variables
        v += self.feature_norm.trainable_variables
        
        # Transformer blocks
        for i in range(len(self.blocks)):
            block = getattr(self, f"block_{i}")
            v += block.trainable_variables
            
        # Output projection
        v += self.output_projection.trainable_variables
        
        # ✅ Upsampler (NEW!)
        v += self.upsampler.trainable_variables
        
        return v

    def call(self, inputs, training=False):
        _, h_freq = inputs 
        B = tf.shape(h_freq)[0]
        
        # =====================================================================
        # STEP 1: Input Processing - Remove singleton dims
        # =====================================================================
        # Input: [B, K, 1, 1, M, OFDM, FFT]
        h_sq = tf.squeeze(h_freq, axis=[2, 3])  # [B, K, M, OFDM, FFT]
        
        # =====================================================================
        # STEP 2: Reshape to Resource Blocks
        # =====================================================================
        h_rb = tf.reshape(h_sq, [B, self.K, self.M, self.num_ofdm, self.num_rb, self.rb_size])
        # [B, K, M, OFDM, num_rb, rb_size]
        
        h_rb = tf.transpose(h_rb, [0, 3, 4, 5, 1, 2])
        # [B, OFDM, num_rb, rb_size, K, M]
        
        # =====================================================================
        # STEP 3: Stack Real/Imag and Aggregate over rb_size
        # =====================================================================
        h_math = tf.stack([tf.math.real(h_rb), tf.math.imag(h_rb)], axis=-1)
        # [B, OFDM, num_rb, rb_size, K, M, 2]
        
        # Aggregate over rb_size dimension (exploit frequency correlation)
        h_math_agg = self.rb_aggregation(tf.transpose(h_math, [0, 1, 2, 4, 5, 6, 3]))
        # After transpose: [B, OFDM, num_rb, K, M, 2, rb_size]
        # After Dense(1): [B, OFDM, num_rb, K, M, 2, 1]
        
        h_math_agg = tf.squeeze(h_math_agg, axis=-1)
        # [B, OFDM, num_rb, K, M, 2]
        
        # =====================================================================
        # STEP 4: Prepare for Transformer - Flatten OFDM into batch
        # =====================================================================
        # Concatenate real/imag as features
        feat = tf.reshape(h_math_agg, [B * self.num_ofdm, self.num_rb, self.K, self.M * 2])
        # [B*OFDM, num_rb, K, M*2]
        
        # =====================================================================
        # STEP 5: Transformer Processing
        # =====================================================================
        x = self.input_embedding(feat)  # [B*OFDM, num_rb, K, embed_dim]
        x = self.feature_norm(x)
        
        for block in self.blocks:
            x = block(x, training=training)
        # x: [B*OFDM, num_rb, K, embed_dim]
        
        # =====================================================================
        # STEP 6: Output Projection
        # =====================================================================
        out = self.output_projection(x)  # [B*OFDM, num_rb, K, M*2]
        
        # Unflatten OFDM dimension
        out = tf.reshape(out, [B, self.num_ofdm, self.num_rb, self.K, self.M * 2])
        # [B, OFDM, num_rb, K, M*2]
        
        # Split into real/imag and create complex
        out = tf.reshape(out, [B, self.num_ofdm, self.num_rb, self.K, self.M, 2])
        w_rb = tf.complex(out[..., 0], out[..., 1])
        # [B, OFDM, num_rb, K, M]
        
        # Transpose to [B, OFDM, num_rb, M, K] for Sionna format
        w_rb = tf.transpose(w_rb, perm=[0, 1, 2, 4, 3])
        # [B, OFDM, num_rb, M, K]
        
        # =====================================================================
        # STEP 7: LEARNED UPSAMPLING (NEW!)
        # =====================================================================
        w_full = self.upsampler(w_rb)  # [B, OFDM, FFT, M, K]
        
        # =====================================================================
        # STEP 8: Power Normalization
        # =====================================================================
        # Normalize per frequency across TX antennas and users
        w_normalized = tf.math.l2_normalize(w_full, axis=[3, 4], epsilon=1e-12) * \
                       tf.cast(self.power_scale, w_full.dtype)
        
        # Add num_tx dimension for Sionna: [B, 1, OFDM, FFT, M, K]
        return tf.expand_dims(w_normalized, axis=1)

    # DATASET (identique à ton code)

# =============================================================================

# =============================================================================
# BASELINE PRECODERS
# =============================================================================
class RZFPrecodedChannel(PrecodedChannel):
    def __init__(self, resource_grid, stream_management, **kwargs):
        super().__init__(resource_grid, stream_management, **kwargs)
    
    def call(self, inputs):
        y, h_freq = inputs
        h_pc_desired = self.get_desired_channels(h_freq)
        g = rzf_precoding_matrix(h_pc_desired, alpha=0.1)
        return g


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

class CachedSionnaDataset:
    """Pre-generate and cache Sionna channels - FIXED VERSION"""
    
    def __init__(self, system, dataset_size=100000, batch_size=512,cache_file='cached_sionna.npy'):
        self.system = system
        self.dataset_size = dataset_size
        self.cache_file = cache_file
        self.batch_size= batch_size
        
        print(f"\n{'='*80}")
        print(f"  CHANNEL DATASET")
        print(f"{'='*80}")
        print(f"  Target size: {dataset_size:,} samples")
        print(f"  Cache file: {cache_file}")
        print(f"{'='*80}\n")
        
        if os.path.exists(cache_file):
            print(f"📂 Loading existing cached dataset...")
            self.load_cached_channels()
        else:
            print(f"🔄 Generating new channel dataset...")
            self.generate_and_cache_channels_optimized()
    
    def generate_and_cache_channels_optimized(self):
        """Generate channels using FAST pre-allocated array"""
        batch_size = self.batch_size
        num_batches = self.dataset_size // batch_size
        
        print(f"Generating {num_batches} batches of {batch_size} samples each...\n")
        
        # Get shape from one batch
        print("📐 Getting channel dimensions...")
        self.system.new_topology(batch_size, seed=42)
        cir = self.system.channel_model(
            batch_size,
            self.system.rg.num_ofdm_symbols,
            1.0 / self.system.rg.ofdm_symbol_duration
        )
        h_freq_sample = cir_to_ofdm_channel(
            self.system.frequencies,
            *cir,
            normalize=True
        )
        sample_shape = h_freq_sample.shape
        print(f"   Shape per batch: {sample_shape}")
        
        # Pre-allocate entire array
        total_shape = (self.dataset_size,) + sample_shape[1:]
        print(f"   Total shape: {total_shape}")
        print(f"   Memory required: ~{self.estimate_memory(total_shape):.2f} GB")
        
        self.h_freq_all = np.zeros(total_shape, dtype=np.complex64)
        print(f"✅ Pre-allocated array!\n")
        
        # Fill array in batches
        import time
        start_time = time.time()
        
        for i in range(num_batches):
            if i % 50 == 0:
                progress = (i / num_batches) * 100
                if i > 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / i * (num_batches - i)
                    print(f"   Progress: {i:4d}/{num_batches} ({progress:5.1f}%) | "
                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
                else:
                    print(f"   Progress: {i:4d}/{num_batches} ({progress:5.1f}%)")
            
            self.system.new_topology(batch_size, seed=None)
            cir = self.system.channel_model(
                batch_size,
                self.system.rg.num_ofdm_symbols,
                1.0 / self.system.rg.ofdm_symbol_duration
            )
            h_freq = cir_to_ofdm_channel(
                self.system.frequencies,
                *cir,
                normalize=True
            )
            
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            self.h_freq_all[start_idx:end_idx] = h_freq.numpy()
        
        total_time = time.time() - start_time
        print(f"\n✅ Generated {len(self.h_freq_all):,} channel samples in {total_time/60:.1f} minutes")
        print(f"   Shape: {self.h_freq_all.shape}")
        print(f"   Dtype: {self.h_freq_all.dtype}")
        
        # Normalize
        self.normalize_channels_sionna_style()
        
        # Save
        print(f"\n💾 Saving to {self.cache_file} (uncompressed for fast loading)...")
        np.save(self.cache_file, self.h_freq_all)
        
        file_size_gb = os.path.getsize(self.cache_file) / (1024**3)
        print(f"✅ Saved!")
        print(f"   File: {self.cache_file}")
        print(f"   Size: {file_size_gb:.2f} GB")
        print(f"✅ Saved successfully!")
    
    def estimate_memory(self, shape):
        """Estimate memory in GB"""
        num_elements = np.prod(shape)
        bytes_total = num_elements * 8  # complex64 = 8 bytes
        gb = bytes_total / (1024**3)
        return gb
    
    def normalize_channels_sionna_style(self):
        """Apply Sionna-compatible normalization"""
        print(f"\n🔧 Applying Sionna-compatible normalization...")
        
        # Compute power per sample (across all dimensions except batch)
        #power_per_sample = np.mean(np.abs(self.h_freq_all)**2, axis=tuple(range(1, len(self.h_freq_all.shape))),keepdims=True)
        
        # Normalize to unit average power
        #norm_factor = np.sqrt(power_per_sample) + 1e-12
        #self.h_freq_all = self.h_freq_all / norm_factor
        
        # Verify
        final_power = np.mean(np.abs(self.h_freq_all)**2)
        print(f"   ✅ Mean power after normalization: {final_power:.6f} (target: 1.0)")
        
        if not (0.95 < final_power < 1.05):
            print(f"   ⚠️  WARNING: Power {final_power:.4f} is outside expected range!")
    
    def load_cached_channels(self):
        """Load pre-generated channels from cache - FIXED"""
        import time
        start = time.time()
        
        print(f"   Loading {self.cache_file}...")
        print(f"   (This may take 30-60 seconds for large files...)")
        
        # ✅ FIX: Handle both .npy and .npz formats correctly
        if self.cache_file.endswith('.npy'):
            # .npy files: direct array load
            self.h_freq_all = np.load(self.cache_file)
        elif self.cache_file.endswith('.npz'):
            # .npz files: dictionary load
            data = np.load(self.cache_file, allow_pickle=True)
            self.h_freq_all = data['h_freq']
        else:
            raise ValueError(f"Unknown file format: {self.cache_file}")
        
        load_time = time.time() - start
        
        print(f"✅ Loaded {len(self.h_freq_all):,} samples in {load_time:.1f}s")
        print(f"   Shape: {self.h_freq_all.shape}")
        print(f"   Mean power: {np.mean(np.abs(self.h_freq_all)**2):.6f}")
    
    def get_batch(self, batch_size):
        """Sample a random batch from cached channels"""
        indices = np.random.choice(len(self.h_freq_all), size=batch_size, replace=True)
        return tf.constant(self.h_freq_all[indices], dtype=tf.complex64)

# =============================================================================
# MU-MIMO SYSTEM WITH CACHED CHANNELS
# =============================================================================
class MU_MIMO_System(tf.keras.Model):
    def __init__(self, num_tx=8, num_rx=4, precoder_type="transformer", rb_size=12, batch_size=32, weights_path=None):
        super().__init__()
        
        self.num_bs_antennas = num_tx
        self.num_users = num_rx
        self.precoder_type = precoder_type
        self.num_bits_per_symbol = 2
        self.defaut_batch_size=batch_size

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
            if weights_path is not None:
                self.load_precoder_weights(weights_path)
        elif precoder_type == "rzf":
            print("✅ Using RZF Precoder")
            self.precoder = RZFPrecodedChannel(self.rg, self.sm)
        elif precoder_type == "wmmse":
            print("✅ Using WMMSE Precoder")
            self.precoder = WMMSEPrecodedChannel(resource_grid=self.rg,stream_management=self.sm, num_tx_ant=self.num_bs_antennas,num_rx_ant=self.num_users,num_iterations=10 )
    
    def load_precoder_weights(self, weights_path):
        """Load trained weights for transformer precoder"""
        import pickle
        import os
        
        # ✅ Check if it's a directory or file
        if os.path.isdir(weights_path):
            # If directory, look for weights.pkl inside
            weights_file = os.path.join(weights_path, 'weights.pkl')
        else:
            weights_file = weights_path
        
        if not os.path.exists(weights_file):
            print(f"❌ Weights file not found: {weights_file}")
            
            # Try to help find the correct path
            parent_dir = os.path.dirname(weights_file)
            if os.path.exists(parent_dir):
                print(f"\n📂 Available files in {parent_dir}:")
                for f in os.listdir(parent_dir):
                    print(f"   - {f}")
            return False
        
        print(f"\n📂 Loading precoder weights from: {weights_file}")
        
        try:
            with open(weights_file, 'rb') as f:
                trained_weights = pickle.load(f)
            
            precoder_vars = self.precoder.trainable_variables
            
            print(f"   Saved weights: {len(trained_weights)} tensors")
            print(f"   Model weights: {len(precoder_vars)} tensors")
            
            if len(trained_weights) != len(precoder_vars):
                print(f"⚠️  WARNING: Weight count mismatch!")
                print(f"   Attempting to load matching weights...")
                
                # Load as many as match
                loaded_count = 0
                for i, (var, weight) in enumerate(zip(precoder_vars, trained_weights)):
                    try:
                        if var.shape == weight.shape:
                            var.assign(weight)
                            loaded_count += 1
                        else:
                            print(f"   ⚠️  Skipping weight {i}: shape mismatch")
                            print(f"      Model: {var.shape}, Saved: {weight.shape}")
                    except Exception as e:
                        print(f"   ❌ Error loading weight {i}: {e}")
                
                print(f"   Loaded {loaded_count}/{len(trained_weights)} weights")
                return loaded_count == len(precoder_vars)
            else:
                # Perfect match - load all
                for var, weight in zip(precoder_vars, trained_weights):
                    var.assign(weight)
                print(f"✅ Successfully loaded all {len(trained_weights)} weights")
                print(f"   (Weights are batch-size independent)\n")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return False
    
    def new_topology(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        bs = batch_size if batch_size is not None else self.default_batch_size
    
        topology = gen_topology(bs, self.num_users, "umi")
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


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class SimpleCheckpoint:
    def __init__(self, save_dir='./training_weight'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_checkpoint_path = None
    
    def save_best(self, precoder, config_dict, metrics):
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            import shutil
            shutil.rmtree(self.best_checkpoint_path)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_name = f"best_{timestamp}"
        ckpt_dir = os.path.join(self.save_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        weights_path = os.path.join(ckpt_dir, 'weights.pkl')
        weights = [v.numpy() for v in precoder.trainable_variables]
        with open(weights_path, 'wb') as f:
            pickle.dump(weights, f)
        
        config_path = os.path.join(ckpt_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        metrics_path = os.path.join(ckpt_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            clean_metrics = {k: float(v) if np.isscalar(v) else v.tolist() 
                           for k, v in metrics.items()}
            json.dump(clean_metrics, f, indent=2)
        
        self.best_checkpoint_path = ckpt_dir
        return ckpt_dir


# =============================================================================
# SIMPLIFIED TRAINER (STABLE & FAST)
# =============================================================================
# =============================================================================
# SIMPLIFIED TRAINER WITH BCE + SUM RATE
# =============================================================================
class SimplifiedTrainer:
    """Simple trainer: Sum Rate + BCE for decodability"""
    
    def __init__(self, system, cached_dataset,
                 snr_db=20.0,
                 total_epochs=200,
                 learning_rate=1e-3,
                 batch_size=512,
                 num_iters=100,
                 bce_weight=0.5,
                 rate_weight=1.0):
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.bce_weight = bce_weight
        self.rate_weight = rate_weight
        
        # Simple Adam with cosine decay
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * num_iters,
            alpha=0.1
        )
        
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
        self.trainable_vars = system.precoder.trainable_variables
        
        self.checkpoint = SimpleCheckpoint('./training_weight')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  SIMPLIFIED TRAINING (SUM RATE + BCE)")
        print(f"{'='*80}")
        print(f"  Cached samples: {len(cached_dataset.h_freq_all):,}")
        print(f"  SNR: {snr_db} dB")
        print(f"  Epochs: {total_epochs}")
        print(f"  LR: {learning_rate} → {learning_rate*0.1}")
        print(f"  BCE weight: {bce_weight}")
        print(f"  Rate weight: {rate_weight}")
        print(f"{'='*80}\n")
    
    @tf.function
    def train_step(self, h_freq_batch, batch_size, snr_db):
        """Training with STABLE BCE + Sum Rate"""
        
        with tf.GradientTape() as tape:
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=True)
            
            # ==========================================
            # 1. STABLE BCE Loss
            # ==========================================
            c_float = tf.cast(c, tf.float32)
            
            # ✅ FIX: Clip LLRs to prevent overflow
            llr_clipped = tf.clip_by_value(llr, -10.0, 10.0)
            
            # ✅ Use stable BCE
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
            bce_loss = tf.reduce_mean(bce)
            
            # ✅ Check for NaN
            bce_loss = tf.where(tf.math.is_finite(bce_loss), bce_loss, tf.constant(1.0))
            
            # ==========================================
            # 2. Sum Rate
            # ==========================================
            sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr_clipped = tf.clip_by_value(sinr, 0.0, 1e4)
            
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            h_pc_desired = self.system.precoder.get_desired_channels(h_freq)

            # ==========================================
            # 3. Combined Loss (SIMPLER WEIGHTS)
            # ==========================================
            # Just minimize BCE and maximize sum rate
            loss =  self.bce_weight*bce_loss - self.rate_weight * sum_rate  # ✅ Simple ratio
                        
            
            # Metrics
            ber = compute_ber(b, b_hat)
            min_rate = tf.reduce_min(rate_per_user)
            K = tf.cast(tf.shape(rate_per_user)[0], tf.float32)
            jain = tf.square(tf.reduce_sum(rate_per_user)) / (K * tf.reduce_sum(tf.square(rate_per_user)))
        
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        return {
            'sum_rate': sum_rate,
            'bce_loss': bce_loss,
            'rate_per_user': rate_per_user,
            'min_rate': min_rate,
            'ber': ber,
            'jain': jain,
            'grad_norm': grad_norm,
            'loss': loss
        }
    
    def train(self, log_interval=1):
        print(f"🚀 Training @ {self.snr_db} dB (BCE + Sum Rate)\n")
        
        print(f"{'='*140}")
        print(f"{'Epoch':>6} | {'Sum':>7} | {'BCE':>7} | {'User Rates':>40} | {'Min':>6} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*140}")
        
        for epoch in range(self.total_epochs):
            metrics = {
                'sum_rate': [], 'bce_loss': [], 'rate_per_user': [], 
                'min_rate': [], 'ber': [], 'jain': [], 'grad_norm': [], 'loss': []
            }
            
            for i in range(self.num_iters):
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                
                if epoch == 0 and i == 0:
                    print("   ⏳ Compiling...")
                
                m = self.train_step(
                    h_freq_batch,
                    tf.constant(self.batch_size, dtype=tf.int32),
                    tf.constant(self.snr_db, dtype=tf.float32)
                )
                
                if epoch == 0 and i == 0:
                    print("   ✅ Compiled!\n")
                
                for k in ['sum_rate', 'bce_loss', 'min_rate', 'ber', 'jain', 'grad_norm', 'loss']:
                    metrics[k].append(float(m[k]))
                metrics['rate_per_user'].append(m['rate_per_user'].numpy())
            
            # Average
            avg_sum = np.mean(metrics['sum_rate'])
            avg_bce = np.mean(metrics['bce_loss'])
            avg_rates = np.mean(metrics['rate_per_user'], axis=0)
            avg_min = np.mean(metrics['min_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad_norm'])
            avg_loss = np.mean(metrics['loss'])
            
            self.history.append({
                'epoch': epoch,
                'sum_rate': avg_sum,
                'bce_loss': avg_bce,
                'ber': avg_ber
            })
            
            if avg_sum > self.best_sum_rate and avg_ber < 0.1:
                self.best_sum_rate = avg_sum
                self.best_epoch = epoch
            
            # Log
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
                marker = " ⭐" if epoch == self.best_epoch else ""
                
                print(f"{epoch+1:6d} | {avg_sum:7.2f} | {avg_bce:7.4f} | {rate_str:>40} | "
                      f"{avg_min:6.2f} | {avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}")
        
        print(f"{'='*140}\n")
        
        # Save
        print(f"💾 Saving best model (epoch {self.best_epoch+1})...")
        final_ckpt = self.checkpoint.save_best(
            self.system.precoder,
            {'best_epoch': self.best_epoch, 'snr_db': self.snr_db,
             'bce_weight': self.bce_weight, 'rate_weight': self.rate_weight},
            {'sum_rate': self.best_sum_rate}
        )
        
        history_path = os.path.join(final_ckpt, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")


class CachedTrainer:
    """
    Hybrid Trainer with WMMSE Warm-Start:
    1. Warm-Up Phase: Learn from WMMSE (better teacher than RZF!)
    2. Fine-Tuning Phase: Improve with BCE + Sum Rate
    """
    
    def __init__(self, system, cached_dataset,
                 snr_db=20.0,
                 total_epochs=100,
                 warmup_epochs=10,
                 learning_rate=5e-4,
                 batch_size=32,  # ✅ Reduced for memory
                 num_iters=None,
                 bce_weight=0.5,  # ✅ Increased from 0.1
                 rate_weight=1.0):
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        
        self.total_epochs = int(total_epochs) if total_epochs is not None else 200
        self.num_iters = int(num_iters) if num_iters is not None else None
        self.warmup_epochs = int(warmup_epochs)
        
        self.batch_size = batch_size
        self.bce_weight = bce_weight
        self.rate_weight = rate_weight
        
        # ✅ Create WMMSE teacher (separate from system's precoder)
        self.wmmse_teacher = WMMSEPrecodedChannel(
            resource_grid=system.rg,
            stream_management=system.sm,
            num_iterations=10  # Full WMMSE iterations for teacher
        )
        
        # Optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * (num_iters or 100),
            alpha=0.1  # Decay to 10% of initial LR
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=5.0)
        
        # Build model
        dummy_in = (
            tf.zeros([1, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64),
            tf.zeros([1, 1, system.num_users, 1, system.num_bs_antennas, 
                     system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
        )
        try:
            if not system.precoder.built:
                system.precoder(dummy_in)
        except:
            pass

        self.trainable_vars = system.precoder.trainable_variables
        self.checkpoint = SimpleCheckpoint('./training_weight')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  HYBRID TRAINING (WMMSE WARM-START + FINE-TUNING)")
        print(f"{'='*80}")
        print(f"  Total Epochs:     {self.total_epochs}")
        print(f"  Warm-up (WMMSE):  {self.warmup_epochs} epochs")
        print(f"  Fine-tune (Hybrid): {self.total_epochs - self.warmup_epochs} epochs")
        print(f"  Batch Size:       {self.batch_size}")
        print(f"  Initial LR:       {learning_rate} → {learning_rate * 0.1}")
        print(f"  BCE Weight:       {bce_weight}")
        print(f"  Rate Weight:      {rate_weight}")
        print(f"{'='*80}\n")

    @tf.function
    def train_step_supervised(self, h_freq_batch, batch_size, snr_db):
        """Phase 1: Learn WMMSE Direction (not just MSE)"""
        
        with tf.GradientTape() as tape:
            # Compute teacher
            no = ebnodb2no(snr_db, self.system.num_bits_per_symbol, 0.5, self.system.rg)
            dummy_y = tf.zeros([batch_size, 1, self.system.num_users, 14, 72], dtype=tf.complex64)
            g_teacher = self.wmmse_teacher((dummy_y, h_freq_batch, no))
            
            # Compute student
            g_pred = self.system.precoder((dummy_y, h_freq_batch), training=True)
            
            # ✅ FIX 1: Flatten to vectors for cosine similarity
            g_teacher_flat = tf.reshape(g_teacher, [batch_size, -1])  # [B, S*F*M*K*2]
            g_pred_flat = tf.reshape(g_pred, [batch_size, -1])
            
            # ✅ FIX 2: Cosine similarity loss (1 - cos(θ))
            # This measures angular difference, not magnitude
            teacher_normalized = tf.math.l2_normalize(g_teacher_flat, axis=-1, epsilon=1e-12)
            pred_normalized = tf.math.l2_normalize(g_pred_flat, axis=-1, epsilon=1e-12)
            
            #teacher_normalized = g_teacher_flat / teacher_norm
            #pred_normalized = g_pred_flat / pred_norm
            
            # Cosine similarity: dot product of normalized vectors
            cosine_sim = tf.reduce_sum(teacher_normalized * tf.math.conj(pred_normalized), axis=-1)
            
            # Loss = 1 - similarity (0 = perfect match, 2 = opposite)
            cosine_loss = 1.0 - tf.abs(cosine_sim)
            loss = 10.0 * tf.reduce_mean(cosine_loss)  # Scale for reasonable magnitude
            
            # ✅ FIX 3: Add MSE as auxiliary loss (10:1 ratio)
            mse_loss = tf.reduce_mean(tf.abs(g_pred - g_teacher)**2)
            loss = loss + 1 * mse_loss
        
        # Gradient application
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        # --- 2. LOGGING STEP (Full System for Metrics) ---
        b, b_hat, c, llr, h_eff, no, _, _, _, _ = \
            self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=False)

        # Compute Metrics
        sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sinr_clipped = tf.clip_by_value(sinr, 1e-9, 1e4)
        
        rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
        rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
        sum_rate = tf.reduce_sum(rate_per_user)
        
        ber = compute_ber(b, b_hat)
        
        # BCE Loss (for logging)
        c_float = tf.cast(c, tf.float32)
        llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
        bce_loss = tf.reduce_mean(bce)

        return sum_rate, bce_loss, ber, grad_norm, loss, rate_per_user

    @tf.function
    def train_step_unsupervised(self, h_freq_batch, batch_size, snr_db):
        """Phase 2: Hybrid Optimization (BCE + Sum Rate)"""
        with tf.GradientTape() as tape:
            # Full E2E Chain
            '''''
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=True)
            
            # --- 1. BCE Loss (Decodability) ---
            c_float = tf.cast(c, tf.float32)
            llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
            bce_loss = tf.reduce_mean(bce)
            
            # --- 2. Sum Rate Loss ---
            sinr = self.system.lmmse_sinr(h_eff, no=no + 1e-6, interference_whitening=True)
            
            # ✅ Use log1p for better numerical stability
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            
            # ✅ 3. Fairness Loss (Optional but recommended)
            min_rate = tf.reduce_min(rate_per_user)
            fairness_loss = -min_rate  # Encourage min user rate
            '''''
            g = self.system.precoder((None, h_freq_batch), training=True)
            h_perm = tf.transpose(h_freq_batch, perm=[0, 5, 6, 1, 3, 2, 4])
            h_matrix = tf.squeeze(h_perm, axis=[4, 5]) # [B, S, F, Rx, Tx]
            w_matrix = tf.squeeze(g, axis=1)           # [B, S, F, Tx, Rx]
            
            # 2. H_eff = H * W
            h_eff_proxy = tf.matmul(h_matrix, w_matrix) # [B, S, F, Rx, Rx]
            
            # 3. SINR & Rate
            diag_sig = tf.abs(tf.linalg.diag_part(h_eff_proxy))**2
            total_pow = tf.reduce_sum(tf.abs(h_eff_proxy)**2, axis=-1)
            interf = total_pow - diag_sig
            
            no = ebnodb2no(snr_db, 2, 0.5, self.system.rg)
            no_val = tf.cast(no, diag_sig.dtype)
            
            sinr = diag_sig / (interf + no_val + 1e-12)
            rates_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rates_element, axis=[0, 1, 2])
            sum_rate = tf.reduce_sum(rate_per_user)
            # Combined Loss
            loss =-(self.rate_weight * sum_rate)
            
        # Gradient Handling
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        loss_safe = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        b, b_hat, _, _, _, _, _, _, _, _ = \
            self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=False)
        
        return sum_rate, tf.constant(0.0), compute_ber(b, b_hat), grad_norm, loss_safe, rate_per_user
    
    def train(self, log_interval=1, patience=10, print_every=10):
        # Auto-calculate iterations
        if self.num_iters is None:
            if hasattr(self.cached_dataset, 'h_freq_all'):
                self.num_iters = len(self.cached_dataset.h_freq_all) // self.batch_size
            else:
                self.num_iters = DATASET_SIZE // self.batch_size
            print(f"  -> Auto-configured: {self.num_iters} iterations/epoch")

        print(f"🚀 Training @ {self.snr_db} dB")
        print(f"{'='*155}")
        print(f"{'Epoch':>6} | {'Mode':>10} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*155}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'bce': [], 'ber': [], 'loss': [], 'rates': [], 'grad': []}
            is_warmup = epoch < self.warmup_epochs
            mode_str = "WMMSE" if is_warmup else "HYBRID"
            
            for i in range(self.num_iters):
                # Get Batch
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                # Train Step
                if is_warmup:
                    rate, bce, ber, g_norm, loss, rates = self.train_step_supervised(
                        h_freq_batch, bs_tensor, snr_tensor
                    )
                else:
                    rate, bce, ber, g_norm, loss, rates = self.train_step_unsupervised(
                        h_freq_batch, bs_tensor, snr_tensor
                    )
                
                # Store Metrics
                metrics['sum_rate'].append(float(rate))
                metrics['bce'].append(float(bce))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())

                # Intra-epoch printing
                if (i + 1) % print_every == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-print_every:])
                    rec_loss = np.mean(metrics['loss'][-print_every:])
                    rec_ber = np.mean(metrics['ber'][-print_every:])
                    rec_grad = np.mean(metrics['grad'][-print_every:])
                    rec_rates = np.mean(metrics['rates'][-print_every:], axis=0)
                    
                    rates_str = "[" + " ".join([f"{r:.1f}" for r in rec_rates]) + "]"
                    print(f"    ... Step {i+1}/{self.num_iters} | Sum: {rec_sum:.2f} | "
                          f"Usr: {rates_str} | BER: {rec_ber:.1e} | "
                          f"Grad: {rec_grad:.2e} | Loss: {rec_loss:.2f}", end='\r')

            # End of Epoch Summary
            avg_loss = np.mean(metrics['loss'])
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            self.history.append({'epoch': epoch, 'sum_rate': avg_sum, 'ber': avg_ber})
            
            # ✅ Save best based on SUM RATE (primary metric)
            if avg_sum > self.best_sum_rate:  # BER constraint
                self.best_sum_rate = avg_sum
                self.best_epoch = epoch
                self.checkpoint.save_best(
                    self.system.precoder, 
                    {'epoch': epoch, 'snr_db': self.snr_db},
                    {'sum_rate': avg_sum, 'ber': avg_ber}
                )
                marker = " ⭐"
                wait = 0
            else:
                marker = ""
                if not is_warmup:
                    wait += 1
            
            # Print epoch summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            print(f"\r{epoch+1:6d} | {mode_str:>10} | {avg_sum:10.2f} | {rate_str:>40} | "
                  f"{avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}      ")
            
            if wait >= patience:
                print(f"\n🛑 Early Stopping: No improvement for {patience} epochs.")
                break

        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}")
        print(f"   Target: Beat RZF (35.83) on rate, beat WMMSE (3e-4) on BER\n")

class UnsupervisedTrainer:
    """
    Unsupervised Trainer:
    Directly optimizes the Neural Network to maximize Sum Rate (Shannon Capacity).
    No Teacher, No Labels. PURE Gradient Descent on the Channel Physics.
    """
    
    def __init__(self, system, cached_dataset,
                 snr_db=20.0,
                 total_epochs=100,
                 learning_rate=5e-4,
                 batch_size=64,
                 num_iters=None,
                 rate_weight=1.0, # Main driver
                 bce_weight=0.01): # Small regularization for BER stability
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        
        # Defensive Integer Conversion
        self.total_epochs = int(total_epochs) if total_epochs is not None else 100
        self.num_iters = int(num_iters) if num_iters is not None else None
        
        self.batch_size = batch_size
        self.rate_weight = rate_weight
        self.bce_weight = bce_weight
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        # Trigger model build
        dummy_in = (
            tf.zeros([1, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64),
            tf.zeros([1, 1, system.num_users, 1, system.num_bs_antennas, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
        )
        try:
            if not system.precoder.built:
                system.precoder(dummy_in)
        except:
            pass

        self.trainable_vars = system.precoder.trainable_variables
        # Assuming SimpleCheckpoint is defined in your utils
        self.checkpoint = SimpleCheckpoint('./training_weight_unsupervised') 
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  UNSUPERVISED TRAINING (MAX SUM-RATE)")
        print(f"{'='*80}")
        print(f"  Total Epochs: {self.total_epochs}")
        print(f"  Batch Size:   {self.batch_size}")
        print(f"  LR:           {learning_rate}")
        print(f"  Loss:         Maximizing SumRate (Weight: {rate_weight})")
        print(f"{'='*80}\n")
        # Re-capture the variables
        self.trainable_vars = system.precoder.trainable_variables
        print(f"✅ Variables found: {len(self.trainable_vars)}")

    @tf.function
    def train_step(self, h_freq_batch, batch_size, snr_db):
        """
        The Core Loop:
        1. Precoder predicts G
        2. Signal goes through Channel
        3. Receiver calculates SINR
        4. Loss = -Sum(log2(1+SINR))
        """
        with tf.GradientTape() as tape:
            # --- 1. Forward Pass (Training=True) ---
            # This must return the effective channel 'h_eff' or 'sinr' to allow gradient flow
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=True)
            
            # --- 2. Differentiable Rate Calculation ---
            # Add epsilon to noise to prevent singular matrices
            # interference_whitening=True is standard for MU-MIMO
            sinr = self.system.lmmse_sinr(h_eff, no=no + 1e-9, interference_whitening=True)
            
            # Clip SINR to prevent exploding gradients in high-SNR regimes
            sinr_clipped = tf.clip_by_value(sinr, 1e-9, 5e4) 
            
            # Shannon Rate: log2(1 + SINR)
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            
            # Average over Streams/Antennas/OFDM to get Per-User Rate
            # Adjust axes based on your exact shape. Usually [Batch, Rx, Streams, OFDM]
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            
            # --- 3. BCE Loss (Optional Stabilizer) ---
            # Keeps the constellation points somewhat separated for BER
            c_float = tf.cast(c, tf.float32)
            llr_clipped = tf.clip_by_value(llr, -20.0, 20.0) 
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
            bce_loss = tf.reduce_mean(bce)
            
            # --- 4. Total Loss ---
            # We want to MAXIMIZE rate, so we MINIMIZE negative rate
            loss = (self.bce_weight * bce_loss) - (self.rate_weight * sum_rate)
            
        # --- Gradient Handling ---
        grads = tape.gradient(loss, self.trainable_vars)
        
        # NaN Guard & Clipping
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 1.0) # Tight clipping for unsupervised stability
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        # Logging safety
        loss_safe = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        
        return sum_rate, bce_loss, compute_ber(b, b_hat), grad_norm, loss_safe, rate_per_user

    def train(self, log_interval=1, patience=10, print_every=10):
        # Auto-configure iterations
        if self.num_iters is None:
             if hasattr(self.cached_dataset, 'shape'):
                 self.num_iters = self.cached_dataset.shape[0] // self.batch_size
             else:
                 self.num_iters = DATASET_SIZE // BATCH_SIZE # Fallback
            
             print(f"  -> Auto-configured: {self.num_iters} iterations/epoch")

        print(f"🚀 Training @ {self.snr_db} dB")
        print(f"{'='*155}")
        print(f"{'Epoch':>6} | {'Mode':>10} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*155}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'bce': [], 'ber': [], 'loss': [], 'rates': [], 'grad': []}
            mode_str = "UNSUPERV"
            
            for i in range(self.num_iters):
                # 1. Get Batch
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                # 2. Train Step (Only one mode now)
                rate, bce, ber, g_norm, loss, rates = self.train_step(h_freq_batch, bs_tensor, snr_tensor)
                
                # 3. Store Metrics
                metrics['sum_rate'].append(float(rate))
                metrics['bce'].append(float(bce))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())

                # --- INTRA-EPOCH PRINTING ---
                if (i + 1) % print_every == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-print_every:])
                    rec_loss = np.mean(metrics['loss'][-print_every:])
                    rec_ber = np.mean(metrics['ber'][-print_every:])
                    rec_grad = np.mean(metrics['grad'][-print_every:])
                    rec_rates = np.mean(metrics['rates'][-print_every:], axis=0)
                    
                    rates_str = "[" + " ".join([f"{r:.1f}" for r in rec_rates]) + "]"
                    print(f"    ... Step {i+1}/{self.num_iters} | Sum: {rec_sum:.2f} | Usr: {rates_str} | BER: {rec_ber:.1e} | Grad: {rec_grad:.2e} | Loss: {rec_loss:.2f}", end='\r')

            # --- END OF EPOCH ---
            avg_loss = np.mean(metrics['loss'])
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            self.history.append({'epoch': epoch, 'sum_rate': avg_sum})
            
            # Save Best
            if avg_sum > self.best_sum_rate:
                self.best_sum_rate = avg_sum
                self.best_epoch = epoch
                self.checkpoint.save_best(self.system.precoder, {}, {'sum_rate': avg_sum})
                marker = " ⭐"
                wait = 0
            else:
                marker = ""
                wait += 1
            
            # Print Summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            print(f"\r{epoch+1:6d} | {mode_str:>10} | {avg_sum:10.2f} | {rate_str:>40} | {avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}      ")
            
            if wait >= patience:
                print(f"\n🛑 Early Stopping triggered! No improvement for {patience} epochs.")
                break

        print(f"\n✅ Best Rate: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")
# =============================================================================
# SIMPLIFIED MAIN
# =============================================================================


# EVALUATION
# =============================================================================
def evaluate_system(system, ebno_range, num_batches=100, batch_size=512, name="System"):
    print(f"\n📊 Evaluating {name}:")
    results = {'sum_rate': [], 'ber': [], 'sinr': []}
    
    for ebno_db in ebno_range:
        rates, bers, sinrs = [], [], []
        
        for _ in range(num_batches):
            system.new_topology(batch_size)
            
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
                tf.constant(batch_size, dtype=tf.int32),
                tf.constant(ebno_db, dtype=tf.float32),
                training=False
            )
            
            ber = compute_ber(b, b_hat)
            bers.append(float(ber))
            
            sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr_mean = tf.reduce_mean(sinr)
            sinrs.append(float(sinr_mean))
            
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])
            sum_rate = tf.reduce_sum(tf.reduce_mean(rate_per_user, axis=0))
            rates.append(float(sum_rate))
        
        avg_rate = np.mean(rates)
        avg_ber = np.mean(bers)
        avg_sinr_db = 10 * np.log10(np.mean(sinrs) + 1e-12)
        
        print(f"  SNR={ebno_db:2.0f}dB: Rate={avg_rate:5.2f} bps/Hz, BER={avg_ber:.2e}, SINR={avg_sinr_db:5.2f}dB")
        
        results['sum_rate'].append(avg_rate)
        results['ber'].append(avg_ber)
        results['sinr'].append(avg_sinr_db)
    
    return {k: np.array(v) for k, v in results.items()}


def plot_comparison(results_all, ebno_range, save_path='comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'RZF': 'blue', 'WMMSE': 'green', 'Transformer': 'red'}
    markers = {'RZF': 'o', 'WMMSE': 's', 'Transformer': '^'}
    
    for name, res in results_all.items():
        color = colors.get(name, 'black')
        marker = markers.get(name, 'x')
        
        axes[0].plot(ebno_range, res['sum_rate'], label=name, 
                    linewidth=2.5, marker=marker, markersize=8, color=color)
        axes[1].semilogy(ebno_range, res['ber'], label=name, 
                        linewidth=2.5, marker=marker, markersize=8, color=color)
        axes[2].plot(ebno_range, res['sinr'], label=name, 
                    linewidth=2.5, marker=marker, markersize=8, color=color)
    
    axes[0].set(xlabel='SNR (dB)', ylabel='Sum Rate (bps/Hz)', title='Spectral Efficiency')
    axes[1].set(xlabel='SNR (dB)', ylabel='BER', title='Bit Error Rate')
    axes[2].set(xlabel='SNR (dB)', ylabel='SINR (dB)', title='Post-Equalization SINR')
    
    for ax in axes:
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {save_path}\n")

def debug_shapes(system):
    print("\n🕵️‍♂️ RUNNING DIMENSION DIAGNOSTIC...")
    
    # 1. Create a dummy batch
    batch_size = 2
    system.new_topology(batch_size)
    
    # Generate a dummy channel
    cir = system.channel_model(batch_size, system.rg.num_ofdm_symbols, 1.0/system.rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
    
    # (2, 4, 1, 1, 8, 14, 72)
    print(f"1. Raw h_freq Input Shape: {h_freq.shape}")
    
    # 2. FIX: Squeeze BOTH singleton dimensions (RxAnt=1 and NumBS=1)
    # We remove axes 2 and 3
    h_sq = tf.squeeze(h_freq, axis=[2, 3]) 
    # New Shape: (2, 4, 8, 14, 72) -> [Batch, Rx, Tx, OFDM, FFT]
    print(f"2. Squeezed Shape:        {h_sq.shape} (Target: [B, Rx, Tx, OFDM, FFT])")
    
    # 3. Transpose to Transformer Format
    # Current Indices: 0:Batch, 1:Rx, 2:Tx, 3:OFDM, 4:FFT
    # Target: [Batch, OFDM, FFT, Rx, Tx] -> Indices [0, 3, 4, 1, 2]
    h_perm = tf.transpose(h_sq, perm=[0, 3, 4, 1, 2])
    print(f"3. Corrected Permutation: {h_perm.shape}")
    
    # 4. Check Feature Flattening
    B, S, F, Rx, Tx = h_perm.shape
    num_rb = F // 12
    rb_size = 12
    
    # [B, S, NumRB, RB_Size, Rx, Tx]
    h_rb = tf.reshape(h_perm, [B, S, num_rb, rb_size, Rx, Tx])
    print(f"4. RB Reshape:            {h_rb.shape}")
    
    print("✅ Diagnostic Complete.\n")

def test_precoder_integration(system):
    print(f"\n{'='*60}")
    print(f"🔬 RUNNING PRECODER DIAGNOSTIC TEST")
    print(f"{'='*60}")
    
    # 1. Setup Dummy Data
    batch_size = 2
    num_tx = system.num_bs_antennas
    num_rx = system.num_users
    num_ofdm = system.rg.num_ofdm_symbols
    fft_size = system.rg.fft_size
    
    print("1️⃣  Generating Dummy Inputs...")
    # Create dummy channel h_freq [Batch, Num_Rx, Rx_Ant, Tx_Ant, OFDM, FFT]
    # Note: Sionna's raw channel output is usually this shape
    h_freq_dummy = tf.ones([batch_size, num_rx, 1, 1, num_tx, num_ofdm, fft_size], dtype=tf.complex64)
    
    # Create dummy data x_rg [Batch, 1, Num_Rx, OFDM, FFT]
    x_rg_dummy = tf.ones([batch_size, 1, num_rx, num_ofdm, fft_size], dtype=tf.complex64)
    
    print(f"    h_freq shape: {h_freq_dummy.shape}")
    print(f"    x_rg shape:   {x_rg_dummy.shape}")

    # 2. Run Precoder Forward Pass
    print("\n2️⃣  Running Precoder Forward Pass...")
    try:
        # We pass the tuple as expected by your call method
        g = system.precoder((x_rg_dummy, h_freq_dummy))
        print(f"    ✅ Forward pass successful!")
        print(f"    Output 'g' shape: {g.shape}")
        
        # EXPECTED: [Batch, 1, OFDM, FFT, Tx, Rx]
        # [2, 1, 14, 72, 8, 4]
        expected_shape = (batch_size, 1, num_ofdm, fft_size, num_tx, num_rx)
        if g.shape == expected_shape:
            print(f"    ✅ Shape MATCHES expected Siona/Matmul format.")
        else:
            print(f"    ❌ Shape MISMATCH! Expected {expected_shape}, got {g.shape}")
            return False

    except Exception as e:
        print(f"    ❌ Forward pass FAILED: {e}")
        return False

    # 3. Test Sionna API Compatibility (Effective Channel)
    print("\n3️⃣  Testing Sionna 'compute_effective_channel'...")
    try:
        # This function internally performs h @ g
        # It verifies if the precoder output aligns with channel dimensions
        h_eff = system.precoder.compute_effective_channel(h_freq_dummy, g)
        print(f"    ✅ compute_effective_channel successful!")
        print(f"    Effective channel shape: {h_eff.shape}")
        # Expected: [Batch, 1, OFDM, FFT, Rx, Rx] (Interference matrix)
    except Exception as e:
        print(f"    ❌ Sionna compatibility check FAILED: {e}")
        print("    Hint: Your 'g' output dimensions likely don't allow matrix multiplication with 'h'.")
        return False

    # 4. Test Manual Matrix Multiplication (Your System Call Logic)
    print("\n4️⃣  Testing System Matrix Multiplication (x_precoded)...")
    try:
        # This replicates the logic inside MU_MIMO_System.call
        W = tf.squeeze(g, axis=1) # [B, S, F, Tx, Rx]
        
        # Prepare x_vec: [B, S, F, Rx, 1]
        x_vec = tf.transpose(x_rg_dummy, perm=[0, 3, 4, 2, 1])
        
        # Perform Matmul: [B, S, F, Tx, Rx] @ [B, S, F, Rx, 1] -> [B, S, F, Tx, 1]
        x_precoded_struct = tf.matmul(W, x_vec)
        
        print(f"    ✅ Matrix Multiplication successful!")
        print(f"    Tx Signal Shape: {x_precoded_struct.shape}")
    except Exception as e:
        print(f"    ❌ Manual Matmul check FAILED: {e}")
        return False
        
    print(f"\n🎉 ALL TESTS PASSED. DIMENSIONS ARE CORRECT.")
    return True

def evaluate_system_detailed(system, ebno_range, num_batches=100, batch_size=512, name="System"):
    print(f"\n📊 Evaluating {name}:")
    results = {'sum_rate': [], 'ber': [], 'sinr': [], 'sinr_per_user': [], 'rate_per_user': []}
    
    for ebno_db in ebno_range:
        rates, bers, sinrs = [], [], []
        all_sinr_per_user = []
        all_rate_per_user = []
        
        for _ in range(num_batches):
            system.new_topology(batch_size)
            
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
                tf.constant(batch_size, dtype=tf.int32),
                tf.constant(ebno_db, dtype=tf.float32),
                training=False
            )
            
            ber = compute_ber(b, b_hat)
            bers.append(float(ber))
            
            sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr_mean = tf.reduce_mean(sinr)
            sinrs.append(float(sinr_mean))
            
            # ✅ Per-user metrics
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])  # [Batch, Users]
            
            # Average across batch for each user
            rate_per_user_avg = tf.reduce_mean(rate_per_user, axis=0)  # [Users]
            all_rate_per_user.append(rate_per_user_avg.numpy())
            
            # Per-user SINR
            sinr_per_user = tf.reduce_mean(sinr, axis=[1, 2, 4])  # [Batch, Users]
            sinr_per_user_avg = tf.reduce_mean(sinr_per_user, axis=0)  # [Users]
            all_sinr_per_user.append(sinr_per_user_avg.numpy())
            
            sum_rate = tf.reduce_sum(rate_per_user_avg)
            rates.append(float(sum_rate))
        
        avg_rate = np.mean(rates)
        avg_ber = np.mean(bers)
        avg_sinr_db = 10 * np.log10(np.mean(sinrs) + 1e-12)
        
        # ✅ Per-user statistics
        rate_per_user_all = np.mean(all_rate_per_user, axis=0)
        sinr_per_user_all = np.mean(all_sinr_per_user, axis=0)
        sinr_per_user_db = 10 * np.log10(sinr_per_user_all + 1e-12)
        
        # Calculate fairness metrics
        rate_std = np.std(rate_per_user_all)
        rate_min = np.min(rate_per_user_all)
        rate_max = np.max(rate_per_user_all)
        
        sinr_std_db = np.std(sinr_per_user_db)
        sinr_min_db = np.min(sinr_per_user_db)
        sinr_max_db = np.max(sinr_per_user_db)
        
        print(f"\n  SNR={ebno_db:2.0f}dB:")
        print(f"    Sum Rate: {avg_rate:5.2f} bps/Hz, BER: {avg_ber:.2e}, Avg SINR: {avg_sinr_db:5.2f}dB")
        print(f"    Per-User Rates: {rate_per_user_all}")
        print(f"      └─ Min: {rate_min:.2f}, Max: {rate_max:.2f}, Std: {rate_std:.2f}")
        print(f"    Per-User SINR (dB): {sinr_per_user_db}")
        print(f"      └─ Min: {sinr_min_db:.2f}, Max: {sinr_max_db:.2f}, Std: {sinr_std_db:.2f}")
        
        results['sum_rate'].append(avg_rate)
        results['ber'].append(avg_ber)
        results['sinr'].append(avg_sinr_db)
        results['sinr_per_user'].append(sinr_per_user_db)
        results['rate_per_user'].append(rate_per_user_all)
    
    return results

# =============================================================================
# MAIN WITH 1M CACHED DATASET
# =============================================================================
def main():
    NUM_TX=4
    NUM_RX=4

    print("="*80)
    print("TRAINING WITH 50k CACHED CHANNELS")

    print("="*80 + "\n")
    # Create system
    system = MU_MIMO_System(
        num_tx=NUM_TX, num_rx=NUM_RX,
        precoder_type="transformer",
        rb_size=6
    )
    
    # ✅ Load existing cached dataset (no regeneration!)
    print("📂 Loading cached dataset...")
    cached_dataset = CachedSionnaDataset(system,dataset_size=DATASET_SIZE, batch_size= BATCH_SIZE*4,
    cache_file=f'/export/tmp/sala/cached_sionna_{DATASET_SIZE//1000}k_4x4.npy'  # ✅ .npy!
    )

    
    # ✅ Train with BCE + Sum Rate
    print ("CachedTrainer")
    trainer = CachedTrainer(
        system,
        cached_dataset,
        snr_db=20.0,
        total_epochs=100,
        learning_rate=1e-3,
        batch_size=BATCH_SIZE,
        num_iters=None,
        bce_weight=0,   # ✅ Balance decodability
        rate_weight=1   # ✅ and efficiency
    )
    debug_shapes(system)
    test_precoder_integration(system)

    trainer.train(log_interval=1)
    
    # Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    ebno_range = np.arange(0, 26, 5)

    system.new_topology(BATCH_SIZE, seed=999)
    results_transformer = evaluate_system(system, ebno_range, 50, BATCH_SIZE, "Transformer")
    sys_wmmse = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type="wmmse")
    sys_wmmse.new_topology(BATCH_SIZE, seed=999)
    _ = sys_wmmse(tf.constant(BATCH_SIZE), tf.constant(20.0))
    results_wmmse = evaluate_system(sys_wmmse, ebno_range, 50, BATCH_SIZE, "WMMSE")

    sys_rzf = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type="rzf")
    sys_rzf.new_topology(BATCH_SIZE, seed=999)
    _ = sys_rzf(tf.constant(BATCH_SIZE), tf.constant(20.0))
    results_rzf = evaluate_system(sys_rzf, ebno_range, 50, BATCH_SIZE, "RZF")
    
    
    # Plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_comparison(
        {
            'RZF': results_rzf,
            'WMMSE': results_wmmse,
            'Transformer': results_transformer
        },
        ebno_range,
        save_path=f'comparison_{timestamp}.png'
    )
    
    # Summary
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)
    print(f"\n🎯 Performance @ 20 dB:")
    idx_20 = np.where(ebno_range == 20)[0][0]
    print(f"   RZF:         {results_rzf['sum_rate'][idx_20]:.2f} bps/Hz, BER={results_rzf['ber'][idx_20]:.2e}")
    print(f"   WMMSE:       {results_wmmse['sum_rate'][idx_20]:.2f} bps/Hz, BER={results_wmmse['ber'][idx_20]:.2e}")
    print(f"   Transformer: {results_transformer['sum_rate'][idx_20]:.2f} bps/Hz, BER={results_transformer['ber'][idx_20]:.2e}")
    
    if results_transformer['sum_rate'][idx_20] > results_wmmse['sum_rate'][idx_20]:
        gain = results_transformer['sum_rate'][idx_20] - results_wmmse['sum_rate'][idx_20]
        print(f"\n   ✅ GAIN over WMMSE: +{gain:.2f} bps/Hz ({gain/results_wmmse['sum_rate'][idx_20]*100:.1f}%)")
    else:
        gap = results_wmmse['sum_rate'][idx_20] - results_transformer['sum_rate'][idx_20]
        print(f"\n   ⚠️  Gap to WMMSE: -{gap:.2f} bps/Hz")
    
    print()


if __name__ == "__main__":
    main()
