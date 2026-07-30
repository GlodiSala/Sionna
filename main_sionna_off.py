import os
import json
import pickle
import logging
from datetime import datetime
import scipy.io as sio

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
from tensorflow.keras import layers


class SwiGLU(layers.Layer):
    def call(self, x):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=-1)
        return (x_a * tf.nn.sigmoid(x_a)) * x_b

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
    @property
    def trainable_variables(self):
        """Force the collection of ALL nested variables from blocks."""
        v = []
        # 1. Collect from standard component layers
        v += self.rb_aggregation.trainable_variables
        v += self.input_embedding.trainable_variables
        v += self.feature_norm.trainable_variables
        
        # 2. Collect from the transformer block list
        # Ensure block.trainable_variables is returning its internal components!
        for block in self.blocks_list:
            v += block.trainable_variables
            
        # 3. Collect from output
        v += self.output_projection.trainable_variables
        return v
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
    """
    Final Synchronized Transformer Precoder with 5-Component Math Features.
    Break the plateau: Real, Imag, Abs, Cos(Phase), Sin(Phase)
    """
    def __init__(self, resource_grid, stream_management, 
                 num_tx=4, num_rx=4, rb_size=12, fft_size=72,
                 embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        
        super().__init__(resource_grid, stream_management, **kwargs)
        
        self.M = num_tx
        self.K = num_rx
        self.rb_size = rb_size
        self.fft_size = fft_size
        self.num_rb = fft_size // rb_size
        self.power_scale = tf.constant(np.sqrt(self.K), dtype=tf.complex64)

        # 1. Component Layers
        self.rb_aggregation = layers.Dense(1, use_bias=False)
        self.input_embedding = layers.Dense(embed_dim, kernel_initializer='glorot_uniform')
        self.feature_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # 2. Block List with Explicit Keras Registration (Crucial for parameter tracking)
        self.blocks_list = []
        for i in range(num_layers):
            block = FactorizedTransformerBlock(
                num_rx=self.K, num_rb=self.num_rb, 
                embed_dim=embed_dim, num_heads=num_heads
            )
            # Register as attribute for self.trainable_variables visibility
            setattr(self, f"trans_block_{i}", block)
            self.blocks_list.append(block)
        
        self.output_projection = layers.Dense(2 * self.M, kernel_initializer='glorot_uniform')

    @property
    def trainable_variables(self):
        """Manually expose variables to trainer to prevent 0-param error"""
        all_vars = []
        all_vars.extend(self.rb_aggregation.trainable_variables)
        all_vars.extend(self.input_embedding.trainable_variables)
        all_vars.extend(self.feature_norm.trainable_variables)
        for i in range(len(self.blocks_list)):
            all_vars.extend(getattr(self, f"trans_block_{i}").trainable_variables)
        all_vars.extend(self.output_projection.trainable_variables)
        return all_vars

    def call(self, inputs, training=False):
        _, h_freq = inputs 
        h_pc_desired = self.get_desired_channels(h_freq) # [B, 1, Time, Freq, K, M]
        h = tf.squeeze(h_pc_desired, axis=1) # [B, 14, 72, Rx, Tx]
        
        B = tf.shape(h)[0]
        # Reshape to group subcarriers inside RBs
        h_rb = tf.reshape(h, [B, 14, self.num_rb, self.rb_size, self.K, self.M])
        
        # --- DOMAIN KNOWLEDGE EXTRACTION (5 Components) ---
        h_re, h_im = tf.math.real(h_rb), tf.math.imag(h_rb)
        h_abs = tf.math.abs(h_rb)
        h_ang = tf.math.angle(h_rb)
        h_cos, h_sin = tf.math.cos(h_ang), tf.math.sin(h_ang)
        
        # Stack Rank-7 Features: [B, 14, RB, Size, Rx, Tx, 5]
        h_math = tf.stack([h_re, h_im, h_abs, h_cos, h_sin], axis=-1)
        
        # Aggregation over Subcarriers (Axis 3 moved to end)
        # Perm list maps current 3 to last index 6
        perm = [0, 1, 2, 4, 5, 6, 3] 
        h_math_agg = self.rb_aggregation(tf.transpose(h_math, perm=perm))
        h_math_agg = tf.squeeze(h_math_agg, axis=-1) # [B, 14, RB, Rx, Tx, 5]
        
        # Flatten for input_embedding: [Batch*14, RB, Rx, M*5]
        feat = tf.reshape(h_math_agg, [-1, self.num_rb, self.K, self.M * 5])
        x = self.feature_norm(self.input_embedding(feat))
        
        # --- ATTENTION BLOCKS ---
        for block in self.blocks_list:
            x = block(x, training=training)
        
        # --- OUTPUT MAPPING ---
        out = self.output_projection(x) # [Batch*14, RB, Rx, Tx*2]
        out = tf.reshape(out, [B, 14, self.num_rb, self.K, self.M, 2])
        
        w = tf.complex(out[..., 0], out[..., 1]) # precoder matrix components
        w = tf.transpose(w, perm=[0, 1, 2, 4, 3]) # [Batch, 14, RB, Tx, Rx]
        
        # Upsample RB to full Resource Grid
        w_up = tf.tile(tf.expand_dims(w, axis=3), [1, 1, 1, self.rb_size, 1, 1])
        w_full = tf.reshape(w_up, [B, 14, self.fft_size, self.M, self.K])
        
        # Power Normalization: Keep total frame energy consistent
        power_total = tf.reduce_sum(tf.abs(w_full)**2, axis=[3, 4], keepdims=True)
        power_sqrt = tf.cast(tf.math.sqrt(power_total + 1e-12), tf.complex64)
        
        w_norm = (w_full / power_sqrt) * self.power_scale
        return tf.expand_dims(w_norm, axis=1) # [B, 1, 14, 72, Tx, Rx]class FactorizedTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_rx, num_rb, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_rx = num_rx
        self.num_rb = num_rb
        self.embed_dim = embed_dim
        
        # Pre-Norm components
        self.norm_freq = layers.LayerNormalization(epsilon=1e-6)
        self.freq_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        
        self.norm_space = layers.LayerNormalization(epsilon=1e-6)
        self.space_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )

        self.norm_ffn = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4), # Standard 4x expansion for Transformers
            SwiGLU(),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        
        # 1. Frequency Attention (Attention across RBs)
        # Input x: [B*14, RB, Rx, Embed]
        x_freq = tf.transpose(x, perm=[0, 2, 1, 3]) # [B*14, Rx, RB, Embed]
        x_freq = tf.reshape(x_freq, [-1, self.num_rb, self.embed_dim])
        
        x_norm = self.norm_freq(x_freq)
        att = self.freq_att(x_norm, x_norm, training=training)
        x_freq = x_freq + att
        
        x_res = tf.reshape(x_freq, [-1, self.num_rx, self.num_rb, self.embed_dim])
        x = tf.transpose(x_res, perm=[0, 2, 1, 3]) # Back to [B*14, RB, Rx, Embed]

        # 2. Spatial Attention (Attention across Users)
        x_space = tf.reshape(x, [-1, self.num_rx, self.embed_dim])
        
        x_norm = self.norm_space(x_space)
        att = self.space_att(x_norm, x_norm, training=training)
        x_space = x_space + att
        
        x = tf.reshape(x_space, [-1, self.num_rb, self.num_rx, self.embed_dim])

        # 3. FFN
        x_flat = tf.reshape(x, [-1, self.embed_dim])
        x_norm = self.norm_ffn(x_flat)
        ffn_out = self.ffn(x_norm, training=training)
        x_flat = x_flat + ffn_out
        
        x = tf.reshape(x_flat, [-1, self.num_rb, self.num_rx, self.embed_dim])
        
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

    @property
    def trainable_variables(self):
        v = []
        v += self.rb_aggregation.trainable_variables
        v += self.input_embedding.trainable_variables
        v += self.feature_norm.trainable_variables
        for i in range(len(self.blocks)):
            v += getattr(self, f"block_{i}").trainable_variables
        v += self.output_projection.trainable_variables
        return v

    def call(self, inputs, training=False):
        _, h_freq = inputs 
        B = tf.shape(h_freq)[0]
        h_sq = tf.squeeze(h_freq, axis=[2, 3]) 
        
        h_rb = tf.reshape(h_sq, [B, self.K, self.M, 14, self.num_rb, self.rb_size])
        h_rb = tf.transpose(h_rb, [0, 3, 4, 5, 1, 2])
        
        # Domain Knowledge Stack
        h_ang = tf.math.angle(h_rb)
        h_math = tf.stack([
            tf.math.real(h_rb), tf.math.imag(h_rb), tf.math.abs(h_rb),
            tf.math.cos(h_ang), tf.math.sin(h_ang)
        ], axis=-1)
        
        # Aggregate Size dim
        h_math_agg = self.rb_aggregation(tf.transpose(h_math, [0, 1, 2, 4, 5, 6, 3]))
        h_math_agg = tf.squeeze(h_math_agg, axis=-1)
        
        feat = tf.reshape(h_math_agg, [-1, self.num_rb, self.K, self.M * 5])
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
        power_tot = tf.reduce_sum(tf.abs(w_full)**2, axis=[3,4], keepdims=True)
        w_normalized = (w_full / tf.cast(tf.math.sqrt(power_tot + 1e-12), tf.complex64)) * self.power_scale
        return tf.expand_dims(w_normalized, axis=1)# DATASET (identique à ton code)
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
        
        return g_out#
# =============================================================================
# DATASET CLASS FOR .mat FILE
# =============================================================================
class CSIDatasetFromMat:
    """
    Loads CSI data from .mat file and provides batching interface.
    Expected H format: (B, K, F, M) where:
        B = batch/samples
        K = num_rx (users)
        F = num_subcarriers
        M = num_tx (antennas)
    """
    def __init__(self, path, batch_size=64):
        self.path = path
        self.batch_size = batch_size
        
        print(f"\n{'='*80}")
        print(f"  LOADING CUSTOM CSI DATASET")
        print(f"{'='*80}")
        print(f"  Path: {path}")
        
        # Load .mat file
        mat_data = sio.loadmat(path)
        
        # ✅ Adjust key name based on your .mat file structure
        # Common keys: 'H', 'CSI', 'channel', 'h_freq', etc.
        if 'H' in mat_data:
            self.h_freq_all = mat_data['H']
        elif 'CSI' in mat_data:
            self.h_freq_all = mat_data['CSI']
        elif 'channel' in mat_data:
            self.h_freq_all = mat_data['channel']
        else:
            # Show available keys to help debug
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            raise KeyError(f"Could not find CSI data. Available keys: {available_keys}")
        
        # Ensure complex type
        if self.h_freq_all.dtype != np.complex64:
            self.h_freq_all = self.h_freq_all.astype(np.complex64)
        
        self.Ndata = self.h_freq_all.shape[0]
        
        print(f"  Shape: {self.h_freq_all.shape}")
        print(f"  Expected: (Samples, K={self.h_freq_all.shape[1]}, "
              f"F={self.h_freq_all.shape[2]}, M={self.h_freq_all.shape[3]})")
        print(f"  Dtype: {self.h_freq_all.dtype}")
        print(f"  Total samples: {self.Ndata:,}")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*80}\n")
        
        # Normalize channels
        self._normalize_channels()
    
    def _normalize_channels(self):
        """Normalize to unit average power"""
        power = np.mean(np.abs(self.h_freq_all)**2)
        print(f"🔧 Normalizing channels:")
        print(f"   Power before: {power:.6f}")
        
        # Normalize
        norm_factor = np.sqrt(power)
        self.h_freq_all = self.h_freq_all / norm_factor
        
        power_after = np.mean(np.abs(self.h_freq_all)**2)
        print(f"   Power after:  {power_after:.6f} (target: 1.0)")
        
        if not (0.95 < power_after < 1.05):
            print(f"   ⚠️  WARNING: Normalization may have failed!")
    
    def get_tf_dataset(self, shuffle=True, batches_per_epoch=100):
        """
        Create a tf.data.Dataset for training/validation.
        
        Args:
            shuffle: Whether to shuffle samples
            batches_per_epoch: Number of batches to yield per epoch
        
        Returns:
            tf.data.Dataset yielding batches of shape (batch_size, K, F, M)
        """
        # Create index dataset
        if shuffle:
            indices = np.random.permutation(self.Ndata)
        else:
            indices = np.arange(self.Ndata)
        
        # Take only enough indices for desired batches_per_epoch
        total_samples = batches_per_epoch * self.batch_size
        indices = indices[:total_samples]
        
        def generator():
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:  # Only full batches
                    yield self.h_freq_all[batch_indices]
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(
                shape=(self.batch_size, *self.h_freq_all.shape[1:]),
                dtype=tf.complex64
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10)
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    def get_batch(self, batch_size=None):
        """Get a random batch (for simple trainer interface)"""
        if batch_size is None:
            batch_size = self.batch_size
        
        indices = np.random.choice(self.Ndata, size=batch_size, replace=True)
        return tf.constant(self.h_freq_all[indices], dtype=tf.complex64)


# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================
def preprocessing_h(H, T):
    """
    Convert dataset format to Sionna's expected format.
    
    Inputs:
        H: (B, K, F, M) - from dataset
        T: int - num_ofdm_symbols (typically 14)
    
    Outputs:
        H: (B, K, 1, 1, M, T, F) - for Sionna precoders
    
    Where:
        B = batch size
        K = num_rx (users)
        F = num_subcarriers (fft_size)
        M = num_tx (antennas)
        T = num_ofdm_symbols
    """
    # Current: (B, K, F, M)
    H = tf.expand_dims(H, axis=2)   # (B, K, 1, F, M)
    H = tf.expand_dims(H, axis=3)   # (B, K, 1, 1, F, M)
    
    # Transpose to move M before F
    H = tf.transpose(H, [0, 1, 2, 3, 5, 4])   # (B, K, 1, 1, M, F)
    
    # Add time dimension and repeat
    H = tf.expand_dims(H, axis=-2)            # (B, K, 1, 1, M, 1, F)
    H = tf.repeat(H, T, axis=-2)              # (B, K, 1, 1, M, T, F)
    
    return H

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




# =============================================================================
# MODIFIED TRAINERS TO USE CUSTOM DATASET
# =============================================================================

class UnsupervisedTrainer:
    """
    Unsupervised Trainer for Custom Dataset.
    Maximizes Sum Rate directly from CSI without labels.
    """
    
    def __init__(self, system, dataset,
                 snr_db=20.0,
                 total_epochs=100,
                 learning_rate=5e-4,
                 batch_size=64,
                 batches_per_epoch=100,
                 rate_weight=1.0,
                 bce_weight=0.01):
        
        self.system = system
        self.dataset = dataset
        self.snr_db = snr_db
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.rate_weight = rate_weight
        self.bce_weight = bce_weight
        
        # Optimizer
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * batches_per_epoch,
            alpha=0.1
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
        
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
        self.checkpoint = SimpleCheckpoint('./training_weight_unsupervised')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  UNSUPERVISED TRAINING (MAX SUM-RATE)")
        print(f"{'='*80}")
        print(f"  Total Epochs:      {self.total_epochs}")
        print(f"  Batches/Epoch:     {self.batches_per_epoch}")
        print(f"  Batch Size:        {self.batch_size}")
        print(f"  LR:                {learning_rate} → {learning_rate * 0.1}")
        print(f"  Dataset Samples:   {dataset.Ndata:,}")
        print(f"{'='*80}\n")

    @tf.function
    def train_step(self, h_batch, batch_size, snr_db):
        """Training step with custom dataset"""
        
        # ✅ Preprocess H from (B, K, F, M) to Sionna format
        h_freq = preprocessing_h(h_batch, self.system.rg.num_ofdm_symbols)
        
        with tf.GradientTape() as tape:
            # Forward pass
            b, b_hat, c, llr, h_eff, no, _, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq, training=True)
            
            # Sum Rate Loss
            sinr = self.system.lmmse_sinr(h_eff, no=no + 1e-9, interference_whitening=True)
            sinr_clipped = tf.clip_by_value(sinr, 1e-9, 5e4)
            
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            
            # BCE Loss (decodability regularization)
            c_float = tf.cast(c, tf.float32)
            llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
            bce_loss = tf.reduce_mean(bce)
            
            # Combined Loss
            loss = (self.bce_weight * bce_loss) - (self.rate_weight * sum_rate)
        
        # Gradient handling
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        loss_safe = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        
        return sum_rate, bce_loss, compute_ber(b, b_hat), grad_norm, loss_safe, rate_per_user

    def train(self, log_interval=1, patience=10):
        """Training loop using tf.data.Dataset"""
        
        # Create training dataset
        train_dataset = self.dataset.get_tf_dataset(
            shuffle=True,
            batches_per_epoch=self.batches_per_epoch
        )
        
        print(f"🚀 Training @ {self.snr_db} dB (Unsupervised)")
        print(f"{'='*155}")
        print(f"{'Epoch':>6} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*155}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'bce': [], 'ber': [], 'loss': [], 'rates': [], 'grad': []}
            
            for i, h_batch in enumerate(train_dataset):
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                # Train step
                rate, bce, ber, g_norm, loss, rates = self.train_step(
                    h_batch, bs_tensor, snr_tensor
                )
                
                # Store metrics
                metrics['sum_rate'].append(float(rate))
                metrics['bce'].append(float(bce))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-10:])
                    print(f"    ... Batch {i+1}/{self.batches_per_epoch} | Sum: {rec_sum:.2f}", end='\r')
            
            # Epoch summary
            avg_loss = np.mean(metrics['loss'])
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            self.history.append({'epoch': epoch, 'sum_rate': avg_sum, 'ber': avg_ber})
            
            # Save best
            if avg_sum > self.best_sum_rate:
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
                wait += 1
            
            # Print summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            print(f"\r{epoch+1:6d} | {avg_sum:10.2f} | {rate_str:>40} | "
                  f"{avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}      ")
            
            if wait >= patience:
                print(f"\n🛑 Early Stopping: No improvement for {patience} epochs.")
                break
        
        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")


class CachedTrainer:
    """
    Hybrid Trainer with WMMSE Warm-Start for Custom Dataset.
    Phase 1: Learn from WMMSE (supervised)
    Phase 2: Fine-tune with BCE + Sum Rate (unsupervised)
    """
    
    def __init__(self, system, dataset,
                 snr_db=20.0,
                 total_epochs=100,
                 warmup_epochs=5,
                 learning_rate=5e-4,
                 batch_size=64,
                 batches_per_epoch=100,
                 bce_weight=0.5,
                 rate_weight=1.0):
        
        self.system = system
        self.dataset = dataset
        self.snr_db = snr_db
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.bce_weight = bce_weight
        self.rate_weight = rate_weight
        
        # Create WMMSE teacher
        self.wmmse_teacher = WMMSEPrecodedChannel(
            resource_grid=system.rg,
            stream_management=system.sm,
            num_iterations=10
        )
        
        # Optimizer
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * batches_per_epoch,
            alpha=0.1
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=5.0)
        
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
        self.checkpoint = SimpleCheckpoint('./training_weight_hybrid')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  HYBRID TRAINING (WMMSE WARM-START + FINE-TUNING)")
        print(f"{'='*80}")
        print(f"  Total Epochs:      {self.total_epochs}")
        print(f"  Warm-up (WMMSE):   {self.warmup_epochs} epochs")
        print(f"  Fine-tune (Hybrid): {self.total_epochs - self.warmup_epochs} epochs")
        print(f"  Batch Size:        {self.batch_size}")
        print(f"  Batches/Epoch:     {self.batches_per_epoch}")
        print(f"  Initial LR:        {learning_rate} → {learning_rate * 0.1}")
        print(f"  BCE Weight:        {bce_weight}")
        print(f"  Rate Weight:       {rate_weight}")
        print(f"{'='*80}\n")

    @tf.function
    def train_step_supervised(self, h_batch, batch_size, snr_db):
        """Phase 1: Learn from WMMSE"""
        
        # Preprocess
        h_freq = preprocessing_h(h_batch, self.system.rg.num_ofdm_symbols)
        
        with tf.GradientTape() as tape:
            # Compute teacher
            no = ebnodb2no(snr_db, self.system.num_bits_per_symbol, 0.5, self.system.rg)
            dummy_y = tf.zeros([batch_size, 1, self.system.num_users, 14, 72], dtype=tf.complex64)
            g_teacher = self.wmmse_teacher((dummy_y, h_freq, no))
            
            # Compute student
            g_pred = self.system.precoder((dummy_y, h_freq), training=True)
            
            # MSE Loss
            mse_loss = tf.reduce_mean(tf.abs(g_pred - g_teacher)**2)
            loss = 100.0 * mse_loss
        
        # Apply gradients
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        # Logging metrics
        b, b_hat, c, llr, h_eff, no, _, _, _, _ = \
            self.system.call_with_cached_channel(batch_size, snr_db, h_freq, training=False)

        sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sinr_clipped = tf.clip_by_value(sinr, 1e-9, 1e4)
        
        rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
        rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
        sum_rate = tf.reduce_sum(rate_per_user)
        
        ber = compute_ber(b, b_hat)
        
        c_float = tf.cast(c, tf.float32)
        llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
        bce_loss = tf.reduce_mean(bce)

        return sum_rate, bce_loss, ber, grad_norm, loss, rate_per_user

    @tf.function
    def train_step_unsupervised(self, h_batch, batch_size, snr_db):
        """Phase 2: Fine-tune with BCE + Sum Rate"""
        
        # Preprocess
        h_freq = preprocessing_h(h_batch, self.system.rg.num_ofdm_symbols)
        
        with tf.GradientTape() as tape:
            # Forward pass
            b, b_hat, c, llr, h_eff, no, _, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq, training=True)
            
            # BCE Loss
            c_float = tf.cast(c, tf.float32)
            llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=c_float, logits=llr_clipped)
            bce_loss = tf.reduce_mean(bce)
            
            # Sum Rate
            sinr = self.system.lmmse_sinr(h_eff, no=no + 1e-6, interference_whitening=True)
            sinr_clipped = tf.clip_by_value(sinr, 1e-9, 1e4)
            
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            
            # Combined Loss
            loss = (self.bce_weight * bce_loss) - (self.rate_weight * sum_rate)
        
        # Gradient handling
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)
        
        loss_safe = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        
        return sum_rate, bce_loss, compute_ber(b, b_hat), grad_norm, loss_safe, rate_per_user
    
    def train(self, log_interval=1, patience=10):
        """Training loop"""
        
        # Create training dataset
        train_dataset = self.dataset.get_tf_dataset(
            shuffle=True,
            batches_per_epoch=self.batches_per_epoch
        )
        
        print(f"🚀 Training @ {self.snr_db} dB")
        print(f"{'='*155}")
        print(f"{'Epoch':>6} | {'Mode':>10} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*155}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'bce': [], 'ber': [], 'loss': [], 'rates': [], 'grad': []}
            is_warmup = epoch < self.warmup_epochs
            mode_str = "WMMSE" if is_warmup else "HYBRID"
            
            for i, h_batch in enumerate(train_dataset):
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                # Train step
                if is_warmup:
                    rate, bce, ber, g_norm, loss, rates = self.train_step_supervised(
                        h_batch, bs_tensor, snr_tensor
                    )
                else:
                    rate, bce, ber, g_norm, loss, rates = self.train_step_unsupervised(
                        h_batch, bs_tensor, snr_tensor
                    )
                
                # Store metrics
                metrics['sum_rate'].append(float(rate))
                metrics['bce'].append(float(bce))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())
                
                # Progress
                if (i + 1) % 10 == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-10:])
                    print(f"    ... Batch {i+1}/{self.batches_per_epoch} | Sum: {rec_sum:.2f}", end='\r')
            
            # Epoch summary
            avg_loss = np.mean(metrics['loss'])
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            self.history.append({'epoch': epoch, 'sum_rate': avg_sum, 'ber': avg_ber})
            
            # Save best
            if avg_sum > self.best_sum_rate:
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
            
            # Print summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            print(f"\r{epoch+1:6d} | {mode_str:>10} | {avg_sum:10.2f} | {rate_str:>40} | "
                  f"{avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}      ")
            
            if wait >= patience:
                print(f"\n🛑 Early Stopping: No improvement for {patience} epochs.")
                break

        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")


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


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Configuration
    CSI_GRID_PATH = "/export/tmp/sala/dataset_stecath_raytracing_allusers_64subcarriers.mat"
    NUM_TX = 4
    NUM_RX = 4
    BATCH_SIZE = 64
    NTRAIN = 6400  # 100 batches per epoch
    NVAL = 640     # 10 batches per epoch
    
    print("="*80)
    print("TRAINING WITH CUSTOM STECATH DATASET")
    print("="*80 + "\n")
    
    # ✅ Step 1: Load Custom Dataset
    dataset = CSIDatasetFromMat(
        path=CSI_GRID_PATH,
        batch_size=BATCH_SIZE
    )
    
    # ✅ Step 2: Create System
    system = MU_MIMO_System(
        num_tx=NUM_TX,
        num_rx=NUM_RX,
        precoder_type="transformer",
        rb_size=12
    )
    
    # ✅ Step 3: Train with Unsupervised Trainer (as requested)
    print("\n" + "="*80)
    print("UNSUPERVISED TRAINING")
    print("="*80)
    
    trainer = UnsupervisedTrainer(
        system,
        dataset,
        snr_db=20.0,
        total_epochs=100,
        learning_rate=1e-3,
        batch_size=BATCH_SIZE,
        batches_per_epoch=NTRAIN // BATCH_SIZE,  # 100 batches
        bce_weight=0.1,   # Small regularization
        rate_weight=1.0   # Main objective
    )
    
    trainer.train(log_interval=1, patience=10)
    
    # ✅ Step 4: Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    ebno_range = np.arange(0, 26, 5)
    
    # Evaluate Transformer
    system.new_topology(BATCH_SIZE, seed=999)
    results_transformer = evaluate_system(system, ebno_range, 50, BATCH_SIZE, "Transformer")
    
    # Evaluate WMMSE
    sys_wmmse = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type="wmmse")
    sys_wmmse.new_topology(BATCH_SIZE, seed=999)
    _ = sys_wmmse(tf.constant(BATCH_SIZE), tf.constant(20.0))
    results_wmmse = evaluate_system(sys_wmmse, ebno_range, 50, BATCH_SIZE, "WMMSE")
    
    # Evaluate RZF
    sys_rzf = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type="rzf")
    sys_rzf.new_topology(BATCH_SIZE, seed=999)
    _ = sys_rzf(tf.constant(BATCH_SIZE), tf.constant(20.0))
    results_rzf = evaluate_system(sys_rzf, ebno_range, 50, BATCH_SIZE, "RZF")
    
    # ✅ Step 5: Plot Results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_comparison(
        {
            'RZF': results_rzf,
            'WMMSE': results_wmmse,
            'Transformer': results_transformer
        },
        ebno_range,
        save_path=f'comparison_stecath_{timestamp}.png'
    )
    
    # ✅ Step 6: Summary
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