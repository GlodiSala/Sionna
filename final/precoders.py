
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers
from tensorflow.keras.layers import RMSNormalization

def rzf_precoder(h_freq, stream_management, alpha=0.1):
    """
    ✅ CORRECTED: RZF precoding following Sionna conventions
    
    Args:
        h_freq: [B, num_rx, num_rx_ant, num_tx, num_tx_ant, ofdm, fft]
        stream_management: StreamManagement object
        alpha: regularization parameter
    
    Returns:
        g: [B, num_tx, ofdm, fft, num_tx_ant, num_streams_per_tx]
    
    ⚠️ WARNING: This function computes ONLY the precoding matrix.
    To get the effective channel, use _compute_effective_channel from main.py
    """
    from sionna.phy.mimo import rzf_precoding_matrix
    
    # ✅ Extract desired channels (mimics Sionna's get_desired_channels)
    # Transpose: [num_tx, num_rx, num_rx_ant, num_tx_ant, ofdm, fft, B]
    h_pc = tf.transpose(h_freq, [3, 1, 2, 4, 5, 6, 0])
    
    # Gather desired RX for each TX
    h_pc = tf.gather(h_pc, stream_management.precoding_ind, 
                     axis=1, batch_dims=1)
    
    # Flatten RX dimensions: [num_tx, num_streams_per_tx, num_tx_ant, ofdm, fft, B]
    s = tf.shape(h_pc)
    num_streams = s[1] * s[2]
    h_pc = tf.reshape(h_pc, [s[0], num_streams, s[3], s[4], s[5], s[6]])
    
    # Transpose: [B, num_tx, ofdm, fft, num_streams_per_tx, num_tx_ant]
    h_pc_desired = tf.transpose(h_pc, [5, 0, 3, 4, 1, 2])
    
    # ✅ Apply RZF using Sionna's function
    g = rzf_precoding_matrix(h_pc_desired, alpha=alpha)
    # Output: [B, num_tx, ofdm, fft, num_tx_ant, num_streams_per_tx]
    
    # ✅ Normalize each column to unit power (Sionna standard)
    #norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    #g = tf.math.divide_no_nan(g, tf.cast(norm, g.dtype))
    
    return g


# =============================================================================
# ✅ CORRECTED: WMMSE Precoder (Function-based)
# =============================================================================

def wmmse_precoder(h_freq, no, stream_management, num_iterations=10):
    """
    ✅ CORRECTED: WMMSE precoding following Sionna conventions
    
    Args:
        h_freq: [B, num_rx, num_rx_ant, num_tx, num_tx_ant, ofdm, fft]
        no: noise variance (scalar or [B])
        stream_management: StreamManagement object
        num_iterations: number of WMMSE iterations
    
    Returns:
        g: [B, num_tx, ofdm, fft, num_tx_ant, num_streams_per_tx]
    
    Implementation follows the WMMSE algorithm from:
    Q. Shi et al., "An iteratively weighted MMSE approach to distributed
    sum-utility maximization for a MIMO interfering broadcast channel," 
    IEEE Trans. Signal Process., 2011.
    """
    
    # ✅ Extract desired channels (same logic as RZF)
    h_pc = tf.transpose(h_freq, [3, 1, 2, 4, 5, 6, 0])
    h_pc = tf.gather(h_pc, stream_management.precoding_ind, 
                     axis=1, batch_dims=1)
    s = tf.shape(h_pc)
    num_streams = s[1] * s[2]
    h_pc = tf.reshape(h_pc, [s[0], num_streams, s[3], s[4], s[5], s[6]])
    h_pc_desired = tf.transpose(h_pc, [5, 0, 3, 4, 1, 2])
    # Shape: [B, num_tx, ofdm, fft, num_streams_per_tx, num_tx_ant]
    
    # Get dimensions
    batch_size = s[6]
    num_tx = s[0]
    ofdm_syms = s[4]
    fft_size = s[5]
    K = num_streams  # num_streams_per_tx
    M = s[3]  # num_tx_ant
    
    # Squeeze num_tx dimension (=1 for single BS)
    h_pc_squeezed = tf.squeeze(h_pc_desired, axis=1)
    # Shape: [B, ofdm, fft, K, M]
    
    H = h_pc_squeezed
    
    # ✅ Initialize with RZF
    V = rzf_precoding_matrix(h_pc_desired, alpha=0.1)
    V = tf.squeeze(V, axis=1)  # [B, ofdm, fft, M, K]
    
    # Prepare noise
    no_val = tf.cast(no, H.dtype.real_dtype)
    if len(tf.shape(no_val)) == 0:  # scalar
        no_val = tf.reshape(no_val, [1, 1, 1])
    else:  # [B]
        no_val = tf.reshape(no_val, [-1, 1, 1])
    no_complex = tf.cast(no_val, H.dtype)
    
    K_float = tf.cast(K, H.dtype.real_dtype)
    eye_M = tf.eye(M, dtype=H.dtype)[None, None, None, :, :]
    
    # ✅ WMMSE iterations
    for iteration in range(num_iterations):
        # H: [B, ofdm, fft, K, M]
        # V: [B, ofdm, fft, M, K]
        
        # Step 1: Compute U (receive filters)
        HV = tf.matmul(H, V)  # [B, ofdm, fft, K, K]
        signal = tf.linalg.diag_part(HV)  # [B, ofdm, fft, K]
        signal_power = tf.abs(signal)**2
        
        total_power = tf.reduce_sum(tf.abs(HV)**2, axis=-1)
        interference_power = total_power - signal_power
        
        denom = signal_power + interference_power + no_val
        U = tf.math.conj(signal) / tf.cast(denom, H.dtype)
        
        # Step 2: Compute MSE weights
        mse = 1.0 - tf.math.real(U * signal)
        W = 1.0 / tf.maximum(mse, 1e-6)
        
        # Step 3: Update V (precoders)
        weights = tf.cast(W * tf.abs(U)**2, H.dtype)
        H_H = tf.transpose(H, [0, 1, 2, 4, 3], conjugate=True)
        
        weighted_H = tf.expand_dims(weights, -1) * H
        A = tf.matmul(H_H, weighted_H)
        A_reg = A + no_complex[..., None, None] * eye_M
        
        target = tf.cast(W, H.dtype) * tf.math.conj(U)
        B = H_H * tf.expand_dims(target, -2)
        
        V = tf.linalg.solve(A_reg, B)
        
        # ✅ Power normalization within iterations
        power = tf.reduce_sum(tf.abs(V)**2, axis=[3, 4], keepdims=True)
        scale = tf.sqrt(K_float / (power + 1e-12))
        V = V * tf.cast(scale, H.dtype)
    
    # ✅ Add back num_tx dimension: [B, 1, ofdm, fft, M, K]
    V = tf.expand_dims(V, axis=1)
    
    # ✅ Final normalization (unit power per column)
    norm = tf.sqrt(tf.reduce_sum(tf.abs(V)**2, axis=-2, keepdims=True))
    V = tf.math.divide_no_nan(V, tf.cast(norm, V.dtype))
    
    return V


# =============================================================================
# ✅ TRANSFORMER BUILDING BLOCKS (Unchanged - these are good)
# =============================================================================

class SwiGLU(layers.Layer):
    """Gated Linear Unit activation"""
    def call(self, x):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=-1)
        return (x_a * tf.nn.sigmoid(x_a)) * x_b


class SeparableAttentionBlock(layers.Layer):
    """
    Separable attention: Process frequency and users independently
    (This implementation is GOOD - no changes needed)
    """
    
    def __init__(self, num_freq, num_users, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_freq = num_freq
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm_freq = RMSNormalization(epsilon=1e-6, name='norm_freq')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn = RMSNormalization(epsilon=1e-6, name='norm_ffn')
        
        self.freq_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name='freq_attention'
        )
        
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name='user_attention'
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu', name='ffn_expand'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim, name='ffn_project')
        ], name='ffn')
        
        self.scale_freq = self.add_weight(
            name='scale_freq', shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        self.scale_user = self.add_weight(
            name='scale_user', shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        self.scale_ffn = self.add_weight(
            name='scale_ffn', shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
    
    def call(self, x, training=False):
        """
        Args:
            x: [B, num_freq, num_users, embed_dim]
        Returns:
            x: [B, num_freq, num_users, embed_dim]
        """
        B = tf.shape(x)[0]
        
        # Frequency attention
        x_freq = tf.reshape(
            tf.transpose(x, [0, 2, 1, 3]),
            [B * self.num_users, self.num_freq, self.embed_dim]
        )
        x_freq_norm = self.norm_freq(x_freq)
        freq_attn_out = self.freq_attn(
            query=x_freq_norm, value=x_freq_norm, training=training
        )
        x_freq = x_freq + self.scale_freq * freq_attn_out
        x = tf.transpose(
            tf.reshape(x_freq, [B, self.num_users, self.num_freq, self.embed_dim]),
            [0, 2, 1, 3]
        )
        
        # User attention
        x_user = tf.reshape(x, [B * self.num_freq, self.num_users, self.embed_dim])
        x_user_norm = self.norm_user(x_user)
        user_attn_out = self.user_attn(
            query=x_user_norm, value=x_user_norm, training=training
        )
        x_user = x_user + self.scale_user * user_attn_out
        x = tf.reshape(x_user, [B, self.num_freq, self.num_users, self.embed_dim])
        
        # FFN
        x_flat = tf.reshape(x, [B * self.num_freq * self.num_users, self.embed_dim])
        x_flat_norm = self.norm_ffn(x_flat)
        ffn_out = self.ffn(x_flat_norm, training=training)
        x_flat = x_flat + self.scale_ffn * ffn_out
        x = tf.reshape(x_flat, [B, self.num_freq, self.num_users, self.embed_dim])
        
        return x


# =============================================================================
# ✅ CORRECTED: TransformerPrecoder
# =============================================================================

class TransformerPrecoder(Model):
    """
    ✅ BALANCED: Better upsampling WITHOUT breaking fairness
    
    Key changes from original:
    1. Larger Conv1DTranspose kernel (better context)
    2. Optional refinement layer (smooth transitions)
    3. NO per-SC independent scaling (preserves fairness!)
    """
    
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=None, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_ofdm = num_ofdm
        self.fft_size = fft_size
        self.rb_size = rb_size
        self.embed_dim = embed_dim
        self.use_rb_grouping = (rb_size is not None)
        self.num_streams_per_tx = num_rx
        
        if self.use_rb_grouping:
            self.num_rb = fft_size // rb_size
            self.num_freq = self.num_rb
            
            # RB aggregation (unchanged - this works!)
            self.rb_aggregation = tf.keras.Sequential([
                layers.Dense(rb_size // 2, activation='gelu'),
                layers.Dense(1)
            ], name='rb_aggregation')
            
            # ✅ IMPROVED: Larger kernel for more frequency context
            # Original: kernel_size = 2 * rb_size = 24
            # New:      kernel_size = 3 * rb_size = 36 (50% more context!)
            self.upsample_conv = layers.Conv1DTranspose(
                filters=num_tx * num_rx * 2,  # real + imag
                kernel_size=3 * rb_size,  # ✅ LARGER for smoother upsampling
                strides=rb_size,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name='learned_upsample'
            )
            
            # ✅ OPTIONAL: Light refinement (preserves structure!)
            # Small kernel = doesn't break global power distribution
            self.refinement_conv = layers.Conv1D(
                filters=num_tx * num_rx * 2,
                kernel_size=5,  # Small kernel preserves fairness
                padding='same',
                activation='tanh',  # Bounded output prevents explosions
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name='light_refinement'
            )
        else:
            self.num_freq = fft_size
        
        # Input processing (unchanged)
        self.input_embedding = layers.Dense(embed_dim)
        self.feature_norm = RMSNormalization(epsilon=1e-6)
        
        # Positional embeddings (unchanged)
        self.pos_embedding_freq = layers.Embedding(
            input_dim=self.num_freq,
            output_dim=embed_dim,
            name='pos_embed_freq'
        )
        
        # Transformer blocks (unchanged)
        self.blocks = [
            SeparableAttentionBlock(self.num_freq, num_rx, embed_dim, num_heads)
            for _ in range(num_layers)
        ]
        
        # Output projection (unchanged)
        self.output_projection = layers.Dense(2 * num_tx)
    
    def call(self, h_freq, training=False, return_real_imag=False):
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        
        h_sq = tf.squeeze(h_freq, axis=[2, 3])
        
        # Feature extraction (unchanged)
        if self.use_rb_grouping:
            h_rb = tf.reshape(h_sq, [B, self.num_rx, self.num_tx, 
                                     self.num_ofdm, self.num_rb, self.rb_size])
            h_rb = tf.transpose(h_rb, [0, 3, 4, 5, 1, 2])
            h_math = tf.stack([tf.math.real(h_rb), tf.math.imag(h_rb)], axis=-1)
            
            h_math_agg = self.rb_aggregation(
                tf.transpose(h_math, [0, 1, 2, 4, 5, 6, 3])
            )
            h_math_agg = tf.squeeze(h_math_agg, axis=-1)
            
            feat = tf.reshape(h_math_agg, 
                            [B * self.num_ofdm, self.num_rb, 
                             self.num_rx, self.num_tx * 2])
        else:
            h_perm = tf.transpose(h_sq, [0, 3, 4, 1, 2])
            h_math = tf.stack([tf.math.real(h_perm), tf.math.imag(h_perm)], axis=-1)
            feat = tf.reshape(h_math, 
                            [B * self.num_ofdm, self.fft_size, 
                             self.num_rx, self.num_tx * 2])
        
        # Transformer processing (unchanged)
        x = self.input_embedding(feat)
        x = self.feature_norm(x)
        
        freq_positions = tf.range(self.num_freq)
        pos_embed = self.pos_embedding_freq(freq_positions)
        pos_embed = tf.reshape(pos_embed, [1, self.num_freq, 1, self.embed_dim])
        x = x + pos_embed
                
        for block in self.blocks:
            x = block(x, training=training)
                
        # Output projection (unchanged)
        out = self.output_projection(x)
        
        # Reshape to [B*ofdm, freq, rx, tx, 2]
        out = tf.reshape(out, [B * self.num_ofdm, self.num_freq, 
                              self.num_rx, self.num_tx, 2])
                
        # Transpose: [B*ofdm, freq, rx, tx, 2] -> [B*ofdm, freq, tx, rx, 2]
        out = tf.transpose(out, [0, 1, 3, 2, 4])
        
        # Add batch dim: [B, ofdm, freq, tx, rx, 2]
        out = tf.reshape(out, [B, self.num_ofdm, self.num_freq,
                              self.num_tx, self.num_rx, 2])        
        
        w_real = out[..., 0]
        w_imag = out[..., 1]
        
        # ✅ IMPROVED UPSAMPLING (conservatif!)
        if self.use_rb_grouping:
            # Stack real/imag for upsampling
            w_combined = tf.stack([w_real, w_imag], axis=-1)  # [B, ofdm, freq, tx, rx, 2]
            
            # Flatten for Conv1DTranspose
            w_flat = tf.reshape(w_combined, 
                               [B * self.num_ofdm, self.num_rb, 
                                self.num_tx * self.num_streams_per_tx * 2])
            
            # ✅ Stage 1: Conv1DTranspose with LARGER kernel
            w_up = self.upsample_conv(w_flat, training=training)
            w_up = w_up[:, :self.fft_size, :]
            
            # ✅ Stage 2: OPTIONAL light refinement
            # Tanh activation keeps values bounded → preserves fairness!
            w_refined = self.refinement_conv(w_up, training=training)
            
            # Blend: 80% original + 20% refined (conservative!)
            w_final = 0.8 * w_up + 0.2 * w_refined
            
            # Reshape back
            w_up_reshaped = tf.reshape(w_final, 
                                      [B, self.num_ofdm, self.fft_size, 
                                       self.num_tx, self.num_streams_per_tx, 2])
            
            w_real_full = w_up_reshaped[..., 0]
            w_imag_full = w_up_reshaped[..., 1]
        else:
            w_real_full = w_real
            w_imag_full = w_imag
        
        # ✅ Power normalization (UNCHANGED - CRITICAL!)
        # This global normalization FORCES fairness across users!
        magnitude_sq = w_real_full**2 + w_imag_full**2
        power_per_tx_ant = tf.reduce_sum(magnitude_sq, axis=4, keepdims=True)
        scale = tf.sqrt(float(self.num_streams_per_tx) / (power_per_tx_ant + 1e-12))
        
        w_real_norm = w_real_full * scale
        w_imag_norm = w_imag_full * scale
        
        # ✅ Final reshape to Sionna format
        if return_real_imag:
            w_real_final = tf.expand_dims(w_real_norm, axis=1)
            w_imag_final = tf.expand_dims(w_imag_norm, axis=1)
            return (w_real_final, w_imag_final)
        else:
            w_complex = tf.complex(w_real_norm, w_imag_norm)
            w_complex = tf.expand_dims(w_complex, axis=1)
            return w_complex
        

# =============================================================================
# ✅ RESIDUAL TRANSFORMER PRECODER (Simple & Efficace)
# =============================================================================

class ResidualTransformerPrecoder(Model):
    """
    ✅ RESIDUAL LEARNING: W = RZF + α × Δ
    
    Improvements over base transformer:
    1. RB features: mean + first + last (captures slope!)
    2. Channel mixing: sees all users together (multi-user interference!)
    3. Learnable alpha: model decides correction strength
    4. RZF baseline: guaranteed performance floor
    
    Args:
        num_tx: Number of transmit antennas
        num_rx: Number of users
        num_ofdm: Number of OFDM symbols
        fft_size: FFT size
        rb_size: Resource block size (None = full SC)
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
    """
    
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_ofdm = num_ofdm
        self.fft_size = fft_size
        self.rb_size = rb_size
        self.num_rb = fft_size // rb_size
        self.embed_dim = embed_dim
        self.num_streams_per_tx = num_rx
        
        # =====================================================================
        # ✅ MODIF 1: RB Feature Extractor (mean + first + last)
        # =====================================================================
        # Project 3 features (mean, first, last) × 2 (real, imag) = 6 dims
        # Down to tx*2 dims for transformer input
        self.rb_feature_projection = layers.Dense(
            num_tx * 2,
            activation='gelu',
            name='rb_feature_projection'
        )
        
        # =====================================================================
        # Transformer Core
        # =====================================================================
        self.input_embedding = layers.Dense(embed_dim)
        self.feature_norm = RMSNormalization(epsilon=1e-6)
        
        self.pos_embedding_freq = layers.Embedding(
            input_dim=self.num_rb,
            output_dim=embed_dim,
            name='pos_embed_freq'
        )
        
        self.blocks = [
            SeparableAttentionBlock(self.num_rb, num_rx, embed_dim, num_heads)
            for _ in range(num_layers)
        ]
        
        self.output_projection = layers.Dense(2 * num_tx)
        
        # =====================================================================
        # ✅ MODIF 2: Channel-Conditioned Upsampler (with user mixing!)
        # =====================================================================
        # Sees ALL users together per SC!
        self.channel_conditioner = tf.keras.Sequential([
            layers.Conv1D(128, kernel_size=3, padding='same', activation='gelu'),
            layers.Dense(64, activation='gelu'),
            layers.Dense(num_tx * num_rx * 2)  # Output: corrections
        ], name='channel_conditioner')
        
        # =====================================================================
        # ✅ MODIF 3: non  Alpha (correction strength)
        # =====================================================================
        self.alpha = 0.15
    
    def call(self, h_freq, training=False, return_real_imag=False):
        """
        Forward pass
        
        Args:
            h_freq: [B, rx, 1, 1, tx, ofdm, fft] - Channel
            
        Returns:
            delta: [B, 1, ofdm, fft, tx, rx] - CORRECTION (not full precoder!)
        """
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        
        h_sq = tf.squeeze(h_freq, axis=[2, 3])  # [B, rx, tx, ofdm, fft]
        
        # =====================================================================
        # ✅ MODIF 1: Extract RICH RB features (mean + first + last)
        # =====================================================================
        h_rb = tf.reshape(h_sq, [B, self.num_rx, self.num_tx, 
                                 self.num_ofdm, self.num_rb, self.rb_size])
        
        # Extract 3 features per RB
        h_rb_mean = tf.reduce_mean(h_rb, axis=-1)   # [B, rx, tx, ofdm, num_rb]
        h_rb_first = h_rb[..., 0]                   # First SC (start of RB)
        h_rb_last = h_rb[..., -1]                   # Last SC (end of RB)
        
        # Real/imag for each feature → 6 total features
        features_list = []
        for feat in [h_rb_mean, h_rb_first, h_rb_last]:
            feat_real = tf.math.real(feat)
            feat_imag = tf.math.imag(feat)
            features_list.extend([feat_real, feat_imag])
        
        # Stack: [B, rx, tx, ofdm, num_rb, 6]
        h_rb_features = tf.stack(features_list, axis=-1)
        
        # Transpose for transformer: [B, ofdm, num_rb, rx, tx, 6]
        h_rb_features = tf.transpose(h_rb_features, [0, 3, 4, 1, 2, 5])
        
        # Flatten and project: [B*ofdm, num_rb, rx, tx*6] → [B*ofdm, num_rb, rx, tx*2]
        h_rb_flat = tf.reshape(h_rb_features, 
            [B * self.num_ofdm, self.num_rb, self.num_rx, self.num_tx * 6])
        
        h_rb_proj = self.rb_feature_projection(h_rb_flat)
        # [B*ofdm, num_rb, rx, tx*2]
        
        # =====================================================================
        # Transformer Processing
        # =====================================================================
        x = self.input_embedding(h_rb_proj)
        x = self.feature_norm(x)
        
        # Add positional embeddings
        freq_positions = tf.range(self.num_rb)
        pos_embed = self.pos_embedding_freq(freq_positions)
        pos_embed = tf.reshape(pos_embed, [1, self.num_rb, 1, self.embed_dim])
        x = x + pos_embed
        
        # Through transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Output projection
        out = self.output_projection(x)  # [B*ofdm, num_rb, rx, tx*2]
        
        # Reshape: [B*ofdm, num_rb, rx, tx, 2]
        out = tf.reshape(out, [B * self.num_ofdm, self.num_rb, 
                              self.num_rx, self.num_tx, 2])
        
        # Transpose: [B*ofdm, num_rb, tx, rx, 2]
        out = tf.transpose(out, [0, 1, 3, 2, 4])
        
        # Add batch dim: [B, ofdm, num_rb, tx, rx, 2]
        out = tf.reshape(out, [B, self.num_ofdm, self.num_rb,
                              self.num_tx, self.num_rx, 2])
        
        delta_rb_real = out[..., 0]
        delta_rb_imag = out[..., 1]
        
        # =====================================================================
        # Coarse Upsampling (Simple repeat)
        # =====================================================================
        delta_coarse_real = tf.repeat(delta_rb_real, self.rb_size, axis=2)
        delta_coarse_imag = tf.repeat(delta_rb_imag, self.rb_size, axis=2)
        # [B, ofdm, fft_size, tx, rx]
        
        # =====================================================================
        # ✅ MODIF 2: Channel-Conditioned Refinement (with USER MIXING!)
        # =====================================================================
        # Get full-resolution channel
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])  # [B, ofdm, fft, rx, tx]
        h_full_real = tf.math.real(h_full)
        h_full_imag = tf.math.imag(h_full)
        
        # Flatten
        B_ofdm = B * self.num_ofdm
        delta_flat = tf.stack([delta_coarse_real, delta_coarse_imag], axis=-1)
        delta_flat = tf.reshape(delta_flat, 
            [B_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
        
        h_flat = tf.stack([h_full_real, h_full_imag], axis=-1)
        h_flat = tf.reshape(h_flat,
            [B_ofdm, self.fft_size, self.num_rx, self.num_tx, 2])
        h_flat = tf.transpose(h_flat, [0, 1, 3, 2, 4])  # [B, fft, tx, rx, 2]
        
        # Concatenate: [B_ofdm, fft, tx, rx, 4]
        combined = tf.concat([delta_flat, h_flat], axis=-1)
        
        # ✅ KEY: Flatten tx-rx to see ALL users together!
        combined_mixed = tf.reshape(combined, 
            [B_ofdm, self.fft_size, self.num_tx * self.num_rx * 4])
        # [B_ofdm, fft, tx*rx*4] → Network sees ALL users per SC!
        
        # Channel-conditioned refinement
        refinement_flat = self.channel_conditioner(combined_mixed, training=training)
        # [B_ofdm, fft, tx*rx*2]
        
        # Reshape back
        refinement = tf.reshape(refinement_flat, 
            [B_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
        
        # Combine: coarse + small refinement
        delta_final_flat = delta_flat + 0.2 * refinement
        
        # Reshape back to [B, ofdm, fft, tx, rx, 2]
        delta_final = tf.reshape(delta_final_flat,
            [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
        
        delta_real_full = delta_final[..., 0]
        delta_imag_full = delta_final[..., 1]
        
        # =====================================================================
        # ✅ MODIF 3: Apply Learnable Alpha
        # =====================================================================
        delta_real_scaled = self.alpha * delta_real_full
        delta_imag_scaled = self.alpha * delta_imag_full
        
        # =====================================================================
        # Output
        # =====================================================================
        if return_real_imag:
            delta_real_final = tf.expand_dims(delta_real_scaled, axis=1)
            delta_imag_final = tf.expand_dims(delta_imag_scaled, axis=1)
            return (delta_real_final, delta_imag_final)
        else:
            delta_complex = tf.complex(delta_real_scaled, delta_imag_scaled)
            delta_complex = tf.expand_dims(delta_complex, axis=1)
            return delta_complex

'''
class TransformerPrecoderV2(Model):
    """
    ✅ IMPROVED TransformerPrecoder with ResidualTransformer insights:
    1. Rich RB features (mean + first + last)
    2. Channel-conditioned upsampling
    3. User mixing in channel conditioner
    """
    
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=None, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_ofdm = num_ofdm
        self.fft_size = fft_size
        self.rb_size = rb_size
        self.embed_dim = embed_dim
        self.use_rb_grouping = (rb_size is not None)
        self.num_streams_per_tx = num_rx
        
        if self.use_rb_grouping:
            self.num_rb = fft_size // rb_size
            self.num_freq = self.num_rb
            
            # ✅ IMPROVED: Rich feature projection (not aggregation!)
            self.rb_feature_projection = layers.Dense(
                num_tx * 2,
                activation='gelu',
                name='rb_feature_projection'
            )
            
            self.input_norm = RMSNormalization(epsilon=1e-6)
            # ✅ IMPROVED: Conv1DTranspose with larger kernel
            self.upsample_conv = layers.Conv1DTranspose(
                filters=num_tx * num_rx * 2,
                kernel_size=3 * rb_size,
                strides=rb_size,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name='learned_upsample'
            )
            
            # ✅ NEW: Channel-conditioned refinement
            self.channel_conditioner = tf.keras.Sequential([
                layers.Dense(128, activation='gelu'),
                layers.Dense(64, activation='gelu'),
                layers.Dense(num_tx * num_rx * 2)
            ], name='channel_conditioner')
        else:
            self.num_freq = fft_size
        
        # Transformer core (unchanged)
        self.input_embedding = layers.Dense(embed_dim)
        self.feature_norm = RMSNormalization(epsilon=1e-6)
        
        self.pos_embedding_freq = layers.Embedding(
            input_dim=self.num_freq,
            output_dim=embed_dim,
            name='pos_embed_freq'
        )
        
        self.blocks = [
            SeparableAttentionBlock(self.num_freq, num_rx, embed_dim, num_heads)
            for _ in range(num_layers)
        ]
        
        self.output_projection = layers.Dense(2 * num_tx)
    
    def call(self, h_freq, training=False, return_real_imag=False):
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        
        h_sq = tf.squeeze(h_freq, axis=[2, 3])
        
        # ✅ IMPROVED: Rich RB feature extraction
        if self.use_rb_grouping:
            h_rb = tf.reshape(h_sq, [B, self.num_rx, self.num_tx, 
                                     self.num_ofdm, self.num_rb, self.rb_size])
            
            # Extract 3 features per RB
            h_rb_mean = tf.reduce_mean(h_rb, axis=-1)
            h_rb_first = h_rb[..., 0]
            h_rb_last = h_rb[..., -1]
            
            # Real/imag for each → 6 features
            features_list = []
            for feat in [h_rb_mean, h_rb_first, h_rb_last]:
                features_list.extend([tf.math.real(feat), tf.math.imag(feat)])
            
            h_rb_features = tf.stack(features_list, axis=-1)  # [..., 6]
            h_rb_features = tf.transpose(h_rb_features, [0, 3, 4, 1, 2, 5])
            
            h_rb_flat = tf.reshape(h_rb_features, 
                [B * self.num_ofdm, self.num_rb, self.num_rx, self.num_tx * 6])
            
            feat = self.rb_feature_projection(h_rb_flat)
        else:
            # Full frequency (no RB grouping)
            h_perm = tf.transpose(h_sq, [0, 3, 4, 1, 2])
            h_math = tf.stack([tf.math.real(h_perm), tf.math.imag(h_perm)], axis=-1)
            feat = tf.reshape(h_math, 
                [B * self.num_ofdm, self.fft_size, self.num_rx, self.num_tx * 2])
        
        # Transformer processing (unchanged)
        x = self.input_embedding(feat)
        x = self.feature_norm(x)
        
        freq_positions = tf.range(self.num_freq)
        pos_embed = self.pos_embedding_freq(freq_positions)
        pos_embed = tf.reshape(pos_embed, [1, self.num_freq, 1, self.embed_dim])
        x = x + pos_embed
        
        for block in self.blocks:
            x = block(x, training=training)
        
        out = self.output_projection(x)
        
        # Reshape and transpose
        out = tf.reshape(out, [B * self.num_ofdm, self.num_freq, 
                              self.num_rx, self.num_tx, 2])
        out = tf.transpose(out, [0, 1, 3, 2, 4])
        out = tf.reshape(out, [B, self.num_ofdm, self.num_freq,
                              self.num_tx, self.num_rx, 2])
        
        w_real = out[..., 0]
        w_imag = out[..., 1]
        
        # ✅ IMPROVED: Upsampling with channel conditioning
        if self.use_rb_grouping:
            # Stage 1: Conv1DTranspose
            w_combined = tf.stack([w_real, w_imag], axis=-1)
            w_flat = tf.reshape(w_combined, 
                [B * self.num_ofdm, self.num_rb, self.num_tx * self.num_rx * 2])
            
            w_up = self.upsample_conv(w_flat, training=training)
            w_up = w_up[:, :self.fft_size, :]
            
            # Stage 2: Channel-conditioned refinement
            w_up_reshaped = tf.reshape(w_up, 
                [B * self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
            
            # Get full-resolution channel
            h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])
            h_full_flat = tf.stack([tf.math.real(h_full), tf.math.imag(h_full)], axis=-1)
            h_full_flat = tf.reshape(h_full_flat,
                [B * self.num_ofdm, self.fft_size, self.num_rx, self.num_tx, 2])
            h_full_flat = tf.transpose(h_full_flat, [0, 1, 3, 2, 4])
            
            # Concatenate and mix users
            combined = tf.concat([w_up_reshaped, h_full_flat], axis=-1)
            combined_mixed = tf.reshape(combined, 
                [B * self.num_ofdm, self.fft_size, self.num_tx * self.num_rx * 4])
            
            refinement_flat = self.channel_conditioner(combined_mixed, training=training)
            refinement = tf.reshape(refinement_flat, 
                [B * self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
            
            # Blend
            w_final = w_up_reshaped + 0.2 * refinement
            
            w_up_reshaped = tf.reshape(w_final, 
                [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
            
            w_real_full = w_up_reshaped[..., 0]
            w_imag_full = w_up_reshaped[..., 1]
        else:
            w_real_full = w_real
            w_imag_full = w_imag
        
        # Power normalization (unchanged - CRITICAL!)
        magnitude_sq = w_real_full**2 + w_imag_full**2
        power_per_tx_ant = tf.reduce_sum(magnitude_sq, axis=[3,4], keepdims=True)
        scale = tf.sqrt(float(self.num_streams_per_tx) / (power_per_tx_ant + 1e-12))
        
        w_real_norm = w_real_full * scale
        w_imag_norm = w_imag_full * scale
        
        # Final output
        if return_real_imag:
            w_real_final = tf.expand_dims(w_real_norm, axis=1)
            w_imag_final = tf.expand_dims(w_imag_norm, axis=1)
            return (w_real_final, w_imag_final)
        else:
            w_complex = tf.complex(w_real_norm, w_imag_norm)
            w_complex = tf.expand_dims(w_complex, axis=1)
            return w_complex
'''
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import RMSNormalization

class TransformerPrecoderV2(Model):
    """
    ✅ TRANSFORMER PRECODER V2 (SOTA Edition)
    
    Key Features:
    1. Rich RB Features: Mean + First + Last SC + **Input Covariance**.
    2. Input Normalization: RMSNorm on input features.
    3. Learned Upsampling: Conv1DTranspose with mixing across users.
    4. Channel Refinement: Conditioned on full channel state.
    5. Global Power Constraint: Allows water-filling optimization.
    """
    
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_ofdm = num_ofdm
        self.fft_size = fft_size
        self.rb_size = rb_size
        self.embed_dim = embed_dim
        self.use_rb_grouping = (rb_size is not None)
        self.num_streams_per_tx = num_rx
        
        if self.use_rb_grouping:
            # --- STRATEGY: 2 TOKENS PER RB ---
            self.tokens_per_rb = 2 
            self.sc_per_token = rb_size // self.tokens_per_rb # 6
            self.num_tokens = (fft_size // rb_size) * self.tokens_per_rb # 12
            
            # [1] Feature Projection
            # Input dim auto-adaptative (gere les features + covariance)
            self.rb_feature_projection = layers.Dense(
                embed_dim, 
                activation='gelu',
                name='rb_feature_projection'
            )
            
            # [2] Upsampling Conv
            self.upsample_conv = layers.Conv1DTranspose(
                filters=num_tx * num_rx * 2, 
                kernel_size=self.sc_per_token * 3,
                strides=self.sc_per_token,
                padding='same',
                name='learned_upsample'
            )
            
            # [3] Channel Conditioner (Refinement)
            # Ajout d'une Conv1D pour le contexte fréquentiel (Amélioration V3)
            self.channel_conditioner = tf.keras.Sequential([
                layers.Conv1D(128, kernel_size=3, padding='same', activation='gelu'),
                layers.Dense(64, activation='gelu'),
                layers.Dense(num_tx * num_rx * 2)
            ], name='channel_conditioner')
            
        else:
            self.num_tokens = fft_size
            self.input_embedding = layers.Dense(embed_dim)
            self.output_projection = layers.Dense(num_tx * num_rx * 2)

        # Common Components
        self.input_norm = RMSNormalization(epsilon=1e-6)
        
        self.pos_embedding = layers.Embedding(
            input_dim=self.num_tokens,
            output_dim=embed_dim,
            name='pos_embed'
        )
        
        self.blocks = [
            SeparableAttentionBlock(self.num_tokens, num_rx, embed_dim, num_heads)
            for _ in range(num_layers)
        ]

    def call(self, h_freq, training=False, return_real_imag=False):
        # h_freq: [B, rx, 1, 1, tx, ofdm, fft]
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        h_sq = tf.squeeze(h_freq, axis=[2, 3]) # [B, rx, tx, ofdm, fft]
        
        if self.use_rb_grouping:
            # --- STEP 1: RICH FEATURE EXTRACTION ---
            # 1. Reshape to tokens: [B, rx, tx, ofdm, num_tokens, sc_per_token]
            h_tokens = tf.reshape(h_sq, [B, self.num_rx, self.num_tx, self.num_ofdm, self.num_tokens, self.sc_per_token])
            
            # 2. Extract Basic Features (Mean, First, Last)
            h_mean = tf.reduce_mean(h_tokens, axis=-1)
            h_first = h_tokens[..., 0]
            h_last = h_tokens[..., -1]
            
            # 3. COMPUTE INPUT COVARIANCE (Crucial for MIMO)
            # We use the mean channel per token to compute covariance
            # Permute for matmul: [B, ofdm, tokens, rx, tx]
            h_mean_perm = tf.transpose(h_mean, [0, 3, 4, 1, 2])
            # R = H * H^H -> [B, ofdm, tokens, rx, rx]
            cov_matrix = tf.matmul(h_mean_perm, h_mean_perm, adjoint_b=True)
            
            # Stack Real/Imag of Covariance
            cov_real = tf.math.real(cov_matrix)
            cov_imag = tf.math.imag(cov_matrix)
            # Flatten to features: [B, ofdm, tokens, rx, rx*2]
            # (Chaque user voit sa ligne de corrélation avec les autres)
            cov_feat = tf.concat([cov_real, cov_imag], axis=-1)
            
            # 4. Prepare Standard Features
            # Stack features: [B, rx, tx, ofdm, tokens] -> [..., 6] (3 features * 2 complex)
            feats_list = [h_mean, h_first, h_last]
            feats_stack = []
            for f in feats_list:
                feats_stack.append(tf.math.real(f))
                feats_stack.append(tf.math.imag(f))
            h_feats = tf.stack(feats_stack, axis=-1)
            
            # Transpose to align with covariance: [B, ofdm, tokens, rx, tx, 6]
            h_feats = tf.transpose(h_feats, [0, 3, 4, 1, 2, 5])
            # Flatten TX: [B, ofdm, tokens, rx, tx*6]
            h_feats_flat = tf.reshape(h_feats, [B, self.num_ofdm, self.num_tokens, self.num_rx, self.num_tx * 6])
            
            # 5. FUSION (Concatenate Channel Features + Covariance)
            # Input to projection: [B, ofdm, tokens, rx, (tx*6 + rx*2)]
            # Note: Covariance is [rx, rx]. We broadcast/align properly.
            # Here covariance is [B, ofdm, tokens, rx, rx*2]. 
            # It naturally concatenates along the last dimension for each user.
            feat_input = tf.concat([h_feats_flat, cov_feat], axis=-1)
            
            # Flatten batch dims for Dense layer
            feat_input = tf.reshape(feat_input, [B * self.num_ofdm, self.num_tokens, self.num_rx, -1])
            
            # Project to Embedding Dim
            x = self.rb_feature_projection(feat_input) 
            
        else:
            # Full SC logic
            h_perm = tf.transpose(h_sq, [0, 3, 4, 1, 2])
            h_math = tf.stack([tf.math.real(h_perm), tf.math.imag(h_perm)], axis=-1)
            feat_input = tf.reshape(h_math, [B * self.num_ofdm, self.fft_size, self.num_rx, self.num_tx * 2])
            x = self.input_embedding(feat_input)

        # --- STEP 2: TRANSFORMER CORE ---
        x = self.input_norm(x)
        
        positions = tf.range(self.num_tokens)
        pos_embed = self.pos_embedding(positions)
        pos_embed = tf.reshape(pos_embed, [1, self.num_tokens, 1, self.embed_dim])
        x = x + pos_embed
        
        for block in self.blocks:
            x = block(x, training=training)
            
        # --- STEP 3: UPSAMPLING & OUTPUT ---
        if self.use_rb_grouping:
            # Upsampling
            x_flat_users = tf.reshape(x, [B * self.num_ofdm, self.num_tokens, self.num_rx * self.embed_dim])
            w_up = self.upsample_conv(x_flat_users) 
            w_up = w_up[:, :self.fft_size, :]
            w_up_reshaped = tf.reshape(w_up, [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
            
            # Channel Conditioned Refinement
            h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])
            h_full_flat = tf.stack([tf.math.real(h_full), tf.math.imag(h_full)], axis=-1)
            h_full_flat = tf.reshape(h_full_flat, [B * self.num_ofdm, self.fft_size, -1])
            
            w_flat_for_cond = tf.reshape(w_up_reshaped, [B * self.num_ofdm, self.fft_size, -1])
            combined = tf.concat([w_flat_for_cond, h_full_flat], axis=-1)
            
            refinement = self.channel_conditioner(combined)
            w_final_flat = w_flat_for_cond + 0.2 * refinement
            
            out = tf.reshape(w_final_flat, [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
            
        else:
            out = self.output_projection(x)
            out = tf.reshape(out, [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])

        w_real = out[..., 0]
        w_imag = out[..., 1]
        
        # --- STEP 4: GLOBAL POWER NORMALIZATION ---
        mag_sq = w_real**2 + w_imag**2
        total_pwr = tf.reduce_sum(mag_sq, axis=[3, 4], keepdims=True)
        scale = tf.sqrt(tf.cast(self.num_rx, w_real.dtype) / (total_pwr + 1e-12))
        
        w_real_norm = w_real * scale
        w_imag_norm = w_imag * scale
        
        # --- STEP 5: OUTPUT ---
        if return_real_imag:
            return (tf.expand_dims(w_real_norm, 1), tf.expand_dims(w_imag_norm, 1))
        else:
            w_complex = tf.complex(w_real_norm, w_imag_norm)
            return tf.expand_dims(w_complex, 1)