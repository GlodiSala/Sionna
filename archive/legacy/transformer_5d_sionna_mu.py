import tensorflow as tf
from tensorflow.keras import layers
from sionna.phy.ofdm import PrecodedChannel


class WorkingTransformerPrecoder(PrecodedChannel):
    """
    FIXED transformer that actually works
    
    Key fixes:
    1. Simpler RB processing (no complex aggregation)
    2. Better initialization
    3. Proper CSI normalization
    """
    
    def __init__(self, resource_grid, stream_management,
                 num_tx_antennas=8, num_rx_antennas=4, rb_size=12,
                 embed_dim=64, num_heads=4, num_layers=2,  # ✅ SMALLER
                 dtype=tf.complex64, **kwargs):
        
        super().__init__(resource_grid, stream_management, dtype=dtype, **kwargs)
        
        self.M = num_tx_antennas
        self.K = num_rx_antennas
        self.rb_size = rb_size
        self.fft_size = resource_grid.fft_size
        self.num_rb = self.fft_size // rb_size
        self.embed_dim = embed_dim
        
        print(f"\n{'='*70}")
        print(f"  WORKING TRANSFORMER PRECODER")
        print(f"{'='*70}")
        print(f"  Architecture: {embed_dim}d × {num_layers}L × {num_heads}H")
        print(f"  MIMO: {self.M} antennas → {self.K} users")
        print(f"  RBs: {self.num_rb}")
        print(f"{'='*70}\n")
        
        # ✅ Input processing (CRITICAL FIX)
        self.csi_projection = layers.Dense(
            embed_dim,
            kernel_initializer='glorot_uniform',  # Standard init
            bias_initializer='zeros',
            activation='relu',  # ✅ Add nonlinearity
            name='csi_projection'
        )
        
        # Transformer
        self.transformer = tf.keras.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=0.0
            ),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim),
        ], name='transformer')
        
        # ✅ Output (CRITICAL FIX)
        self.output_dense = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),  # Hidden layer
            layers.Dense(
                2 * self.M * self.K,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),  # ✅ Larger init
                bias_initializer='zeros'
            )
        ], name='output')
        
        # Build
        dummy_h = tf.zeros([1, self.K, 1, 1, self.M, 
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        dummy_y = tf.zeros([1, self.K, 1,
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        _ = self.call((dummy_y, dummy_h))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Built! Parameters: {total_params:,}\n")
    
    def call(self, inputs, training=False):
        y, h = inputs
        
        h = tf.squeeze(h, axis=[2, 3])  # [B, K, M, S, F]
        B = tf.shape(h)[0]
        S = tf.shape(h)[3]
        F = tf.shape(h)[4]
        
        # ✅ SIMPLE RB aggregation (just average, no learning)
        h_rb = tf.reshape(h, [B, self.K, self.M, S, self.num_rb, self.rb_size])
        h_avg = tf.reduce_mean(h_rb, axis=-1)  # [B, K, M, S, num_rb]
        
        # Reshape: [B, S, num_rb, K*M]
        h_trans = tf.transpose(h_avg, perm=[0, 3, 4, 1, 2])
        h_flat = tf.reshape(h_trans, [B*S, self.num_rb, self.K * self.M])
        
        # ✅ Convert to real and NORMALIZE
        h_real = tf.math.real(h_flat)
        h_imag = tf.math.imag(h_flat)
        h_concat = tf.concat([h_real, h_imag], axis=-1)  # [B*S, num_rb, 2*K*M]
        
        # ✅ CRITICAL: Normalize CSI
        h_norm = h_concat / (tf.math.reduce_std(h_concat, axis=-1, keepdims=True) + 1e-6)
        
        # Project
        x = self.csi_projection(h_norm)  # [B*S, num_rb, embed_dim]
        
        # Transform
        x = self.transformer(x)  # [B*S, num_rb, embed_dim]
        
        # Average over RBs (global pooling)
        x_pooled = tf.reduce_mean(x, axis=1)  # [B*S, embed_dim]
        
        # Output
        out = self.output_dense(x_pooled)  # [B*S, 2*M*K]
        
        # Reshape to precoder
        out = tf.reshape(out, [B, S, self.M, self.K, 2])
        out = tf.tile(tf.expand_dims(out, 2), [1, 1, F, 1, 1, 1])  # Broadcast to all freqs
        
        w = tf.complex(out[..., 0], out[..., 1])
        
        # ✅ Per-user normalization
        power_per_user = tf.reduce_sum(tf.abs(w)**2, axis=3, keepdims=True)
        w_normalized = w / tf.cast(tf.sqrt(power_per_user + 1e-12), w.dtype)
        
        return tf.expand_dims(w_normalized, axis=1)


import tensorflow as tf
from tensorflow.keras import layers
from sionna.phy.ofdm import PrecodedChannel


# =============================================================================
# ULTRA-SIMPLE MLP PRECODER (FIXED)
# =============================================================================
class MLPPrecoder(PrecodedChannel):
    """
    Simple MLP: H → W
    
    Minimal version to test if training loop works
    """
    
    def __init__(self, resource_grid, stream_management,
                 num_tx_antennas=8, num_rx_antennas=4,
                 dtype=tf.complex64, **kwargs):
        
        super().__init__(resource_grid, stream_management, dtype=dtype, **kwargs)
        
        self.M = num_tx_antennas
        self.K = num_rx_antennas
        self.fft_size = resource_grid.fft_size
        
        input_dim = 2 * self.K * self.M
        output_dim = 2 * self.M * self.K
        
        # Simple 3-layer MLP
        self.mlp = tf.keras.Sequential([
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dense(128, activation='relu', name='dense2'),
            layers.Dense(output_dim,
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                        name='output'),
        ], name='mlp')
        
        print(f"\n✅ MLP Precoder")
        print(f"   Input: {input_dim}")
        print(f"   Hidden: 128 → 128")
        print(f"   Output: {output_dim}\n")
        
        # Build
        dummy_h = tf.zeros([1, self.K, 1, 1, self.M, 
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        dummy_y = tf.zeros([1, self.K, 1,
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        _ = self.call((dummy_y, dummy_h))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Parameters: {total_params:,}\n")
    
    @property
    def trainable_variables(self):
        """✅ CRITICAL: Expose trainable variables"""
        return self.mlp.trainable_variables
    
    def call(self, inputs, training=False):
        y, h = inputs
        
        h = tf.squeeze(h, axis=[2, 3])  # [B, K, M, S, F]
        B = tf.shape(h)[0]
        S = tf.shape(h)[3]
        F = tf.shape(h)[4]
        
        # Average over frequency
        h_avg = tf.reduce_mean(h, axis=-1)  # [B, K, M, S]
        
        # Flatten
        h_flat = tf.reshape(h_avg, [B*S, self.K * self.M])
        
        # To real
        h_real = tf.math.real(h_flat)
        h_imag = tf.math.imag(h_flat)
        h_features = tf.concat([h_real, h_imag], axis=-1)
        
        # MLP
        out = self.mlp(h_features)
        
        # Reshape to precoder
        out = tf.reshape(out, [B, S, self.M, self.K, 2])
        
        # Broadcast to all frequencies
        out = tf.tile(tf.expand_dims(out, 2), [1, 1, F, 1, 1, 1])
        
        # To complex
        w = tf.complex(out[..., 0], out[..., 1])
        
        # Per-user power normalization
        power_per_user = tf.reduce_sum(tf.abs(w)**2, axis=3, keepdims=True)
        w_normalized = w / tf.cast(tf.sqrt(power_per_user + 1e-12), w.dtype)
        
        return tf.expand_dims(w_normalized, axis=1)


# =============================================================================
# SIMPLE LINEAR PRECODER (FIXED)
# =============================================================================
class SimpleLinearPrecoder(PrecodedChannel):
    """
    Simplest possible precoder: Single linear layer H → W
    """
    
    def __init__(self, resource_grid, stream_management,
                 num_tx_antennas=8, num_rx_antennas=4,
                 dtype=tf.complex64, **kwargs):
        
        super().__init__(resource_grid, stream_management, dtype=dtype, **kwargs)
        
        self.M = num_tx_antennas
        self.K = num_rx_antennas
        self.fft_size = resource_grid.fft_size
        
        input_dim = 2 * self.K * self.M
        output_dim = 2 * self.M * self.K
        
        # Single dense layer
        self.dense = layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            name='precoder'
        )
        
        print(f"\n✅ Linear Precoder")
        print(f"   Input: H [{self.K}, {self.M}]")
        print(f"   Output: W [{self.M}, {self.K}]")
        print(f"   Single dense layer: {input_dim} → {output_dim}\n")
        
        # Build
        dummy_h = tf.zeros([1, self.K, 1, 1, self.M, 
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        dummy_y = tf.zeros([1, self.K, 1,
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        _ = self.call((dummy_y, dummy_h))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Parameters: {total_params:,}\n")
    
    @property
    def trainable_variables(self):
        """✅ CRITICAL: Expose trainable variables"""
        return self.dense.trainable_variables
    
    def call(self, inputs, training=False):
        y, h = inputs
        
        h = tf.squeeze(h, axis=[2, 3])  # [B, K, M, S, F]
        B = tf.shape(h)[0]
        S = tf.shape(h)[3]
        F = tf.shape(h)[4]
        
        # Average over frequency
        h_avg = tf.reduce_mean(h, axis=-1)  # [B, K, M, S]
        
        # Flatten
        h_flat = tf.reshape(h_avg, [B*S, self.K * self.M])
        
        # To real
        h_real = tf.math.real(h_flat)
        h_imag = tf.math.imag(h_flat)
        h_features = tf.concat([h_real, h_imag], axis=-1)
        
        # Linear layer
        out = self.dense(h_features)
        
        # Reshape to precoder
        out = tf.reshape(out, [B, S, self.M, self.K, 2])
        
        # Broadcast to all frequencies
        out = tf.tile(tf.expand_dims(out, 2), [1, 1, F, 1, 1, 1])
        
        # To complex
        w = tf.complex(out[..., 0], out[..., 1])
        
        # Per-user power normalization
        power_per_user = tf.reduce_sum(tf.abs(w)**2, axis=3, keepdims=True)
        w_normalized = w / tf.cast(tf.sqrt(power_per_user + 1e-12), w.dtype)
        
        return tf.expand_dims(w_normalized, axis=1)


# =============================================================================
# WORKING TRANSFORMER (FIXED)
# =============================================================================
class WorkingTransformerPrecoder(PrecodedChannel):
    """
    Fixed transformer that should actually work
    """
    
    def __init__(self, resource_grid, stream_management,
                 num_tx_antennas=8, num_rx_antennas=4, rb_size=12,
                 embed_dim=64, num_heads=4, num_layers=2,
                 dtype=tf.complex64, **kwargs):
        
        super().__init__(resource_grid, stream_management, dtype=dtype, **kwargs)
        
        self.M = num_tx_antennas
        self.K = num_rx_antennas
        self.rb_size = rb_size
        self.fft_size = resource_grid.fft_size
        self.num_rb = self.fft_size // rb_size
        self.embed_dim = embed_dim
        
        print(f"\n{'='*70}")
        print(f"  WORKING TRANSFORMER PRECODER")
        print(f"{'='*70}")
        print(f"  Architecture: {embed_dim}d × {num_layers}L × {num_heads}H")
        print(f"  MIMO: {self.M} antennas → {self.K} users")
        print(f"{'='*70}\n")
        
        # Input projection
        self.input_proj = layers.Dense(
            embed_dim,
            kernel_initializer='glorot_uniform',
            activation='relu',
            name='input_proj'
        )
        
        # Simple transformer
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            name='attention'
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim)
        ], name='ffn')
        
        # Output
        self.output_net = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(2 * self.M * self.K,
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        ], name='output')
        
        # Build
        dummy_h = tf.zeros([1, self.K, 1, 1, self.M, 
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        dummy_y = tf.zeros([1, self.K, 1,
                            resource_grid.num_ofdm_symbols,
                            resource_grid.fft_size], dtype=dtype)
        _ = self.call((dummy_y, dummy_h))
        
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Built! Parameters: {total_params:,}\n")
    
    @property
    def trainable_variables(self):
        """✅ CRITICAL: Expose all trainable variables"""
        all_vars = []
        all_vars.extend(self.input_proj.trainable_variables)
        all_vars.extend(self.norm1.trainable_variables)
        all_vars.extend(self.attn.trainable_variables)
        all_vars.extend(self.norm2.trainable_variables)
        all_vars.extend(self.ffn.trainable_variables)
        all_vars.extend(self.output_net.trainable_variables)
        return all_vars
    
    def call(self, inputs, training=False):
        y, h = inputs
        
        h = tf.squeeze(h, axis=[2, 3])  # [B, K, M, S, F]
        B = tf.shape(h)[0]
        S = tf.shape(h)[3]
        F = tf.shape(h)[4]
        
        # Simple RB averaging
        h_rb = tf.reshape(h, [B, self.K, self.M, S, self.num_rb, self.rb_size])
        h_avg = tf.reduce_mean(h_rb, axis=-1)  # [B, K, M, S, num_rb]
        
        # Reshape
        h_trans = tf.transpose(h_avg, perm=[0, 3, 4, 1, 2])
        h_flat = tf.reshape(h_trans, [B*S, self.num_rb, self.K * self.M])
        
        # To real
        h_real = tf.math.real(h_flat)
        h_imag = tf.math.imag(h_flat)
        h_concat = tf.concat([h_real, h_imag], axis=-1)
        
        # Project
        x = self.input_proj(h_concat)  # [B*S, num_rb, embed_dim]
        
        # Transformer block
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, training=training)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + ffn_out
        
        # Pool over RBs
        x_pooled = tf.reduce_mean(x, axis=1)  # [B*S, embed_dim]
        
        # Output
        out = self.output_net(x_pooled)  # [B*S, 2*M*K]
        
        # Reshape
        out = tf.reshape(out, [B, S, self.M, self.K, 2])
        out = tf.tile(tf.expand_dims(out, 2), [1, 1, F, 1, 1, 1])
        
        w = tf.complex(out[..., 0], out[..., 1])
        
        # Normalize
        power_per_user = tf.reduce_sum(tf.abs(w)**2, axis=3, keepdims=True)
        w_normalized = w / tf.cast(tf.sqrt(power_per_user + 1e-12), w.dtype)
        
        return tf.expand_dims(w_normalized, axis=1)