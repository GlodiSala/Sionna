import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import RMSNormalization
from sionna.phy.ofdm import PrecodedChannel


class SwiGLU(layers.Layer):
    def call(self, x):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=-1)
        return (x_a * tf.nn.sigmoid(x_a)) * x_b


class SimpleTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0, **kwargs):
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


class SimplifiedTransformerPrecoder(PrecodedChannel):
    def __init__(self, resource_grid, stream_management, 
                 num_tx_antennas=8, num_rx_antennas=4, rb_size=12,
                 embed_dim=128, num_heads=4, num_layers=4,
                 use_learned_rb_aggregation=True,  # ✅ NEW
                 dtype=tf.complex64, **kwargs):
        
        super().__init__(resource_grid, stream_management, dtype=dtype, **kwargs)
        
        self.M = num_tx_antennas
        self.K = num_rx_antennas
        self.rb_size = rb_size
        self.fft_size = resource_grid.fft_size
        self.num_rb = self.fft_size // rb_size
        self.embed_dim = embed_dim
        self.use_learned_rb_aggregation = use_learned_rb_aggregation
        
        # ✅ Learned RB aggregation (patch embedding)
        if use_learned_rb_aggregation:
            self.rb_aggregation = layers.Dense(
                1, 
                use_bias=True,
                kernel_initializer=tf.keras.initializers.Constant(1.0 / rb_size),
                name='rb_aggregation'
            )
        
        self.input_embedding = layers.Dense(embed_dim, name='input_embedding')
        self.feature_norm = RMSNormalization(axis=-1, epsilon=1e-6)
        
        self.blocks = []
        for i in range(num_layers):
            block = SimpleTransformerBlock(
                embed_dim, num_heads, 
                hidden_dim=embed_dim*4, 
                dropout=0, 
                name=f'block_{i}'
            )
            self.blocks.append(block)
        
        self.output_projection = layers.Dense(
            2 * self.M * self.K, 
            kernel_initializer=tf.keras.initializers.Identity(gain=0.1),  # ✅ Identity!
            name='output_projection'
        )
        
        # Build
        print(f"🔧 Building Transformer with {'learned' if use_learned_rb_aggregation else 'simple'} RB aggregation...")
        dummy_h = tf.zeros([1, self.K, 1, 1, self.M, 
                            resource_grid.num_ofdm_symbols, 
                            resource_grid.fft_size], dtype=dtype)
        dummy_y = tf.zeros([1, self.K, 1, 
                            resource_grid.num_ofdm_symbols, 
                            resource_grid.fft_size], dtype=dtype)
        _ = self.call((dummy_y, dummy_h))
        
        # Count parameters
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"✅ Total parameters: {total_params:,}\n")

    @property
    def trainable_variables(self):
        all_vars = []
        
        # RB aggregation (if learned)
        if self.use_learned_rb_aggregation:
            all_vars.extend(self.rb_aggregation.trainable_variables)
        
        all_vars.extend(self.input_embedding.trainable_variables)
        all_vars.extend(self.feature_norm.trainable_variables)
        
        for block in self.blocks:
            all_vars.extend(block.trainable_variables)
        
        all_vars.extend(self.output_projection.trainable_variables)
        return all_vars

    def call(self, inputs, training=False):
        y, h = inputs 
        
        h = tf.squeeze(h, axis=[2, 3])  # [B, K, M, S, F]
        B = tf.shape(h)[0]
        S = tf.shape(h)[3]
        
        # Reshape to RBs
        h_rb = tf.reshape(h, [B, self.K, self.M, S, self.num_rb, self.rb_size])
        # Shape: [B, K, M, S, num_rb, rb_size]
        
        # ✅ RB aggregation: Learned or Simple
        if self.use_learned_rb_aggregation:
            # Apply dense layer to compress rb_size → 1
            # Process real and imaginary separately
            h_real = tf.math.real(h_rb)
            h_imag = tf.math.imag(h_rb)
            
            # Dense layer learns weighted combination of subcarriers
            h_real_agg = self.rb_aggregation(h_real)  # [B,K,M,S,num_rb,1]
            h_imag_agg = self.rb_aggregation(h_imag)  # [B,K,M,S,num_rb,1]
            
            # Squeeze and recombine
            h_real_agg = tf.squeeze(h_real_agg, axis=-1)
            h_imag_agg = tf.squeeze(h_imag_agg, axis=-1)
            h_avg = tf.complex(h_real_agg, h_imag_agg)
        else:
            # Simple averaging (baseline)
            h_avg = tf.reduce_mean(h_rb, axis=-1)
        
        # Reshape for transformer
        h_trans = tf.transpose(h_avg, perm=[0, 3, 4, 1, 2])
        h_flat = tf.reshape(h_trans, [B*S, self.num_rb, self.K * self.M])
        
        # Convert to real features
        feat_real = tf.math.real(h_flat)
        feat_imag = tf.math.imag(h_flat)
        feat = tf.concat([
            tf.cast(feat_real, tf.float32), 
            tf.cast(feat_imag, tf.float32)
        ], axis=-1)
        
        feat = self.feature_norm(feat)
        
        # Transformer
        x = self.input_embedding(feat)
        for block in self.blocks:
            x = block(x, training=training)
        
        out = self.output_projection(x)
        
        # Reshape to precoding matrix
        out = tf.reshape(out, [B, S, self.num_rb, self.M, self.K, 2])
        w = tf.complex(out[..., 0], out[..., 1])
        
        # ✅ RZF-COMPATIBLE NORMALIZATION
        power_per_user = tf.reduce_sum(tf.abs(w)**2, axis=[3,4], keepdims=True)
        k_factor = tf.cast(tf.sqrt(tf.cast(self.K, tf.float32)), w.dtype)
        w_normalized = w / tf.cast(tf.sqrt(power_per_user + 1e-12), w.dtype)*k_factor

        
        # Upsample
        w = tf.expand_dims(w_normalized, axis=3)
        w = tf.tile(w, [1, 1, 1, self.rb_size, 1, 1])
        w = tf.reshape(w, [B, S, self.fft_size, self.M, self.K])
        
        return tf.expand_dims(w, axis=1)