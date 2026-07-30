import os
import sys

import tensorflow as tf
import numpy as np
from sionna.phy.ofdm import RemoveNulledSubcarriers
from tensorflow.keras import layers, Model

# Put this in transformer_5d_sionna_rebuilt.py or inline in your script.

import tensorflow as tf
from tensorflow.keras import layers

class SwiGLU(tf.keras.layers.Layer):
    def call(self, x):
        x_a, x_b = tf.split(x, num_or_size_splits=2, axis=-1)
        swish = x_a * tf.sigmoid(x_a)
        return swish * x_b

class TransformerBlock(layers.Layer):
    def __init__(self, input_dim=128, num_heads=12, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_dim // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2),
            SwiGLU(),
            layers.Dropout(dropout),
            layers.Dense(input_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + self.dropout(attn, training=training))
        ff = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout(ff, training=training))
        return x

class StackedTransformer5D_OldStyle(tf.keras.Model):
    """
    Reconstructs the old 5D Transformer behavior, but inputs Sionna h_freq and outputs
    RB-based per-user precoder.
    """
    def __init__(self,
                 num_tx_antennas=8,
                 num_rx_antennas=4,        # number of users / streams
                 num_ofdm_symbols=14,
                 fft_size=72,
                 rb_size=12,
                 embedding_dim=128,
                 num_heads=4,
                 hidden_dim=1024,
                 num_layers=4,
                 dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.M = num_tx_antennas
        self.U = num_rx_antennas
        self.ofdm = num_ofdm_symbols
        self.fft = fft_size
        self.rb_size = rb_size
        assert fft_size % rb_size == 0, "fft_size must be divisible by rb_size"
        self.num_rb = fft_size // rb_size

        # token dim per RB per user: real+imag of (M * rb_size)
        self.token_dim = 2 * (self.M * self.rb_size)

        # Embedding & transformer
        self.embedding = layers.Dense(embedding_dim, name="embed_dense")
        self.transformer_blocks = [
            TransformerBlock(input_dim=embedding_dim, num_heads=num_heads,
                             hidden_dim=hidden_dim, dropout=0.0)
            for _ in range(num_layers)
        ]

        # Output layer
        self.output_layer = layers.Dense(self.token_dim, name="out_dense")

    def call(self, h_freq, training=False):
        """
        h_freq: Sionna style [B, 1, num_rx, 1, num_tx, num_ofdm, fft]
        Returns:
          W_user_rb: [B, U, num_rb, rb_size, M] (complex) - per-user RB precoder
        """
        # 1) Clean Sionna dims -> [B, U, M, ofdm, fft]
        h_clean = tf.squeeze(h_freq, axis=[1, 3])    # -> [B, U, M, ofdm, fft]
        B = tf.shape(h_clean)[0]
        U = tf.shape(h_clean)[1]
        M = tf.shape(h_clean)[2]
        ofdm = tf.shape(h_clean)[3]
        fft = tf.shape(h_clean)[4]

        # 2) Average over OFDM symbols
        h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, U, M, fft]

        # 3) Reshape into RB bins: [B, U, M, num_rb, rb_size]
        h_rb = tf.reshape(h_avg, [B, U, M, self.num_rb, self.rb_size])

        # 4) Reorder to [B, U, num_rb, rb_size, M] then flatten for tokens
        h_rb_reordered = tf.transpose(h_rb, perm=[0, 1, 3, 4, 2])  # [B, U, RB, rb_size, M]
        
        # Flatten to [B*U, RB, rb_size*M]
        h_flat = tf.reshape(h_rb_reordered, [B * U, self.num_rb, self.rb_size * M])
        
        # Split into real and imaginary
        h_real = tf.math.real(h_flat)
        h_imag = tf.math.imag(h_flat)
        tokens = tf.concat([h_real, h_imag], axis=-1)  # [B*U, RB, 2*rb_size*M]

        # 5) Embedding & transformer
        x = self.embedding(tokens)  # [B*U, RB, embedding_dim]
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # 6) Output dense -> split into real+imag
        out = self.output_layer(x)  # [B*U, RB, token_dim]
        half = self.token_dim // 2
        out_r = out[:, :, :half]
        out_i = out[:, :, half:]
        precoder_tokens = tf.complex(out_r, out_i)  # [B*U, RB, rb_size*M]

        # 7) Reshape back to [B, U, RB, rb_size, M]
        precoder_tokens = tf.reshape(precoder_tokens, [B, U, self.num_rb, self.rb_size, M])

        # 8) Normalize per RB per stream
        norm = tf.norm(precoder_tokens, axis=[3, 4], keepdims=True)  # [B, U, RB, 1, 1]
        W_user_rb = precoder_tokens / (norm + 1e-12)

        #tf.print("StackedTransformer5D output shape:", tf.shape(W_user_rb), "expected: [B, U, num_rb, rb_size, M]", output_stream=sys.stderr)

        return W_user_rb  # [B, U, num_rb, rb_size, M]


class TransformerPrecoder5D_Fixed(tf.keras.layers.Layer):
    """
    Layer used by Sionna pipeline. Converts 5D precoder to Sionna format.
    ✅ CORRECT POWER NORMALIZATION
    """
    def __init__(self, resource_grid, stream_management, num_tx_antennas=8,
                 num_rx_antennas=4, rb_size=4, **kwargs):
        super().__init__(**kwargs)
        self._rg = resource_grid
        self._sm = stream_management
        self.M = num_tx_antennas
        self.U = num_rx_antennas
        self.rb_size = rb_size
        self.num_ofdm = resource_grid.num_ofdm_symbols
        self.fft = resource_grid.fft_size
        self.num_rb = self.fft // self.rb_size

        # Internal transformer that outputs [B, U, RB, rb_size, M]
        self.transformer = StackedTransformer5D_OldStyle(
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            num_ofdm_symbols=self.num_ofdm,
            fft_size=self.fft,
            rb_size=self.rb_size
        )

        # Utility to remove nulled carriers
        self.remove_nulled = RemoveNulledSubcarriers(resource_grid)

    def call(self, x_rg, h_freq, training=None):
        """
        ✅ CORRECT: Normalize to unit power per user
        """
        batch = tf.shape(x_rg)[0]

        # 1) Get per-user RB precoder from transformer
        W_user_rb = self.transformer(h_freq, training=training)  # [B, U, RB, rb_size, M]
        
        # ✅ CORRECT NORMALIZATION
        # Total power per user = sum of |W|² over all dimensions
        power_per_user = tf.reduce_sum(
            tf.abs(W_user_rb) ** 2,
            axis=[2, 3, 4],  # Sum over [RB, rb_size, M]
            keepdims=True
        )  # [B, U, 1, 1, 1]
        
        # Normalize so each user has total power = 1.0
        W_user_rb_normalized = W_user_rb / tf.cast(
            tf.sqrt(power_per_user + 1e-12),
            tf.complex64
        )
        
        # ✅ OPTIONAL: Verify power during training
        if training:
            actual_power = tf.reduce_sum(
                tf.abs(W_user_rb_normalized) ** 2, 
                axis=[2, 3, 4]
            )
            tf.print("Power per user:", actual_power[0], 
                     "Target: [1.0, 1.0, 1.0, 1.0]", 
                     output_stream=sys.stderr)
        
        # Store for sum rate calculation
        W_for_sum_rate = W_user_rb_normalized
        
        # 2) Convert to frequency domain
        W_user_freq = tf.reshape(W_user_rb_normalized, [batch, self.U, self.fft, self.M])
        W_user_freq = tf.transpose(W_user_freq, perm=[0, 1, 3, 2])  # [B, U, M, fft]
        
        # 3) Expand to OFDM symbols
        W_user_freq = tf.expand_dims(W_user_freq, axis=3)
        W_full = tf.tile(W_user_freq, [1, 1, 1, self.num_ofdm, 1])  # [B, U, M, ofdm, fft]
        
        # ✅ NO ADDITIONAL SCALING
        W_scaled = W_full
        
        # 4) Apply precoding
        x_data = tf.squeeze(x_rg, axis=1)
        x_precoded = tf.einsum('bsmof,bsof->bmof', W_scaled, x_data)
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        # 5) Effective channel
        h_clean = tf.squeeze(h_freq, axis=[1, 3])
        h_eff = tf.einsum('brmof,bsmof->brsof', h_clean, tf.math.conj(W_scaled))
        h_eff = tf.expand_dims(h_eff, axis=1)
        h_eff = tf.expand_dims(h_eff, axis=3)
        h_eff = self.remove_nulled(h_eff)

        return x_precoded, h_eff, W_for_sum_rate
