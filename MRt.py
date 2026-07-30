import tensorflow as tf
from sionna.phy.ofdm import RemoveNulledSubcarriers

class SimpleMRTPrecoder(tf.keras.layers.Layer):
    """Simple MRT precoding - no learning, just for testing"""
    def __init__(self, resource_grid, stream_management, num_tx_antennas, num_users, **kwargs):
        super().__init__(**kwargs)
        self._rg = resource_grid
        self._sm = stream_management
        self.M = num_tx_antennas
        self.K = num_users
        self.num_ofdm = resource_grid.num_ofdm_symbols
        self.fft = resource_grid.fft_size
        self.rb_size = 12
        self.num_rb = self.fft // self.rb_size
        self.remove_nulled = RemoveNulledSubcarriers(resource_grid)
    
    def call(self, x_rg, h_freq, training=None):
        batch = tf.shape(x_rg)[0]
        
        # MRT: W = conj(H^T)
        h_clean = tf.squeeze(h_freq, axis=[2, 3])  # [B, K, M, ofdm, fft]
        h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, K, M, fft]
        
        # Reshape to RB
        h_rb = tf.reshape(h_avg, [batch, self.K, self.M, self.num_rb, self.rb_size])
        h_rb = tf.transpose(h_rb, perm=[0, 1, 3, 4, 2])  # [B, K, num_rb, rb_size, M]
        
        # MRT precoding
        W_rb = tf.math.conj(h_rb)
        power = tf.reduce_sum(tf.abs(W_rb)**2, axis=[3, 4], keepdims=True)
        W_rb_normalized = W_rb / tf.cast(tf.sqrt(tf.maximum(power, 1e-10)), tf.complex64)
        
        # Convert to freq
        W_freq = tf.reshape(W_rb_normalized, [batch, self.K, self.fft, self.M])
        W_freq = tf.transpose(W_freq, perm=[0, 1, 3, 2])  # [B, K, M, fft]
        W_freq = tf.expand_dims(W_freq, axis=3)
        W_full = tf.tile(W_freq, [1, 1, 1, self.num_ofdm, 1])
        
        # Apply precoding
        x_data = tf.squeeze(x_rg, axis=2)  # [B, K, ofdm, fft]
        x_precoded_per_user = W_full * tf.expand_dims(x_data, axis=2)
        x_precoded = tf.reduce_sum(x_precoded_per_user, axis=1)
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        # Effective channel
        h_expanded = tf.expand_dims(h_clean, axis=2)  # [B, K_rx, 1, M, ofdm, fft]
        W_expanded = tf.expand_dims(W_full, axis=1)   # [B, 1, K_tx, M, ofdm, fft]
        
        h_eff_full = tf.reduce_sum(h_expanded * tf.math.conj(W_expanded), axis=3)
        h_eff_sionna = tf.transpose(h_eff_full, perm=[0, 2, 1, 3, 4])
        h_eff_sionna = tf.expand_dims(h_eff_sionna, axis=3)
        h_eff_sionna = tf.expand_dims(h_eff_sionna, axis=4)
        
        h_eff_final = self.remove_nulled(h_eff_sionna)
        
        return x_precoded, h_eff_final, W_rb_normalized