"""
WMMSE (Weighted Minimum Mean Square Error) Precoder for Sionna
Fixed to handle Sionna's 7D channel format correctly
"""

import tensorflow as tf
from sionna.phy.mimo import StreamManagement


class WMMSEPrecoder(tf.keras.layers.Layer):
    """
    WMMSE Precoding for Multi-User MIMO Downlink
    Compatible with Sionna's 7D channel format: [B, num_rx, num_tx, 1, num_tx_streams, num_ofdm, fft]
    
    Simplified implementation that returns effective channel in Sionna format
    """
    
    def __init__(self, 
                 resource_grid,
                 stream_management,
                 num_iterations=5,  # Reduced for faster computation
                 power_constraint=1.0,
                 return_effective_channel=True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._num_iterations = num_iterations
        self._power_constraint = power_constraint
        self._return_effective_channel = return_effective_channel
        
        # Get system dimensions
        self._num_tx = stream_management.num_tx
        self._num_streams_per_tx = stream_management.num_streams_per_tx
        self._num_rx = stream_management.num_rx_per_tx
        
        print(f"\n[WMMSE PRECODER CONFIG]")
        print(f"  Iterations: {num_iterations}")
        print(f"  Power constraint: {power_constraint}")
        print(f"  TX antennas: {self._num_tx}")
        print(f"  RX antennas: {self._num_rx}")
        print(f"  Streams per TX: {self._num_streams_per_tx}")
    
    def _clean_channel_format(self, h_freq):
        """
        Convert Sionna 7D format to cleaner format
        
        Input: [B, num_rx, num_tx, 1, num_tx_streams, num_ofdm, fft]
        Output: [B, num_rx, num_tx, fft] (averaged over OFDM)
        """
        # Squeeze singleton dimensions
        if len(h_freq.shape) == 7:
            h_clean = tf.squeeze(h_freq, axis=[1, 3])  # [B, num_rx, num_tx, num_ofdm, fft]
        else:
            h_clean = h_freq
        
        # Average over OFDM symbols
        h_avg = tf.reduce_mean(h_clean, axis=-2)  # [B, num_rx, num_tx, fft]
        
        return h_avg
    
    def _zero_forcing_precoder(self, h):
        """
        Simple Zero-Forcing precoder initialization
        
        Args:
            h: [B, num_rx, num_tx, fft]
        
        Returns:
            W: [B, num_tx, num_rx, fft] - ZF precoding matrix
        """
        # For each frequency bin: W = H^H (H H^H)^-1
        
        # H^H: [B, num_tx, num_rx, fft]
        h_herm = tf.transpose(h, perm=[0, 2, 1, 3], conjugate=True)
        
        # For each subcarrier: compute H H^H
        # Expand dims for matmul: [B, fft, num_rx, num_tx]
        h_batch = tf.transpose(h, perm=[0, 3, 1, 2])  # [B, fft, num_rx, num_tx]
        h_herm_batch = tf.transpose(h_herm, perm=[0, 3, 1, 2])  # [B, fft, num_tx, num_rx]
        
        # H H^H: [B, fft, num_rx, num_rx]
        h_hh = tf.matmul(h_batch, h_herm_batch)
        
        # Add regularization
        eye = tf.eye(self._num_rx, dtype=h_hh.dtype)
        eye = tf.reshape(eye, [1, 1, self._num_rx, self._num_rx])
        h_hh_reg = h_hh + 1e-6 * eye
        
        # Invert: [B, fft, num_rx, num_rx]
        h_hh_inv = tf.linalg.inv(h_hh_reg)
        
        # W = H^H (H H^H)^-1: [B, fft, num_tx, num_rx]
        w_batch = tf.matmul(h_herm_batch, h_hh_inv)
        
        # Reshape back: [B, num_tx, num_rx, fft]
        w = tf.transpose(w_batch, perm=[0, 2, 3, 1])
        
        return w
    
    def _wmmse_iteration(self, W, h, noise_var):
        """
        One WMMSE iteration: Update U, Lambda, W
        
        Args:
            W: [B, num_tx, num_rx, fft] - current precoding matrix
            h: [B, num_rx, num_tx, fft] - channel
            noise_var: scalar (float32)
        
        Returns:
            W_new: [B, num_tx, num_rx, fft] - updated precoding matrix
        """
        batch_size = tf.shape(h)[0]
        fft_size = tf.shape(h)[3]
        
        # Convert noise_var to complex to match matrix dtype
        noise_var_complex = tf.cast(noise_var, h.dtype)
        
        # Reshape for batch processing
        h_batch = tf.transpose(h, perm=[0, 3, 1, 2])  # [B, fft, num_rx, num_tx]
        W_batch = tf.transpose(W, perm=[0, 3, 1, 2])  # [B, fft, num_tx, num_rx]
        
        # 1. Update equalizers U_k for each user
        # R_k = sum_j H_k W_j W_j^H H_k^H + sigma^2 I
        # U_k = R_k^-1 H_k W_k
        
        # Compute H W: [B, fft, num_rx, num_rx]
        HW = tf.matmul(h_batch, W_batch)
        
        # Interference covariance: sum_j |H_k W_j|^2
        HWW = tf.matmul(HW, tf.transpose(HW, perm=[0, 1, 3, 2], conjugate=True))
        
        # Add noise (with correct type)
        eye = tf.eye(self._num_rx, dtype=h.dtype)
        eye = tf.reshape(eye, [1, 1, self._num_rx, self._num_rx])
        R = HWW + noise_var_complex * eye  # ✅ FIXED: cast noise_var
        
        # Regularize and invert
        R_reg = R + tf.cast(1e-8, h.dtype) * eye  # ✅ FIXED: cast regularization too
        R_inv = tf.linalg.inv(R_reg)  # [B, fft, num_rx, num_rx]
        
        # U = R^-1 H W: [B, fft, num_rx, num_rx]
        U_batch = tf.matmul(R_inv, HW)
        
        # 2. Update weights Lambda_k = (I - U_k^H H_k W_k)^-1
        # For simplicity: lambda_k = 1 / (1 - tr(U_k^H H_k W_k))
        
        # U^H H W: [B, fft, num_rx, num_rx]
        UHW = tf.matmul(tf.transpose(U_batch, perm=[0, 1, 3, 2], conjugate=True), HW)
        
        # Diagonal elements (MSE for each stream)
        mse = 1.0 - tf.linalg.diag_part(tf.math.real(UHW))  # [B, fft, num_rx]
        
        # Lambda = 1 / mse
        Lambda = 1.0 / (mse + 1e-8)  # [B, fft, num_rx]
        Lambda = tf.clip_by_value(Lambda, 0.1, 10.0)  # Numerical stability
        
        # 3. Update precoding matrix W
        # W = (sum_k H_k^H U_k Lambda_k U_k^H H_k + (sigma^2/P) I)^-1 (sum_k H_k^H U_k Lambda_k)
        
        # Expand Lambda for matrix operations: [B, fft, num_rx, num_rx] diagonal
        Lambda_diag = tf.linalg.diag(tf.cast(Lambda, h.dtype))
        
        # H^H: [B, fft, num_tx, num_rx]
        h_herm_batch = tf.transpose(h_batch, perm=[0, 1, 3, 2], conjugate=True)
        
        # A = H^H U Lambda U^H H
        U_Lambda = tf.matmul(U_batch, Lambda_diag)  # [B, fft, num_rx, num_rx]
        U_Lambda_Uh = tf.matmul(U_Lambda, tf.transpose(U_batch, perm=[0, 1, 3, 2], conjugate=True))
        A = tf.matmul(tf.matmul(h_herm_batch, U_Lambda_Uh), h_batch)  # [B, fft, num_tx, num_tx]
        
        # Add regularization: (sigma^2 / P) I
        eye_tx = tf.eye(self._num_tx, dtype=h.dtype)
        eye_tx = tf.reshape(eye_tx, [1, 1, self._num_tx, self._num_tx])
        reg_term = (noise_var_complex / tf.cast(self._power_constraint, h.dtype)) * eye_tx  # ✅ FIXED
        A = A + reg_term
        
        # B = H^H U Lambda
        B = tf.matmul(h_herm_batch, U_Lambda)  # [B, fft, num_tx, num_rx]
        
        # Solve: W = A^-1 B
        W_new_batch = tf.linalg.solve(A, B)  # [B, fft, num_tx, num_rx]
        
        # Reshape back: [B, num_tx, num_rx, fft]
        W_new = tf.transpose(W_new_batch, perm=[0, 2, 3, 1])
        
        return W_new    
    def _normalize_power(self, W):
        """
        Normalize precoding matrix to satisfy power constraint
        
        Args:
            W: [B, num_tx, num_rx, fft]
        
        Returns:
            W_normalized: [B, num_tx, num_rx, fft]
        """
        # Compute total power: sum over all dimensions except batch
        power = tf.reduce_sum(tf.abs(W) ** 2, axis=[1, 2, 3], keepdims=True)
        
        # Normalize
        scaling = tf.sqrt(self._power_constraint / (power + 1e-12))
        W_normalized = W * tf.cast(scaling, W.dtype)
        
        return W_normalized
    
    def _convert_to_sionna_format(self, W, num_ofdm_symbols):
        """
        Convert precoding matrix to Sionna format
        
        Args:
            W: [B, num_tx, num_rx, fft]
            num_ofdm_symbols: number of OFDM symbols
        
        Returns:
            W_sionna: [B, 1, num_rx, 1, num_tx, num_ofdm, fft]
        """
        # Expand dimensions
        W = tf.expand_dims(W, axis=1)  # [B, 1, num_tx, num_rx, fft]
        W = tf.expand_dims(W, axis=3)  # [B, 1, num_tx, 1, num_rx, fft]
        
        # Transpose to get [B, 1, num_rx, 1, num_tx, fft]
        W = tf.transpose(W, perm=[0, 1, 4, 3, 2, 5])
        
        # Replicate across OFDM symbols
        W = tf.expand_dims(W, axis=-2)  # [B, 1, num_rx, 1, num_tx, 1, fft]
        W = tf.repeat(W, num_ofdm_symbols, axis=-2)  # [B, 1, num_rx, 1, num_tx, ofdm, fft]
        
        return W
    
    @tf.function
    def call(self, x, h_freq, **kwargs):
        """
        Apply WMMSE precoding
        
        Args:
            x: Input symbols [B, 1, num_rx, num_ofdm, fft]
            h_freq: Channel [B, 1, num_rx, 1, num_tx, num_ofdm, fft]
        
        Returns:
            x_precoded: [B, 1, num_tx, num_ofdm, fft]
            h_eff: Effective channel [B, 1, num_rx, 1, num_rx, num_ofdm, fft]
        """
        # Get dimensions
        batch_size = tf.shape(x)[0]
        num_ofdm_symbols = tf.shape(x)[3]
        fft_size = tf.shape(x)[4]
        
        # Clean channel format: [B, num_rx, num_tx, fft]
        h_clean = self._clean_channel_format(h_freq)
        
        # Estimate noise variance
        noise_var = tf.constant(0.1, dtype=tf.float32)
        
        # Initialize with Zero-Forcing
        W = self._zero_forcing_precoder(h_clean)  # [B, num_tx, num_rx, fft]
        
        # WMMSE iterations
        for _ in range(self._num_iterations):
            W = self._wmmse_iteration(W, h_clean, noise_var)
        
        # Normalize power
        W = self._normalize_power(W)  # [B, num_tx, num_rx, fft]
        
        # ========== CORRECTED: Apply precoding using einsum ==========
        # x: [B, 1, num_rx, ofdm, fft]
        # W: [B, num_tx, num_rx, fft]
        # Want: x_precoded[B, 1, num_tx, ofdm, fft] = sum_rx W[B, tx, rx, fft] * x[B, 1, rx, ofdm, fft]
        
        # Remove singleton dimension from x for einsum
        x_squeezed = tf.squeeze(x, axis=1)  # [B, num_rx, ofdm, fft]
        
        # Apply precoding: sum over num_rx dimension
        # einsum: 'btrf,brof->btof'
        # b=batch, t=num_tx, r=num_rx, o=ofdm, f=fft
        x_precoded = tf.einsum('btrf,brof->btof', W, x_squeezed)  # [B, num_tx, ofdm, fft]
        
        # Add back the singleton dimension to match Sionna format
        x_precoded = tf.expand_dims(x_precoded, axis=1)  # [B, 1, num_tx, ofdm, fft]
        
        # ========== Compute effective channel ==========
        if self._return_effective_channel:
            # h_eff = H * W
            # h: [B, num_rx, num_tx, fft]
            # W: [B, num_tx, num_rx, fft]
            
            # Reshape for matmul: [B, fft, num_rx, num_tx] @ [B, fft, num_tx, num_rx]
            h_batch = tf.transpose(h_clean, perm=[0, 3, 1, 2])  # [B, fft, num_rx, num_tx]
            W_batch = tf.transpose(W, perm=[0, 3, 1, 2])  # [B, fft, num_tx, num_rx]
            
            # Matrix multiply: [B, fft, num_rx, num_rx]
            h_eff_batch = tf.matmul(h_batch, W_batch)
            
            # Reshape to: [B, num_rx, num_rx, fft]
            h_eff = tf.transpose(h_eff_batch, perm=[0, 2, 3, 1])
            
            # Convert to Sionna format: [B, 1, num_rx, 1, num_rx, ofdm, fft]
            h_eff = tf.expand_dims(h_eff, axis=1)  # [B, 1, num_rx, num_rx, fft]
            h_eff = tf.expand_dims(h_eff, axis=3)  # [B, 1, num_rx, 1, num_rx, fft]
            h_eff = tf.expand_dims(h_eff, axis=-2)  # [B, 1, num_rx, 1, num_rx, 1, fft]
            h_eff = tf.repeat(h_eff, num_ofdm_symbols, axis=-2)  # [B, 1, num_rx, 1, num_rx, ofdm, fft]
            
            return x_precoded, h_eff
        else:
            return x_precoded