import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import datetime
from evaluate_per_user import *
import time

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)
from wmmse_precoder import WMMSEPrecoder

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration failed: {e}")

# Sionna imports - CORRECTED

import sionna
sionna.phy.config.seed = 42

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer,
                             OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers)
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, ApplyOFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber
print(f"✅ Sionna version: {sionna.__version__}")

from transformer_5d_sionna import TransformerPrecoder5D_Fixed
def plot_constellation_debug(simulator, batch_size=64, ebno_db=10.0, save_path=None):
    """
    Plot constellation points at different stages:
    1. Transmitted symbols (after mapping)
    2. Received symbols (before equalization)
    3. Equalized symbols (after LMMSE equalization)
    4. Decoded symbols (after demapping and decoding)
    
    This helps diagnose where the constellation is getting distorted
    """
    ebno_tf = tf.constant(ebno_db, dtype=tf.float32)
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
    
    # Get noise power
    no = simulator.get_noise_power(ebno_tf) if hasattr(simulator, 'get_noise_power') else \
         ebnodb2no(ebno_tf, simulator._num_bits_per_symbol, simulator._coderate, simulator._rg)
    
    # === FORWARD PASS WITH INTERMEDIATE OUTPUTS ===
    b = simulator._binary_source([batch_size, 1, simulator._num_streams_per_tx, simulator._k])
    c = simulator._encoder(b)
    x_original = simulator._mapper(c)  # Original transmitted symbols
    x_rg = simulator._rg_mapper(x_original)
    
    # Channel
    cir = simulator._cdl(batch_size, simulator._rg.num_ofdm_symbols, 
                         1 / simulator._rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(simulator._frequencies, *cir, normalize=True)
    
    # Precoding
    if simulator._precoder_type == "transformer_5d":
        x_rg_precoded, g, _ = simulator._transformer_precoder(x_rg, h_freq)
    elif simulator._precoder_type == "wmmse":
        x_rg_precoded, g = simulator._wmmse_precoder(x_rg, h_freq)
    else:
        x_rg_precoded, g = simulator._zf_precoder(x_rg, h_freq)
    
    # Apply channel (received signal)
    y = simulator._channel_freq(x_rg_precoded, h_freq, no)
    
    # Equalization
    if simulator._perfect_csi:
        h_hat = g
        err_var = 0.0
    else:
        h_hat, err_var = simulator._ls_est(y, no)
    
    x_hat_equalized, no_eff = simulator._lmmse_equ(y, h_hat, err_var, no)
    
    # Demapping and decoding
    llr = simulator._demapper(x_hat_equalized, no_eff)
    b_hat = simulator._decoder(llr)
    c_hat = simulator._encoder(b_hat)  # Re-encode to get transmitted codewords
    x_decoded = simulator._mapper(c_hat)  # Map decoded bits back to symbols
    
    # === EXTRACT SYMBOLS FOR PLOTTING ===
    # Get data symbols only (remove pilots)
    def extract_data_symbols(x_rg_tensor):
        """Extract data symbols from resource grid"""
        # x_rg shape: [batch, num_tx, num_streams, num_ofdm_symbols, fft_size]
        x_squeezed = tf.squeeze(x_rg_tensor, axis=1)  # Remove tx dimension
        # Extract data subcarriers (skip pilots)
        data_mask = simulator._rg.pilot_pattern.mask
        # Flatten and extract
        x_flat = tf.reshape(x_squeezed, [-1])
        return x_flat.numpy()
    
    # Original symbols (transmitted)
    x_orig_np = x_original.numpy().flatten()
    
    # Equalized symbols
    x_eq_np = x_hat_equalized.numpy().flatten()
    
    # Decoded symbols
    x_dec_np = x_decoded.numpy().flatten()
    
    # Take subset for plotting (too many points otherwise)
    n_points = min(2000, len(x_orig_np))
    idx = np.random.choice(len(x_orig_np), n_points, replace=False)
    
    x_orig_plot = x_orig_np[idx]
    x_eq_plot = x_eq_np[idx]
    x_dec_plot = x_dec_np[idx]
    
    # === CREATE PLOTS ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original Transmitted Symbols
    ax1 = axes[0, 0]
    ax1.scatter(x_orig_plot.real, x_orig_plot.imag, alpha=0.5, s=10, c='blue')
    ax1.set_xlabel('In-Phase', fontsize=11)
    ax1.set_ylabel('Quadrature', fontsize=11)
    ax1.set_title('Transmitted Symbols\n(After Mapper, Before Precoding)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_aspect('equal')
    
    # Plot ideal QPSK constellation
    qpsk_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    ax1.scatter(qpsk_points.real, qpsk_points.imag, 
               c='red', s=100, marker='x', linewidths=3, label='Ideal QPSK')
    ax1.legend()
    
    # Plot 2: Received Symbols (before equalization)
    ax2 = axes[0, 1]
    y_np = y.numpy()
    y_plot = y_np.flatten()[idx]
    ax2.scatter(y_plot.real, y_plot.imag, alpha=0.3, s=10, c='orange')
    ax2.scatter(qpsk_points.real, qpsk_points.imag, 
               c='red', s=100, marker='x', linewidths=3, label='Ideal QPSK')
    ax2.set_xlabel('In-Phase', fontsize=11)
    ax2.set_ylabel('Quadrature', fontsize=11)
    ax2.set_title('Received Symbols\n(After Channel, Before Equalization)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Plot 3: Equalized Symbols
    ax3 = axes[0, 2]
    ax3.scatter(x_eq_plot.real, x_eq_plot.imag, alpha=0.5, s=10, c='green')
    ax3.scatter(qpsk_points.real, qpsk_points.imag, 
               c='red', s=100, marker='x', linewidths=3, label='Ideal QPSK')
    ax3.set_xlabel('In-Phase', fontsize=11)
    ax3.set_ylabel('Quadrature', fontsize=11)
    ax3.set_title('Equalized Symbols\n(After LMMSE Equalization)', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.set_aspect('equal')
    ax3.legend()
    
    # Plot 4: Decoded Symbols
    ax4 = axes[1, 0]
    ax4.scatter(x_dec_plot.real, x_dec_plot.imag, alpha=0.5, s=10, c='purple')
    ax4.scatter(qpsk_points.real, qpsk_points.imag, 
               c='red', s=100, marker='x', linewidths=3, label='Ideal QPSK')
    ax4.set_xlabel('In-Phase', fontsize=11)
    ax4.set_ylabel('Quadrature', fontsize=11)
    ax4.set_title('Decoded Symbols\n(After Decoder + Re-mapping)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.set_aspect('equal')
    ax4.legend()
    
    # Plot 5: Error Vector Magnitude (EVM)
    ax5 = axes[1, 1]
    evm = np.abs(x_eq_plot - x_orig_plot)
    ax5.hist(evm, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.set_xlabel('Error Magnitude', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title(f'Error Vector Magnitude\nMean EVM: {np.mean(evm):.4f}', 
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: BER Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate BER
    ber_total = compute_ber(b, b_hat).numpy()
    
    # Calculate symbol error rate
    symbol_errors = np.sum(np.abs(x_dec_plot - x_orig_plot) > 0.1)
    ser = symbol_errors / len(x_orig_plot)
    
    stats_text = (
        f"Statistics @ Eb/N0 = {ebno_db:.1f} dB\n"
        f"{'='*40}\n\n"
        f"Precoder Type: {simulator._precoder_type}\n"
        f"Batch Size: {batch_size}\n"
        f"Num Symbols Plotted: {n_points}\n\n"
        f"Performance Metrics:\n"
        f"  BER: {ber_total:.2e}\n"
        f"  SER: {ser:.2e}\n"
        f"  Mean EVM: {np.mean(evm):.4f}\n"
        f"  Std EVM: {np.std(evm):.4f}\n\n"
        f"Constellation Analysis:\n"
        f"  TX Power: {np.mean(np.abs(x_orig_plot)**2):.4f}\n"
        f"  RX Power: {np.mean(np.abs(x_eq_plot)**2):.4f}\n"
        f"  Noise Var: {float(no):.2e}\n"
    )
    
    ax6.text(0.1, 0.9, stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            family='monospace')
    
    plt.suptitle(f'Constellation Debugging: {simulator._precoder_type.upper()} Precoder', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Constellation plot saved: {save_path}")
    
    plt.show()
    
    return fig, {
        'ber': ber_total,
        'ser': ser,
        'evm_mean': np.mean(evm),
        'evm_std': np.std(evm)
    }
class OFDMSimulator5D_Fixed(tf.keras.Model):
    """Corrected OFDM Simulator with 5D Transformer Precoding"""
    
    def __init__(self, cdl_model="A", delay_spread=30e-9, perfect_csi=True,
                 precoder_type="transformer_5d", transformer_model_path=None, rb_size=4):
        super().__init__()
        
        # System parameters
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 4
        self._num_bs_ant = 8
        self._num_streams_per_tx = self._num_ut_ant
        self._subcarrier_spacing = 15e3
        self._carrier_frequency = 2.6e9
        self._cyclic_prefix_length = 6
        self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bits_per_symbol = 2
        self._coderate = 0.5
        self._rb_size = rb_size
        self._perfect_csi = perfect_csi
        self._precoder_type = precoder_type
        
        # Stream management
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)
        
        # Resource grid
        self._rg = ResourceGrid(
            num_ofdm_symbols=self._num_ofdm_symbols,
            fft_size=self._fft_size,
            subcarrier_spacing=self._subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self._num_streams_per_tx,
            cyclic_prefix_length=self._cyclic_prefix_length,
            num_guard_carriers=[5, 6],
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices
        )
        
        # Antenna arrays
        self._ut_array = AntennaArray(
            num_rows=1, num_cols=self._num_ut_ant//2,
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=self._carrier_frequency
        )
        self._bs_array = AntennaArray(
            num_rows=1, num_cols=self._num_bs_ant//2,
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=self._carrier_frequency
        )
        
        # Channel
        self._cdl = CDL(
            model=cdl_model, delay_spread=delay_spread,
            carrier_frequency=self._carrier_frequency,
            ut_array=self._ut_array, bs_array=self._bs_array,
            direction="downlink", min_speed=0.0
        )
        
        # Communication chain
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)
        
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        
        # Precoders
        self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)
        if precoder_type == "transformer_5d":
            self._transformer_precoder = TransformerPrecoder5D_Fixed(
                self._rg, self._sm,
                num_tx_antennas=self._num_bs_ant,
                num_rx_antennas=self._num_ut_ant,
                rb_size=rb_size
            )
        if precoder_type == "wmmse":
            self._wmmse_precoder = WMMSEPrecoder(
                self._rg, 
                self._sm,
                num_iterations=10,  # Can tune this
                power_constraint=1.0,
                return_effective_channel=True
            )

        
        # Channel and receiver
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
        print(f"\n[SIMULATOR CONFIG]")
        print(f"  FFT size: {self._fft_size}")
        print(f"  OFDM symbols: {self._num_ofdm_symbols}")
        print(f"  TX antennas: {self._num_bs_ant}")
        print(f"  RX antennas: {self._num_ut_ant}")
        print(f"  Streams: {self._num_streams_per_tx}")
        print(f"  RB size: {self._rb_size}")
        print(f"  Precoder: {precoder_type}")
    
    @tf.function
    def call(self, batch_size, ebno_db, training=False, return_llr=False):
        """
        Execute one forward pass through the system
        
        Returns:
            If return_llr=False: (b, b_hat, h_freq, g, precoding_matrix)
            If return_llr=True: (b, b_hat, h_freq, g, precoding_matrix, llr, c)
        """
        # ✅ CORRECTION: Convert batch_size to tensor
        batch_size = tf.constant(batch_size) if not isinstance(batch_size, tf.Tensor) else batch_size
        
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        
        # Transmitter
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)  # Coded bits
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        
        # Channel
        cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1 / self._rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)
        
        precoding_matrix = None
        
        # Precoding
        if self._precoder_type == "wmmse":
            x_rg_precoded, g = self._wmmse_precoder(x_rg, h_freq)
            precoding_matrix = None  # WMMSE doesn't return explicit matrix in same format
        elif self._precoder_type == "transformer_5d":
            precoder_output = self._transformer_precoder(x_rg, h_freq)
            x_rg_precoded, g, precoding_matrix = precoder_output
        else:
            x_rg_precoded, g = self._zf_precoder(x_rg, h_freq)
        
        # Apply channel
        y = self._channel_freq(x_rg_precoded, h_freq, no)
        
        # Receiver
        if self._perfect_csi:
            h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est(y, no)
        
        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)
        
        # Return with coded bits if LLR requested
        if return_llr:
            return b, b_hat, h_freq, g, precoding_matrix, llr, c
        else:
            return b, b_hat, h_freq, g, precoding_matrix

import gc
import tensorflow as tf

class EndToEndTrainer_Fixed:
    """
    Memory-Optimized Trainer
    """
    
    def __init__(self, simulator_5d, learning_rate=5e-4):
        self.simulator = simulator_5d
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        
        print(f"\n[TRAINER CONFIG - MEMORY OPTIMIZED]")
        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Loss: -SumRate + λ_ber·BCE + λ_power·||W||²")
    
    def safe_numpy(self, value):
        """Safely convert TF tensor to numpy"""
        if hasattr(value, 'numpy'):
            return float(value.numpy())
        return float(value)
    
    def get_noise_power(self, ebno_db):
        """Calculate noise power from Eb/No"""
        return ebnodb2no(
            ebno_db,
            self.simulator._num_bits_per_symbol,
            self.simulator._coderate,
            self.simulator._rg
        )
    
    def calculate_sum_rate_corrected(self, precoding_matrix, h_freq, noise_power=None):
        """Calculate sum rate - memory optimized version"""
        rb_size = self.simulator._rb_size
        
        if noise_power is None:
            noise_power = tf.constant(1e-13, dtype=tf.float32)
        
        # Clean up Sionna dimensions
        if len(h_freq.shape) == 7:
            h_clean = tf.squeeze(h_freq, axis=[1, 3])
        else:
            h_clean = h_freq
        
        # Average over OFDM symbols
        h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, R, T, fft]
        h_avg = tf.transpose(h_avg, perm=[0, 1, 3, 2])  # [B, R, fft, T]
        
        B = tf.shape(h_avg)[0]
        R = tf.shape(h_avg)[1]
        fft_size = tf.shape(h_avg)[2]
        T = tf.shape(h_avg)[3]
        
        num_rb = fft_size // rb_size
        
        # Reshape h into RB format
        h_rb = tf.reshape(h_avg, [B, R, num_rb, rb_size, T])
        
        # Handle precoding matrix format
        W_shape = tf.shape(precoding_matrix)
        
        if len(precoding_matrix.shape) == 5 and W_shape[3] == rb_size:
            W_rb = precoding_matrix
            # ✅ CRITICAL: Normalize per RB to prevent power explosion
            norm_per_rb = tf.norm(W_rb, axis=[3, 4], keepdims=True)
            W_rb_normalized = W_rb / (norm_per_rb + 1e-12)
        else:
            W = tf.squeeze(precoding_matrix, axis=[1, 3]) if len(precoding_matrix.shape) == 7 else precoding_matrix
            W_avg = tf.reduce_mean(W, axis=3)
            W_avg = tf.transpose(W_avg, perm=[0, 1, 3, 2])
            
            S = tf.shape(W_avg)[1]
            W_rb = tf.reshape(W_avg, [B, S, num_rb, rb_size, T])
            
            norm_per_rb = tf.norm(W_rb, axis=[3, 4], keepdims=True)
            W_rb_normalized = W_rb / (norm_per_rb + 1e-12)
        
        # Flatten to frequency dimension
        F = num_rb * rb_size
        h_flat = tf.reshape(h_rb, [B, R, F, T])
        W_flat = tf.reshape(W_rb_normalized, [B, R, F, T])
        
        # Calculate SINR - memory efficient version
        # Use einsum instead of matmul to save memory
        W_eff = tf.einsum('bufi,bujf->buij', h_flat, tf.math.conj(
            tf.transpose(W_flat, perm=[0, 1, 3, 2])))
        
        diag_W = tf.linalg.diag_part(tf.abs(W_eff) ** 2)
        total_power = tf.reduce_sum(tf.abs(W_eff) ** 2, axis=3)
        interference = total_power - diag_W
        
        SINR = diag_W / (interference + noise_power + 1e-12)
        rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)
        
        # Calculate rate per user
        rate_per_user_per_batch = tf.reduce_sum(rate_per_subcarrier, axis=2)
        rate_per_user = tf.reduce_mean(rate_per_user_per_batch, axis=0)
        
        # Total sum rate
        sum_rate_per_batch = tf.reduce_sum(rate_per_subcarrier, axis=[1, 2])
        avg_sum_rate = tf.reduce_mean(sum_rate_per_batch)
        
        avg_sinr = tf.reduce_mean(SINR)
        avg_sinr_db = 10.0 * tf.math.log(avg_sinr + 1e-12) / tf.math.log(10.0)
        
        metrics = {
            'sinr_db': avg_sinr_db,
            'signal_power': tf.reduce_mean(diag_W),
            'interference_power': tf.reduce_mean(interference),
            'rate_per_subcarrier': tf.reduce_mean(rate_per_subcarrier),
            'rate_per_user': rate_per_user
        }
        
        return avg_sum_rate, metrics
    
    @tf.function
    def compute_power_constraint_loss(self, precoding_matrix):
        """Soft power constraint"""
        power_per_rb = tf.reduce_sum(
            tf.abs(precoding_matrix)**2, 
            axis=[3, 4]
        )
        power_deviation = tf.reduce_mean((power_per_rb - 1.0)**2)
        return power_deviation
    
    @tf.function
    def compute_bce_loss(self, coded_bits, llr):
        """Binary Cross-Entropy loss"""
        coded_bits_flat = tf.reshape(coded_bits, [-1])
        llr_flat = tf.reshape(llr, [-1])
        labels = tf.cast(coded_bits_flat, tf.float32)
        
        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=llr_flat
        )
        
        return tf.reduce_mean(bce_loss)
    
    @tf.function(reduce_retracing=True)  # ✅ FIX: Reduce retracing
    def train_step(self, batch_size, ebno_db, lambda_ber, lambda_power):
        """
        FIXED Training Step - No OOM
        
        Loss = -SumRate + λ_ber·BCE + λ_power·PowerConstraint
        
        NOTE: lambda_ber is now passed as argument (no Python int in @tf.function)
        """
        batch_size_tensor = tf.constant(batch_size, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            # Forward pass with LLRs
            b, b_hat, h_freq, g, precoding_matrix, llr, c = self.simulator(
                batch_size_tensor, ebno_db, training=True, return_llr=True
            )
            
            # 1. MAIN OBJECTIVE: Maximize Sum Rate
            noise_power = self.get_noise_power(ebno_db)
            sum_rate, metrics = self.calculate_sum_rate_corrected(
                precoding_matrix, h_freq, noise_power
            )
            rate_loss = -sum_rate
            
            # 2. SOFT DECODING LOSS
            bce_loss = self.compute_bce_loss(c, llr)
            
            # 3. POWER CONSTRAINT
            power_loss = self.compute_power_constraint_loss(precoding_matrix)
            
            # 4. TOTAL LOSS
            total_loss = rate_loss + lambda_ber * bce_loss + lambda_power * power_loss
            
            # Metrics
            ber_total = compute_ber(b, b_hat)
            ber_per_user = self._compute_ber_per_user_tf(b, b_hat)
        
        # Backpropagation with gradient clipping
        gradients = tape.gradient(total_loss, self.simulator.trainable_variables)
        
        # ✅ FIX: More aggressive gradient clipping to prevent memory spikes
        gradients = [
            tf.clip_by_norm(g, 10) if g is not None else g 
            for g in gradients
        ]
        
        self.optimizer.apply_gradients(
            zip(gradients, self.simulator.trainable_variables)
        )
        
        # Extended metrics
        metrics.update({
            'rate_loss': rate_loss,
            'bce_loss': bce_loss,
            'power_loss': power_loss,
            'lambda_ber': lambda_ber
        })
        
        return total_loss, ber_total, sum_rate, metrics, ber_per_user
    
    @tf.function
    def _compute_ber_per_user_tf(self, b, b_hat):
        """Compute BER per user"""
        b = tf.squeeze(b, axis=1)
        b_hat = tf.squeeze(b_hat, axis=1)
        
        batch_size = tf.cast(tf.shape(b)[0], tf.float32)
        k = tf.cast(tf.shape(b)[2], tf.float32)
        
        errors_per_bit = tf.not_equal(b, b_hat)
        errors_per_user = tf.reduce_sum(tf.cast(errors_per_bit, tf.float32), axis=[0, 2])
        
        total_bits_per_user = batch_size * k
        ber_per_user = errors_per_user / total_bits_per_user
        
        return ber_per_user
    
    def train(self, num_epochs=100, batch_size=128, ebno_db_range=(0, 10)):  # ✅ Reduced batch_size
        """
        Training loop - MEMORY OPTIMIZED
        """
        print(f"\n{'='*80}")
        print(f"TRAINING WITH SIMPLIFIED LOSS (MEMORY OPTIMIZED)")
        print(f"Loss = -SumRate + λ_ber(t)·BCE + λ_power·||W||²")
        print(f"Batch size: {batch_size} (reduced to avoid OOM)")
        print(f"{'='*80}\n")
        
        training_history = {
            'losses': [], 'bers': [], 'sum_rates': [], 'sinr': [],
            'rate_losses': [], 'bce_losses': [], 'power_losses': [],
            'bers_per_user': [[] for _ in range(self.simulator._num_streams_per_tx)],
            'rates_per_user': [[] for _ in range(self.simulator._num_streams_per_tx)]
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Random Eb/N0
            ebno_db = tf.random.uniform(
                [], ebno_db_range[0], ebno_db_range[1], dtype=tf.float32
            )
            
            # ✅ FIX: Calculate lambda_ber OUTSIDE @tf.function
            progress = epoch / num_epochs
            lambda_ber = 100  # Numpy calculation
            lambda_ber_tf = tf.constant(lambda_ber, dtype=tf.float32)  # Convert to tensor
            lambda_power_tf = tf.constant(0, dtype=tf.float32)
            
            # Training step
            loss, ber, sum_rate, metrics, ber_per_user = self.train_step(
                batch_size, ebno_db, lambda_ber_tf, lambda_power_tf
            )
            
            # Store metrics
            training_history['losses'].append(self.safe_numpy(loss))
            training_history['bers'].append(self.safe_numpy(ber))
            training_history['sum_rates'].append(self.safe_numpy(sum_rate))
            training_history['sinr'].append(self.safe_numpy(metrics['sinr_db']))
            training_history['rate_losses'].append(self.safe_numpy(metrics['rate_loss']))
            training_history['bce_losses'].append(self.safe_numpy(metrics['bce_loss']))
            training_history['power_losses'].append(self.safe_numpy(metrics['power_loss']))
            
            # Per user metrics
            ber_per_user_np = ber_per_user.numpy()
            rate_per_user_np = metrics['rate_per_user'].numpy()
            for u in range(self.simulator._num_streams_per_tx):
                training_history['bers_per_user'][u].append(float(ber_per_user_np[u]))
                training_history['rates_per_user'][u].append(float(rate_per_user_np[u]))
            
            # ✅ FIX: Clear GPU memory every 10 epochs
            if (epoch + 1) % 10 == 0:
                tf.keras.backend.clear_session()
                gc.collect()
            
            # Log every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"EbN0: {self.safe_numpy(ebno_db):5.1f} dB | "
                      f"Loss: {self.safe_numpy(loss):7.3f} | "
                      f"Rate: {self.safe_numpy(sum_rate):6.2f} bps/Hz | "
                      f"BER: {self.safe_numpy(ber):.2e} | "
                      f"SINR: {self.safe_numpy(metrics['sinr_db']):5.1f} dB | "
                      f"λ_ber: {lambda_ber:.3f} | "
                      f"Time: {epoch_time:.2f}s")
                
                # Display per user every 10 epochs
                if (epoch + 1) % 10 == 0:
                    print(f"  BER per user: ", end="")
                    for u in range(self.simulator._num_streams_per_tx):
                        print(f"U{u+1}={ber_per_user_np[u]:.2e} ", end="")
                    print()
                    
                    print(f"  Rate per user: ", end="")
                    for u in range(self.simulator._num_streams_per_tx):
                        print(f"U{u+1}={rate_per_user_np[u]:5.2f} ", end="")
                    print()
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Final sum rate: {training_history['sum_rates'][-1]:.2f} bps/Hz")
        print(f"Final BER: {training_history['bers'][-1]:.2e}")
        print(f"Final SINR: {training_history['sinr'][-1]:.1f} dB")
        
        # Final per user
        print(f"\nFinal BER per user:")
        for u in range(self.simulator._num_streams_per_tx):
            print(f"  User {u+1}: {training_history['bers_per_user'][u][-1]:.2e}")
        
        print(f"\nFinal Rate per user:")
        for u in range(self.simulator._num_streams_per_tx):
            print(f"  User {u+1}: {training_history['rates_per_user'][u][-1]:.2f} bps/Hz")
        
        print(f"{'='*80}\n")
        
        return training_history
    
    def save_checkpoint(self, name="checkpoint"):
        """Save model weights"""
        checkpoint_path = f"checkpoints/transformer_5d_{name}.weights.h5"
        os.makedirs("checkpoints", exist_ok=True)
        self.simulator.save_weights(checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")

def simulate_ber(simulator, ebno_range, batch_size=128, num_batches=10, return_sum_rate=False):
    """Evaluate BER over Eb/N0 range"""
    bers = []
    sum_rates = [] if return_sum_rate else None
    
    for ebno_db in ebno_range:
        ber_list = []
        rate_list = []
        ebno_tf = tf.constant(ebno_db, dtype=tf.float32)
        
        for _ in range(num_batches):
            b, b_hat, h_freq, g, precoding = simulator(batch_size, ebno_tf, training=False)
            ber = compute_ber(b, b_hat)
            ber_list.append(ber.numpy())
            
            if return_sum_rate and precoding is not None:
                # Calculate sum rate for this batch
                noise_power = ebnodb2no(ebno_tf, 2, 0.5, simulator._rg)
                # Simple SINR calculation for comparison
                # (you can call calculate_sum_rate_corrected here)
                
        avg_ber = np.mean(ber_list)
        bers.append(avg_ber)
        
        if return_sum_rate:
            avg_rate = np.mean(rate_list)
            sum_rates.append(avg_rate)
            print(f"  Eb/N0 = {ebno_db:5.1f} dB: BER = {avg_ber:.2e}, Sum Rate = {avg_rate:.1f} bps/Hz")
        else:
            print(f"  Eb/N0 = {ebno_db:5.1f} dB: BER = {avg_ber:.2e}")
    
    if return_sum_rate:
        return np.array(bers), np.array(sum_rates)
    return np.array(bers)

"""
Complete Plotting Functions for 5D Transformer Precoding
Supports: Transformer, Zero-Forcing, and WMMSE baselines
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_training_progress(training_history, params_dict):
    """
    Plot training progress: Loss, Sum Rate, BER, SINR, and component losses
    """
    fig = plt.figure(figsize=(20, 12))
    
    param_text = (
        f"Training Parameters:\n"
        f"Epochs: {params_dict['num_epochs']}\n"
        f"Batch Size: {params_dict['batch_size']}\n"
        f"Learning Rate: {params_dict['learning_rate']}\n"
        f"Eb/N0 Range: {params_dict['ebno_range']} dB\n"
        f"RB Size: {params_dict['rb_size']}\n"
        f"Precoder: {params_dict['precoder_type']}\n"
        f"TX Ant: {params_dict['num_tx']}, RX Ant: {params_dict['num_rx']}"
    )
    
    epochs = np.arange(1, len(training_history['losses']) + 1)
    
    # 1. Total Loss
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, training_history['losses'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Sum Rate
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, training_history['sum_rates'], 'g-', linewidth=2, label='Sum Rate')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
    ax2.set_title('Sum Rate Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    final_rate = training_history['sum_rates'][-1]
    ax2.annotate(f'Final: {final_rate:.2f}', 
                xy=(epochs[-1], final_rate), 
                xytext=(epochs[-1]*0.7, final_rate*0.9),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')
    
    # 3. BER (log scale)
    ax3 = plt.subplot(3, 3, 3)
    ax3.semilogy(epochs, training_history['bers'], 'm-', linewidth=2, label='BER')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('BER (log scale)', fontsize=11)
    ax3.set_title('BER During Training', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend()
    
    # 4. SINR
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(epochs, training_history['sinr'], 'r-', linewidth=2, label='SINR')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('SINR (dB)', fontsize=11)
    ax4.set_title('SINR Evolution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.legend()
    
    # 5. Component Losses
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(epochs, training_history['rate_losses'], 'b-', linewidth=2, label='Rate Loss', alpha=0.7)
    ax5.plot(epochs, training_history['bce_losses'], 'orange', linewidth=2, label='BCE Loss', alpha=0.7)
    ax5.plot(epochs, training_history['power_losses'], 'purple', linewidth=2, label='Power Loss', alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Loss Components', fontsize=11)
    ax5.set_title('Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. BER per User
    ax6 = plt.subplot(3, 3, 6)
    num_users = len(training_history['bers_per_user'])
    colors = ['blue', 'orange', 'green', 'red']
    for u in range(num_users):
        ax6.semilogy(epochs, training_history['bers_per_user'][u], 
                    color=colors[u], linewidth=2, label=f'User {u+1}', alpha=0.7)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('BER per User (log scale)', fontsize=11)
    ax6.set_title('BER per User Evolution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend()
    
    # 7. Rate per User
    ax7 = plt.subplot(3, 3, 7)
    for u in range(num_users):
        ax7.plot(epochs, training_history['rates_per_user'][u], 
                color=colors[u], linewidth=2, label=f'User {u+1}', alpha=0.7)
    ax7.set_xlabel('Epoch', fontsize=11)
    ax7.set_ylabel('Rate per User (bps/Hz)', fontsize=11)
    ax7.set_title('Rate per User Evolution', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 8. Learning Progress (normalized)
    ax8 = plt.subplot(3, 3, 8)
    normalized_rate = np.array(training_history['sum_rates']) / max(training_history['sum_rates'])
    normalized_loss = 1 - (np.array(training_history['losses']) / max(training_history['losses']))
    ax8.plot(epochs, normalized_rate, 'g-', linewidth=2, label='Sum Rate (norm)', alpha=0.7)
    ax8.plot(epochs, normalized_loss, 'b-', linewidth=2, label='Loss Improvement', alpha=0.7)
    ax8.set_xlabel('Epoch', fontsize=11)
    ax8.set_ylabel('Normalized Progress', fontsize=11)
    ax8.set_title('Training Progress (Normalized)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.set_ylim([0, 1.1])
    
    # 9. Parameters Text Box
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    ax9.text(0.1, 0.9, param_text, 
            transform=ax9.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    final_stats = (
        f"\nFinal Statistics:\n"
        f"Sum Rate: {training_history['sum_rates'][-1]:.2f} bps/Hz\n"
        f"BER: {training_history['bers'][-1]:.2e}\n"
        f"SINR: {training_history['sinr'][-1]:.1f} dB\n"
        f"Rate Loss: {training_history['rate_losses'][-1]:.3f}\n"
        f"BCE Loss: {training_history['bce_losses'][-1]:.3f}\n"
        f"Power Loss: {training_history['power_losses'][-1]:.4f}"
    )
    ax9.text(0.1, 0.35, final_stats,
            transform=ax9.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            family='monospace')
    
    plt.suptitle('5D Transformer Precoding - Training Progress', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def plot_ber_comparison_all_baselines(ber_transformer, ber_zf, ber_wmmse, 
                                      ebno_range, params_dict):
    """
    Plot BER comparison: Transformer vs ZF vs WMMSE
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    param_text = (
        f"System Parameters:\n"
        f"TX Ant: {params_dict['num_tx']}\n"
        f"RX Ant: {params_dict['num_rx']}\n"
        f"Users: {params_dict['num_users']}\n"
        f"RB Size: {params_dict['rb_size']}\n"
        f"FFT Size: {params_dict['fft_size']}\n"
        f"Modulation: QPSK\n"
        f"Code Rate: {params_dict['code_rate']}"
    )
    
    # 1. Total BER Comparison
    ax1 = axes[0, 0]
    ax1.semilogy(ebno_range, ber_zf['ber_total'], 
                'r-o', label='Zero-Forcing', linewidth=2.5, markersize=7, alpha=0.8)
    ax1.semilogy(ebno_range, ber_wmmse['ber_total'], 
                'g-^', label='WMMSE', linewidth=2.5, markersize=7, alpha=0.8)
    ax1.semilogy(ebno_range, ber_transformer['ber_total'], 
                'b-s', label='5D Transformer', linewidth=2.5, markersize=7, alpha=0.8)
    
    ax1.set_xlabel('Eb/N0 (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax1.set_title('BER Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # Gain annotation at 10 dB
    idx_10db = np.argmin(np.abs(ebno_range - 10))
    ber_zf_10 = ber_zf['ber_total'][idx_10db]
    ber_wmmse_10 = ber_wmmse['ber_total'][idx_10db]
    ber_tf_10 = ber_transformer['ber_total'][idx_10db]
    
    if ber_tf_10 > 0 and ber_zf_10 > 0:
        gain_vs_zf = 10 * np.log10(ber_zf_10 / ber_tf_10)
        gain_vs_wmmse = 10 * np.log10(ber_wmmse_10 / ber_tf_10)
        
        annotation_text = (
            f'Gains @ 10dB:\n'
            f'vs ZF: {gain_vs_zf:.1f} dB\n'
            f'vs WMMSE: {gain_vs_wmmse:.1f} dB'
        )
        ax1.annotate(annotation_text, 
                    xy=(10, ber_tf_10), 
                    xytext=(12, ber_tf_10 * 5),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=10, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 2. BER per User @ 10 dB
    ax2 = axes[0, 1]
    num_users = ber_transformer['ber_per_user'].shape[0]
    users = np.arange(1, num_users + 1)
    width = 0.25
    
    ber_tf_users = ber_transformer['ber_per_user'][:, idx_10db]
    ber_zf_users = ber_zf['ber_per_user'][:, idx_10db]
    ber_wmmse_users = ber_wmmse['ber_per_user'][:, idx_10db]
    
    ax2.bar(users - width, ber_zf_users, width, 
           label='ZF', color='red', alpha=0.7, edgecolor='black')
    ax2.bar(users, ber_wmmse_users, width, 
           label='WMMSE', color='green', alpha=0.7, edgecolor='black')
    ax2.bar(users + width, ber_tf_users, width, 
           label='Transformer', color='blue', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('User', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BER', fontsize=12, fontweight='bold')
    ax2.set_title(f'BER per User @ Eb/N0 = 10 dB', fontsize=14, fontweight='bold')
    ax2.set_xticks(users)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both', axis='y')
    ax2.legend(fontsize=11)
    
    # 3. Gain in dB vs Eb/N0
    ax3 = axes[1, 0]
    
    # Calculate gains
    gain_vs_zf_all = np.zeros_like(ebno_range, dtype=float)
    gain_vs_wmmse_all = np.zeros_like(ebno_range, dtype=float)
    
    valid_idx = (ber_transformer['ber_total'] > 0) & (ber_zf['ber_total'] > 0)
    gain_vs_zf_all[valid_idx] = 10 * np.log10(
        ber_zf['ber_total'][valid_idx] / ber_transformer['ber_total'][valid_idx]
    )
    
    valid_idx_wmmse = (ber_transformer['ber_total'] > 0) & (ber_wmmse['ber_total'] > 0)
    gain_vs_wmmse_all[valid_idx_wmmse] = 10 * np.log10(
        ber_wmmse['ber_total'][valid_idx_wmmse] / ber_transformer['ber_total'][valid_idx_wmmse]
    )
    
    ax3.plot(ebno_range, gain_vs_zf_all, 'r-o', linewidth=2.5, markersize=7, label='Gain vs ZF')
    ax3.plot(ebno_range, gain_vs_wmmse_all, 'g-^', linewidth=2.5, markersize=7, label='Gain vs WMMSE')
    ax3.set_xlabel('Eb/N0 (dB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Gain (dB)', fontsize=12, fontweight='bold')
    ax3.set_title('BER Gain: Transformer vs Baselines', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No gain')
    ax3.legend(fontsize=11)
    
    # 4. Parameters
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.1, 0.9, param_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
            family='monospace')
    
    # Add performance summary
    summary_text = (
        f"\n=== PERFORMANCE @ 10 dB ===\n"
        f"Transformer: {ber_tf_10:.2e}\n"
        f"WMMSE:       {ber_wmmse_10:.2e}\n"
        f"ZF:          {ber_zf_10:.2e}\n"
        f"\n=== GAINS ===\n"
        f"vs ZF:    {gain_vs_zf:.2f} dB\n"
        f"vs WMMSE: {gain_vs_wmmse:.2f} dB\n"
        f"\n=== RANKING ===\n"
    )
    
    # Rank the methods
    bers_at_10 = [
        ('Transformer', ber_tf_10),
        ('WMMSE', ber_wmmse_10),
        ('ZF', ber_zf_10)
    ]
    bers_at_10.sort(key=lambda x: x[1])
    
    for rank, (method, ber) in enumerate(bers_at_10, 1):
        summary_text += f"{rank}. {method}: {ber:.2e}\n"
    
    ax4.text(0.1, 0.4, summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
            family='monospace')
    
    plt.suptitle('BER Performance: Transformer vs ZF vs WMMSE', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_complete_results_with_wmmse(training_history, ber_transformer, ber_zf, ber_wmmse,
                                     rate_transformer, ebno_range, params_dict, timestamp):
    """
    Complete visualization with all baselines
    """
    fig = plt.figure(figsize=(22, 16))
    
    param_text = (
        f"=== SYSTEM ===\n"
        f"TX: {params_dict['num_tx']}, RX: {params_dict['num_rx']}\n"
        f"Users: {params_dict['num_users']}\n"
        f"FFT: {params_dict['fft_size']}, RB: {params_dict['rb_size']}\n"
        f"\n=== TRAINING ===\n"
        f"Epochs: {params_dict['num_epochs']}\n"
        f"Batch: {params_dict['batch_size']}\n"
        f"LR: {params_dict['learning_rate']}\n"
        f"Eb/N0: {params_dict['ebno_range']} dB\n"
        f"\n=== RESULTS ===\n"
        f"Sum Rate: {training_history['sum_rates'][-1]:.2f}\n"
        f"BER: {training_history['bers'][-1]:.2e}\n"
        f"SINR: {training_history['sinr'][-1]:.1f} dB"
    )
    
    epochs = np.arange(1, len(training_history['losses']) + 1)
    colors = ['blue', 'orange', 'green', 'red']
    num_users = ber_transformer['ber_per_user'].shape[0]
    
    # Row 1: Training Progress
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(epochs, training_history['losses'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Training Loss', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(epochs, training_history['sum_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Sum Rate (bps/Hz)', fontsize=10)
    ax2.set_title('Sum Rate (Training)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 4, 3)
    ax3.semilogy(epochs, training_history['bers'], 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('BER', fontsize=10)
    ax3.set_title('BER (Training)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    ax4 = plt.subplot(4, 4, 4)
    ax4.plot(epochs, training_history['sinr'], 'r-', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('SINR (dB)', fontsize=10)
    ax4.set_title('SINR Evolution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Row 2: BER Comparison
    ax5 = plt.subplot(4, 4, 5)
    ax5.semilogy(ebno_range, ber_zf['ber_total'], 'r-o', label='ZF', linewidth=2, markersize=5)
    ax5.semilogy(ebno_range, ber_wmmse['ber_total'], 'g-^', label='WMMSE', linewidth=2, markersize=5)
    ax5.semilogy(ebno_range, ber_transformer['ber_total'], 'b-s', label='Transformer', linewidth=2, markersize=5)
    ax5.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax5.set_ylabel('BER', fontsize=10)
    ax5.set_title('BER: All Methods', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    ax6 = plt.subplot(4, 4, 6)
    for u in range(num_users):
        ax6.semilogy(ebno_range, ber_transformer['ber_per_user'][u, :],
                    color=colors[u], linewidth=2, label=f'U{u+1}', marker='s', markersize=4)
    ax6.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax6.set_ylabel('BER', fontsize=10)
    ax6.set_title('BER/User (Transformer)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=8)
    
    ax7 = plt.subplot(4, 4, 7)
    for u in range(num_users):
        ax7.semilogy(ebno_range, ber_wmmse['ber_per_user'][u, :],
                    color=colors[u], linewidth=2, label=f'U{u+1}', marker='^', markersize=4)
    ax7.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax7.set_ylabel('BER', fontsize=10)
    ax7.set_title('BER/User (WMMSE)', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, which='both')
    ax7.legend(fontsize=8)
    
    ax8 = plt.subplot(4, 4, 8)
    for u in range(num_users):
        ax8.semilogy(ebno_range, ber_zf['ber_per_user'][u, :],
                    color=colors[u], linewidth=2, label=f'U{u+1}', marker='o', markersize=4)
    ax8.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax8.set_ylabel('BER', fontsize=10)
    ax8.set_title('BER/User (ZF)', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, which='both')
    ax8.legend(fontsize=8)
    
    # Row 3: Sum Rate Analysis
    ax9 = plt.subplot(4, 4, 9)
    ax9.plot(ebno_range, rate_transformer['sum_rate_total'], 
            'b-s', linewidth=2, markersize=5, label='Transformer')
    ax9.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax9.set_ylabel('Sum Rate (bps/Hz)', fontsize=10)
    ax9.set_title('Sum Rate vs Eb/N0', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=9)
    
    ax10 = plt.subplot(4, 4, 10)
    for u in range(num_users):
        ax10.plot(ebno_range, rate_transformer['rate_per_user'][u, :],
                 color=colors[u], linewidth=2, label=f'U{u+1}', marker='s', markersize=4)
    ax10.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax10.set_ylabel('Rate (bps/Hz)', fontsize=10)
    ax10.set_title('Rate/User', fontsize=11, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=8)
    
    ax11 = plt.subplot(4, 4, 11)
    fairness = np.zeros_like(ebno_range, dtype=float)
    for i in range(len(ebno_range)):
        rates = rate_transformer['rate_per_user'][:, i]
        fairness[i] = np.min(rates) / (np.max(rates) + 1e-12)
    ax11.plot(ebno_range, fairness, 'purple', linewidth=2, marker='o', markersize=5)
    ax11.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax11.set_ylabel('Fairness Index', fontsize=10)
    ax11.set_title('User Fairness', fontsize=11, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 1.1])
    ax11.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
    
    ax12 = plt.subplot(4, 4, 12)
    ax12.axis('off')
    ax12.text(0.05, 0.95, param_text,
             transform=ax12.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
             family='monospace')
    
    # Row 4: Detailed Comparisons
    ax13 = plt.subplot(4, 4, 13)
    # Gain vs ZF
    gain_vs_zf = np.zeros_like(ebno_range, dtype=float)
    valid = (ber_transformer['ber_total'] > 0) & (ber_zf['ber_total'] > 0)
    gain_vs_zf[valid] = 10 * np.log10(ber_zf['ber_total'][valid] / ber_transformer['ber_total'][valid])
    ax13.plot(ebno_range, gain_vs_zf, 'r-o', linewidth=2, markersize=5)
    ax13.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax13.set_ylabel('Gain (dB)', fontsize=10)
    ax13.set_title('Gain vs ZF', fontsize=11, fontweight='bold')
    ax13.grid(True, alpha=0.3)
    ax13.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax14 = plt.subplot(4, 4, 14)
    # Gain vs WMMSE
    gain_vs_wmmse = np.zeros_like(ebno_range, dtype=float)
    valid = (ber_transformer['ber_total'] > 0) & (ber_wmmse['ber_total'] > 0)
    gain_vs_wmmse[valid] = 10 * np.log10(ber_wmmse['ber_total'][valid] / ber_transformer['ber_total'][valid])
    ax14.plot(ebno_range, gain_vs_wmmse, 'g-^', linewidth=2, markersize=5)
    ax14.set_xlabel('Eb/N0 (dB)', fontsize=10)
    ax14.set_ylabel('Gain (dB)', fontsize=10)
    ax14.set_title('Gain vs WMMSE', fontsize=11, fontweight='bold')
    ax14.grid(True, alpha=0.3)
    ax14.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # BER per user during training
    ax15 = plt.subplot(4, 4, 15)
    for u in range(num_users):
        ax15.semilogy(epochs, training_history['bers_per_user'][u],
                     color=colors[u], linewidth=2, label=f'U{u+1}', alpha=0.8)
    ax15.set_xlabel('Epoch', fontsize=10)
    ax15.set_ylabel('BER', fontsize=10)
    ax15.set_title('BER/User (Training)', fontsize=11, fontweight='bold')
    ax15.grid(True, alpha=0.3, which='both')
    ax15.legend(fontsize=8)
    
    # Rate per user during training
    ax16 = plt.subplot(4, 4, 16)
    for u in range(num_users):
        ax16.plot(epochs, training_history['rates_per_user'][u],
                 color=colors[u], linewidth=2, label=f'U{u+1}', alpha=0.8)
    ax16.set_xlabel('Epoch', fontsize=10)
    ax16.set_ylabel('Rate (bps/Hz)', fontsize=10)
    ax16.set_title('Rate/User (Training)', fontsize=11, fontweight='bold')
    ax16.grid(True, alpha=0.3)
    ax16.legend(fontsize=8)
    
    plt.suptitle(f'5D Transformer: Complete Results with All Baselines ({timestamp})', 
                fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    return fig


def save_and_show_all_plots(training_history, ber_transformer, ber_zf, ber_wmmse,
                            rate_transformer, ebno_range, params_dict):
    """
    Generate and save all plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("GENERATING ALL PLOTS...")
    print("="*80)
    
    # Plot 1: Training Progress
    print("[1/3] Creating training progress plot...")
    fig1 = plot_training_progress(training_history, params_dict)
    filename1 = f'training_progress_{timestamp}.png'
    fig1.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {filename1}")
    
    # Plot 2: BER Comparison (all baselines)
    print("[2/3] Creating BER comparison plot (with WMMSE)...")
    fig2 = plot_ber_comparison_all_baselines(ber_transformer, ber_zf, ber_wmmse,
                                             ebno_range, params_dict)
    filename2 = f'ber_comparison_all_{timestamp}.png'
    fig2.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {filename2}")
    
    # Plot 3: Complete Results
    print("[3/3] Creating complete results plot...")
    fig3 = plot_complete_results_with_wmmse(training_history, ber_transformer, 
                                            ber_zf, ber_wmmse, rate_transformer,
                                            ebno_range, params_dict, timestamp)
    filename3 = f'complete_results_wmmse_{timestamp}.png'
    fig3.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {filename3}")
    
    print("\n" + "="*80)
    print("ALL PLOTS SAVED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    plt.show()
    
    return fig1, fig2, fig3

def main():
    """Main training and evaluation - COMPLETE & CLEAN"""
    print("\n" + "="*80)
    print("5D TRANSFORMER PRECODING - COMPLETE EVALUATION WITH ALL BASELINES")
    print("="*80)
    
    # ========== PARAMETERS ==========
    num_epochs = 100
    batch_size = 128
    learning_rate = 5e-4
    rb_size = 12
    ebno_db_range = (0, 10)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== PHASE 1: CREATE AND TRAIN TRANSFORMER ==========
    print("\n[1/6] Creating 5D Transformer simulator...")
    simulator_5d = OFDMSimulator5D_Fixed(
        perfect_csi=True,
        precoder_type="transformer_5d",
        rb_size=rb_size
    )
    
    print("\n[2/6] Building model (dummy forward pass)...")
    batch_size_dummy = tf.constant(4, dtype=tf.int32)
    ebno_dummy = tf.constant(10.0, dtype=tf.float32)
    _ = simulator_5d(batch_size_dummy, ebno_dummy, training=False)
    print("✅ Model built!")
    
    print("\n[3/6] Training Transformer...")
    trainer = EndToEndTrainer_Fixed(simulator_5d, learning_rate=learning_rate)
    training_history = trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        ebno_db_range=ebno_db_range
    )
    trainer.save_checkpoint("final")
    
    # ========== PHASE 2: EVALUATE ALL METHODS ==========
    ebno_eval_range = np.arange(-5, 20, 2.0)
     # After training, before evaluation
    print("\n[DEBUG] Plotting constellation for trained Transformer...")
    plot_constellation_debug(
        simulator_5d, 
        batch_size=128, 
        ebno_db=10.0,
        save_path=f'constellation_transformer_{timestamp}.png'
    )
    simulator_zf = OFDMSimulator5D_Fixed(
        perfect_csi=True, 
        precoder_type="zf", 
        rb_size=rb_size
    )
    
    print("\n[DEBUG] Plotting constellation for ZF...")
    plot_constellation_debug(
        simulator_zf, 
        batch_size=128, 
        ebno_db=10.0,
        save_path=f'constellation_zf_{timestamp}.png'
    )
    print("\n[4/6] Evaluating BER - Transformer...")
    ber_results_transformer = evaluate_ber_vs_ebno_per_user(
        simulator_5d, 
        ebno_eval_range,
        batch_size=128,
        num_target_block_errors=1000,
        max_mc_iter=1000
    )
    
    print("\n[4/6] Evaluating BER - Zero-Forcing...")
    
    ber_results_zf = evaluate_ber_vs_ebno_per_user(
        simulator_zf,
        ebno_eval_range,
        batch_size=128,
        num_target_block_errors=1000,
        max_mc_iter=1000
    )
    
    print("\n[4/6] Evaluating BER - WMMSE...")
    simulator_wmmse = OFDMSimulator5D_Fixed(
        perfect_csi=True, 
        precoder_type="wmmse", 
        rb_size=rb_size
    )
    ber_results_wmmse = evaluate_ber_vs_ebno_per_user(
        simulator_wmmse,
        ebno_eval_range,
        batch_size=128,
        num_target_block_errors=1000,
        max_mc_iter=1000
    )
    
    # ========== PHASE 3: EVALUATE SUM RATE ==========
    print("\n[5/6] Evaluating Sum Rate (Transformer)...")
    rate_results_transformer = evaluate_sum_rate_per_user(
        simulator_5d,
        ebno_eval_range,
        batch_size=128,
        num_batches=100
    )
    
    # ========== PHASE 4: GENERATE ALL PLOTS ==========
    print("\n[6/6] Generating all plots...")
    
    params_dict = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'ebno_range': ebno_db_range,
        'rb_size': rb_size,
        'precoder_type': '5D Transformer',
        'num_tx': simulator_5d._num_bs_ant,
        'num_rx': simulator_5d._num_ut_ant,
        'num_users': simulator_5d._num_streams_per_tx,
        'fft_size': simulator_5d._fft_size,
        'num_ofdm_symbols': simulator_5d._num_ofdm_symbols,
        'code_rate': simulator_5d._coderate
    }
    
    # Import plotting functions
    from plotting_functions import save_and_show_all_plots
    
    fig1, fig2, fig3 = save_and_show_all_plots(
        training_history, 
        ber_results_transformer, 
        ber_results_zf,
        ber_results_wmmse,
        rate_results_transformer,
        ebno_eval_range,
        params_dict
    )
    
    # ========== PHASE 5: FINAL SUMMARY ==========
    print_final_summary(ber_results_transformer, ber_results_zf, ber_results_wmmse,
                       rate_results_transformer, ebno_eval_range)
    
    return (simulator_5d, training_history, 
            ber_results_transformer, ber_results_zf, ber_results_wmmse,
            rate_results_transformer)


def print_final_summary(ber_tf, ber_zf, ber_wmmse, rate_tf, ebno_range):
    """Print comprehensive final summary"""
    print(f"\n{'='*80}")
    print("FINAL COMPREHENSIVE SUMMARY @ 10 dB")
    print(f"{'='*80}")
    
    idx_10db = np.argmin(np.abs(ebno_range - 10.0))
    
    # BER Comparison
    print("\n📊 BER COMPARISON:")
    ber_tf_10 = ber_tf['ber_total'][idx_10db]
    ber_wmmse_10 = ber_wmmse['ber_total'][idx_10db]
    ber_zf_10 = ber_zf['ber_total'][idx_10db]
    
    print(f"  Transformer:  {ber_tf_10:.2e}")
    print(f"  WMMSE:        {ber_wmmse_10:.2e}")
    print(f"  Zero-Forcing: {ber_zf_10:.2e}")
    
    # Gains
    gain_vs_zf = 10*np.log10(ber_zf_10 / ber_tf_10) if ber_tf_10 > 0 else 0
    gain_vs_wmmse = 10*np.log10(ber_wmmse_10 / ber_tf_10) if ber_tf_10 > 0 else 0
    
    print(f"\n📈 GAINS:")
    print(f"  Transformer vs ZF:    {gain_vs_zf:+.2f} dB")
    print(f"  Transformer vs WMMSE: {gain_vs_wmmse:+.2f} dB")
    
    # BER per user
    print(f"\n📡 BER PER USER:")
    print(f"{'User':<8} {'Transformer':<15} {'WMMSE':<15} {'ZF':<15}")
    print("-" * 60)
    for u in range(4):
        ber_tf_u = ber_tf['ber_per_user'][u, idx_10db]
        ber_wmmse_u = ber_wmmse['ber_per_user'][u, idx_10db]
        ber_zf_u = ber_zf['ber_per_user'][u, idx_10db]
        print(f"User {u+1:<3} {ber_tf_u:<15.2e} {ber_wmmse_u:<15.2e} {ber_zf_u:<15.2e}")
    
    # Rate per user
    print(f"\n📶 RATE PER USER (Transformer):")
    for u in range(4):
        rate = rate_tf['rate_per_user'][u, idx_10db]
        print(f"  User {u+1}: {rate:.2f} bps/Hz")
    
    print(f"\n📊 TOTAL SUM RATE: {rate_tf['sum_rate_total'][idx_10db]:.2f} bps/Hz")
    
    # Fairness
    rates_at_10db = rate_tf['rate_per_user'][:, idx_10db]
    fairness = np.min(rates_at_10db) / np.max(rates_at_10db)
    print(f"⚖️  FAIRNESS INDEX: {fairness:.3f}")
    
    # Ranking
    print(f"\n🏆 RANKING @ 10 dB:")
    methods = [
        ('Transformer', ber_tf_10),
        ('WMMSE', ber_wmmse_10),
        ('Zero-Forcing', ber_zf_10)
    ]
    methods.sort(key=lambda x: x[1])
    
    for rank, (method, ber) in enumerate(methods, 1):
        print(f"  {rank}. {method:<15} BER = {ber:.2e}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    results = main()