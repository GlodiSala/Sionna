"""
Diagnostic Script for BER Analysis with Noiseless Transmission Option
======================================================================

This script allows testing the transmission chain without noise to verify
that the precoding, channel modeling, and equalization are working correctly.

Key features:
- Noiseless transmission mode (set noise power to 0)
- Detailed constellation plots at each stage
- Per-user BER analysis
- Comparison of ZF, WMMSE, and Transformer precoders in noiseless conditions

Usage:
    python main_sionna_5d_noiseless_diagnostic.py

The noiseless mode will help identify if BER issues are due to:
1. Incorrect channel modeling
2. Precoding matrix issues
3. Equalization problems
4. Or genuinely noise-related

"""

import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import datetime

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration failed: {e}")

# Sionna imports
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

# Import custom modules (adjust paths as needed)
import sys
sys.path.append('/mnt/project')
from transformer_5d_sionna import TransformerPrecoder5D_Fixed
from wmmse_precoder import WMMSEPrecoder


class OFDMSimulatorDiagnostic(tf.keras.Model):
    """
    OFDM Simulator with Noiseless Mode for Diagnostics
    """
    
    def __init__(self, cdl_model="A", delay_spread=30e-9, perfect_csi=True,
                 precoder_type="transformer_5d", rb_size=4, noiseless=False):
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
        self._noiseless = noiseless  # NEW: Noiseless mode flag
        
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
                num_iterations=10,
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
        
        print(f"\n[DIAGNOSTIC SIMULATOR CONFIG]")
        print(f"  FFT size: {self._fft_size}")
        print(f"  OFDM symbols: {self._num_ofdm_symbols}")
        print(f"  TX antennas: {self._num_bs_ant}")
        print(f"  RX antennas: {self._num_ut_ant}")
        print(f"  Streams: {self._num_streams_per_tx}")
        print(f"  RB size: {self._rb_size}")
        print(f"  Precoder: {precoder_type}")
        print(f"  🔍 NOISELESS MODE: {noiseless}")
    
    @tf.function
    def call(self, batch_size, ebno_db, training=False):
        """
        Execute forward pass with optional noiseless mode
        """
        batch_size = tf.constant(batch_size) if not isinstance(batch_size, tf.Tensor) else batch_size
        
        # Calculate noise power (will be set to 0 in noiseless mode)
        if self._noiseless:
            no = tf.constant(0.0, dtype=tf.float32)
        else:
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        
        # Transmitter
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        
        # Channel
        cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1 / self._rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)
        
        # Precoding
        if self._precoder_type == "wmmse":
            x_rg_precoded, g = self._wmmse_precoder(x_rg, h_freq)
        elif self._precoder_type == "transformer_5d":
            x_rg_precoded, g, _ = self._transformer_precoder(x_rg, h_freq)
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
        
        return b, b_hat, x, x_hat, h_freq, g


def compute_ber_per_user(b, b_hat, num_users=4):
    """Compute BER for each user separately"""
    bers = []
    for user_idx in range(num_users):
        b_user = b[:, :, user_idx, :]
        b_hat_user = b_hat[:, :, user_idx, :]
        ber_user = compute_ber(b_user, b_hat_user).numpy()
        bers.append(ber_user)
    return np.array(bers)


def plot_noiseless_diagnostic(simulator, batch_size=128, save_path=None):
    """
    Plot constellation and compute BER in noiseless mode
    """
    ebno_db = tf.constant(10.0, dtype=tf.float32)
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
    
    # Forward pass
    b, b_hat, x_original, x_equalized, h_freq, g = simulator(batch_size_tf, ebno_db, training=False)
    
    # Convert to numpy
    x_orig_np = x_original.numpy().flatten()
    x_eq_np = x_equalized.numpy().flatten()
    
    # Sample for plotting
    n_points = min(2000, len(x_orig_np))
    idx = np.random.choice(len(x_orig_np), n_points, replace=False)
    x_orig_plot = x_orig_np[idx]
    x_eq_plot = x_eq_np[idx]
    
    # Compute BER
    ber_total = compute_ber(b, b_hat).numpy()
    ber_per_user = compute_ber_per_user(b, b_hat)
    
    # Compute EVM
    evm = np.abs(x_eq_plot - x_orig_plot)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Transmitted symbols
    ax1 = axes[0]
    ax1.scatter(x_orig_plot.real, x_orig_plot.imag, alpha=0.5, s=20, c='blue', label='Transmitted')
    qpsk_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    ax1.scatter(qpsk_points.real, qpsk_points.imag, c='red', s=150, marker='x', linewidths=3, label='Ideal QPSK')
    ax1.set_xlabel('In-Phase')
    ax1.set_ylabel('Quadrature')
    ax1.set_title('Transmitted Symbols\n(Before Precoding)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Equalized symbols
    ax2 = axes[1]
    ax2.scatter(x_eq_plot.real, x_eq_plot.imag, alpha=0.5, s=20, c='green', label='Equalized')
    ax2.scatter(qpsk_points.real, qpsk_points.imag, c='red', s=150, marker='x', linewidths=3, label='Ideal QPSK')
    ax2.set_xlabel('In-Phase')
    ax2.set_ylabel('Quadrature')
    mode_str = "NOISELESS" if simulator._noiseless else f"Eb/N0={10} dB"
    ax2.set_title(f'Equalized Symbols\n({mode_str})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Plot 3: Statistics
    ax3 = axes[2]
    ax3.axis('off')
    
    stats_text = (
        f"{'='*45}\n"
        f"NOISELESS DIAGNOSTIC RESULTS\n"
        f"{'='*45}\n\n"
        f"Precoder: {simulator._precoder_type.upper()}\n"
        f"Noiseless Mode: {simulator._noiseless}\n"
        f"Batch Size: {batch_size}\n\n"
        f"Overall BER: {ber_total:.2e}\n"
        f"Mean EVM: {np.mean(evm):.6f}\n"
        f"Max EVM: {np.max(evm):.6f}\n\n"
        f"BER Per User:\n"
    )
    
    for i, ber in enumerate(ber_per_user):
        stats_text += f"  User {i+1}: {ber:.2e}\n"
    
    stats_text += f"\n{'='*45}\n"
    
    if simulator._noiseless and ber_total > 1e-6:
        stats_text += "\n⚠️  WARNING: Non-zero BER in noiseless mode!\n"
        stats_text += "This indicates issues with:\n"
        stats_text += "- Channel modeling\n"
        stats_text += "- Precoding\n"
        stats_text += "- Equalization\n"
    elif simulator._noiseless:
        stats_text += "\n✅ Perfect transmission in noiseless mode!\n"
    
    ax3.text(0.05, 0.95, stats_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Noiseless Diagnostic: {simulator._precoder_type.upper()}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Diagnostic plot saved: {save_path}")
    
    return fig, {'ber_total': ber_total, 'ber_per_user': ber_per_user, 'evm_mean': np.mean(evm)}


def run_noiseless_comparison():
    """
    Run noiseless transmission test for all precoder types
    """
    print("\n" + "="*80)
    print("NOISELESS TRANSMISSION DIAGNOSTIC")
    print("="*80)
    print("\nThis test runs transmission without noise to verify that:")
    print("  1. Channel is correctly modeled")
    print("  2. Precoding matrices are correct")
    print("  3. Equalization works properly")
    print("\nIn noiseless mode, BER should be zero (or very close to zero).")
    print("="*80 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rb_size = 12
    batch_size = 128
    
    results = {}
    precoders = ["zf", "wmmse", "transformer_5d"]
    
    for precoder in precoders:
        print(f"\n[Testing {precoder.upper()} in noiseless mode...]")
        
        simulator = OFDMSimulatorDiagnostic(
            perfect_csi=True,
            precoder_type=precoder,
            rb_size=rb_size,
            noiseless=True  # KEY: Enable noiseless mode
        )
        
        # Build model
        _ = simulator(tf.constant(4, dtype=tf.int32), tf.constant(10.0), training=False)
        
        # Run diagnostic
        save_path = f'noiseless_diagnostic_{precoder}_{timestamp}.png'
        fig, metrics = plot_noiseless_diagnostic(simulator, batch_size=batch_size, save_path=save_path)
        
        results[precoder] = metrics
        
        print(f"  ✅ {precoder.upper()}: BER = {metrics['ber_total']:.2e}, Mean EVM = {metrics['evm_mean']:.6f}")
    
    # Summary
    print("\n" + "="*80)
    print("NOISELESS DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"\n{'Precoder':<20} {'Total BER':<15} {'Mean EVM':<15} {'Status':<20}")
    print("-" * 70)
    
    for precoder, metrics in results.items():
        ber = metrics['ber_total']
        evm = metrics['evm_mean']
        status = "✅ PASS" if ber < 1e-5 else "⚠️  ISSUE DETECTED"
        print(f"{precoder.upper():<20} {ber:<15.2e} {evm:<15.6f} {status:<20}")
    
    print("\n" + "="*80)
    print("\nInterpretation:")
    print("  - BER < 1e-6: Transmission chain is correct")
    print("  - BER > 1e-5: There are issues with precoding/equalization")
    print("  - High EVM: Channel inversion or equalization problems")
    print("="*80 + "\n")
    
    return results


def run_noisy_vs_noiseless_comparison():
    """
    Compare noisy vs noiseless transmission for a single precoder
    """
    print("\n" + "="*80)
    print("NOISY VS NOISELESS COMPARISON (ZF Precoder)")
    print("="*80 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = 128
    
    # Noiseless
    print("[1/2] Testing NOISELESS transmission...")
    sim_noiseless = OFDMSimulatorDiagnostic(
        perfect_csi=True,
        precoder_type="zf",
        rb_size=12,
        noiseless=True
    )
    _ = sim_noiseless(tf.constant(4), tf.constant(10.0), training=False)
    fig1, metrics_noiseless = plot_noiseless_diagnostic(
        sim_noiseless, 
        batch_size=batch_size, 
        save_path=f'comparison_noiseless_{timestamp}.png'
    )
    
    # Noisy
    print("\n[2/2] Testing NOISY transmission (Eb/N0 = 10 dB)...")
    sim_noisy = OFDMSimulatorDiagnostic(
        perfect_csi=True,
        precoder_type="zf",
        rb_size=12,
        noiseless=False
    )
    _ = sim_noisy(tf.constant(4), tf.constant(10.0), training=False)
    fig2, metrics_noisy = plot_noiseless_diagnostic(
        sim_noisy, 
        batch_size=batch_size, 
        save_path=f'comparison_noisy_{timestamp}.png'
    )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Mode':<15} {'BER':<15} {'Mean EVM':<15}")
    print("-" * 45)
    print(f"{'Noiseless':<15} {metrics_noiseless['ber_total']:<15.2e} {metrics_noiseless['evm_mean']:<15.6f}")
    print(f"{'Noisy (10dB)':<15} {metrics_noisy['ber_total']:<15.2e} {metrics_noisy['evm_mean']:<15.6f}")
    print("="*80 + "\n")
    
    if metrics_noiseless['ber_total'] > 1e-5:
        print("⚠️  WARNING: Noiseless BER is high! This indicates fundamental issues.")
        print("   Check: precoding matrix, channel model, equalization")
    else:
        print("✅ Noiseless transmission is perfect. BER issues are noise-related.")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BER DIAGNOSTIC TOOL - NOISELESS TRANSMISSION MODE")
    print("="*80)
    print("\nThis tool helps diagnose BER issues by testing transmission")
    print("without noise. This isolates channel modeling, precoding, and")
    print("equalization issues from noise-related problems.\n")
    
    # Run diagnostics
    print("Choose diagnostic mode:")
    print("  1. Run noiseless test for all precoders")
    print("  2. Compare noisy vs noiseless (ZF only)")
    print("  3. Run both\n")
    
    # For automation, run option 3
    choice = "3"
    
    if choice in ["1", "3"]:
        results = run_noiseless_comparison()
    
    if choice in ["2", "3"]:
        run_noisy_vs_noiseless_comparison()
    
    print("\n✅ Diagnostic complete! Check the generated plots.")
    plt.show()