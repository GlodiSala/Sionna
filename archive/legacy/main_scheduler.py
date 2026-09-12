# =============================================================================
# SYSTEM-LEVEL EVALUATION WITH SCHEDULER + LINK ADAPTATION
# =============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from datetime import datetime

# Sionna System-Level imports
from sionna.sys import (
    PHYAbstraction, 
    OuterLoopLinkAdaptation, 
    PFSchedulerSUMIMO,
    spread_across_subcarriers
)
from sionna.phy.utils import log2, insert_dims
from sionna.phy.ofdm import LMMSEPostEqualizationSINR

# Your existing imports
import sionna
sionna.phy.config.seed = 42

from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber
from main_sionna_simple import MU_MIMO_System

# =============================================================================
# SYSTEM-LEVEL SIMULATOR WITH SCHEDULER + PHY ABSTRACTION
# =============================================================================

class SystemLevelSimulator_WithScheduler(tf.keras.Model):
    """
    Evaluates precoder performance with:
    - Proportional Fair Scheduler
    - Link Adaptation (MCS selection)
    - PHY Abstraction (realistic throughput)
    """
    
    def __init__(self, base_system, num_slots=100, bler_target=0.1):
        super().__init__()
        
        self.base_system = base_system
        self.num_slots = num_slots
        self.bler_target = bler_target
        
        # Extract system parameters
        self.rg = base_system.rg
        self.sm = base_system.sm
        self.num_users = base_system.num_users
        self.num_bs_ant = base_system.num_bs_antennas
        self.num_streams_per_ut = self.rg.num_streams_per_tx
        
        # Initialize Sionna System Components
        self._init_system_components()
        
        print(f"\n{'='*80}")
        print(f"  SYSTEM-LEVEL SIMULATOR")
        print(f"{'='*80}")
        print(f"  Mode: Proportional Fair Scheduling + Link Adaptation")
        print(f"  Slots: {num_slots}")
        print(f"  BLER Target: {bler_target}")
        print(f"  Users: {self.num_users}")
        print(f"  Batch Size: 1 (required for scheduler state)")
        print(f"{'='*80}\n")
    
    def _init_system_components(self):
        """Initialize scheduler, link adaptation, and PHY abstraction"""
        
        # ✅ All components use batch_size=1
        system_batch_size = [1, 1]  # [num_batches=1, num_sectors=1]
        
        # 1. PHY Abstraction (SINR → Throughput mapping)
        self.phy_abs = PHYAbstraction()
        
        # 2. Outer-Loop Link Adaptation (MCS selection)
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            num_ut=self.num_users,
            batch_size=system_batch_size
        )
        
        # 3. Proportional Fair Scheduler
        self.scheduler = PFSchedulerSUMIMO(
            num_ut=self.num_users,
            num_freq_res=self.rg.fft_size,
            num_ofdm_sym=self.rg.num_ofdm_symbols,
            batch_size=system_batch_size,
            num_streams_per_ut=self.num_streams_per_ut,
            beta=0.98  # PF forgetting factor
        )
        
        # 4. Post-Equalization SINR Calculator
        self.lmmse_sinr = LMMSEPostEqualizationSINR(
            resource_grid=self.rg,
            stream_management=self.sm
        )
        
        print("✅ System components initialized (batch_size=1)")
    
    def _reset_state(self):
        """Reset scheduler and OLLA state"""
        self.olla.reset()
        self.olla.bler_target = self.bler_target
        self.olla.olla_delta_up = 0.2  # OLLA step size
        
        # Initialize feedback
        harq_feedback = -tf.ones([1, 1, self.num_users], dtype=tf.int32)
        sinr_eff_feedback = tf.ones([1, 1, self.num_users], dtype=tf.float32)
        num_decoded_bits = tf.zeros([1, 1, self.num_users], dtype=tf.int32)
        
        return harq_feedback, sinr_eff_feedback, num_decoded_bits
    
    @tf.function
    def simulate_slot(self, 
                     h_freq, 
                     no,
                     harq_feedback,
                     sinr_eff_feedback,
                     num_decoded_bits,
                     mcs_table_index=1):
        """Simulate one time slot with scheduling (batch_size=1)"""
        
        # ✅ Ensure batch_size=1
        h_freq = h_freq[:1, ...]
        
        # ============================================
        # 1. ESTIMATE ACHIEVABLE RATE
        # ============================================
        # Use last SINR feedback to estimate current rates
        # [1, 1, num_users]
        rate_achievable_est = log2(1.0 + tf.pow(10.0, self.olla.sinr_eff_db_last / 10.0))
        
        # Expand to full grid: [1, 1, num_ofdm, num_subcarriers, num_users]
        rate_achievable_est = insert_dims(rate_achievable_est, 2, axis=-2)
        rate_achievable_est = tf.tile(
            rate_achievable_est,
            [1, 1, self.rg.num_ofdm_symbols, self.rg.fft_size, 1]
        )
        
        # ============================================
        # 2. PROPORTIONAL FAIR SCHEDULING
        # ============================================
        # Output: [1, 1, num_ofdm, num_subcarriers, num_users, num_streams]
        is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)
        
        # Convert to float
        is_scheduled_float = tf.cast(is_scheduled, tf.float32)
        
        # Count allocated resources per user
        # [1, 1, num_users]
        num_allocated_re = tf.reduce_sum(
            tf.cast(is_scheduled, tf.int32),
            axis=[-1, -3, -4]  # Sum over streams, ofdm, subcarriers
        )
        
        # ============================================
        # 3. PRECODING (Your Transformer!)
        # ============================================
        # Create dummy input signal (batch_size=1)
        x_dummy = tf.ones([1, 1, self.num_users, 
                          self.rg.num_ofdm_symbols, self.rg.fft_size],
                         dtype=tf.complex64)
        
        # Get precoding matrix from trained transformer
        g = self.base_system.precoder((x_dummy, h_freq))
        
        # ============================================
        # 4. EFFECTIVE CHANNEL & SINR
        # ============================================
        h_eff = self.base_system.precoder.compute_effective_channel(h_freq, g)
        
        # Compute post-equalization SINR
        # [1, num_ofdm, num_subcarriers, num_users, num_streams]
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        
        # ✅ Ensure batch_size=1
        sinr = sinr[:1, ...]
        
        # Apply scheduling mask
        # is_scheduled_float: [1, 1, num_ofdm, num_subcarriers, num_users, num_streams]
        is_scheduled_mask = tf.squeeze(is_scheduled_float, axis=1)
        sinr_scheduled = sinr * is_scheduled_mask
        
        # ============================================
        # 5. LINK ADAPTATION (MCS Selection)
        # ============================================
        # [1, 1, num_users]
        mcs_index = self.olla(
            num_allocated_re,
            harq_feedback=harq_feedback,
            sinr_eff=sinr_eff_feedback
        )
        
        # ============================================
        # 6. PHY ABSTRACTION (Throughput Estimation)
        # ============================================
        # Expand dimensions to match expected input
        # sinr_scheduled: [1, num_ofdm, num_subcarriers, num_users, num_streams]
        # Need: [1, num_sectors, num_ofdm, num_subcarriers, num_users, num_streams]
        sinr_for_phy_abs = tf.expand_dims(sinr_scheduled, axis=1)
        
        # Returns: [1, 1, num_users] for all outputs
        num_decoded_bits_new, harq_feedback_new, sinr_eff_new, _, _ = \
            self.phy_abs(
                mcs_index,
                sinr=sinr_for_phy_abs,
                mcs_table_index=mcs_table_index,
                mcs_category=1  # Downlink
            )
        
        # Update feedback for next slot
        sinr_eff_feedback_new = tf.where(
            num_allocated_re > 0,
            sinr_eff_new,
            tf.zeros_like(sinr_eff_new)
        )
        
        # Calculate mean SINR only for scheduled resources
        sinr_mean = tf.reduce_sum(sinr_scheduled) / (tf.reduce_sum(is_scheduled_mask) + 1e-12)
        
        return {
            'num_decoded_bits': num_decoded_bits_new,
            'harq_feedback': harq_feedback_new,
            'sinr_eff_feedback': sinr_eff_feedback_new,
            'mcs_index': mcs_index,
            'num_allocated_re': num_allocated_re,
            'is_scheduled': is_scheduled,
            'sinr_mean': sinr_mean
        }
    
    def evaluate(self, batch_size, ebno_db):
        """
        Run system-level evaluation over multiple slots
        
        Args:
            batch_size: IGNORED (system-level always uses 1)
            ebno_db: SNR in dB
        
        Returns:
            results: Dict with per-user throughput, fairness, etc.
        """
        
        print(f"\n🔄 Running {self.num_slots} slots @ {ebno_db} dB...")
        print(f"   (Using batch_size=1, ignoring passed batch_size={batch_size})")
        
        # Initialize
        harq_fb, sinr_fb, decoded_bits = self._reset_state()
        
        # Storage
        total_bits_per_user = np.zeros(self.num_users)
        total_allocated_re = np.zeros(self.num_users)
        mcs_history = []
        
        for slot in range(self.num_slots):
            # ✅ Generate single channel (batch_size=1)
            self.base_system.new_topology(1)
            
            cir = self.base_system.channel_model(
                1,  # batch_size=1
                self.rg.num_ofdm_symbols,
                1.0 / self.rg.ofdm_symbol_duration
            )
            
            h_freq = cir_to_ofdm_channel(
                self.base_system.frequencies,
                *cir,
                normalize=True
            )
            
            # Compute noise power
            no = ebnodb2no(
                ebno_db,
                self.base_system.num_bits_per_symbol,
                0.5,
                self.rg
            )
            
            # Simulate slot
            results = self.simulate_slot(
                h_freq, no, harq_fb, sinr_fb, decoded_bits
            )
            
            # Update state
            harq_fb = results['harq_feedback']
            sinr_fb = results['sinr_eff_feedback']
            decoded_bits = results['num_decoded_bits']
            
            # Accumulate stats
            bits = results['num_decoded_bits'].numpy()[0, 0, :]
            allocated = results['num_allocated_re'].numpy()[0, 0, :]
            
            total_bits_per_user += bits
            total_allocated_re += allocated
            mcs_history.append(results['mcs_index'].numpy()[0, 0, :])
            
            if (slot + 1) % 20 == 0:
                avg_throughput = np.mean(total_bits_per_user) / (slot + 1)
                print(f"   Slot {slot+1}/{self.num_slots} | "
                      f"Avg Throughput: {avg_throughput:.1f} bits/slot")
        
        # Calculate metrics
        avg_throughput_per_user = total_bits_per_user / self.num_slots
        sum_throughput = np.sum(avg_throughput_per_user)
        
        # Fairness (Jain's index)
        jain_fairness = (np.sum(avg_throughput_per_user)**2) / \
                       (self.num_users * np.sum(avg_throughput_per_user**2) + 1e-12)
        
        # Spectral efficiency (bits/slot → bps/Hz)
        slot_duration = self.rg.ofdm_symbol_duration * self.rg.num_ofdm_symbols
        bandwidth = self.rg.fft_size * self.rg.subcarrier_spacing
        spectral_efficiency = sum_throughput / slot_duration / bandwidth
        
        print(f"\n✅ Evaluation Complete!")
        print(f"   Sum Throughput:     {sum_throughput:.1f} bits/slot")
        print(f"   Spectral Efficiency: {spectral_efficiency:.2f} bps/Hz")
        print(f"   Jain Fairness:      {jain_fairness:.3f}")
        print(f"   Per-User Throughput: {avg_throughput_per_user}")
        
        return {
            'sum_throughput': sum_throughput,
            'spectral_efficiency': spectral_efficiency,
            'per_user_throughput': avg_throughput_per_user,
            'jain_fairness': jain_fairness,
            'mcs_history': np.array(mcs_history),
            'total_allocated_re': total_allocated_re
        }
# =============================================================================
# USAGE EXAMPLE
# =============================================================================
def plot_system_level_results_with_mcs(all_results, snr_range):
    """
    Plot system-level metrics + MCS adaptation
    
    Creates 2 rows:
    - Top row: Spectral Efficiency, Sum Throughput, User Fairness
    - Bottom row: MCS Evolution, MCS Distribution, Per-User Throughput
    """
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = {'Transformer': 'red', 'RZF': 'blue', 'WMMSE': 'green'}
    markers = {'Transformer': 'o', 'RZF': 's', 'WMMSE': '^'}
    
    # =========================================================================
    # TOP ROW: System-Level Metrics (same as before)
    # =========================================================================
    
    # --- Plot 1: Spectral Efficiency ---
    ax1 = fig.add_subplot(gs[0, 0])
    for name, results in all_results.items():
        spectral_eff = [results[snr]['spectral_efficiency'] for snr in snr_range]
        ax1.plot(snr_range, spectral_eff, 'o-', label=name, 
                color=colors[name], linewidth=2.5, markersize=8,
                marker=markers[name])
    
    ax1.set_xlabel('SNR (dB)', fontsize=11)
    ax1.set_ylabel('Spectral Efficiency (bps/Hz)', fontsize=11)
    ax1.set_title('With Scheduler + Link Adaptation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Sum Throughput ---
    ax2 = fig.add_subplot(gs[0, 1])
    for name, results in all_results.items():
        sum_throughput = [results[snr]['sum_throughput'] for snr in snr_range]
        ax2.plot(snr_range, sum_throughput, 'o-', label=name,
                color=colors[name], linewidth=2.5, markersize=8,
                marker=markers[name])
    
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Sum Throughput (bits/slot)', fontsize=11)
    ax2.set_title('Total Throughput', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Fairness ---
    ax3 = fig.add_subplot(gs[0, 2])
    for name, results in all_results.items():
        fairness = [results[snr]['jain_fairness'] for snr in snr_range]
        ax3.plot(snr_range, fairness, 'o-', label=name,
                color=colors[name], linewidth=2.5, markersize=8,
                marker=markers[name])
    
    ax3.set_xlabel('SNR (dB)', fontsize=11)
    ax3.set_ylabel('Jain Fairness Index', fontsize=11)
    ax3.set_title('User Fairness', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.4, 1.05])
    
    # =========================================================================
    # BOTTOM ROW: MCS Analysis (NEW!)
    # =========================================================================
    
    # --- Plot 4: MCS Evolution Over Time (20 dB example) ---
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Pick SNR = 20 dB to show MCS adaptation
    target_snr = 20
    if target_snr in snr_range:
        for name, results in all_results.items():
            mcs_history = results[target_snr]['mcs_history']  # [num_slots, num_users]
            
            # Plot average MCS across users
            avg_mcs = np.mean(mcs_history, axis=1)
            slots = np.arange(len(avg_mcs))
            
            ax4.plot(slots, avg_mcs, '-', label=name, 
                    color=colors[name], linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Time Slot', fontsize=11)
    ax4.set_ylabel('Average MCS Index', fontsize=11)
    ax4.set_title(f'MCS Adaptation @ {target_snr} dB', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 100])
    
    # --- Plot 5: MCS Distribution (Box Plot) ---
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Collect MCS data for all SNRs
    mcs_data = {name: [] for name in all_results.keys()}
    x_positions = []
    x_labels = []
    
    for i, snr in enumerate(snr_range):
        for name in all_results.keys():
            mcs_history = all_results[name][snr]['mcs_history']
            # Flatten: all users, all slots
            mcs_flat = mcs_history.flatten()
            mcs_data[name].append(mcs_flat)
    
    # Create grouped box plots
    width = 0.25
    x = np.arange(len(snr_range))
    
    for i, (name, color) in enumerate(colors.items()):
        positions = x + (i - 1) * width
        
        # Box plot data
        data_to_plot = [mcs_data[name][j] for j in range(len(snr_range))]
        
        bp = ax5.boxplot(data_to_plot, positions=positions, widths=width*0.8,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color))
    
    ax5.set_xlabel('SNR (dB)', fontsize=11)
    ax5.set_ylabel('MCS Index', fontsize=11)
    ax5.set_title('MCS Distribution', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(snr_range)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[name], alpha=0.6, label=name) 
                      for name in all_results.keys()]
    ax5.legend(handles=legend_elements, fontsize=10)
    
    # --- Plot 6: Per-User Throughput @ 20 dB ---
    ax6 = fig.add_subplot(gs[1, 2])
    
    target_snr = 20
    if target_snr in snr_range:
        x = np.arange(4)  # 4 users
        width = 0.25
        
        for i, (name, color) in enumerate(colors.items()):
            per_user_tput = all_results[name][target_snr]['per_user_throughput']
            positions = x + (i - 1) * width
            
            bars = ax6.bar(positions, per_user_tput, width, 
                          label=name, color=color, alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show if non-zero
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('User Index', fontsize=11)
        ax6.set_ylabel('Throughput (bits/slot)', fontsize=11)
        ax6.set_title(f'Per-User Throughput @ {target_snr} dB', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['User 0', 'User 1', 'User 2', 'User 3'])
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Save Figure
    # =========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'system_level_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {filename}\n")
    
    plt.tight_layout()
    return fig


def main_system_level_evaluation():
    """Evaluate with MCS visualization"""
    
    print("\n" + "="*80)
    print("  SYSTEM-LEVEL EVALUATION WITH SCHEDULER")
    print("="*80 + "\n")
    
    # Create systems
    system_transformer = MU_MIMO_System(
        num_tx=4, num_rx=4, precoder_type="transformer", rb_size=4,
        batch_size=32,
        weights_path='/users/sala/test_projet/Trans/freq/Sionna/training_weight/best_20251208_123918'
    )
    
    system_rzf = MU_MIMO_System(num_tx=4, num_rx=4, precoder_type="rzf", batch_size=32)
    system_wmmse = MU_MIMO_System(num_tx=4, num_rx=4, precoder_type="wmmse", batch_size=32)
    
    # Wrap in system-level simulators
    sys_level_transformer = SystemLevelSimulator_WithScheduler(system_transformer, num_slots=100)
    sys_level_rzf = SystemLevelSimulator_WithScheduler(system_rzf, num_slots=100)
    sys_level_wmmse = SystemLevelSimulator_WithScheduler(system_wmmse, num_slots=100)
    
    # Evaluate
    snr_range = [10, 15, 20, 25]
    batch_size = 32
    
    print("\n" + "="*80)
    print("  TRANSFORMER EVALUATION")
    print("="*80)
    results_transformer = {}
    for snr in snr_range:
        print(f"\n{'='*60}")
        print(f"  SNR = {snr} dB")
        print(f"{'='*60}")
        results_transformer[snr] = sys_level_transformer.evaluate(batch_size, float(snr))
    
    print("\n" + "="*80)
    print("  RZF BASELINE")
    print("="*80)
    results_rzf = {}
    for snr in snr_range:
        print(f"\n{'='*60}")
        print(f"  SNR = {snr} dB")
        print(f"{'='*60}")
        results_rzf[snr] = sys_level_rzf.evaluate(batch_size, float(snr))
    
    print("\n" + "="*80)
    print("  WMMSE BASELINE")
    print("="*80)
    results_wmmse = {}
    for snr in snr_range:
        print(f"\n{'='*60}")
        print(f"  SNR = {snr} dB")
        print(f"{'='*60}")
        results_wmmse[snr] = sys_level_wmmse.evaluate(batch_size, float(snr))
    
    # ✅ Plot with MCS analysis
    all_results = {
        'Transformer': results_transformer,
        'RZF': results_rzf,
        'WMMSE': results_wmmse
    }
    
    fig = plot_system_level_results_with_mcs(all_results, snr_range)
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY @ 20 dB")
    print("="*80)
    
    for name in ['Transformer', 'RZF', 'WMMSE']:
        res = all_results[name][20]
        print(f"\n{name}:")
        print(f"  Spectral Efficiency: {res['spectral_efficiency']:.2f} bps/Hz")
        print(f"  Sum Throughput:      {res['sum_throughput']:.1f} bits/slot")
        print(f"  Jain Fairness:       {res['jain_fairness']:.3f}")
        print(f"  Per-User Throughput: {res['per_user_throughput']}")
        
        # MCS statistics
        mcs_hist = res['mcs_history']
        print(f"  MCS Range:           [{np.min(mcs_hist):.0f}, {np.max(mcs_hist):.0f}]")
        print(f"  MCS Mean:            {np.mean(mcs_hist):.1f}")
    
    return all_results


if __name__ == "__main__":
    results = main_system_level_evaluation()


def plot_system_level_results(all_results, snr_range):
    """Plot system-level metrics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Transformer': 'red', 'RZF': 'blue', 'WMMSE': 'green'}
    
    for name, results in all_results.items():
        # Extract metrics
        spectral_eff = [results[snr]['spectral_efficiency'] for snr in snr_range]
        sum_throughput = [results[snr]['sum_throughput'] for snr in snr_range]
        fairness = [results[snr]['jain_fairness'] for snr in snr_range]
        
        # Plot
        axes[0].plot(snr_range, spectral_eff, 'o-', label=name, 
                    color=colors[name], linewidth=2.5, markersize=8)
        axes[1].plot(snr_range, sum_throughput, 'o-', label=name,
                    color=colors[name], linewidth=2.5, markersize=8)
        axes[2].plot(snr_range, fairness, 'o-', label=name,
                    color=colors[name], linewidth=2.5, markersize=8)
    
    axes[0].set(xlabel='SNR (dB)', ylabel='Spectral Efficiency (bps/Hz)',
               title='With Scheduler + Link Adaptation')
    axes[1].set(xlabel='SNR (dB)', ylabel='Sum Throughput (bits/slot)',
               title='Total Throughput')
    axes[2].set(xlabel='SNR (dB)', ylabel='Jain Fairness Index',
               title='User Fairness')
    
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'system_level_comparison_{timestamp}.png', dpi=300)
    print(f"\n✅ Plot saved: system_level_comparison_{timestamp}.png\n")


