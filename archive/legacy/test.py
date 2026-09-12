"""
Frequency Correlation Study for Sionna CDL Channels
===================================================
Analyze how frequency correlation affects RB grouping performance
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime

# Sionna imports

import sionna

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer,
                             OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers)
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, ApplyOFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber
sionna.phy.config.seed = 42

print("✅ Imports successful")


class SionnaFrequencyAnalyzer:
    """Analyze frequency correlation in Sionna CDL channels"""
    
    def __init__(self, cdl_model="A", delay_spread=30e-9, num_samples=1000):
        self.cdl_model = cdl_model
        self.delay_spread = delay_spread
        self.num_samples = num_samples
        
        # System parameters (match your main training)
        self.carrier_frequency = 2.6e9
        self.fft_size = 72
        self.subcarrier_spacing = 30e3
        self.num_ofdm_symbols = 14
        self.num_bs_ant = 8
        self.num_ut_ant = 4
        
        # Create antenna arrays
        self.ut_array = AntennaArray(
            num_rows=1, num_cols=self.num_ut_ant//2,
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=self.carrier_frequency
        )
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=self.num_bs_ant//2,
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=self.carrier_frequency
        )
        
        # Create CDL channel
        self.cdl = CDL(
            model=cdl_model, delay_spread=delay_spread,
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction="downlink", min_speed=0.0
        )
        
        # Resource grid for frequency grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self.num_ut_ant,
            cyclic_prefix_length=6,
            num_guard_carriers=[5, 6],
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11]
        )
        
        self.frequencies = subcarrier_frequencies(self.fft_size, self.subcarrier_spacing)
        
        print(f"\n[ANALYZER CONFIG]")
        print(f"  CDL Model: {cdl_model}")
        print(f"  Delay Spread: {delay_spread*1e9:.1f} ns")
        print(f"  FFT Size: {self.fft_size}")
        print(f"  Num samples: {num_samples}")
        print(f"  TX antennas: {self.num_bs_ant}")
        print(f"  RX antennas: {self.num_ut_ant}")
    
    def generate_channel_samples(self):
        """Generate channel realizations"""
        print("\n[1/5] Generating channel samples...")
        
        # Generate CIR
        a, tau = self.cdl(
            batch_size=self.num_samples,
            num_time_steps=self.num_ofdm_symbols,
            sampling_frequency=1/self.rg.ofdm_symbol_duration
        )
        
        # Convert to frequency domain
        h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        # Shape: [batch, num_rx=1, num_users, num_tx=1, num_bs_ant, num_ofdm, fft_size]
        
        # Clean up dimensions
        h_clean = tf.reshape(h_freq, [self.num_samples, self.num_ut_ant, self.num_bs_ant,
                              self.num_ofdm_symbols, self.fft_size]).numpy()  # [num_samples, num_users, num_bs_ant, num_ofdm, fft_size]
        
        print(f"  ✅ Generated shape: {h_clean.shape}")
        print(f"  ✅ Channel samples: {self.num_samples}")
        
        return h_clean
    
    def compute_frequency_correlation_matrix(self, h_data):
        """Compute correlation between all subcarrier pairs"""
        print("\n[2/5] Computing frequency correlation matrix...")
        
        # Average over users, antennas, and OFDM symbols
        # h_data: [num_samples, num_users, num_bs_ant, num_ofdm, fft_size]
        h_avg = np.mean(np.abs(h_data), axis=(1, 2, 3))  # [num_samples, fft_size]
        
        num_subcarriers = h_avg.shape[1]
        corr_matrix = np.zeros((num_subcarriers, num_subcarriers))
        
        for i in range(num_subcarriers):
            for j in range(num_subcarriers):
                corr_matrix[i, j] = pearsonr(h_avg[:, i], h_avg[:, j])[0]
        
        print(f"  ✅ Correlation matrix: {corr_matrix.shape}")
        print(f"  ✅ Mean correlation: {np.mean(corr_matrix):.4f}")
        
        return corr_matrix
    
    def compute_adjacent_correlation(self, h_data, max_distance=20):
        """Compute correlation vs subcarrier distance"""
        print("\n[3/5] Computing adjacent subcarrier correlation...")
        
        h_avg = np.mean(np.abs(h_data), axis=(1, 2, 3))  # [num_samples, fft_size]
        num_subcarriers = h_avg.shape[1]
        
        distances = list(range(1, min(max_distance + 1, num_subcarriers)))
        correlations = []
        
        for d in distances:
            corr_values = []
            for i in range(num_subcarriers - d):
                corr = pearsonr(h_avg[:, i], h_avg[:, i + d])[0]
                corr_values.append(corr)
            correlations.append(np.mean(corr_values))
        
        print(f"  ✅ Computed for distances 1-{max(distances)}")
        print(f"  ✅ Correlation at distance 1: {correlations[0]:.4f}")
        print(f"  ✅ Correlation at distance {max(distances)}: {correlations[-1]:.4f}")
        
        return distances, correlations
    
    def estimate_coherence_bandwidth(self, distances, correlations, threshold=0.7):
        """Estimate coherence bandwidth in subcarriers"""
        print("\n[4/5] Estimating coherence bandwidth...")
        
        for i, corr in enumerate(correlations):
            if corr < threshold:
                coherence_bw = distances[i]
                print(f"  ✅ Coherence bandwidth: {coherence_bw} subcarriers")
                print(f"  ✅ (correlation drops below {threshold} at distance {coherence_bw})")
                return coherence_bw
        
        coherence_bw = distances[-1]
        print(f"  ✅ Coherence bandwidth: >{coherence_bw} subcarriers")
        print(f"  ✅ (correlation stays above {threshold} for all distances tested)")
        return coherence_bw
    
    def plot_correlation_results(self, corr_matrix, distances, correlations, 
                                 coherence_bw, output_dir="frequency_study_sionna"):
        """Plot all correlation analysis results"""
        print("\n[5/5] Plotting results...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Correlation Matrix Heatmap
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   vmin=-0.2, vmax=1.0, square=False, ax=ax1,
                   cbar_kws={'label': 'Correlation'})
        ax1.set_title(f'Frequency Correlation Matrix\n({self.cdl_model} channel, delay spread={self.delay_spread*1e9:.0f}ns)')
        ax1.set_xlabel('Subcarrier Index')
        ax1.set_ylabel('Subcarrier Index')
        
        # 2. Correlation vs Distance
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(distances, correlations, 'b-o', linewidth=2, markersize=6)
        ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=1.5, label='Correlation = 0.5')
        ax2.axhline(y=0.7, color='g', linestyle='--', linewidth=1.5, label='Correlation = 0.7')
        ax2.axvline(x=coherence_bw, color='orange', linestyle=':', linewidth=2, 
                   label=f'Coherence BW = {coherence_bw}')
        ax2.set_xlabel('Subcarrier Distance')
        ax2.set_ylabel('Average Correlation')
        ax2.set_title('Frequency Correlation vs Subcarrier Separation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1.05])
        
        # 3. Zoomed Correlation (first 10 subcarriers)
        ax3 = plt.subplot(2, 3, 3)
        zoom_distance = min(10, len(distances))
        ax3.plot(distances[:zoom_distance], correlations[:zoom_distance], 
                'b-o', linewidth=2, markersize=8)
        ax3.axhline(y=0.7, color='g', linestyle='--', linewidth=1.5, label='Threshold = 0.7')
        ax3.set_xlabel('Subcarrier Distance')
        ax3.set_ylabel('Average Correlation')
        ax3.set_title('Correlation (Zoomed: First 10 Subcarriers)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([0.5, 1.05])
        
        # 4. Diagonal slice of correlation matrix
        ax4 = plt.subplot(2, 3, 4)
        diagonal_slice = np.diagonal(corr_matrix, offset=0)
        for offset in [1, 2, 4, 8]:
            if offset < corr_matrix.shape[0]:
                diag = np.diagonal(corr_matrix, offset=offset)
                ax4.plot(diag, label=f'Offset={offset}', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Subcarrier Index')
        ax4.set_ylabel('Correlation')
        ax4.set_title('Correlation Along Diagonals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. RB Size Recommendations
        ax5 = plt.subplot(2, 3, 5)
        rb_sizes = [1,2,4,6,8,10,12,14,16,18,20,24,26,28,30,36,42,48,54,60,66,72]

        rb_efficiency = []
        for rb in rb_sizes:
            if rb <= coherence_bw:
                efficiency = 1.0
            else:
                efficiency = coherence_bw / rb
            rb_efficiency.append(efficiency)
        
        colors = ['green' if eff >= 0.8 else 'orange' if eff >= 0.6 else 'red' 
                 for eff in rb_efficiency]
        ax5.bar(range(len(rb_sizes)), rb_efficiency, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(rb_sizes)))
        ax5.set_xticklabels(rb_sizes)
        ax5.set_xlabel('RB Size (subcarriers)')
        ax5.set_ylabel('Expected Efficiency')
        ax5.set_title(f'RB Size Recommendations\n(Coherence BW = {coherence_bw} subcarriers)')
        ax5.axhline(y=1.0, color='g', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axhline(y=0.8, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim([0, 1.1])
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
FREQUENCY CORRELATION ANALYSIS
{'='*40}

Channel Model: {self.cdl_model}
Delay Spread: {self.delay_spread*1e9:.1f} ns
Samples: {self.num_samples}

RESULTS:
{'='*40}

Coherence Bandwidth: {coherence_bw} subcarriers
  ({coherence_bw * self.subcarrier_spacing / 1e3:.1f} kHz)

Mean Correlation: {np.mean(corr_matrix):.4f}

Correlation at distance 1: {correlations[0]:.4f}
Correlation at distance 4: {correlations[3]:.4f}
Correlation at distance 8: {correlations[7] if len(correlations) > 7 else 'N/A'}

RECOMMENDED RB SIZES:
{'='*40}

✅ Excellent (>80% efficiency):
   RB ≤ {coherence_bw}

⚠️  Acceptable (60-80% efficiency):
   RB = {coherence_bw + 1} to {int(coherence_bw * 1.33)}

❌ Poor (<60% efficiency):
   RB > {int(coherence_bw * 1.33)}

CURRENT SYSTEM:
{'='*40}

FFT Size: {self.fft_size} subcarriers
Current RB size: 4 subcarriers
Status: {'✅ GOOD' if 4 <= coherence_bw else '⚠️ RECONSIDER'}
        """
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        filename = f'frequency_analysis_{self.cdl_model}_{self.delay_spread*1e9:.0f}ns_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {filepath}")
        
        plt.show()
        
        return filepath


def analyze_multiple_delay_spreads():
    """Compare frequency correlation for different delay spreads"""
    print("\n" + "="*80)
    print("ANALYZING MULTIPLE DELAY SPREADS")
    print("="*80)
    
    delay_spreads = [30e-9, 100e-9, 300e-9]  # 30ns à 300ns

    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, ds in enumerate(delay_spreads):
        print(f"\n--- Delay Spread: {ds*1e9:.0f} ns ---")
        
        analyzer = SionnaFrequencyAnalyzer(
            cdl_model="A",
            delay_spread=ds,
            num_samples=500  # Fewer samples for speed
        )
        
        h_data = analyzer.generate_channel_samples()
        distances, correlations = analyzer.compute_adjacent_correlation(h_data, max_distance=20)
        coherence_bw = analyzer.estimate_coherence_bandwidth(distances, correlations)
        
        results[ds] = {
            'distances': distances,
            'correlations': correlations,
            'coherence_bw': coherence_bw
        }
        
        # Plot
        ax = axes[idx]
        ax.plot(distances, correlations, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=0.7, color='r', linestyle='--', linewidth=1.5, label='Threshold = 0.7')
        ax.axvline(x=coherence_bw, color='orange', linestyle=':', linewidth=2,
                  label=f'Coherence BW = {coherence_bw}')
        ax.set_xlabel('Subcarrier Distance')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Delay Spread = {ds*1e9:.0f} ns\nCoherence BW = {coherence_bw} subcarriers')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('frequency_study_sionna/delay_spread_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Comparison plot saved: frequency_study_sionna/delay_spread_comparison.png")
    plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Delay Spread vs Coherence Bandwidth")
    print("="*80)
    for ds in delay_spreads:
        coh_bw = results[ds]['coherence_bw']
        print(f"  Delay Spread {ds*1e9:5.0f} ns → Coherence BW: {coh_bw:2d} subcarriers " +
              f"({coh_bw * 15:.0f} kHz)")
    print("="*80)


def main():
    """Main frequency study execution"""
    print("\n" + "="*80)
    print("SIONNA CHANNEL FREQUENCY CORRELATION STUDY")
    print("="*80)
    
    # Single analysis with default parameters
    analyzer = SionnaFrequencyAnalyzer(
    cdl_model="A",  # plus dispersé que A
    delay_spread=300e-9,
    num_samples=500)
    
    # Generate channels
    h_data = analyzer.generate_channel_samples()
    
    # Compute correlations
    corr_matrix = analyzer.compute_frequency_correlation_matrix(h_data)
    distances, correlations = analyzer.compute_adjacent_correlation(h_data, max_distance=20)
    coherence_bw = analyzer.estimate_coherence_bandwidth(distances, correlations, threshold=0.7)
    
    # Plot results
    analyzer.plot_correlation_results(corr_matrix, distances, correlations, coherence_bw)
    
    # Multi-delay analysis
    print("\n" + "="*80)

    #analyze_multiple_delay_spreads()
    
    print("\n" + "="*80)
    print("✅ FREQUENCY STUDY COMPLETE!")
    print("="*80)
    print(f"\nKey Finding: Coherence Bandwidth = {coherence_bw} subcarriers")
    print(f"Recommendation: Use RB size ≤ {coherence_bw} for optimal performance")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()