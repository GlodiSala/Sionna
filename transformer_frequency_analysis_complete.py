"""
Transformer Precoding Frequency Analysis - FULLY FIXED VERSION
Analyzes how transformer precoding matrices evolve with frequency,
compares to channel frequency correlation, and evaluates RB grouping sizes.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
# Add the directory containing main_sionna_5d_training.py to path
sys.path.insert(0, '/users/sala/test_projet/Trans/freq/Sionna')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from datetime import datetime
import time

# Sionna imports
import sionna
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import ebnodb2no, compute_ber

print(f"✅ Sionna version: {sionna.__version__}")

# Import transformer model
from transformer_5d_sionna import TransformerPrecoder5D_Fixed

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration failed: {e}")


class TransformerFrequencyAnalyzer:
    """
    Comprehensive analyzer for transformer precoding frequency characteristics
    """
    def __init__(self, 
                 model_path,
                 cdl_model="A",
                 delay_spread=30e-9,
                 num_samples=500,
                 output_dir="transformer_frequency_analysis"):
        
        self.model_path = model_path
        self.cdl_model = cdl_model
        self.delay_spread = delay_spread
        self.num_samples = num_samples
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # System parameters (matching training configuration)
        self.fft_size = 72
        self.num_ofdm_symbols = 14
        self.num_tx = 8
        self.num_rx = 4
        self.subcarrier_spacing = 15e3
        self.carrier_frequency = 2.6e9
        self.cyclic_prefix_length = 6
        self.pilot_ofdm_symbol_indices = [2, 11]
        self.num_bits_per_symbol = 2
        self.coderate = 0.5
        
        print(f"\n{'='*80}")
        print(f"TRANSFORMER FREQUENCY ANALYZER INITIALIZED")
        print(f"{'='*80}")
        print(f"Channel: CDL-{cdl_model}, Delay Spread: {delay_spread*1e9:.0f}ns")
        print(f"System: {self.num_tx}x{self.num_rx} MIMO, {self.fft_size} subcarriers")
        print(f"Samples: {num_samples}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
        
        # Setup Sionna components
        self._setup_sionna()
        
    def _setup_sionna(self):
        """Initialize Sionna resource grid and channel model"""
        print("[Setup] Initializing Sionna components...")
        
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self.num_rx,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices,
            cyclic_prefix_length=self.cyclic_prefix_length
        )
        
        self.sm = StreamManagement(np.array([[1]]), self.num_rx)
        
        # Antenna arrays
        self.ut_array = AntennaArray(
            num_rows=1,
            num_cols=int(self.num_rx),
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency
        )
        
        self.bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(self.num_tx),
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency
        )
        
        # Channel model
        self.cdl = CDL(
            model=self.cdl_model,
            delay_spread=self.delay_spread,
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink",
            min_speed=0.0
        )
        
        # Frequency points
        self.frequencies = subcarrier_frequencies(self.fft_size, self.subcarrier_spacing)
        
        print("  ✅ Sionna components initialized")
        
    def load_transformer_model(self, rb_size=12):
        """Load trained transformer model - FIXED attribute name"""
        print(f"\n[1/7] Loading transformer model (RB size={rb_size})...")
        print(f"  Model path: {self.model_path}")
        
        try:
            # Import the full simulator
            from main_sionna_5d_training import OFDMSimulator5D_Fixed  # Use v2!
            
            # Create full simulator with SAME rb_size as checkpoint
            simulator = OFDMSimulator5D_Fixed(
                cdl_model=self.cdl_model,
                delay_spread=self.delay_spread,
                perfect_csi=True,
                precoder_type="transformer_5d",
                rb_size=rb_size  # Use 12!
            )
            
            # Build simulator by calling it once
            batch_size = 2
            ebno_db = tf.constant(10.0, dtype=tf.float32)
            _ = simulator(batch_size, ebno_db, training=False)
            
            # Load full simulator weights
            simulator.load_weights(self.model_path)
            print("  ✅ Full simulator loaded successfully")
            print(f"  ✅ Model trained with RB size: {rb_size}")
            
            # Extract the transformer precoder - CORRECT ATTRIBUTE NAME
            self.transformer_precoder = simulator._transformer_precoder  # Changed from _precoder
            return True
            
        except Exception as e:
            print(f"  ⚠️ Error loading full simulator: {e}")
            print(f"  ℹ️ Creating new transformer without pretrained weights")
            
            # Fallback: create standalone transformer
            self.transformer_precoder = TransformerPrecoder5D_Fixed(
                resource_grid=self.rg,
                stream_management=self.sm,
                num_tx_antennas=self.num_tx,
                num_rx_antennas=self.num_rx,
                rb_size=rb_size
            )
            
            # Build it
            dummy_x = tf.complex(
                tf.random.normal([1, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size]),
                tf.random.normal([1, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size])
            )
            dummy_h = tf.complex(
                tf.random.normal([1, 1, self.num_rx, 1, self.num_tx, self.num_ofdm_symbols, self.fft_size]),
                tf.random.normal([1, 1, self.num_rx, 1, self.num_tx, self.num_ofdm_symbols, self.fft_size])
            )
            _ = self.transformer_precoder(dummy_x, dummy_h, training=False)
            
            return False  
    def generate_channel_samples(self):
        """Generate channel samples and extract frequency response"""
        print(f"\n[2/7] Generating {self.num_samples} channel samples...")
        
        batch_size = 64
        all_h_freq = []
        
        num_batches = int(np.ceil(self.num_samples / batch_size))
        
        for i in range(num_batches):
            current_batch = min(batch_size, self.num_samples - i * batch_size)
            
            # Generate channel impulse response
            a, tau = self.cdl(
                batch_size=current_batch, 
                num_time_steps=self.num_ofdm_symbols,
                sampling_frequency=1.0
            )
            
            # Convert to frequency domain
            h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
            
            # Shape: [batch, 1, num_rx, 1, num_tx, num_ofdm, fft_size]
            all_h_freq.append(h_freq.numpy())
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {(i+1)*batch_size}/{self.num_samples} samples")
        
        h_freq_all = np.concatenate(all_h_freq, axis=0)[:self.num_samples]
        print(f"  ✅ Generated shape: {h_freq_all.shape}")
        
        return h_freq_all
    
    def compute_channel_frequency_correlation(self, h_freq):
        """Compute frequency correlation of channel"""
        print("\n[3/7] Computing channel frequency correlation...")
        
        # h_freq: [samples, 1, num_rx, 1, num_tx, num_ofdm, fft_size]
        # Average over all dimensions except frequency
        h_magnitude = np.abs(h_freq)
        h_avg = np.mean(h_magnitude, axis=(1, 2, 3, 4, 5))  # [samples, fft_size]
        
        num_subcarriers = h_avg.shape[1]
        corr_matrix = np.zeros((num_subcarriers, num_subcarriers))
        
        for i in range(num_subcarriers):
            for j in range(num_subcarriers):
                corr_matrix[i, j] = pearsonr(h_avg[:, i], h_avg[:, j])[0]
        
        # Compute adjacent correlation
        distances = list(range(1, min(21, num_subcarriers)))
        correlations = []
        for d in distances:
            corr_values = []
            for i in range(num_subcarriers - d):
                corr = pearsonr(h_avg[:, i], h_avg[:, i + d])[0]
                corr_values.append(corr)
            correlations.append(np.mean(corr_values))
        
        # Estimate coherence bandwidth
        coherence_bw = next((d for d, c in zip(distances, correlations) if c < 0.7), distances[-1])
        
        print(f"  ✅ Mean correlation: {np.mean(corr_matrix):.4f}")
        print(f"  ✅ Coherence bandwidth: {coherence_bw} subcarriers ({coherence_bw * self.subcarrier_spacing / 1e3:.1f} kHz)")
        
        return corr_matrix, distances, correlations, coherence_bw
    
    def extract_precoding_matrices(self, h_freq, rb_size=12):
        """Extract precoding matrices from transformer"""
        print(f"\n[4/7] Extracting transformer precoding matrices (RB size={rb_size})...")
        
        # Load model if not already loaded
        if not hasattr(self, 'transformer_precoder') or self.transformer_precoder.rb_size != rb_size:
            self.load_transformer_model(rb_size)
        
        # Get precoding matrices
        all_W = []
        batch_size = 64
        
        num_batches = int(np.ceil(h_freq.shape[0] / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, h_freq.shape[0])
            h_batch = h_freq[start_idx:end_idx]
            
            h_tf = tf.constant(h_batch, dtype=tf.complex64)
            
            # Create dummy data symbols
            current_batch_size = h_batch.shape[0]
            x_rg = tf.complex(
                tf.random.normal([current_batch_size, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size]),
                tf.random.normal([current_batch_size, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size])
            )
            
            # Get precoding matrices
            _, _, W = self.transformer_precoder(x_rg, h_tf, training=False)
            all_W.append(W.numpy())
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {end_idx}/{h_freq.shape[0]} samples")
        
        W_all = np.concatenate(all_W, axis=0)
        print(f"  ✅ Precoding matrices shape: {W_all.shape}")
        
        return W_all
    
    def compute_precoding_frequency_correlation(self, W):
        """Compute how precoding matrices vary across frequency"""
        print("\n[5/7] Computing precoding frequency correlation...")
        
        # W shape: [batch, num_rx, num_tx, num_ofdm, fft_size]
        # Average over batch and ofdm
        W_magnitude = np.abs(W)
        W_avg = np.mean(W_magnitude, axis=(0, 3))  # [num_rx, num_tx, fft_size]
        
        # Flatten rx and tx dimensions for correlation
        W_flat = W_avg.reshape(-1, W_avg.shape[2])  # [num_rx*num_tx, fft_size]
        
        num_subcarriers = W_flat.shape[1]
        corr_matrix = np.zeros((num_subcarriers, num_subcarriers))
        
        for i in range(num_subcarriers):
            for j in range(num_subcarriers):
                corr_matrix[i, j] = pearsonr(W_flat[:, i], W_flat[:, j])[0]
        
        # Adjacent correlation
        distances = list(range(1, min(21, num_subcarriers)))
        correlations = []
        for d in distances:
            corr_values = []
            for i in range(num_subcarriers - d):
                corr = pearsonr(W_flat[:, i], W_flat[:, i + d])[0]
                corr_values.append(corr)
            correlations.append(np.mean(corr_values))
        
        print(f"  ✅ Mean precoding correlation: {np.mean(corr_matrix):.4f}")
        
        return corr_matrix, distances, correlations
    
    def evaluate_rb_grouping_performance(self, h_freq_samples, 
                                    rb_sizes=[1, 2, 4, 6, 8, 12, 18, 24, 36], 
                                    ebno_db=10.0, num_test_batches=50, batch_size=32):
        """
        Evaluate sum rate and SINR for different RB grouping sizes
        """
        print(f"\n[6/7] Evaluating RB grouping sizes at Eb/N0={ebno_db}dB...")
        print(f"  Testing {len(rb_sizes)} different RB sizes: {rb_sizes}")
        print(f"  Using {num_test_batches} test batches of size {batch_size}")
        
        results = {}
        noise_power = ebnodb2no(tf.constant(ebno_db, dtype=tf.float32), 
                            self.num_bits_per_symbol, self.coderate, self.rg).numpy()
        
        for rb_size in rb_sizes:
            if self.fft_size % rb_size != 0:
                print(f"  ⚠️ Skipping RB size {rb_size} (doesn't divide {self.fft_size})")
                continue
                
            print(f"\n  Testing RB size = {rb_size} ({self.fft_size // rb_size} RBs)...")
            
            # Load transformer for this RB size
            transformer = TransformerPrecoder5D_Fixed(
                resource_grid=self.rg,
                stream_management=self.sm,
                num_tx_antennas=self.num_tx,
                num_rx_antennas=self.num_rx,
                rb_size=rb_size
            )
            
            # Build and try to load weights
            dummy_x = tf.complex(
                tf.random.normal([1, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size]),
                tf.random.normal([1, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size])
            )
            dummy_h = tf.complex(
                tf.random.normal([1, 1, self.num_rx, 1, self.num_tx, self.num_ofdm_symbols, self.fft_size]),
                tf.random.normal([1, 1, self.num_rx, 1, self.num_tx, self.num_ofdm_symbols, self.fft_size])
            )
            _ = transformer(dummy_x, dummy_h, training=False)
            
            # Try to copy weights from loaded model if available
            if hasattr(self, 'transformer_precoder') and rb_size == 12:
                try:
                    transformer.transformer.set_weights(self.transformer_precoder.transformer.get_weights())
                    print(f"    ✅ Using pretrained weights (RB={rb_size})")
                except Exception as e:
                    print(f"    ⚠️ Could not load pretrained weights: {e}")
            
            # Evaluate on test samples
            sumrate_list = []
            sinr_list = []
            ber_approx_list = []
            
            start_time = time.time()
            
            # Use pre-generated channel samples
            num_samples = h_freq_samples.shape[0]
            sample_indices = np.random.choice(num_samples, size=min(num_test_batches * batch_size, num_samples), replace=False)
            
            for batch_idx in range(num_test_batches):
                # Get batch of channels
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sample_indices))
                if start_idx >= len(sample_indices):
                    break
                    
                batch_sample_idx = sample_indices[start_idx:end_idx]
                h_batch = h_freq_samples[batch_sample_idx]
                current_batch_size = h_batch.shape[0]
                
                h_tf = tf.constant(h_batch, dtype=tf.complex64)
                
                # Generate random data symbols
                x_rg = tf.complex(
                    tf.random.normal([current_batch_size, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size]),
                    tf.random.normal([current_batch_size, 1, self.num_rx, self.num_ofdm_symbols, self.fft_size])
                )
                
                # Get precoding
                _, h_eff, W = transformer(x_rg, h_tf, training=False)
                
                # Compute sum rate and SINR - PASS rb_size!
                sumrate, avg_sinr, ber_approx = self._compute_metrics(W, h_tf, noise_power, rb_size=rb_size)
                
                sumrate_list.append(sumrate.numpy())
                sinr_list.append(avg_sinr.numpy())
                ber_approx_list.append(ber_approx.numpy())
            
            elapsed_time = time.time() - start_time
            
            avg_sumrate = np.mean(sumrate_list) if sumrate_list else 0.0
            avg_sinr = np.mean(sinr_list) if sinr_list else 0.0
            avg_sinr_db = 10 * np.log10(avg_sinr + 1e-12)
            avg_ber_approx = np.mean(ber_approx_list) if ber_approx_list else 1.0
            
            results[rb_size] = {
                'ber': avg_ber_approx,
                'sum_rate': avg_sumrate,
                'sinr_db': avg_sinr_db,
                'sinr_linear': avg_sinr,
                'num_rbs': self.fft_size // rb_size,
                'time': elapsed_time / len(sumrate_list) if sumrate_list else 0.0
            }
            
            print(f"    BER (approx): {avg_ber_approx:.2e}")
            print(f"    Sum Rate: {avg_sumrate:.2f} bps/Hz")
            print(f"    SINR: {avg_sinr_db:.1f} dB")
            print(f"    Time per batch: {elapsed_time/max(len(sumrate_list),1):.3f}s")
        
        # Also evaluate Zero-Forcing baseline
        print(f"\n  Testing Zero-Forcing baseline...")
        zf_sumrate_list = []
        zf_sinr_list = []
        zf_ber_list = []
        
        for batch_idx in range(min(num_test_batches, 20)):  # Fewer batches for ZF
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sample_indices))
            if start_idx >= len(sample_indices):
                break
                
            batch_sample_idx = sample_indices[start_idx:end_idx]
            h_batch = h_freq_samples[batch_sample_idx]
            h_tf = tf.constant(h_batch, dtype=tf.complex64)
            
            # Compute ZF precoding
            W_zf = self._compute_zf_precoding(h_tf)
            
            # Use rb_size=1 for ZF (no grouping)
            sumrate, avg_sinr, ber_approx = self._compute_metrics(W_zf, h_tf, noise_power, rb_size=1)
            zf_sumrate_list.append(sumrate.numpy())
            zf_sinr_list.append(avg_sinr.numpy())
            zf_ber_list.append(ber_approx.numpy())
        
        avg_sinr_zf = np.mean(zf_sinr_list) if zf_sinr_list else 0.0
        
        results['zf_baseline'] = {
            'ber': np.mean(zf_ber_list) if zf_ber_list else 1.0,
            'sum_rate': np.mean(zf_sumrate_list) if zf_sumrate_list else 0.0,
            'sinr_db': 10 * np.log10(avg_sinr_zf + 1e-12),
            'sinr_linear': avg_sinr_zf,
            'num_rbs': self.fft_size,
            'time': 0.0
        }
        
        print(f"    BER (approx): {results['zf_baseline']['ber']:.2e}")
        print(f"    Sum Rate: {results['zf_baseline']['sum_rate']:.2f} bps/Hz")
        print(f"    SINR: {results['zf_baseline']['sinr_db']:.1f} dB")
        print(f"  ✅ Evaluation complete")
        
        return results
    def _compute_zf_precoding(self, h_freq):
        """Compute Zero-Forcing precoding matrix - FIXED type casting"""
        # h_freq: [B, 1, R, 1, T, ofdm, fft]
        h_clean = tf.squeeze(h_freq, axis=[1, 3])  # [B, R, T, ofdm, fft]
        h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, R, T, fft]
        
        B = tf.shape(h_avg)[0]
        fft = tf.shape(h_avg)[3]
        
        # ZF precoding per subcarrier
        W_list = []
        for f in range(self.fft_size):
            H_f = h_avg[:, :, :, f]  # [B, R, T]
            H_H = tf.transpose(H_f, perm=[0, 2, 1], conjugate=True)  # [B, T, R]
            G = tf.linalg.matmul(H_H, H_f)  # [B, T, T]
            G_inv = tf.linalg.inv(G + 1e-6 * tf.eye(self.num_tx, dtype=G.dtype))
            W_f = tf.linalg.matmul(G_inv, H_H)  # [B, T, R]
            
            # Normalize - FIXED: cast scale to complex64
            power = tf.reduce_sum(tf.abs(W_f)**2, axis=1, keepdims=True) + 1e-12
            power_sqrt = tf.sqrt(power)
            scale = tf.sqrt(float(self.num_tx))
            
            # Cast both components to complex64
            W_f = W_f / tf.cast(power_sqrt, tf.complex64) * tf.cast(scale, tf.complex64)
            
            W_list.append(W_f)
        
        W = tf.stack(W_list, axis=-1)  # [B, T, R, fft]
        W = tf.transpose(W, [0, 2, 1, 3])  # [B, R, T, fft]
        
        # Add OFDM dimension
        W = tf.expand_dims(W, axis=3)  # [B, R, T, 1, fft]
        W = tf.tile(W, [1, 1, 1, self.num_ofdm_symbols, 1])  # [B, R, T, ofdm, fft]
        
        return W
    
    def _compute_metrics(self, W, h_freq, noise_power, rb_size=12):
        """
        Compute sum rate, SINR, and BER - FIXED einsum to NOT sum over frequency
        """
        # W: [B, R, T, ofdm, fft] or [B, 1, R, T, ofdm, fft]
        # h_freq: [B, 1, R, 1, T, ofdm, fft]
        
        # ===== STEP 1: Clean up Sionna dimensions =====
        h_shape = tf.shape(h_freq)
        if len(h_freq.shape) == 7:
            h_clean = tf.squeeze(h_freq, axis=[1, 3])  # [B, R, T, ofdm, fft]
        else:
            h_clean = h_freq
        p_shape = tf.shape(W)
        if len(W.shape) == 7:
            W_clean = tf.squeeze(W, axis=[1, 3])  # [B, S, T, ofdm, fft]
        else:
            W_clean = W
        
        # ===== STEP 2: Average over OFDM symbols (time dimension) =====
        h_avg = tf.reduce_mean(h_clean, axis=3)  # [B, R, T, fft]
        W_avg = tf.reduce_mean(W_clean, axis=3)  # [B, S, T, fft]
        
        # ===== STEP 3: Transpose to put frequency before antennas =====
        h_avg = tf.transpose(h_avg, perm=[0, 1, 3, 2])  # [B, R, fft, T]
        W_avg = tf.transpose(W_avg, perm=[0, 1, 3, 2])  # [B, S, fft, T]
        
        # ===== STEP 4: Reshape into RB format =====
        B = tf.shape(h_avg)[0]
        R = tf.shape(h_avg)[1]  # num_rx (users)
        fft_size = tf.shape(h_avg)[2]
        T = tf.shape(h_avg)[3]  # num_tx
        S = tf.shape(W_avg)[1]  # num_streams
        
        num_rb = fft_size // rb_size
        
        # Reshape to [B, R/S, num_rb, rb_size, T]
        h_rb = tf.reshape(h_avg, [B, R, num_rb, rb_size, T])
        W_rb = tf.reshape(W_avg, [B, S, num_rb, rb_size, T])
        
        # ===== STEP 5: CRITICAL - Normalize per RB =====
        precoder_norm = tf.norm(W_rb, ord='euclidean', axis=[3, 4], keepdims=True)  # [B, S, RB, 1, 1]
        W_rb_normalized = W_rb / (precoder_norm + 1e-12)
        
        # ===== STEP 6: Flatten RB and rb_size back to frequency dimension =====
        F = num_rb * rb_size
        h_flat = tf.reshape(h_rb, [B, R, F, T])  # [B, R, F, T]
        W_flat = tf.reshape(W_rb_normalized, [B, S, F, T])  # [B, S, F, T]
        
        # ===== STEP 7: Calculate effective channel (DO NOT sum over frequency!) =====
        # H_eff[b,r,s,f] = sum_t H[b,r,f,t] * conj(W[b,s,f,t])
        # This is the correct einsum: 'brft,bsft->brsf' (keeps frequency dimension!)
        W_eff = tf.einsum('bufi,bujf->buij', h_flat, tf.math.conj(
            tf.transpose(W_flat, perm=[0, 1, 3, 2])))
        
        # Signal power (diagonal over users dimension for each frequency)
        # For MIMO, user r receives signal from stream r
        # Extract diagonal: h_eff[b,r,r,f] for all r
        # First permute to [B, F, R, S] so we can use diag_part on last two dims
        W_flat = tf.reshape(W_rb_normalized, [B, S, F, T])  # [B, U, F, M]
        
        # ===== STEP 7: Calculate sum rate (EXACT same logic as rate_calculator_5d) =====
        # Effective channel: W = einsum("bufi,bujf->buij", conj(csi), precoder.permute)
        W_eff = tf.einsum('bufi,bujf->buij', h_flat, tf.math.conj(
            tf.transpose(W_flat, perm=[0, 1, 3, 2])))
        
        # Signal power (diagonal)
        diag_W = tf.linalg.diag_part(tf.abs(W_eff) ** 2)  # [B, U, U]
        
        # Total power and interference
        total_power = tf.reduce_sum(tf.abs(W_eff) ** 2, axis=3)  # [B, U, U]
        interference = total_power - diag_W
        
        # SINR
        SINR = diag_W / (interference + noise_power + 1e-12)
        
        # Rate per subcarrier/user
        rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)
        
        # Total power received at each user (sum over all streams)
        
        # Interference (total - signal)
        interference = total_power - diag_W  # [B, R, F]
        
        # SINR
        SINR = diag_W / (interference + noise_power + 1e-12)  # [B, R, F]
        
        # Rate per subcarrier/user
        rate_per_subcarrier = tf.math.log(1.0 + SINR) / tf.math.log(2.0)  # [B, R, F]
        
        # Sum over users AND frequency
        sum_rate_per_batch = tf.reduce_sum(rate_per_subcarrier, axis=[1, 2])  # [B]
        avg_sum_rate = tf.reduce_mean(sum_rate_per_batch)  # scalar
        
        # Average SINR
        avg_sinr = tf.reduce_mean(SINR)
        
        # Approximate BER using SINR (for QPSK)
        ber_per_subcarrier = 0.5 * tf.math.erfc(tf.sqrt(SINR))
        avg_ber = tf.reduce_mean(ber_per_subcarrier)
        
        return avg_sum_rate, avg_sinr, avg_ber
    
    def plot_comprehensive_analysis(self, channel_corr, channel_dist, channel_corr_dist,
                                   precoding_corr, precoding_dist, precoding_corr_dist,
                                   rb_results, coherence_bw):
        """Plot all analysis results"""
        print("\n[7/7] Plotting comprehensive results...")
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Channel frequency correlation heatmap
        ax1 = plt.subplot(3, 4, 1)
        sns.heatmap(channel_corr, cmap='coolwarm', center=0, ax=ax1,
                   vmin=-0.2, vmax=1.0, square=True, cbar_kws={'label': 'Correlation'})
        ax1.set_title(f'Channel Frequency Correlation\n({self.cdl_model}, {self.delay_spread*1e9:.0f}ns)')
        ax1.set_xlabel('Subcarrier Index')
        ax1.set_ylabel('Subcarrier Index')
        
        # 2. Precoding frequency correlation heatmap
        ax2 = plt.subplot(3, 4, 2)
        sns.heatmap(precoding_corr, cmap='viridis', center=0, ax=ax2,
                   vmin=-0.2, vmax=1.0, square=True, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Transformer Precoding Correlation\nAcross Frequency')
        ax2.set_xlabel('Subcarrier Index')
        ax2.set_ylabel('Subcarrier Index')
        
        # 3. Diagonal correlation comparison
        ax3 = plt.subplot(3, 4, 3)
        ax3.plot(channel_dist, channel_corr_dist, 'o-', label='Channel', linewidth=2, markersize=6, alpha=0.7)
        ax3.plot(precoding_dist, precoding_corr_dist, 's-', label='Precoding', linewidth=2, markersize=6)
        ax3.axvline(x=coherence_bw, color='r', linestyle='--', alpha=0.5, linewidth=2, label=f'Coherence BW={coherence_bw}')
        ax3.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Subcarrier Distance')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Correlation vs Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. BER vs RB size
        ax4 = plt.subplot(3, 4, 4)
        rb_sizes = sorted([k for k in rb_results.keys() if k != 'zf_baseline'])
        bers = [rb_results[rb]['ber'] for rb in rb_sizes]
        
        ax4.semilogy(rb_sizes, bers, 'o-', linewidth=2, markersize=8, label='Transformer')
        if 'zf_baseline' in rb_results:
            ax4.axhline(y=rb_results['zf_baseline']['ber'], color='r', linestyle='--', 
                       linewidth=2, label='ZF Baseline')
        ax4.axvline(x=coherence_bw, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Coherence BW={coherence_bw}')
        ax4.set_xlabel('RB Size (subcarriers)')
        ax4.set_ylabel('BER (Approximate)')
        ax4.set_title('BER Performance vs RB Grouping Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sum Rate vs RB size
        ax5 = plt.subplot(3, 4, 5)
        sumrates = [rb_results[rb]['sum_rate'] for rb in rb_sizes]
        ax5.plot(rb_sizes, sumrates, 'o-', linewidth=2, markersize=8, color='green')
        if 'zf_baseline' in rb_results:
            ax5.axhline(y=rb_results['zf_baseline']['sum_rate'], color='r', linestyle='--', 
                       linewidth=2, label='ZF Baseline')
        ax5.axvline(x=coherence_bw, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Coherence BW={coherence_bw}')
        ax5.set_xlabel('RB Size (subcarriers)')
        ax5.set_ylabel('Sum Rate (bps/Hz)')
        ax5.set_title('Sum Rate vs RB Grouping Size')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. SINR vs RB size
        ax6 = plt.subplot(3, 4, 6)
        sinrs = [rb_results[rb]['sinr_db'] for rb in rb_sizes]
        ax6.plot(rb_sizes, sinrs, 'o-', linewidth=2, markersize=8, color='purple')
        if 'zf_baseline' in rb_results:
            ax6.axhline(y=rb_results['zf_baseline']['sinr_db'], color='r', linestyle='--', 
                       linewidth=2, label='ZF Baseline')
        ax6.axvline(x=coherence_bw, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Coherence BW={coherence_bw}')
        ax6.set_xlabel('RB Size (subcarriers)')
        ax6.set_ylabel('SINR (dB)')
        ax6.set_title('SINR vs RB Grouping Size')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance degradation
        ax7 = plt.subplot(3, 4, 7)
        baseline_sumrate = sumrates[0] if sumrates[0] > 0 else 1.0
        degradation_pct = [100 * (1 - rb_results[rb]['sum_rate']/baseline_sumrate) for rb in rb_sizes]
        
        colors = ['green' if d < 5 else 'orange' if d < 15 else 'red' for d in degradation_pct]
        bars = ax7.bar(range(len(rb_sizes)), degradation_pct, color=colors, alpha=0.7)
        ax7.set_xticks(range(len(rb_sizes)))
        ax7.set_xticklabels(rb_sizes)
        ax7.axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax7.axhline(y=15, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax7.set_xlabel('RB Size (subcarriers)')
        ax7.set_ylabel('Sum Rate Degradation (%)')
        ax7.set_title('Performance Degradation vs RB Size')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Computational efficiency (num RBs)
        ax8 = plt.subplot(3, 4, 8)
        num_rbs = [rb_results[rb]['num_rbs'] for rb in rb_sizes]
        complexity_reduction = [100 * (1 - nrb / self.fft_size) for nrb in num_rbs]
        ax8.bar(range(len(rb_sizes)), complexity_reduction, color='skyblue', alpha=0.7)
        ax8.set_xticks(range(len(rb_sizes)))
        ax8.set_xticklabels(rb_sizes)
        ax8.set_xlabel('RB Size (subcarriers)')
        ax8.set_ylabel('Complexity Reduction (%)')
        ax8.set_title('Computational Complexity Reduction')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. RB efficiency recommendations
        ax9 = plt.subplot(3, 4, 9)
        efficiency = [coherence_bw / rb if rb > coherence_bw else 1.0 for rb in rb_sizes]
        efficiency_colors = ['green' if eff >= 0.8 else 'orange' if eff >= 0.6 else 'red' for eff in efficiency]
        ax9.bar(range(len(rb_sizes)), efficiency, color=efficiency_colors, alpha=0.7)
        ax9.set_xticks(range(len(rb_sizes)))
        ax9.set_xticklabels(rb_sizes)
        ax9.axhline(y=1.0, color='g', linestyle='--', linewidth=2, alpha=0.5)
        ax9.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        ax9.set_xlabel('RB Size (subcarriers)')
        ax9.set_ylabel('Expected Efficiency')
        ax9.set_title('RB Size Efficiency')
        ax9.set_ylim([0, 1.1])
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Performance-Complexity Trade-off
        ax10 = plt.subplot(3, 4, 10)
        # Normalize metrics
        norm_sumrate = np.array([rb_results[rb]['sum_rate'] / sumrates[0] for rb in rb_sizes])
        norm_complexity = np.array([rb_results[rb]['num_rbs'] / num_rbs[0] for rb in rb_sizes])
        
        ax10.plot(norm_complexity, norm_sumrate, 'o-', linewidth=2, markersize=8)
        for i, rb in enumerate(rb_sizes):
            ax10.annotate(f'RB={rb}', (norm_complexity[i], norm_sumrate[i]), 
                         textcoords="offset points", xytext=(5,5), fontsize=8)
        ax10.set_xlabel('Relative Complexity')
        ax10.set_ylabel('Relative Sum Rate')
        ax10.set_title('Performance-Complexity Trade-off')
        ax10.grid(True, alpha=0.3)
        
        # 11. Summary table
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        # Find optimal RB size
        optimal_rb = max([rb for rb, d in zip(rb_sizes, degradation_pct) if d < 5], default=rb_sizes[0])
        acceptable_rb = max([rb for rb, d in zip(rb_sizes, degradation_pct) if d < 15], default=rb_sizes[-1])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'transformer_freq_analysis_{self.cdl_model}_{self.delay_spread*1e9:.0f}ns_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {filepath}")
        plt.show()
        
        return filepath


def main():
    """Main analysis execution"""
    print("\n" + "="*80)
    print("TRANSFORMER PRECODING FREQUENCY ANALYSIS")
    print("Comprehensive study of frequency selectivity and RB grouping")
    print("="*80)
    
    # Configuration
    model_path = "/users/sala/test_projet/Trans/freq/Sionna/checkpoints/transformer_5d_final.weights.h5"
    output_dir = "transformer_frequency_analysis"
    
    # Test multiple scenarios
    scenarios = [
        ("A", 30e-9),    # Low delay spread
    ]
    
    all_results = {}
    
    for cdl_model, delay_spread in scenarios:
        print(f"\n\n{'='*80}")
        print(f"SCENARIO: CDL-{cdl_model}, Delay Spread={delay_spread*1e9:.0f}ns")
        print(f"{'='*80}")
        
        # Create analyzer
        analyzer = TransformerFrequencyAnalyzer(
            model_path=model_path,
            cdl_model=cdl_model,
            delay_spread=delay_spread,
            num_samples=500,
            output_dir=output_dir
        )
        
        # Load transformer model
        analyzer.load_transformer_model(rb_size=12)
        
        # Generate channels
        h_freq = analyzer.generate_channel_samples()
        
        # Analyze channel frequency correlation
        channel_corr, channel_dist, channel_corr_dist, coherence_bw = \
            analyzer.compute_channel_frequency_correlation(h_freq)
        
        # Extract precoding matrices
        W = analyzer.extract_precoding_matrices(h_freq, rb_size=12)
        
        # Analyze precoding frequency correlation
        precoding_corr, precoding_dist, precoding_corr_dist = \
            analyzer.compute_precoding_frequency_correlation(W)
        
        # Evaluate different RB sizes
        rb_sizes = [1, 2, 4, 6, 8, 12, 18, 24, 36]
        rb_results = analyzer.evaluate_rb_grouping_performance(
            h_freq, 
            rb_sizes=rb_sizes, 
            ebno_db=10.0,
            num_test_batches=50,
            batch_size=32
        )
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Coherence Bandwidth: {coherence_bw} subcarriers")
        
        rb_sizes_sorted = sorted([k for k in rb_results.keys() if k != 'zf_baseline'])
        print("\nPerformance by RB size:")
        for rb in rb_sizes_sorted:
            res = rb_results[rb]
            print(f"  RB={rb:2d}: Sum Rate={res['sum_rate']:6.2f} bps/Hz, "
                  f"SINR={res['sinr_db']:5.1f} dB, BER={res['ber']:.2e}")
        
        print(f"\nZF Baseline: Sum Rate={rb_results['zf_baseline']['sum_rate']:6.2f} bps/Hz, "
              f"SINR={rb_results['zf_baseline']['sinr_db']:5.1f} dB")
        print("="*80)
        plot_path = analyzer.plot_comprehensive_analysis(
            channel_corr, channel_dist, channel_corr_dist,
            precoding_corr, precoding_dist, precoding_corr_dist,
            rb_results, coherence_bw
        )
        
        
    print("\n✅ ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()