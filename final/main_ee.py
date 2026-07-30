"""
✅ ENERGY EFFICIENCY EVALUATION - COMPLETE & CORRECTED
=======================================================

Training:
- Follow second script's approach (WMMSE supervised warmup → Sum rate fine-tuning)
- Train @ 20 dB (high SNR)
- Proper noise calculation
- Detailed logging with per-user rates

Evaluation:
- Baselines: RZF, WMMSE
- Transformers: RB-grouped (trained)
- Metrics: Sum Rate, BER, Power, Energy Efficiency
- Ericsson KPIs & Reports
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# Imports
from dataset import CachedSionnaDataset
from precoders import TransformerPrecoder, rzf_precoder, wmmse_precoder, compute_effective_channel

import sionna
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer, LMMSEPostEqualizationSINR, RemoveNulledSubcarriers
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import compute_ber

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
DATASET_SIZE = 50_000
NUM_TX, NUM_RX = 8, 4
TRAINING_SNR = 20  # ✅ High SNR training
EVALUATION_SNR_RANGE = np.array([0, 5, 10, 15, 20])

print("="*100)
print("  ⚡ ENERGY EFFICIENCY EVALUATION")
print("="*100)
print(f"  Configuration: {NUM_TX}×{NUM_RX} MU-MIMO, QPSK, LDPC Rate-1/2")
print(f"  Training SNR: {TRAINING_SNR} dB (HIGH SNR)")
print(f"  Evaluation: {EVALUATION_SNR_RANGE[0]}-{EVALUATION_SNR_RANGE[-1]} dB")
print(f"  Dataset: {DATASET_SIZE:,} samples")
print("="*100 + "\n")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_noise_power_correct(ebno_db, bits_per_symbol=2, code_rate=0.5):
    """Correct noise calculation"""
    ebno_linear = 10**(ebno_db / 10)
    esno_linear = ebno_linear * bits_per_symbol * code_rate
    no = 1.0 / esno_linear
    return no


def compute_transmit_power_correct(g):
    """Average transmit power per OFDM symbol"""
    power_per_resource = tf.reduce_sum(tf.abs(g)**2, axis=[4, 5])  # [B, num_tx, ofdm, fft]
    avg_power_per_tx = tf.reduce_mean(power_per_resource, axis=[2, 3])  # [B, num_tx]
    total_power = tf.reduce_sum(avg_power_per_tx, axis=1)  # [B]
    return total_power


def compute_energy_efficiency_kpis(sum_rate, power, resource_grid, baseline_power=None):
    """
    Compute Ericsson KPIs corrected for OFDM overhead and actual grid bandwidth.
    
    Args:
        sum_rate: Average Sum Rate in bps/Hz
        power: Transmit power in Watts
        resource_grid: The Sionna ResourceGrid object used in simulation
        baseline_power: Power from a baseline (e.g., RZF) for savings calculation
    """
    # 1. Calculate the actual occupied bandwidth (Sampling Rate)
    # Total bandwidth occupied by the FFT bins
    actual_bandwidth = resource_grid.fft_size * resource_grid.subcarrier_spacing
    
    # 2. Account for Time-Domain Efficiency (CP Overhead)
    # T_useful / (T_useful + T_CP)
    t_useful = 1.0 / resource_grid.subcarrier_spacing
    t_total = t_useful + (resource_grid.cyclic_prefix_length / (resource_grid.fft_size * resource_grid.subcarrier_spacing))
    time_efficiency = t_useful / t_total
    
    # 3. Net Throughput Calculation
    # We multiply Spectral Efficiency by the bandwidth and scale down by the CP overhead
    throughput_bps = sum_rate * actual_bandwidth * time_efficiency
    
    # 4. Energy Efficiency (bits/Joule)
    ee_bits_per_joule = throughput_bps / (power + 1e-12)
    
    # 5. Energy Savings
    energy_savings_pct = None
    if baseline_power is not None:
        energy_savings_pct = (1.0 - power / (baseline_power + 1e-12)) * 100.0
    
    return {
        'energy_efficiency_bits_per_joule': ee_bits_per_joule,
        'energy_savings_percent': energy_savings_pct,
        'sum_rate_bps_hz': sum_rate,
        'power_watts': power,
        'throughput_mbps': throughput_bps / 1e6,
        'effective_bw_mhz': actual_bandwidth / 1e6
    }
# ==============================================================================
# MU-MIMO SYSTEM
# ==============================================================================

class MU_MIMO_System_EE(tf.keras.Model):
    """MU-MIMO System with corrected power calculation"""
    
    def __init__(self, num_tx=8, num_rx=4, precoder_type="rzf", rb_size=None, **kwargs):
        super().__init__(**kwargs)
        
        self.num_bs_antennas = num_tx
        self.num_users = num_rx
        self.precoder_type = precoder_type
        self.rb_size = rb_size
        self.num_bits_per_symbol = 2
        
        # Stream management
        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_rx)
        
        # Resource grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11]
        )
        
        # Antennas & Channel
        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization="single",
            polarization_type="V", antenna_pattern="omni", carrier_frequency=2.6e9
        )
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx/2), polarization="dual",
            polarization_type="cross", antenna_pattern="38.901", carrier_frequency=2.6e9
        )
        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=False, enable_shadow_fading=False
        )
        
        # PHY layers
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(int(self.rg.num_data_symbols), int(self.rg.num_data_symbols * 2))
        self.mapper = Mapper("qam", 2)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)
        self.demapper = Demapper("app", "qam", 2)
        self.decoder = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(resource_grid=self.rg, stream_management=self.sm)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)
        
        # Initialize precoder
        if precoder_type in ["transformer_rb", "transformer_full"]:
            self.precoder = TransformerPrecoder(
                num_tx=num_tx, num_rx=num_rx,
                num_ofdm=self.rg.num_ofdm_symbols,
                fft_size=self.rg.fft_size,
                rb_size=rb_size if precoder_type == "transformer_rb" else None,
                embed_dim=128, num_heads=4, num_layers=4
            )
        else:
            self.precoder = None
    
    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, self.num_users, "umi")
        self.channel_model.set_topology(*topology)
    
    @tf.function
    def call(self, batch_size, ebno_db, h_freq_cached=None, training=False):
        """Forward pass with corrected power calculation"""
        
        # Noise
        no = compute_noise_power_correct(ebno_db, self.num_bits_per_symbol, 0.5)
        
        # Data
        b = self.binary_source([batch_size, 1, self.num_users, int(self.rg.num_data_symbols)])
        c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)
        
        # Channel
        if h_freq_cached is not None:
            h_freq = h_freq_cached
        else:
            cir = self.channel_model(batch_size, self.rg.num_ofdm_symbols, 
                                     1.0 / self.rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        
        # Precoding
        if self.precoder_type in ["transformer_rb", "transformer_full"]:
            g = self.precoder(h_freq, training=training)
        elif self.precoder_type == "wmmse":
            g = wmmse_precoder(h_freq, no, self.sm, num_iterations=10)
        elif self.precoder_type == "rzf":
            g = rzf_precoder(h_freq, self.sm, alpha=0.1)
        else:
            raise ValueError(f"Unknown precoder: {self.precoder_type}")
        
        # Apply precoding
        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded_struct = tf.matmul(W, x_vec)
        x_precoded_struct = tf.squeeze(x_precoded_struct, axis=-1)
        x_precoded = tf.transpose(x_precoded_struct, perm=[0, 3, 1, 2])
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        # Effective channel
        h_eff = compute_effective_channel(h_freq, g, self.remove_nulled_scs)
        
        # Receiver
        y = self.apply_channel(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)
        
        # Metrics
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate_per_element = tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0)
        rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])  # [B, num_rx]
        sum_rate = tf.reduce_mean(tf.reduce_sum(rate_per_user, axis=1))
        
        ber = compute_ber(b, b_hat)
        power = compute_transmit_power_correct(g)
        avg_power = tf.reduce_mean(power)
        
        return b, b_hat, sum_rate, ber, avg_power, g


# ==============================================================================
# TRAINING: SUPERVISED WARMUP → SUM RATE FINE-TUNING
# ==============================================================================

class SimpleCheckpoint:
    """Simple checkpoint manager"""
    def __init__(self, save_dir='./training_weights_ee'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_checkpoint_path = None
    
    def save_best(self, precoder, config_dict, metrics):
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            import shutil
            shutil.rmtree(self.best_checkpoint_path)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_name = f"best_{timestamp}"
        ckpt_dir = os.path.join(self.save_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        weights_path = os.path.join(ckpt_dir, 'weights.pkl')
        weights = [v.numpy() for v in precoder.trainable_variables]
        with open(weights_path, 'wb') as f:
            pickle.dump(weights, f)
        
        config_path = os.path.join(ckpt_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.best_checkpoint_path = ckpt_dir
        return ckpt_dir


class SupervisedTrainer:
    """
    Two-phase training:
    1. Warmup: Learn from WMMSE teacher
    2. Fine-tune: Maximize sum rate
    """
    
    def __init__(self, system, cached_dataset, snr_db=20.0,
                 warmup_epochs=20, finetune_epochs=30,
                 learning_rate=1e-3, batch_size=256, num_iters=None):
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        self.warmup_epochs = int(warmup_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.total_epochs = self.warmup_epochs + self.finetune_epochs
        self.batch_size = batch_size
        
        # Calculate iterations
        if num_iters is None:
            self.num_iters = len(cached_dataset.h_freq_all) // batch_size
        else:
            self.num_iters = int(num_iters)
        
        # Optimizers
        warmup_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate * 0.5,
            decay_steps=self.warmup_epochs * self.num_iters,
            alpha=0.1
        )
        
        finetune_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate * 0.1,
            decay_steps=self.finetune_epochs * self.num_iters,
            alpha=0.1
        )
        
        self.optimizer_warmup = tf.keras.optimizers.Adam(warmup_lr, clipnorm=1.0)
        self.optimizer_finetune = tf.keras.optimizers.Adam(finetune_lr, clipnorm=2.0)
        
        # Build model
        dummy_h = tf.zeros([1, system.num_users, 1, 1, system.num_bs_antennas,
                           system.rg.num_ofdm_symbols, system.rg.fft_size],
                          dtype=tf.complex64)
        _ = system.precoder(dummy_h, training=False)
        
        self.trainable_vars = system.precoder.trainable_variables
        self.checkpoint = SimpleCheckpoint()
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        
        print(f"\n{'='*140}")
        print(f"  🚀 SUPERVISED TRAINING @ {snr_db} dB")
        print(f"{'='*140}")
        print(f"  Warmup:    {self.warmup_epochs} epochs (WMMSE teacher)")
        print(f"  Fine-tune: {self.finetune_epochs} epochs (Sum Rate)")
        print(f"  Batch:     {self.batch_size}")
        print(f"  LR:        {learning_rate} → {learning_rate * 0.1}")
        print(f"{'='*140}\n")
    
    @tf.function
    def warmup_step(self, h_freq_batch, batch_size, snr_db):
        """Warmup: Learn from WMMSE"""
        with tf.GradientTape() as tape:
            no = compute_noise_power_correct(snr_db, 2, 0.5)
            
            # Teacher
            #g_teacher = wmmse_precoder(h_freq_batch, no, self.system.sm, num_iterations=10)
            g_teacher = wmmse_precoder(h_freq_batch, no, self.system.sm, num_iterations=10)
            
            # Student
            g_pred = self.system.precoder(h_freq_batch, training=True)
            
            # Cosine similarity loss
            g_teacher_flat = tf.reshape(g_teacher, [batch_size, -1])
            g_pred_flat = tf.reshape(g_pred, [batch_size, -1])
            teacher_norm = tf.math.l2_normalize(g_teacher_flat, axis=-1, epsilon=1e-12)
            pred_norm = tf.math.l2_normalize(g_pred_flat, axis=-1, epsilon=1e-12)
            cosine_sim = tf.reduce_sum(teacher_norm * tf.math.conj(pred_norm), axis=-1)
            cosine_loss = 1.0 - tf.abs(cosine_sim)
            
            # MSE auxiliary
            mse_loss = tf.reduce_mean(tf.abs(g_pred - g_teacher)**2)
            
            loss = 10.0 * tf.reduce_mean(cosine_loss) + 5.0 * mse_loss
        
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, grad_norm = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer_warmup.apply_gradients(zip(grads, self.trainable_vars))
        
        # Evaluate with student precoder
        _, _, sum_rate, ber, power, _ = self.system(
            batch_size, snr_db, h_freq_batch, training=False
        )
        
        # Per-user rates
        h_eff = compute_effective_channel(h_freq_batch, g_pred, self.system.remove_nulled_scs)
        sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate_per_element = tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0)
        rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
        
        return sum_rate, ber, grad_norm, loss, rate_per_user
    
    @tf.function
    def finetune_step(self, h_freq_batch, batch_size, snr_db):
        """Fine-tune: Maximize sum rate"""
        with tf.GradientTape() as tape:
            _, _, sum_rate, _, _, _ = self.system(
                batch_size, snr_db, h_freq_batch, training=True
            )
            loss = -sum_rate/10
        
        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer_finetune.apply_gradients(zip(grads, self.trainable_vars))
        
        # Evaluate
        _, _, sum_rate_eval, ber, power, g_eval = self.system(
            batch_size, snr_db, h_freq_batch, training=False
        )
        
        # Per-user rates
        no = compute_noise_power_correct(snr_db, 2, 0.5)
        h_eff = compute_effective_channel(h_freq_batch, g_eval, self.system.remove_nulled_scs)
        sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate_per_element = tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0)
        rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
        
        return sum_rate_eval, ber, grad_norm, loss, rate_per_user
    
    def train(self, print_every=10, patience=20):
        """Train the model"""
        
        print(f"🚀 Training @ {self.snr_db} dB")
        print(f"{'='*160}")
        print(f"{'Epoch':>6} | {'Phase':>10} | {'Sum Rate':>10} | {'User Rates':>45} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*160}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            is_warmup = epoch < self.warmup_epochs
            phase = "WARMUP" if is_warmup else "FINETUNE"
            
            if epoch == self.warmup_epochs:
                print(f"\n{'='*160}")
                print(f"  🔄 SWITCHING TO FINE-TUNING")
                print(f"{'='*160}\n")
            
            metrics = {'rate': [], 'ber': [], 'loss': [], 'grad': [], 'rates': []}
            
            for i in range(self.num_iters):
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                if is_warmup:
                    rate, ber, g_norm, loss, rates = self.warmup_step(
                        h_freq_batch, bs_tensor, snr_tensor
                    )
                else:
                    rate, ber, g_norm, loss, rates = self.finetune_step(
                        h_freq_batch, bs_tensor, snr_tensor
                    )
                
                metrics['rate'].append(float(rate))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())
                
                # Print progress
                if (i + 1) % print_every == 0:
                    rec_rate = np.mean(metrics['rate'][-print_every:])
                    rec_ber = np.mean(metrics['ber'][-print_every:])
                    rec_grad = np.mean(metrics['grad'][-print_every:])
                    rec_loss = np.mean(metrics['loss'][-print_every:])
                    rec_rates = np.mean(metrics['rates'][-print_every:], axis=0)
                    
                    rate_str = "[" + " ".join([f"{r:.1f}" for r in rec_rates]) + "]"
                    status = "✅" if rec_grad > 0.05 else "⚠️" if rec_grad > 0.001 else "❌"
                    
                    print(f"    ... Step {i+1:3d}/{self.num_iters} | Sum: {rec_rate:5.2f} | "
                          f"Usr: {rate_str:>45} | BER: {rec_ber:5.1e} | "
                          f"Grad: {rec_grad:5.2e} | Loss: {rec_loss:6.2f} {status}", end='\r')
            
            # End of epoch
            avg_rate = np.mean(metrics['rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_loss = np.mean(metrics['loss'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            marker = ""
            if avg_rate > self.best_sum_rate:
                self.best_sum_rate = avg_rate
                self.best_epoch = epoch
                self.checkpoint.save_best(
                    self.system.precoder,
                    {'epoch': epoch, 'snr_db': self.snr_db},
                    {'sum_rate': avg_rate, 'ber': avg_ber}
                )
                marker = " ⭐"
                wait = 0
            else:
                if not is_warmup:
                    wait += 1
            
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            
            print(f"\r{epoch+1:6d} | {phase:>10} | {avg_rate:10.2f} | {rate_str:>45} | "
                  f"{avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker}")
            
            if wait >= patience:
                print(f"\n🛑 Early stopping: No improvement for {patience} epochs")
                break
        
        print(f"{'='*160}\n")
        print(f"✅ Training Complete! Best Rate: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")
        
        return self.checkpoint.best_checkpoint_path


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_energy_efficiency(system, snr_range, num_batches=100, batch_size=256, 
                               name="System", baseline_power=None):
    """Comprehensive evaluation with corrected energy efficiency context"""
    
    print(f"\n📊 Evaluating {name}:")
    
    results = {
        'snr': [], 'sum_rate': [], 'ber': [], 'power': [],
        'ee_bits_per_joule': [], 'energy_savings_pct': [], 'throughput_mbps': []
    }
    
    for snr in snr_range:
        rates, bers, powers = [], [], []
        
        for _ in range(num_batches):
            system.new_topology(batch_size)
            _, _, rate, ber, power, _ = system(
                tf.constant(batch_size), tf.constant(snr, dtype=tf.float32), training=False
            )
            
            rates.append(float(rate))
            bers.append(float(ber))
            powers.append(float(power))
        
        avg_rate = np.mean(rates)
        avg_ber = np.mean(bers)
        avg_power = np.mean(powers)
        
        # Reference power for savings calculation
        ref_power = None
        if baseline_power is not None and len(results['snr']) < len(baseline_power):
            ref_power = baseline_power[len(results['snr'])]
        
        # ✅ CALLING THE CORRECTED KPI FUNCTION WITH RESOURCE GRID
        kpis = compute_energy_efficiency_kpis(
            avg_rate, 
            avg_power, 
            system.rg,  # Pass the grid for BW and CP correction
            ref_power
        )
        
        results['snr'].append(snr)
        results['sum_rate'].append(avg_rate)
        results['ber'].append(avg_ber)
        results['power'].append(avg_power)
        results['ee_bits_per_joule'].append(kpis['energy_efficiency_bits_per_joule'])
        results['energy_savings_pct'].append(kpis['energy_savings_percent'])
        results['throughput_mbps'].append(kpis['throughput_mbps'])
        
        print(f"  SNR={snr:2.0f}dB: Rate={avg_rate:5.2f} bps/Hz, "
              f"Power={avg_power:5.2f}W, EE={kpis['energy_efficiency_bits_per_joule']:.2e} bits/J")
    
    return results
# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_ericsson_kpis(results_all, snr_range, save_dir='./results_ee'):
    """Generate Ericsson-style KPI plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Energy Efficiency Analysis - Ericsson KPIs', fontsize=16, fontweight='bold')
    
    colors = {'RZF': '#1f77b4', 'WMMSE': '#ff7f0e', 'Transformer (RB)': '#2ca02c'}
    
    # 1. Sum Rate
    ax = axes[0, 0]
    for name, res in results_all.items():
        ax.plot(snr_range, res['sum_rate'], 'o-', linewidth=2.5, markersize=8,
                color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Spectral Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. BER
    ax = axes[0, 1]
    for name, res in results_all.items():
        ax.semilogy(snr_range, res['ber'], 'o-', linewidth=2.5, markersize=8,
                   color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title('BER Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # 3. Transmit Power
    ax = axes[0, 2]
    for name, res in results_all.items():
        ax.plot(snr_range, res['power'], 'o-', linewidth=2.5, markersize=8,
               color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('TX Power (W)', fontsize=12)
    ax.set_title('Transmit Power', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Energy Efficiency
    ax = axes[1, 0]
    for name, res in results_all.items():
        ax.plot(snr_range, res['ee_bits_per_joule'], 'o-', linewidth=2.5, markersize=8,
               color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (bits/Joule)', fontsize=12)
    ax.set_title('KPI 1: Energy Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # 5. Energy Savings vs RZF
    ax = axes[1, 1]
    if 'RZF' in results_all:
        for name, res in results_all.items():
            if name != 'RZF' and res['energy_savings_pct'][0] is not None:
                ax.plot(snr_range, res['energy_savings_pct'], 'o-', linewidth=2.5, markersize=8,
                       color=colors.get(name, 'gray'), label=f"{name} vs RZF")
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Energy Savings (%)', fontsize=12)
    ax.set_title('KPI 2: Energy Savings vs Baseline', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. Throughput
    ax = axes[1, 2]
    for name, res in results_all.items():
        ax.plot(snr_range, res['throughput_mbps'], 'o-', linewidth=2.5, markersize=8,
               color=colors.get(name, 'gray'), label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Throughput (Mbps)', fontsize=12)
    ax.set_title('System Throughput (20 MHz BW)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ericsson_kpis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/ericsson_kpis.pdf', bbox_inches='tight')
    print(f"\n✅ Saved KPI plots to {save_dir}/")
    plt.close()


def generate_ericsson_report(results_all, snr_range, save_dir='./results_ee'):
    """Generate detailed report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_file = f'{save_dir}/ericsson_report_{datetime.now().strftime("%Y%m%d")}.txt'
    
    eval_snr = 20
    idx = list(snr_range).index(eval_snr)
    
    with open(report_file, 'w') as f:
        header = f"""
{'='*100}
ENERGY EFFICIENCY REPORT - TRANSFORMER PRECODER
{'='*100}
Generated: {timestamp}
Configuration: 8×4 MU-MIMO, QPSK, LDPC Rate-1/2, 20 MHz Bandwidth
Evaluation SNR: {eval_snr} dB

{'='*100}
ERICSSON KPI SUMMARY @ {eval_snr} dB SNR
{'='*100}

"""
        print(header)
        f.write(header)
        
        table_header = f"{'Method':<25} | {'Sum Rate':>12} | {'Power':>10} | {'EE (bits/J)':>15} | {'Savings':>12}\n"
        table_header += f"{'':<25} | {'(bps/Hz)':>12} | {'(W)':>10} | {'':>15} | {'vs RZF (%)':>12}\n"
        table_header += "-" * 100 + "\n"
        
        print(table_header)
        f.write(table_header)
        
        for name, res in results_all.items():
            rate = res['sum_rate'][idx]
            power = res['power'][idx]
            ee = res['ee_bits_per_joule'][idx]
            savings = res['energy_savings_pct'][idx] if res['energy_savings_pct'][idx] is not None else 0.0
            
            line = f"{name:<25} | {rate:12.2f} | {power:10.2f} | {ee:15.2e} | {savings:12.1f}\n"
            print(line, end='')
            f.write(line)
        
        print(f"{'='*100}\n")
        f.write(f"{'='*100}\n")
    
    print(f"✅ Report saved: {report_file}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main execution with corrected BW-aware evaluation"""
    
    # Seed initialization
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    
    # Load dataset
    print("📂 Loading dataset...")
    # We use RZF to initialize the cached dataset parameters
    dummy_system = MU_MIMO_System_EE(NUM_TX, NUM_RX, "rzf")
    cached_dataset = CachedSionnaDataset(
        dummy_system,
        dataset_size=DATASET_SIZE,
        batch_size=512,
        cache_file=f'/export/tmp/sala/cached_sionna_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npy'
    )
    
    results_all = {}
    
    # =========================================================================
    # STEP 1: RZF Baseline (The Power Reference)
    # =========================================================================
    print("\n" + "="*100)
    print("  STEP 1: RZF Baseline")
    print("="*100)
    
    system_rzf = MU_MIMO_System_EE(NUM_TX, NUM_RX, "rzf")
    results_all['RZF'] = evaluate_energy_efficiency(
        system_rzf, EVALUATION_SNR_RANGE, num_batches=50, batch_size=256, name="RZF"
    )
    
    # =========================================================================
    # STEP 2: WMMSE Baseline
    # =========================================================================
    print("\n" + "="*100)
    print("  STEP 2: WMMSE Baseline")
    print("="*100)
    
    system_wmmse = MU_MIMO_System_EE(NUM_TX, NUM_RX, "wmmse")
    # Pass RZF power to calculate energy savings
    results_all['WMMSE'] = evaluate_energy_efficiency(
        system_wmmse, EVALUATION_SNR_RANGE, num_batches=50, batch_size=256,
        name="WMMSE", baseline_power=results_all['RZF']['power']
    )
    
    # =========================================================================
    # STEP 3: Train Transformer (RB-Grouped)
    # =========================================================================
    print("\n" + "="*100)
    print("  STEP 3: Train Transformer (RB-Grouped)")
    print("="*100)
    
    system_tfmr = MU_MIMO_System_EE(NUM_TX, NUM_RX, "transformer_full", rb_size=None)
    trainer = SupervisedTrainer(
        system_tfmr, cached_dataset,
        snr_db=TRAINING_SNR,
        warmup_epochs=10,
        finetune_epochs=40,
        learning_rate=5e-3,
        batch_size=64
    )
    
    # Train the model (Checkpoints will be saved in ./training_weights_ee)
    weights_path = trainer.train(print_every=10, patience=20)
    
    # =========================================================================
    # STEP 4: Evaluate Transformer with Corrected EE
    # =========================================================================
    print("\n" + "="*100)
    print("  STEP 4: Evaluate Transformer")
    print("="*100)
    
    results_all['Transformer (RB)'] = evaluate_energy_efficiency(
        system_tfmr, EVALUATION_SNR_RANGE, num_batches=50, batch_size=256,
        name="Transformer (RB)", baseline_power=results_all['RZF']['power']
    )
    
    # =========================================================================
    # STEP 5: Generate Ericsson-Style Reports
    # =========================================================================
    save_dir = './results_ee'
    os.makedirs(save_dir, exist_ok=True)
    
    # Plots will now reflect the real net throughput and bits/Joule
    plot_ericsson_kpis(results_all, EVALUATION_SNR_RANGE, save_dir)
    generate_ericsson_report(results_all, EVALUATION_SNR_RANGE, save_dir)
    
    print("\n" + "="*100)
    print("  ✅ EVALUATION COMPLETE")
    print("="*100)
    print(f"  📁 Results saved to: {save_dir}/")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
