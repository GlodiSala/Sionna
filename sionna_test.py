import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

# Set random seeds
SEED = 42
LR = 5e-3
tf.random.set_seed(SEED)
np.random.seed(SEED)

# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration failed: {e}")

import sionna
sionna.phy.config.seed = SEED

from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LMMSEEqualizer, 
                             LMMSEPostEqualizationSINR, PrecodedChannel)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber

from transformer_5d_sionna_simple import SimplifiedTransformerPrecoder
from transformer_5d_sionna_mu import *
from tensorflow.keras import layers

# =============================================================================
# RZF BASELINE
# =============================================================================
class RZFPrecodedChannelCustom(PrecodedChannel):
    def __init__(self, resource_grid, stream_management, **kwargs):
        super().__init__(resource_grid, stream_management, **kwargs)
    
    def call(self, inputs):
        y, h_freq = inputs
        h_pc_desired = self.get_desired_channels(h_freq)
        from sionna.phy.mimo import rzf_precoding_matrix
        g = rzf_precoding_matrix(h_pc_desired, alpha=0.0)
        return g


# =============================================================================
# MU-MIMO SYSTEM
# =============================================================================
class MU_MIMO_System(tf.keras.Model):
    def __init__(self, num_tx=8, num_rx=4, precoder_type="transformer", rb_size=12):
        super().__init__()
        
        self.num_bs_antennas = num_tx
        self.num_users = num_rx
        self.precoder_type = precoder_type
        
        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_rx)
        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11]
        )
        
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
        
        if precoder_type == "transformer":
            print("✅ Using Transformer Precoder")
            self.precoder = SimplifiedTransformerPrecoder(
                self.rg, self.sm,
                num_tx_antennas=num_tx,
                num_rx_antennas=num_rx,
                rb_size=rb_size,
                embed_dim=128,
                num_heads=4,
                num_layers=4,
                dropout=0
            )
        elif precoder_type == "rzf":
            print("✅ Using RZF Precoder")
            self.precoder = RZFPrecodedChannelCustom(self.rg, self.sm)
    
    def new_topology(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        topology = gen_topology(batch_size, self.num_users, "umi")
        self.channel_model.set_topology(*topology)
    
    @tf.function
    def call(self, batch_size, ebno_db, training=False):
        no = ebnodb2no(ebno_db, 2, 0.5, self.rg)
        
        b = self.binary_source([batch_size, 1, self.num_users, int(self.rg.num_data_symbols)])
        c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)
        
        cir = self.channel_model(batch_size, self.rg.num_ofdm_symbols, 1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        
        if self.precoder_type == "transformer":
            g = self.precoder((x_rg, h_freq), training=training)
        else:
            g = self.precoder((x_rg, h_freq))
        
        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded_struct = tf.matmul(W, x_vec)
        x_precoded_struct = tf.squeeze(x_precoded_struct, axis=-1)
        x_precoded = tf.transpose(x_precoded_struct, perm=[0, 3, 1, 2])
        x_precoded = tf.expand_dims(x_precoded, axis=1)
        
        h_eff = self.precoder.compute_effective_channel(h_freq, g)
        
        y = self.apply_channel(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)
        
        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g

class MinimalDebugTrainer:
    """
    Simplest possible setup to verify learning works
    
    Test: Can the model improve from random initialization?
    """
    
    def __init__(self, system, learning_rate=LR):  # Very high LR
        self.system = system
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=10.0)
        self.trainable_vars = system.precoder.trainable_variables
        
        print(f"\n{'='*80}")
        print(f"  🔍 MINIMAL DEBUG TRAINER")
        print(f"{'='*80}")
        print(f"  Goal: Verify model CAN learn")
        print(f"  LR: {learning_rate} (very high)")
        print(f"  Loss: Pure sum rate (simplest)")
        print(f"  SNR: 50 dB (maximum gradient signal)")
        print(f"{'='*80}\n")
    
    @tf.function(reduce_retracing=True)
    def train_step(self, batch_size, snr_db):
        with tf.GradientTape() as tape:
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = self.system(
                batch_size, snr_db, training=True
            )
            
            # SIMPLEST POSSIBLE LOSS: Just sum rate
            sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            sum_rate = tf.reduce_sum(tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4]))
                        
            # Metrics
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])

            loss = -tf.reduce_sum(tf.reduce_sum(tf.math.log(1.0 + rate_per_user)))

            ber = compute_ber(b, b_hat)
        
        grads = tape.gradient(loss, self.trainable_vars)
        
        # Check for NaN/Inf gradients
        grad_norms = [tf.reduce_max(tf.abs(g)) if g is not None else 0.0 for g in grads]
        max_grad = tf.reduce_max(grad_norms) if grad_norms else 0.0
        
        grad_var_pairs = [(g, v) for g, v in zip(grads, self.trainable_vars) if g is not None]
        
        if grad_var_pairs:
            grads_only = [g for g, v in grad_var_pairs]
            vars_only = [v for g, v in grad_var_pairs]
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads_only, 10.0)
            self.optimizer.apply_gradients(zip(clipped_grads, vars_only))
        else:
            grad_norm = tf.constant(0.0)
        
        return {
            'sum_rate': sum_rate,
            'rate_per_user': rate_per_user,
            'ber': ber,
            'grad_norm': grad_norm,
            'max_grad': max_grad
        }
    
    def test(self, batch_size=512, snr_db=50.0, num_epochs=50):
        """Quick test: Should see improvement in 50 epochs"""
        
        print(f"🔬 Testing if model can learn...")
        print(f"   Epochs: {num_epochs}")
        print(f"   SNR: {snr_db} dB")
        print(f"   Expected: Sum rate should increase from ~10 to 30+\n")
        
        print(f"{'='*120}")
        print(f"{'Epoch':>6} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>10} | {'MaxGrad':>10}")
        print(f"{'='*120}")
        
        batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
        snr_tf = tf.constant(snr_db, dtype=tf.float32)
        
        initial_rate = None
        
        for epoch in range(num_epochs):
            self.system.new_topology(batch_size, seed=42)
            
            # Single iteration per epoch for speed
            m = self.train_step(batch_size_tf, snr_tf)
            
            sum_rate = float(m['sum_rate'])
            rates = m['rate_per_user'].numpy()
            ber = float(m['ber'])
            grad_norm = float(m['grad_norm'])
            max_grad = float(m['max_grad'])
            
            if epoch == 0:
                initial_rate = sum_rate
            
            if epoch % 5 == 0 or epoch < 5:
                rate_str = "[" + ", ".join([f"{r:5.2f}" for r in rates]) + "]"
                print(f"{epoch+1:6d} | {sum_rate:10.2f} | {rate_str:>40} | {ber:10.2e} | {grad_norm:10.2e} | {max_grad:10.2e}")
        
        print(f"{'='*120}\n")
        
        improvement = sum_rate - initial_rate
        
        print(f"📊 Results:")
        print(f"   Initial rate: {initial_rate:.2f} bps/Hz")
        print(f"   Final rate:   {sum_rate:.2f} bps/Hz")
        print(f"   Improvement:  {improvement:.2f} bps/Hz")
        print(f"   Final BER:    {ber:.2e}\n")
        
        if improvement > 10.0:
            print(f"✅ SUCCESS! Model is learning (+{improvement:.2f} bps/Hz)")
            print(f"   → Continue with full training")
        elif improvement > 2.0:
            print(f"⚠️  WEAK learning (+{improvement:.2f} bps/Hz)")
            print(f"   → Model can learn but slowly")
            print(f"   → Try: Higher LR, longer training, different architecture")
        else:
            print(f"❌ FAILURE! Model not learning (+{improvement:.2f} bps/Hz)")
            print(f"   → Fundamental issue detected")
            print(f"   → Diagnose below:")
            
            if max_grad < 1e-6:
                print(f"      • Gradients too small (max={max_grad:.2e})")
                print(f"      • Issue: Vanishing gradients")
            elif max_grad > 1e3:
                print(f"      • Gradients too large (max={max_grad:.2e})")
                print(f"      • Issue: Exploding gradients")
            
            if ber > 0.4:
                print(f"      • BER very high ({ber:.2e})")
                print(f"      • Issue: Signals not decodable")
            
            print(f"\n   🔧 Suggested fixes:")
            print(f"      1. Check transformer architecture")
            print(f"      2. Verify power normalization")
            print(f"      3. Test with simpler precoder (e.g., linear layer)")


# =============================================================================
# STEP 2: Diagnostic - Check What's Happening Inside
# =============================================================================
def diagnose_transformer(system, batch_size=512, snr_db=50.0):
    """
    Detailed diagnostics of transformer behavior
    """
    
    print(f"\n{'='*80}")
    print(f"  🔬 TRANSFORMER DIAGNOSTICS")
    print(f"{'='*80}\n")
    
    system.new_topology(batch_size, seed=42)
    
    batch_size_tf = tf.constant(batch_size, dtype=tf.int32)
    snr_tf = tf.constant(snr_db, dtype=tf.float32)
    
    # Forward pass
    print("1️⃣ Running forward pass...")
    b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
        batch_size_tf, snr_tf, training=True
    )
    
    # Extract precoding matrix
    W = tf.squeeze(g, axis=1)  # [B, S, F, M, K]
    
    print(f"   Precoder shape: {W.shape}")
    print(f"   Precoder dtype: {W.dtype}")
    
    # Check power distribution
    W_sample = W[0, 0, 0, :, :]  # [M, K] - first sample
    power_per_user = tf.reduce_sum(tf.abs(W_sample)**2, axis=0)
    total_power = tf.reduce_sum(tf.abs(W_sample)**2)
    
    print(f"\n2️⃣ Power analysis:")
    print(f"   Total power: {float(total_power):.4f}")
    print(f"   Per-user power: {[f'{float(p):.4f}' for p in power_per_user]}")
    print(f"   Expected per-user: 1.0 (if per-user normalized)")
    
    # Check orthogonality
    gram = tf.matmul(W_sample, W_sample, transpose_a=True)  # [K, K]
    diag = tf.linalg.diag_part(gram)
    off_diag = gram - tf.linalg.diag(diag)
    
    print(f"\n3️⃣ Orthogonality analysis:")
    print(f"   Gram matrix diagonal: {[f'{float(d):.4f}' for d in diag]}")
    print(f"   Off-diagonal mean: {float(tf.reduce_mean(tf.abs(off_diag))):.4f}")
    print(f"   Expected: diag ≈ [1,1,1,1], off-diag ≈ 0 for ZF")
    
    # Check rates
    sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
    rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
    rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
    sum_rate = tf.reduce_sum(rate_per_user)
    
    print(f"\n4️⃣ Performance:")
    print(f"   Sum rate: {float(sum_rate):.2f} bps/Hz")
    print(f"   Per-user rates: {[f'{float(r):.2f}' for r in rate_per_user]}")
    print(f"   Expected @ 50 dB: 60+ bps/Hz for RZF")
    
    ber = compute_ber(b, b_hat)
    print(f"   BER: {float(ber):.2e}")
    
    # Gradient check
    print(f"\n5️⃣ Gradient check:")
    with tf.GradientTape() as tape:
        b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
            batch_size_tf, snr_tf, training=True
        )
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
        sum_rate = tf.reduce_sum(tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4]))
        loss = -sum_rate
    
    grads = tape.gradient(loss, system.precoder.trainable_variables)
    
    for i, (var, grad) in enumerate(zip(system.precoder.trainable_variables, grads)):
        if grad is not None:
            grad_norm = float(tf.reduce_max(tf.abs(grad)))
            grad_mean = float(tf.reduce_mean(tf.abs(grad)))
            print(f"   Layer {i} ({var.name}): max={grad_norm:.2e}, mean={grad_mean:.2e}")
            
            if grad_norm < 1e-8:
                print(f"      ⚠️  Gradient too small!")
            elif grad_norm > 1e3:
                print(f"      ⚠️  Gradient too large!")
        else:
            print(f"   Layer {i} ({var.name}): NONE (no gradient!)")
    
    print(f"\n{'='*80}\n")


# =============================================================================
# MAIN DEBUG WORKFLOW
# =============================================================================
def debug_workflow():
    """
    Complete debugging workflow
    """
    
    print("\n" + "="*80)
    print("  🔧 TRANSFORMER DEBUG WORKFLOW")
    print("="*80 + "\n")
    
    BATCH_SIZE = 512
    RB_SIZE = 12
    
    # Create transformer system
    print("STEP 1: Create transformer system")
    system = MU_MIMO_System(
        num_tx=8, num_rx=4,
        precoder_type="transformer",
        rb_size=RB_SIZE
    )
    system.new_topology(BATCH_SIZE, seed=42)
    _ = system(tf.constant(BATCH_SIZE), tf.constant(50.0), training=True)
    
    # Diagnose
    print("\nSTEP 2: Diagnose transformer")
    diagnose_transformer(system, batch_size=512, snr_db=50.0)
    
    # Quick learning test
    print("\nSTEP 3: Test if transformer can learn")
    trainer = MinimalDebugTrainer(system, learning_rate=LR)
    trainer.test(batch_size=BATCH_SIZE, snr_db=50.0, num_epochs=50)
    
    # If transformer fails, try simple precoder
    response = input("\n❓ Did transformer show improvement? (y/n): ")
    
    if response.lower() != 'y':
        print("\n" + "="*80)
        print("STEP 4: Testing with simple linear precoder...")
        print("="*80 + "\n")
        
        # Replace precoder
        system_simple = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="rzf")
        system_simple.precoder = SimpleLinearPrecoder(
            system_simple.rg, system_simple.sm,
            num_tx_antennas=8, num_rx_antennas=4
        )
        system_simple.precoder_type = "transformer"  # Trick to enable training
        
        system_simple.new_topology(BATCH_SIZE, seed=42)
        _ = system_simple(tf.constant(BATCH_SIZE), tf.constant(50.0), training=True)
        
        trainer_simple = MinimalDebugTrainer(system_simple, learning_rate=LR)
        trainer_simple.test(batch_size=BATCH_SIZE, snr_db=50.0, num_epochs=50)
    
    print("\n" + "="*80)
    print("  DEBUG COMPLETE")
    print("="*80 + "\n")


def test_mlp():
    """Test if MLP precoder works"""
    
    print("\n" + "="*80)
    print("  🧪 TESTING MLP PRECODER")
    print("="*80 + "\n")
    
    BATCH_SIZE = 512
    
    # Create system with MLP
    system = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="rzf")
    system.precoder = MLPPrecoder(
        system.rg, system.sm,
        num_tx_antennas=8, num_rx_antennas=4
    )
    system.precoder_type = "transformer"  # Enable training
    
    system.new_topology(BATCH_SIZE, seed=42)
    _ = system(tf.constant(BATCH_SIZE), tf.constant(50.0), training=True)
    
    # Test
    trainer = MinimalDebugTrainer(system, learning_rate=LR)
    trainer.test(batch_size=BATCH_SIZE, snr_db=50.0, num_epochs=50)


def test_all_precoders():
    """
    Test all three precoders to see which works
    """
    
    print("\n" + "="*80)
    print("  🧪 TESTING ALL PRECODERS")
    print("="*80 + "\n")
    
    BATCH_SIZE = 512
    SNR_DB = 50.0
    
    precoders_to_test = [
        ("Linear", SimpleLinearPrecoder),
        ("MLP", MLPPrecoder),
        ("Transformer", WorkingTransformerPrecoder),
    ]
    
    results = {}
    
    for name, PrecoderClass in precoders_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING: {name} Precoder")
        print(f"{'='*80}\n")
        
        try:
            # Create system
            system = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="rzf")
            
            if name == "Transformer":
                system.precoder = PrecoderClass(
                    system.rg, system.sm,
                    num_tx_antennas=8, num_rx_antennas=4,
                    rb_size=12
                )
            else:
                system.precoder = PrecoderClass(
                    system.rg, system.sm,
                    num_tx_antennas=8, num_rx_antennas=4
                )
            
            system.precoder_type = "transformer"  # Enable training
            
            system.new_topology(BATCH_SIZE, seed=42)
            _ = system(tf.constant(BATCH_SIZE), tf.constant(SNR_DB), training=True)
            
            # Quick test (10 iterations only)
            trainer = MinimalDebugTrainer(system, learning_rate=LR)
            print(f"\n🔬 Quick test (10 iterations)...\n")
            
            rates = []
            for i in range(10):
                system.new_topology(BATCH_SIZE, seed=42 + i)
                
                with tf.GradientTape() as tape:
                    b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
                        tf.constant(BATCH_SIZE), tf.constant(SNR_DB), training=True
                    )
                    sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
                    rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
                    sum_rate = tf.reduce_sum(tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4]))
                    loss = -sum_rate
                
                grads = tape.gradient(loss, trainer.trainable_vars)
                grad_var_pairs = [(g, v) for g, v in zip(grads, trainer.trainable_vars) if g is not None]
                
                if grad_var_pairs:
                    grads_only = [g for g, v in grad_var_pairs]
                    vars_only = [v for g, v in grad_var_pairs]
                    clipped_grads, _ = tf.clip_by_global_norm(grads_only, 10.0)
                    trainer.optimizer.apply_gradients(zip(clipped_grads, vars_only))
                
                rates.append(float(sum_rate))
                
                if i == 0 or i == 9:
                    print(f"   Iter {i+1}: {float(sum_rate):.2f} bps/Hz")
            
            initial = rates[0]
            final = rates[-1]
            improvement = final - initial
            
            results[name] = {
                'initial': initial,
                'final': final,
                'improvement': improvement,
                'success': improvement > 1.0
            }
            
            if improvement > 1.0:
                print(f"\n   ✅ {name}: WORKS! (+{improvement:.2f} bps/Hz)")
            else:
                print(f"\n   ❌ {name}: FAILS (+{improvement:.2f} bps/Hz)")
        
        except Exception as e:
            print(f"\n   ❌ {name}: ERROR - {str(e)}")
            results[name] = {'error': str(e)}
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"  📊 SUMMARY")
    print(f"{'='*80}\n")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"  {name:15s}: ERROR")
        elif result['success']:
            print(f"  {name:15s}: ✅ WORKS (init={result['initial']:.1f}, final={result['final']:.1f}, +{result['improvement']:.1f})")
        else:
            print(f"  {name:15s}: ❌ FAILS (init={result['initial']:.1f}, final={result['final']:.1f}, +{result['improvement']:.1f})")
    
    print()


if __name__ == "__main__":
    debug_workflow()
    test_all_precoders()
