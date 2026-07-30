"""
CORRECTED: Complete System Comparison with Proper Noise Calculation
"""
import os
import json
import pickle
import logging
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

SEED = 42

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
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer, LMMSEPostEqualizationSINR, RemoveNulledSubcarriers,RZFPrecoder
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import compute_ber,ebnodb2no

# Import your modules
from datasets import CachedSionnaDataset
from precoders_w import (rzf_precoder, wmmse_precoder, TransformerPrecoderV2,TransformerPrecoderV4)

# =============================================================================
# ✅ CORRECTED: Noise Calculation Function
# =============================================================================



"""
Complete System Comparison with Energy Efficiency Analysis
Training @ 20 dB SNR with 10 epochs max
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# System Parameters
NUM_TX = 8
NUM_RX = 4
BATCH_SIZE = 256

# Dataset
DATASET_SIZE = 5000

# Training Configuration - 20 dB HIGH SNR
TRAINING_SNR = 15
EVALUATION_SNR_RANGE = np.array([0, 5, 10, 15, 20])

# Quick Training: 10 epochs max
TRAINING_CONFIG = {
    'rb_grouped': {
        'warmup_epochs': 3,
        'finetune_epochs': 12,
        'batch_size': BATCH_SIZE,
        'learning_rate': 2e-3
    },
    'full_sc': {
        'warmup_epochs': 2,
        'finetune_epochs': 2,
        'batch_size': 64,
        'learning_rate': 1e-4
    }
}

print("="*80)
print("  🚀 COMPREHENSIVE SYSTEM COMPARISON")
print("="*80)
print(f"  Configuration: {NUM_TX}×{NUM_RX} MIMO")
print(f"  Training SNR: {TRAINING_SNR} dB ⚡ (HIGH SNR)")
print(f"  Max Training: 10 epochs per model")
print(f"  Dataset: {DATASET_SIZE:,} samples")
print("="*80 + "\n")
# =============================================================================
# MU-MIMO SYSTEM - CORRECTED
# =============================================================================
class MU_MIMO_System(tf.keras.Model):

    def __init__(self, num_tx=8, num_rx=4, precoder_type="rzf",
                 rb_size=12, batch_size=32, weights_path=None,
                 tokens_per_rb=1,
                 num_intra_layers=2, num_inter_layers=1,
                 embed_dim=128, num_heads=4):
        super().__init__()

        self.num_bs_antennas  = num_tx
        self.num_users        = num_rx
        self.num_rx           = num_rx
        self.precoder_type    = precoder_type
        self.num_bits_per_symbol = 2
        self.default_batch_size  = batch_size
        self.tokens_per_rb    = tokens_per_rb
        self.num_intra_layers = num_intra_layers
        self.num_inter_layers = num_inter_layers
        self.embed_dim        = embed_dim
        self.num_heads        = num_heads

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association,
                                   num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72,
            subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx,
            cyclic_prefix_length=6,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11])

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single", polarization_type="V",
            antenna_pattern="omni", carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx / 2),
            polarization="dual", polarization_type="cross",
            antenna_pattern="38.901", carrier_frequency=2.6e9)

        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        self.binary_source  = BinarySource()
        self.encoder        = LDPC5GEncoder(
            int(self.rg.num_data_symbols),
            int(self.rg.num_data_symbols * 2))
        self.mapper         = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper      = ResourceGridMapper(self.rg)
        self.frequencies    = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.channel_freq   = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ      = LMMSEEqualizer(self.rg, self.sm)
        self.demapper       = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder        = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr     = LMMSEPostEqualizationSINR(
            resource_grid=self.rg, stream_management=self.sm)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        from sionna.phy.ofdm import RZFPrecodedChannel
        self.precoded_channel_helper = RZFPrecodedChannel(
            resource_grid=self.rg, stream_management=self.sm)

        self._initialize_precoder(precoder_type, rb_size, weights_path)

    def _initialize_precoder(self, precoder_type, rb_size, weights_path):

        if precoder_type == "transformer_rb":
            print(f"✅ Using TransformerPrecoderV4 ({self.tokens_per_rb} Tok/RB)")
            self.precoder = TransformerPrecoderV4(
                num_tx=self.num_bs_antennas,
                num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols,
                fft_size=self.rg.fft_size,
                rb_size=rb_size,
                tokens_per_rb=self.tokens_per_rb,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=4)
            if weights_path:
                self.load_precoder_weights(weights_path)
            self.rzf_precoder = None

        elif precoder_type == "transformer_v5":
            print(f"✅ Using TransformerPrecoderV5 (per-RB shared, Swin)")
            self.precoder = TransformerPrecoderV5(
                num_tx=self.num_bs_antennas,
                num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols,
                fft_size=self.rg.fft_size,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_intra_layers=self.num_intra_layers,
                num_inter_layers=self.num_inter_layers)
            if weights_path:
                self.load_precoder_weights(weights_path)
            self.rzf_precoder = None

        elif precoder_type == "transformer_full":
            print("✅ Using TransformerPrecoderV2 (Full 72 SC)")
            self.precoder = TransformerPrecoderV2(
                num_tx=self.num_bs_antennas,
                num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols,
                fft_size=self.rg.fft_size,
                rb_size=None, tokens_per_rb=None,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_layers=4)
            if weights_path:
                self.load_precoder_weights(weights_path)
            self.rzf_precoder = None

        elif precoder_type == "rzf":
            print("✅ Using RZF Precoder")
            self.precoder = None
            self.rzf_precoder = None

        elif precoder_type == "wmmse":
            print("✅ Using WMMSE Precoder")
            self.precoder = None
            self.rzf_precoder = None

        else:
            raise ValueError(f"Unknown precoder type: {precoder_type}")

    # call() et call_with_cached_channel() — uniquement les blocs modifiés

    @tf.function
    def call(self, batch_size, ebno_db, training=False):
        no    = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        b     = self.binary_source([batch_size, 1, self.num_users,
                                    int(self.rg.num_data_symbols)])
        c     = self.encoder(b)
        x     = self.mapper(c)
        x_rg  = self.rg_mapper(x)

        cir    = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                    1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled_scs(h_freq)

        if self.precoder_type == "rzf":
            g = rzf_precoder(h_freq, stream_management=self.sm, alpha=0.0)
        elif self.precoder_type == "wmmse":
            g = wmmse_precoder(h_freq, no, num_iterations=10,
                               stream_management=self.sm)
        elif self.precoder_type in ["transformer_rb", "transformer_full",
                                     "transformer_v5"]:
            g = self.precoder(h_freq, training=training)

        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr   = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g

    @tf.function
    def call_with_cached_channel(self, batch_size, ebno_db,
                                  h_freq_cached, training=False):
        no   = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        b    = self.binary_source([batch_size, 1, self.num_users,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)
        h_freq = h_freq_cached

        if self.precoder_type in ["transformer_rb", "transformer_full",
                                   "transformer_v5"]:
            if training:
                g_re, g_im = self.precoder(h_freq, training=True,
                                            return_real_imag=True)
                g = tf.complex(g_re, g_im)
            else:
                g = self.precoder(h_freq, training=False)

        elif self.precoder_type == "wmmse":
            g = wmmse_precoder(h_freq, no, num_iterations=10,
                               stream_management=self.sm)
        elif self.precoder_type == "rzf":
            g = rzf_precoder(h_freq, stream_management=self.sm, alpha=0.1)
        else:
            raise ValueError(f"Unknown precoder: {self.precoder_type}")

        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr   = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g
        
# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class SimpleCheckpoint:
    def __init__(self, save_dir='./training_weights'):
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
        
        metrics_path = os.path.join(ckpt_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            clean_metrics = {k: float(v) if np.isscalar(v) else v.tolist() 
                           for k, v in metrics.items()}
            json.dump(clean_metrics, f, indent=2)
        
        self.best_checkpoint_path = ckpt_dir
        return ckpt_dir




# =============================================================================
# SUPERVISED TRAINER — tous les fixes intégrés
# =============================================================================
# =============================================================================
# SUPERVISED TRAINER — refactorisé : gradient et eval séparés
# =============================================================================
# =============================================================================
# SUPERVISED TRAINER — gradient et eval séparés, affichage identique à v1
# =============================================================================
class SupervisedTrainer:

    def __init__(self, system, cached_dataset,
                 snr_db=15.0,
                 warmup_epochs=5,
                 finetune_epochs=10,
                 learning_rate=2e-3,
                 batch_size=256,
                 num_iters=None):

        self.system          = system
        self.cached_dataset  = cached_dataset
        self.snr_db          = snr_db
        self.warmup_epochs   = int(warmup_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.total_epochs    = self.warmup_epochs + self.finetune_epochs
        self.rate_norm       = float(system.num_users) * 9.0

        if num_iters is None:
            if hasattr(cached_dataset, 'effective_dataset_size'):
                self.num_iters = cached_dataset.effective_dataset_size // batch_size
            else:
                self.num_iters = len(cached_dataset.h_freq_all) // batch_size
        else:
            self.num_iters = int(num_iters)

        self.batch_size = batch_size

        warmup_steps   = self.warmup_epochs   * self.num_iters
        finetune_steps = self.finetune_epochs * self.num_iters

        self.warmup_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=warmup_steps,
            alpha=0.1)

        self.finetune_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate * 0.5,
            decay_steps=finetune_steps,
            alpha=0.01)

        # Optimizer unique — momentum jamais réinitialisé
        self.optimizer = tf.keras.optimizers.Adam(
            self.warmup_lr, clipnorm=1.0)

        # Build model
        dummy_h = tf.zeros(
            [1, system.num_users, 1, 1, system.num_bs_antennas,
             system.rg.num_ofdm_symbols, system.rg.fft_size],
            dtype=tf.complex64)
        try:
            _ = system.precoder(dummy_h, training=False)
        except Exception:
            pass

        self.trainable_vars = system.precoder.trainable_variables
        total_params = sum(tf.size(v).numpy() for v in self.trainable_vars)
        print(f"  Model: {len(self.trainable_vars)} weight tensors, "
              f"{total_params:,} parameters")

        self.checkpoint    = SimpleCheckpoint(
            f'./training_weights_{system.precoder_type}')
        self.best_sum_rate = -np.inf
        self.best_epoch    = 0
        self.history       = []

        print(f"\n{'='*80}")
        print(f"  SUPERVISED TRAINING (RZF warmup → SUM RATE)")
        print(f"  Warm-up:   {self.warmup_epochs} epochs  LR={learning_rate:.2e}")
        print(f"  Fine-tune: {self.finetune_epochs} epochs  "
              f"LR={learning_rate*0.1:.2e}")
        print(f"  Batch: {self.batch_size}   Iters/epoch: {self.num_iters}")
        print(f"  Rate norm: {self.rate_norm:.1f} bps/Hz")
        print(f"{'='*80}\n")

    # =========================================================================
    # GRADIENT STEPS — légers, un forward pass, pas de métriques eval
    # =========================================================================

    @tf.function
    def warmup_step(self, h_freq_batch, snr_db):
        """Gradient update warmup — RZF teacher, transition MSE→rate.
        Retourne seulement loss + grad_norm + diagnostics transition.
        Pas de call_with_cached_channel ici.
        """
        snr_db = tf.random.uniform([], 5.0, 20.0) 
        no = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                       0.5, self.system.rg)

        g_teacher = tf.stop_gradient(
            rzf_precoder(h_freq_batch,
                         stream_management=self.system.sm,
                         alpha=0.1))

        with tf.GradientTape() as tape:
            g_pred   = self.system.precoder(h_freq_batch, training=True)
            mse_loss = 2.0 * tf.reduce_mean(
                tf.abs(g_pred - g_teacher) ** 2)

            h_eff_pred    = self.system.precoded_channel_helper \
                                .compute_effective_channel(h_freq_batch, g_pred)
            sinr_pred     = self.system.lmmse_sinr(
                h_eff_pred, no=no, interference_whitening=True)
            rate_per_user = tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr_pred, 1e-9, 1e4))
                / tf.math.log(2.0), axis=[0, 1, 2, 4])
            sum_rate_pred       = tf.reduce_sum(rate_per_user)
            sum_rate_normalized = sum_rate_pred / self.rate_norm

            transition_steps = tf.constant(
                float(self.warmup_epochs * self.num_iters), dtype=tf.float32)
            current_step = tf.cast(self.optimizer.iterations, tf.float32)
            progress     = tf.minimum(current_step / transition_steps, 1.0)
            mse_weight   = tf.maximum(1.0 - progress, 0.05)
            rate_weight  = progress

            loss = mse_weight * mse_loss - rate_weight * sum_rate_normalized

        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                 for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)

        # Teacher rate pour diagnostic — coût marginal, g_teacher déjà calculé
        h_eff_t      = self.system.precoded_channel_helper \
                           .compute_effective_channel(h_freq_batch, g_teacher)
        sinr_t       = self.system.lmmse_sinr(
            h_eff_t, no=no, interference_whitening=True)
        rate_teacher = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr_t, 1e-9, 1e4))
            / tf.math.log(2.0), axis=[0, 1, 2, 4]))

        return loss, grad_norm, mse_weight, rate_weight, rate_teacher

    @tf.function
    def finetune_step(self, h_freq_batch, snr_db):
        """Gradient update fine-tune — sum-rate pur.
        Retourne seulement loss + grad_norm.
        Pas de call_with_cached_channel ici.
        """
        with tf.GradientTape() as tape:
            snr_db = tf.random.uniform([], 5.0, 20.0) 
            no     = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                               0.5, self.system.rg)
            g_pred = self.system.precoder(h_freq_batch, training=True)

            h_eff = self.system.precoded_channel_helper \
                        .compute_effective_channel(h_freq_batch, g_pred)
            sinr  = self.system.lmmse_sinr(
                h_eff, no=no, interference_whitening=True)
            rate_per_user = tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
                / tf.math.log(2.0), axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            loss     = -tf.where(tf.math.is_finite(sum_rate),
                                 sum_rate / self.rate_norm,
                                 tf.constant(0.0))

        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                 for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        grad_norm = tf.linalg.global_norm(grads)

        return loss, grad_norm

    # =========================================================================
    # EVAL STEP — coûteux, appelé seulement tous les print_every steps
    # =========================================================================

    @tf.function
    def eval_step(self, h_freq_batch, batch_size, snr_db):
        """Métriques complètes via call_with_cached_channel.
        Identique à ce que retournait l'ancienne version à chaque step,
        mais maintenant appelé seulement tous les print_every steps.
        """
        b, b_hat, c, llr, h_eff, no_eval, _, _, _, _ = \
            self.system.call_with_cached_channel(
                batch_size, snr_db, h_freq_batch, training=False)

        sinr_eval = self.system.lmmse_sinr(
            h_eff, no=no_eval, interference_whitening=True)
        rate_per_user_eval = tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr_eval, 1e-9, 1e4))
            / tf.math.log(2.0), axis=[0, 1, 2, 4])
        sum_rate_eval = tf.reduce_sum(rate_per_user_eval)
        ber           = compute_ber(b, b_hat)

        c_float  = tf.cast(c, tf.float32)
        llr_clip = tf.clip_by_value(llr, -20.0, 20.0)
        bce_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(c_float, llr_clip))

        return sum_rate_eval, ber, bce_loss, rate_per_user_eval

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    def train(self, print_every=100, patience=8):
        print(f"\n🚀 Training @ {self.snr_db} dB")
        print("=" * 89)
        print(f"  {'Epoch':>6} | {'Mode':>10} | {'Sum Rate':>10} | "
              f"{'User Rates':>35} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print("=" * 89)

        no_improve_count = 0

        for epoch in range(self.total_epochs):
            is_warmup = epoch < self.warmup_epochs
            mode_str  = "RZF-WU" if is_warmup else "SUM-RATE"

            # Switch LR in-place — momentum préservé
            if epoch == self.warmup_epochs:
                self.optimizer.learning_rate = self.finetune_lr
                print(f"\n🔄 Switching to fine-tune LR "
                      f"({self.finetune_lr.initial_learning_rate:.2e})"
                      f" — momentum preserved")
                no_improve_count = 0

            # Métriques légères — collectées à chaque step
            train_losses     = []
            train_grad_norms = []

            # Métriques eval — collectées tous les print_every steps
            eval_sum_rates  = []
            eval_bers       = []
            eval_bces       = []
            eval_rates_list = []

            # Pour le diagnostic warmup — seulement aux steps d'eval
            last_mse_w       = 0.0
            last_rate_w      = 0.0
            last_rate_teacher = 0.0

            for i in range(self.num_iters):
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                snr_t = tf.constant(self.snr_db, tf.float32)

                # ── Gradient step ─────────────────────────────────────────
                if is_warmup:
                    (loss, grad_norm,
                     mse_w, rate_w, rate_teacher) = self.warmup_step(
                        h_freq_batch, snr_t)
                    last_mse_w        = float(mse_w)
                    last_rate_w       = float(rate_w)
                    last_rate_teacher = float(rate_teacher)
                else:
                    loss, grad_norm = self.finetune_step(h_freq_batch, snr_t)

                train_losses.append(float(loss))
                train_grad_norms.append(float(grad_norm))

                # ── Eval step — seulement tous les print_every steps ──────
                if (i + 1) % print_every == 0:
                    bs_t = tf.constant(self.batch_size, tf.int32)
                    rate_eval, ber, bce, rates = self.eval_step(
                        h_freq_batch, bs_t, snr_t)

                    eval_sum_rates.append(float(rate_eval))
                    eval_bers.append(float(ber))
                    eval_bces.append(float(bce))
                    eval_rates_list.append(rates.numpy())

                    rs_str = "[" + " ".join(
                        f"{v:.1f}" for v in rates.numpy()) + "]"
                    g_mean = float(np.mean(
                        train_grad_norms[-print_every:]))
                    l_mean = float(np.mean(
                        train_losses[-print_every:]))

                    if is_warmup:
                        print(f"\n🔍 Step {i+1}/{self.num_iters} | "
                              f"Teacher(RZF): {last_rate_teacher:.1f} | "
                              f"Student: {float(rate_eval):.1f} | "
                              f"MSE/Rate: {last_mse_w:.2f}/"
                              f"{last_rate_w:.2f}")
                        print(f"    ... Step {i+1}/{self.num_iters} | "
                              f"Sum: {float(rate_eval):.2f} | "
                              f"Usr: {rs_str} | "
                              f"BER: {float(ber):.1e} | "
                              f"Grad: {g_mean:.2e} | "
                              f"Loss: {l_mean:.3f} ✅")
                    else:
                        print(f"    ... Step {i+1}/{self.num_iters} | "
                              f"Sum: {float(rate_eval):.2f} | "
                              f"Usr: {rs_str} | "
                              f"BER: {float(ber):.1e} | "
                              f"Grad: {g_mean:.2e} | "
                              f"Loss: {l_mean:.3f} ✅")

            # ── Résumé epoch ──────────────────────────────────────────────
            avg_loss = float(np.mean(train_losses))
            avg_grad = float(np.mean(train_grad_norms))

            if eval_sum_rates:
                avg_rate  = float(np.mean(eval_sum_rates))
                avg_ber   = float(np.mean(eval_bers))
                avg_rates = np.mean(eval_rates_list, axis=0)
            else:
                # Ne devrait pas arriver si print_every < num_iters
                avg_rate  = 0.0
                avg_ber   = float('nan')
                avg_rates = np.zeros(self.system.num_users)

            rs_str  = "[" + ", ".join(f"{v:5.2f}" for v in avg_rates) + "]"
            is_best = avg_rate > self.best_sum_rate
            star    = ""

            if is_best:
                self.best_sum_rate = avg_rate
                self.best_epoch    = epoch + 1
                self.checkpoint.save_best(
                    self.system.precoder,
                    {'epoch': epoch + 1, 'snr_db': self.snr_db},
                    {'sum_rate': avg_rate})
                star = " ⭐"
                no_improve_count = 0
            else:
                if not is_warmup:
                    no_improve_count += 1

            print(f"\n{epoch+1:6d} | {mode_str:>10} | "
                  f"{avg_rate:10.2f} | {rs_str} | "
                  f"{avg_ber:10.2e} | {avg_grad:.2e} | "
                  f"{avg_loss:8.3f}{star} ✅")

            self.history.append({
                'epoch'    : epoch + 1,
                'mode'     : mode_str,
                'sum_rate' : avg_rate,
                'ber'      : avg_ber,
                'loss'     : avg_loss})

            if not is_warmup and patience is not None:
                if no_improve_count >= patience:
                    print(f"\n⏹  Early stopping @ epoch {epoch+1} "
                          f"({patience} epochs sans amélioration)")
                    break

        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz "
              f"@ epoch {self.best_epoch}")
        return self.history 
# =============================================================================        
# RESIDUAL TRAINER (Direct Sum Rate - NO WARMUP!)
# =============================================================================

class ResidualTrainer:
    """
    ✅ DIRECT SUM RATE TRAINING pour Residual Transformer
    
    Pas de teacher!
    Pas de warmup!
    Juste: maximiser sum_rate directement!
    
    Pourquoi ça marche:
    - W = RZF + α×Δ
    - Au début: α=0.1, Δ≈0 → W ≈ RZF (déjà bon!)
    - Gradient pousse Δ pour améliorer sum_rate
    - Stable car baseline RZF est frozen!
    """
    
    def __init__(self, system, cached_dataset,
                 snr_db=15.0,
                 total_epochs=30,
                 learning_rate=1e-3,
                 batch_size=256,
                 num_iters=None):
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        
        # Auto-calculate iterations
        if num_iters is None:
            if hasattr(cached_dataset, 'h_freq_all'):
                self.num_iters = len(cached_dataset.h_freq_all) // batch_size
            else:
                self.num_iters = 50000 // batch_size
        else:
            self.num_iters = num_iters
        
        # ✅ Simple optimizer (cosine decay)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * self.num_iters,
            alpha=0.1
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
        
        # Build model
        dummy_h = tf.zeros([1, system.num_users, 1, 1, system.num_bs_antennas, 
                           system.rg.num_ofdm_symbols, system.rg.fft_size], 
                          dtype=tf.complex64)
        _ = system.precoder(dummy_h, training=False)
        
        self.trainable_vars = system.precoder.trainable_variables
        self.checkpoint = SimpleCheckpoint(f'./training_weights_residual')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  RESIDUAL TRANSFORMER TRAINING (Direct Sum Rate)")
        print(f"{'='*80}")
        print(f"  Epochs:     {total_epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  LR:         {learning_rate} → {learning_rate * 0.1}")
        print(f"  Baseline:   RZF (frozen)")
        print(f"  Strategy:   Direct gradient on sum rate")
        print(f"{'='*80}\n")
    
    @tf.function
    def train_step(self, h_freq_batch, batch_size, snr_db):
        """
        Single training step - PURE SUM RATE MAXIMIZATION
        
        Loss = -sum_rate (simple!)
        """
        with tf.GradientTape() as tape:
            # Full forward pass (includes RZF + correction)
            b, b_hat, c, llr, h_eff, no, _, _, _, g = \
                self.system.call_with_cached_channel(
                    batch_size, snr_db, h_freq_batch, training=True
                )
            
            # ✅ LOSS: Negative sum rate (simple!)
            sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr_clipped = tf.clip_by_value(sinr, 1e-9, 1e4)
            
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[0, 1, 2, 4])
            sum_rate = tf.reduce_sum(rate_per_user)
            
            sum_rate_safe = tf.where(tf.math.is_finite(sum_rate), 
                                    sum_rate, tf.constant(0.0))
            loss = -sum_rate_safe  # Scale for stability
        
        # Gradients
        grads = tape.gradient(loss, self.trainable_vars)
        
        if grads is None or any(g is None for g in grads):
            print("⚠️ Gradient is None!")
            grad_norm = tf.constant(0.0)
        else:
            # Clean and clip
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) 
                    for g in grads]
            grads, global_norm = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
            grad_norm = global_norm
        
        ber = compute_ber(b, b_hat)
        
        return sum_rate_safe, ber, grad_norm, loss, rate_per_user
    
    def train(self, print_every=50, patience=15):
        """Training loop"""
        
        print(f"🚀 Training @ {self.snr_db} dB")
        print(f"{'='*120}")
        print(f"{'Epoch':>6} | {'Sum Rate':>10} | {'User Rates':>40} | "
              f"{'BER':>10} | {'Grad':>8} | {'Alpha':>6}")
        print(f"{'='*120}")
        
        wait = 0
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'ber': [], 'grad': [], 'rates': []}
            
            for i in range(self.num_iters):
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                bs = tf.constant(self.batch_size, dtype=tf.int32)
                snr = tf.constant(self.snr_db, dtype=tf.float32)
                
                rate, ber, g_norm, loss, rates = self.train_step(
                    h_freq_batch, bs, snr
                )
                
                metrics['sum_rate'].append(float(rate))
                metrics['ber'].append(float(ber))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())
                
                # Intra-epoch printing
                if (i + 1) % print_every == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-print_every:])
                    rec_ber = np.mean(metrics['ber'][-print_every:])
                    rec_grad = np.mean(metrics['grad'][-print_every:])
                    rec_rates = np.mean(metrics['rates'][-print_every:], axis=0)
                    alpha_val = float(self.system.precoder.alpha)
                    
                    rates_str = "[" + " ".join([f"{r:.1f}" for r in rec_rates]) + "]"
                    print(f"    ... Step {i+1}/{self.num_iters} | "
                          f"Sum: {rec_sum:.2f} | Usr: {rates_str} | "
                          f"BER: {rec_ber:.1e} | Grad: {rec_grad:.2e} | "
                          f"α: {alpha_val:.3f} ✅", end='\r')
            
            # Epoch summary
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            alpha_val = float(self.system.precoder.alpha)
            
            self.history.append({
                'epoch': epoch, 
                'sum_rate': avg_sum, 
                'ber': avg_ber,
                'alpha': alpha_val
            })
            
            # Save best
            marker = ""
            if avg_sum > self.best_sum_rate:
                self.best_sum_rate = avg_sum
                self.best_epoch = epoch
                self.checkpoint.save_best(
                    self.system.precoder,
                    {'epoch': epoch, 'snr_db': self.snr_db, 'alpha': alpha_val},
                    {'sum_rate': avg_sum, 'ber': avg_ber}
                )
                marker = " ⭐"
                wait = 0
            else:
                wait += 1
            
            # Print epoch summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            print(f"\r{epoch+1:6d} | {avg_sum:10.2f} | {rate_str:>40} | "
                  f"{avg_ber:10.2e} | {avg_grad:8.2e} | {alpha_val:6.3f}{marker}")
            
            # Early stopping
            if wait >= patience:
                print(f"\n🛑 Early Stopping: No improvement for {patience} epochs")
                break
        
        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}")
        print(f"   Final Alpha: {self.history[self.best_epoch]['alpha']:.3f}\n")
        
        return self.checkpoint.best_checkpoint_path
        
class SimpleGradientTest:
    """Minimal test to find gradient issue"""
    
    def __init__(self, system, cached_dataset, batch_size=256):
        self.system = system
        self.cached_dataset = cached_dataset
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
    
    @tf.function
    def test_step_1_abs_squared(self, h_freq_batch, batch_size):
        """Test 1: Loss = |g|² (complex abs)"""
        with tf.GradientTape() as tape:
            # Get precoder output
            h_freq_batch_input = h_freq_batch
            g = self.system.precoder(h_freq_batch_input, training=True)
            
            # Simple loss
            loss = tf.reduce_mean(tf.abs(g)**2)
        
        grads = tape.gradient(loss, self.system.precoder.trainable_variables)
        return self._check_grads(grads, loss, "Test 1: |g|²")
    
    @tf.function
    def test_step_2_manual_squared(self, h_freq_batch, batch_size):
        """Test 2: Loss = real² + imag² (manual)"""
        with tf.GradientTape() as tape:
            g = self.system.precoder(h_freq_batch, training=True)
            
            # Manual magnitude squared
            g_real = tf.math.real(g)
            g_imag = tf.math.imag(g)
            loss = tf.reduce_mean(g_real**2 + g_imag**2)
        
        grads = tape.gradient(loss, self.system.precoder.trainable_variables)
        return self._check_grads(grads, loss, "Test 2: real²+imag²")
    
    @tf.function
    def test_step_3_real_only(self, h_freq_batch, batch_size):
        """Test 3: Loss = real² (no complex ops)"""
        with tf.GradientTape() as tape:
            g = self.system.precoder(h_freq_batch, training=True)
            
            # Only real part
            g_real = tf.math.real(g)
            loss = tf.reduce_mean(g_real**2)
        
        grads = tape.gradient(loss, self.system.precoder.trainable_variables)
        return self._check_grads(grads, loss, "Test 3: real² only")
    
    @tf.function
    def test_step_4_before_complex(self, h_freq_batch, batch_size):
        """Test 4: Loss on model output BEFORE tf.complex()"""
        with tf.GradientTape() as tape:
            # Manually call precoder internals
            B = tf.shape(h_freq_batch)[0]
            h_freq = tf.stop_gradient(h_freq_batch)
            h_sq = tf.squeeze(h_freq, axis=[2, 3])
            
            # Go through precoder up to BEFORE complex conversion
            if self.system.precoder.use_rb_grouping:
                h_rb = tf.reshape(h_sq, [B, self.system.precoder.num_rx, 
                                        self.system.precoder.num_tx, 
                                        self.system.precoder.num_ofdm, 
                                        self.system.precoder.num_rb, 
                                        self.system.precoder.rb_size])
                h_rb = tf.transpose(h_rb, [0, 3, 4, 5, 1, 2])
                h_math = tf.stack([tf.math.real(h_rb), tf.math.imag(h_rb)], axis=-1)
                h_math_agg = self.system.precoder.rb_aggregation(
                    tf.transpose(h_math, [0, 1, 2, 4, 5, 6, 3])
                )
                h_math_agg = tf.squeeze(h_math_agg, axis=-1)
                feat = tf.reshape(h_math_agg, [B * self.system.precoder.num_ofdm, 
                                               self.system.precoder.num_rb, 
                                               self.system.precoder.num_rx, 
                                               self.system.precoder.num_tx * 2])
            else:
                h_perm = tf.transpose(h_sq, [0, 3, 4, 1, 2])
                h_math = tf.stack([tf.math.real(h_perm), tf.math.imag(h_perm)], axis=-1)
                feat = tf.reshape(h_math, [B * self.system.precoder.num_ofdm, 
                                          self.system.precoder.fft_size, 
                                          self.system.precoder.num_rx, 
                                          self.system.precoder.num_tx * 2])
            
            # Through transformer
            x = self.system.precoder.input_embedding(feat)
            x = self.system.precoder.feature_norm(x)
            for block in self.system.precoder.blocks:
                x = block(x, training=True)
            
            # Output projection (BEFORE complex conversion)
            out = self.system.precoder.output_projection(x)
            
            # Loss on raw output (real-valued)
            loss = tf.reduce_mean(out**2)
        
        grads = tape.gradient(loss, self.system.precoder.trainable_variables)
        return self._check_grads(grads, loss, "Test 4: before complex")
    
    def _check_grads(self, grads, loss, test_name):
        """Check gradient validity"""
        valid_count = 0
        none_count = 0
        nan_count = 0
        zero_count = 0
        
        for grad in grads:
            if grad is None:
                none_count += 1
            elif not tf.reduce_all(tf.math.is_finite(grad)):
                nan_count += 1
            elif tf.reduce_mean(tf.abs(grad)) < 1e-12:
                zero_count += 1
            else:
                valid_count += 1
        
        total = len(grads)
        
        return {
            'test': test_name,
            'loss': float(loss),
            'total': total,
            'valid': valid_count,
            'none': none_count,
            'nan': nan_count,
            'zero': zero_count
        }
    
    def run_all_tests(self, num_batches=50):
        """Run all tests and aggregate results"""
        print("\n" + "="*80)
        print("  GRADIENT TEST SUITE")
        print("="*80)
        
        results = {
            'test_1': [],
            'test_2': [],
            'test_3': [],
            'test_4': []
        }
        
        for i in range(num_batches):
            h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
            bs = tf.constant(self.batch_size, dtype=tf.int32)
            
            results['test_1'].append(self.test_step_1_abs_squared(h_freq_batch, bs))
            results['test_2'].append(self.test_step_2_manual_squared(h_freq_batch, bs))
            results['test_3'].append(self.test_step_3_real_only(h_freq_batch, bs))
            results['test_4'].append(self.test_step_4_before_complex(h_freq_batch, bs))
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_batches} batches tested...")
        
        print("\n" + "="*80)
        print("  RESULTS")
        print("="*80)
        
        for test_name, test_results in results.items():
            total_valid = sum([r['valid'] for r in test_results])
            total_batches = len(test_results)
            avg_valid = total_valid / total_batches
            
            none_rate = sum([r['none'] for r in test_results]) / total_batches
            nan_rate = sum([r['nan'] for r in test_results]) / total_batches
            zero_rate = sum([r['zero'] for r in test_results]) / total_batches
            
            success_rate = (sum([1 for r in test_results if r['valid'] == r['total']]) / total_batches) * 100
            
            print(f"\n{test_results[0]['test']}:")
            print(f"  Success rate: {success_rate:.1f}% ({total_batches - sum([1 for r in test_results if r['valid'] == r['total']])} failures)")
            print(f"  Avg valid grads: {avg_valid:.1f}/{test_results[0]['total']}")
            print(f"  Avg None: {none_rate:.1f}, NaN: {nan_rate:.1f}, Zero: {zero_rate:.1f}")
        
        print("\n" + "="*80)
        
        return results
# =============================================================================
# UNSUPERVISED TRAINER (Pure Sum Rate)
# =============================================================================
class UnsupervisedTrainer:
    def __init__(self, system, cached_dataset,
                 snr_db=20.0,
                 total_epochs=100,
                 learning_rate=1e-3,
                 batch_size=256,
                 num_iters=None,
                 rate_weight=1.0,
                 bce_weight=0.01):
        
        self.system = system
        self.cached_dataset = cached_dataset
        self.snr_db = snr_db
        
        self.total_epochs = int(total_epochs)
        self.num_iters = int(num_iters) if num_iters else None
        
        self.batch_size = batch_size
        self.rate_weight = rate_weight
        self.bce_weight = bce_weight
        self.remove_nulled_scs = RemoveNulledSubcarriers(system.rg)
        # Optimizer with gradient clipping
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_epochs * (num_iters or 100),
            alpha=0.1
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
        
        # ✅ FIXED: Build model with correct input shape
        if system.precoder is not None:  # Only for transformer
            dummy_h = tf.zeros(
                [1, system.num_users, 1, 1, system.num_bs_antennas,
                 system.rg.num_ofdm_symbols, system.rg.fft_size], 
                dtype=tf.complex64
            )
            
            try:
                # ✅ Pass ONLY h_freq, not a tuple
                _ = system.precoder(dummy_h, training=False)
                self.trainable_vars = system.precoder.trainable_variables
                total_params = sum([tf.size(v).numpy() for v in self.trainable_vars])
                num_vars = len(self.trainable_vars)
                print(f"✅ Model built: {num_vars} weight tensors, {total_params:,} total parameters")
                
                # Debug: print variables
                if num_vars < 50:  # Something's wrong
                    print(f"\n⚠️  WARNING: Only {num_vars} variables found!")
                    print("🔍 Trainable variables:")
                    for i, v in enumerate(self.trainable_vars):
                        print(f"  {i+1:3d}. {v.name:60s} shape={str(v.shape):20s}")
            except Exception as e:
                print(f"❌ Failed to build precoder: {e}")
                raise ValueError("Unsupervised training requires transformer precoder")
        else:
            raise ValueError("Unsupervised training requires transformer precoder")
        
        self.checkpoint = SimpleCheckpoint(f'./training_weights_{system.precoder_type}_unsupervised')
        self.best_sum_rate = -np.inf
        self.best_epoch = 0
        self.history = []
        
        print(f"\n{'='*80}")
        print(f"  UNSUPERVISED TRAINING (MAX SUM-RATE) - FIXED")
        print(f"{'='*80}")
        print(f"  Total Epochs: {self.total_epochs}")
        print(f"  Batch Size:   {self.batch_size}")
        print(f"  LR:           {learning_rate} → {learning_rate * 0.1}")
        print(f"  Loss:         -{rate_weight}×SumRate + {bce_weight}×BCE")
        print(f"{'='*80}\n")
    @tf.function
    def train_step_debug(self, h_freq_batch, batch_size, snr_db):
        """Complete gradient flow debugging - trace every step"""
        
        with tf.GradientTape(persistent=True) as tape:
            # ============================================================
            # FORWARD PASS
            # ============================================================
            b, b_hat, c, llr, _, no, h_freq, x_rg, x_precoded, g = \
                self.system.call_with_cached_channel(batch_size, snr_db, h_freq_batch, training=True)
            
            # Watch intermediate values
            tape.watch(h_freq)
            h_freq = tf.stop_gradient(h_freq)
            
            # Manual h_eff
            h = tf.squeeze(h_freq, axis=[2, 3])
            h = tf.transpose(h, [0, 1, 3, 4, 2])
            g_work = tf.squeeze(g, axis=1)
            h_exp = tf.expand_dims(tf.expand_dims(h, axis=2), axis=-1)
            g_exp = tf.expand_dims(tf.expand_dims(g_work, axis=1), axis=2)
            h_eff_manual = tf.reduce_sum(h_exp * g_exp, axis=5)
            h_eff_manual = tf.squeeze(h_eff_manual, axis=2)
            tape.watch(h_eff_manual)
            
            # SINR
            h_eff_diag = tf.linalg.diag_part(tf.transpose(h_eff_manual, [0, 2, 3, 1, 4]))
            h_eff_users = tf.transpose(h_eff_diag, [0, 3, 1, 2])
            signal_power = tf.abs(h_eff_users)**2
            tape.watch(signal_power)
            
            total_power = tf.reduce_sum(signal_power, axis=1, keepdims=True)
            interference = total_power - signal_power
            
            no_val = tf.cast(no, signal_power.dtype)
            if len(no.shape) == 1:
                no_val = tf.reshape(no_val, [batch_size, 1, 1, 1])
            
            denominator = interference + no_val + 1e-8
            denominator = tf.maximum(denominator, 1e-8)
            sinr = signal_power / denominator
            sinr_clipped = tf.clip_by_value(sinr, 1e-8, 1e3)
            tape.watch(sinr_clipped)
            
            # Rate
            rate_per_element = tf.math.log(1.0 + sinr_clipped) / tf.math.log(2.0)
            tape.watch(rate_per_element)
            
            rate_per_user_per_sample = tf.reduce_mean(rate_per_element, axis=[2, 3])
            sum_rate_per_sample = tf.reduce_sum(rate_per_user_per_sample, axis=1)
            sum_rate = tf.reduce_mean(sum_rate_per_sample)
            tape.watch(sum_rate)
            
            sum_rate_safe = tf.where(tf.math.is_finite(sum_rate), sum_rate, tf.constant(0.0))
            loss = -sum_rate_safe
        
        # ============================================================
        # GRADIENT FLOW ANALYSIS - BACKWARD THROUGH ENTIRE CHAIN
        # ============================================================
        tf.print("\n" + "="*80)
        tf.print("GRADIENT FLOW ANALYSIS")
        tf.print("="*80)
        
        # Step 1: loss → sum_rate
        grad_sum_rate = tape.gradient(loss, sum_rate)
        if grad_sum_rate is None:
            tf.print("❌ BREAK AT: loss → sum_rate")
            return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), loss, tf.zeros(4)
        else:
            tf.print("✅ loss → sum_rate:", tf.reduce_mean(tf.abs(grad_sum_rate)))
        
        # Step 2: loss → rate_per_element
        grad_rate = tape.gradient(loss, rate_per_element)
        if grad_rate is None:
            tf.print("❌ BREAK AT: loss → rate_per_element")
        else:
            has_nan = tf.reduce_any(tf.math.is_nan(grad_rate))
            mean_val = tf.reduce_mean(tf.abs(grad_rate))
            tf.print("✅ loss → rate_per_element: mean=", mean_val, "NaN=", has_nan)
        
        # Step 3: loss → sinr
        grad_sinr = tape.gradient(loss, sinr_clipped)
        if grad_sinr is None:
            tf.print("❌ BREAK AT: loss → sinr")
        else:
            has_nan = tf.reduce_any(tf.math.is_nan(grad_sinr))
            mean_val = tf.reduce_mean(tf.abs(grad_sinr))
            tf.print("✅ loss → sinr: mean=", mean_val, "NaN=", has_nan)
        
        # Step 4: loss → signal_power
        grad_sig_pow = tape.gradient(loss, signal_power)
        if grad_sig_pow is None:
            tf.print("❌ BREAK AT: loss → signal_power")
        else:
            has_nan = tf.reduce_any(tf.math.is_nan(grad_sig_pow))
            mean_val = tf.reduce_mean(tf.abs(grad_sig_pow))
            tf.print("✅ loss → signal_power: mean=", mean_val, "NaN=", has_nan)
        
        # Step 5: loss → h_eff_manual
        grad_heff = tape.gradient(loss, h_eff_manual)
        if grad_heff is None:
            tf.print("❌ BREAK AT: loss → h_eff_manual")
        else:
            has_nan_real = tf.reduce_any(tf.math.is_nan(tf.math.real(grad_heff)))
            has_nan_imag = tf.reduce_any(tf.math.is_nan(tf.math.imag(grad_heff)))
            mean_val = tf.reduce_mean(tf.abs(grad_heff))
            tf.print("✅ loss → h_eff_manual: mean=", mean_val, "NaN=", tf.logical_or(has_nan_real, has_nan_imag))
        
        # Step 6: loss → g (precoder output)
        grad_g = tape.gradient(loss, g)
        if grad_g is None:
            tf.print("❌ BREAK AT: loss → g (PRECODER OUTPUT)")
        else:
            has_nan_real = tf.reduce_any(tf.math.is_nan(tf.math.real(grad_g)))
            has_nan_imag = tf.reduce_any(tf.math.is_nan(tf.math.imag(grad_g)))
            mean_val = tf.reduce_mean(tf.abs(grad_g))
            tf.print("✅ loss → g: mean=", mean_val, "NaN=", tf.logical_or(has_nan_real, has_nan_imag))
        
        # Step 7: loss → h_freq (should be zero or ignored, just checking)
        grad_h = tape.gradient(loss, h_freq)
        if grad_h is None:
            tf.print("   loss → h_freq: None (expected)")
        else:
            tf.print("   loss → h_freq: exists (unexpected)")
        
        tf.print("="*80)
        
        # ============================================================
        # MODEL PARAMETER GRADIENTS - CHECK EACH LAYER
        # ============================================================
        tf.print("\nMODEL PARAMETER GRADIENTS:")
        tf.print("-"*80)
        
        grads = tape.gradient(loss, self.trainable_vars)
        
        if all([g is not None for g in grads]):
            # Analyze each gradient
            num_nan = 0
            num_zero = 0
            num_ok = 0
            
            for i, (var, grad) in enumerate(zip(self.trainable_vars, grads)):
                is_finite = tf.reduce_all(tf.math.is_finite(grad))
                mean_abs = tf.reduce_mean(tf.abs(grad))
                
                if not is_finite:
                    tf.print(f"  [{i}] {var.name}: ❌ NaN/Inf")
                    num_nan += 1
                elif mean_abs < 1e-12:
                    tf.print(f"  [{i}] {var.name}: ⚠️  Near-zero (mean={mean_abs})")
                    num_zero += 1
                else:
                    # Only print first few OK gradients to avoid spam
                    if num_ok < 5:
                        tf.print(f"  [{i}] {var.name}: ✅ OK (mean={mean_abs})")
                    num_ok += 1
            
            tf.print(f"\nSummary: {num_ok} OK, {num_zero} near-zero, {num_nan} NaN/Inf")
            tf.print("="*80 + "\n")
            
            # Apply gradients (replace NaN with zeros)
            clean_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
            clipped_grads, global_norm = tf.clip_by_global_norm(clean_grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, self.trainable_vars))
            grad_norm = global_norm
        else:
            grad_norm = tf.constant(0.0)
            tf.print("❌ Some parameter gradients are None!")
        
        del tape
        
        ber = compute_ber(b, b_hat)
        rate_per_user_avg = tf.reduce_mean(rate_per_user_per_sample, axis=0)
        
        return sum_rate_safe, ber, grad_norm, loss, rate_per_user_avg
    @tf.function  
    def train_step(self, h_freq_batch, batch_size, snr_db):
        """Using Sionna's compute_effective_channel"""
        with tf.GradientTape() as tape:
            # Get data
            no = ebnodb2no(snr_db, self.system.num_bits_per_symbol, 0.5, self.system.rg)
            b = self.system.binary_source([batch_size, 1, self.system.num_users, 
                                        int(self.system.rg.num_data_symbols)])
            c = self.system.encoder(b)
            x = self.system.mapper(c)
            x_rg = self.system.rg_mapper(x)
            
            h_freq = tf.stop_gradient(h_freq_batch)
            
            # Get precoder output as REAL/IMAG
            g_real, g_imag = self.system.precoder(h_freq, training=True, return_real_imag=True)
            
            # Convert to complex
            g_complex = tf.complex(g_real, g_imag)
            
            # Utilise Sionna's compute_effective_channel
            h_eff = self.system.precoded_channel_helper.compute_effective_channel(h_freq, g_complex)
            
            # SINR
            sinr = self.system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr = tf.clip_by_value(sinr, 1e-8, 1e3)
            
            # Compute rate
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user_per_sample = tf.reduce_mean(rate_per_element, axis=[1, 2])
            rate_per_user_per_sample = tf.reduce_sum(rate_per_user_per_sample, axis=-1)
            
            sum_rate_per_sample = tf.reduce_sum(rate_per_user_per_sample, axis=1)
            sum_rate = tf.reduce_mean(sum_rate_per_sample)
            
            sum_rate_safe = tf.where(tf.math.is_finite(sum_rate), sum_rate, tf.constant(0.0))
            loss = -sum_rate_safe
        
        # Gradients
        grads = tape.gradient(loss, self.trainable_vars)
        
        if all([g is not None for g in grads]):
            clean_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
            clipped_grads, global_norm = tf.clip_by_global_norm(clean_grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, self.trainable_vars))
            grad_norm = global_norm
        else:
            grad_norm = tf.constant(0.0)
        
        # Compute BER
        x_precoded = self.system._apply_precoding_to_signal(x_rg, g_complex)
        
        # ✅ CORRIGÉ: Arguments séparés
        y = self.system.apply_channel(x_precoded, h_freq, no)
        x_hat, no_eff = self.system.lmmse_equ(y, h_eff, 0.0, no)
        llr = self.system.demapper(x_hat, no_eff)
        b_hat = self.system.decoder(llr)
        
        ber = compute_ber(b, b_hat)
        
        # Per-user rates for logging
        rate_per_user_avg = tf.reduce_mean(rate_per_user_per_sample, axis=0)
        
        return sum_rate_safe, ber, grad_norm, loss, rate_per_user_avg
    def train(self, log_interval=1, patience=20, print_every=10):
        # Auto-calculate iterations
        if self.num_iters is None:
            if hasattr(self.cached_dataset, 'h_freq_all'):
                self.num_iters = len(self.cached_dataset.h_freq_all) // self.batch_size
            else:
                self.num_iters = DATASET_SIZE // BATCH_SIZE
            print(f"  -> Auto-configured: {self.num_iters} iterations/epoch")
        
        print(f"\n🚀 Training @ {self.snr_db} dB")
        print(f"{'='*155}")
        print(f"{'Epoch':>6} | {'Mode':>10} | {'Sum Rate':>10} | {'User Rates':>40} | {'BER':>10} | {'Grad':>8} | {'Loss':>8}")
        print(f"{'='*155}")
        
        wait = 0
        nan_count = 0  # Track NaN occurrences
        
        for epoch in range(self.total_epochs):
            metrics = {'sum_rate': [], 'ber': [], 'loss': [], 'rates': [], 'grad': []}
            mode_str = "UNSUPERV"
            
            for i in range(self.num_iters):
                h_freq_batch = self.cached_dataset.get_batch(self.batch_size)
                bs_tensor = tf.constant(self.batch_size, dtype=tf.int32)
                snr_tensor = tf.constant(self.snr_db, dtype=tf.float32)
                
                rate, ber, g_norm, loss, rates = self.train_step(
                    h_freq_batch, bs_tensor, snr_tensor
                )
                
                metrics['sum_rate'].append(float(rate))
                metrics['ber'].append(float(ber))
                metrics['loss'].append(float(loss))
                metrics['grad'].append(float(g_norm))
                metrics['rates'].append(rates.numpy())
                
                # Intra-epoch printing
                if (i + 1) % print_every == 0:
                    rec_sum = np.mean(metrics['sum_rate'][-print_every:])
                    rec_loss = np.mean(metrics['loss'][-print_every:])
                    rec_ber = np.mean(metrics['ber'][-print_every:])
                    rec_grad = np.mean(metrics['grad'][-print_every:])
                    rec_rates = np.mean(metrics['rates'][-print_every:], axis=0)
                    
                    rates_str = "[" + " ".join([f"{r:.1f}" for r in rec_rates]) + "]"
                    expected_loss = -rec_sum * self.rate_weight
                    loss_match = "✅" if abs(rec_loss - expected_loss) < 0.5 else "⚠️"
                    
                    print(f"    ... Step {i+1}/{self.num_iters} | Sum: {rec_sum:.2f} | "
                        f"Usr: {rates_str} | BER: {rec_ber:.1e} | "
                        f"Grad: {rec_grad:.2e} | Loss: {rec_loss:.2f} {loss_match}", end='\r')
            
            # End of Epoch
            avg_loss = np.mean(metrics['loss'])
            avg_sum = np.mean(metrics['sum_rate'])
            avg_ber = np.mean(metrics['ber'])
            avg_grad = np.mean(metrics['grad'])
            avg_rates = np.mean(metrics['rates'], axis=0)
            
            # ✅ Count zero gradients (outside @tf.function)
            zero_grad_count = sum([1 for g in metrics['grad'] if abs(g) < 1e-12])
            zero_grad_pct = (zero_grad_count / len(metrics['grad'])) * 100
            
            
            self.history.append({'epoch': epoch, 'sum_rate': avg_sum, 'ber': avg_ber})
            
            # Save best
            if avg_sum > self.best_sum_rate:
                self.best_sum_rate = avg_sum
                self.best_epoch = epoch
                self.checkpoint.save_best(
                    self.system.precoder,
                    {'epoch': epoch, 'snr_db': self.snr_db},
                    {'sum_rate': avg_sum, 'ber': avg_ber}
                )
                marker = " ⭐"
                wait = 0
            else:
                marker = ""
                wait += 1
            
            # Print epoch summary
            rate_str = "[" + ", ".join([f"{r:5.2f}" for r in avg_rates]) + "]"
            expected_loss = -avg_sum * self.rate_weight
            loss_match = "✅" if abs(avg_loss - expected_loss) < 0.5 else "⚠️"
            
            print(f"\r{epoch+1:6d} | {mode_str:>10} | {avg_sum:10.2f} | {rate_str:>40} | "
            f"{avg_ber:10.2e} | {avg_grad:8.2e} | {avg_loss:8.3f}{marker} {loss_match}      ")
        
            # ✅ Only warn if REALLY high
            if zero_grad_pct > 50:
                print(f"    ⚠️  WARNING: {zero_grad_pct:.0f}% of iterations had zero gradients")
            
            # ✅ Warning if too many NaN occurrences
            if nan_count > self.num_iters * 0.1:  # >10% of iterations
                print(f"    ⚠️  WARNING: {nan_count}/{self.num_iters} iterations had zero gradients!")
            
            nan_count = 0  # Reset for next epoch
            
            if wait >= patience:
                print(f"\n🛑 Early Stopping: No improvement for {patience} epochs.")
                break
        
        print(f"\n✅ Best Rate: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch+1}\n")
        return self.checkpoint.best_checkpoint_path


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_system_detailed(system, ebno_range, num_batches=100, batch_size=512, name="System"):
    """Comprehensive evaluation with per-user metrics"""
    print(f"\n📊 Evaluating {name}:")
    results = {
        'sum_rate': [], 'ber': [], 'sinr': [],
        'sinr_per_user': [], 'rate_per_user': [],
        'jain_index': [], 'min_rate': []
    }
    
    for ebno_db in ebno_range:
        rates, bers, sinrs = [], [], []
        all_sinr_per_user = []
        all_rate_per_user = []
        
        for _ in range(num_batches):
            system.new_topology(batch_size)
            
            b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g = system(
                tf.constant(batch_size, dtype=tf.int32),
                tf.constant(ebno_db, dtype=tf.float32),
                training=False
            )
            
            ber = compute_ber(b, b_hat)
            bers.append(float(ber))
            
            sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            sinr_mean = tf.reduce_mean(sinr)
            sinrs.append(float(sinr_mean))
            
            # Per-user metrics
            rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])  # [Batch, Users]
            
            rate_per_user_avg = tf.reduce_mean(rate_per_user, axis=0)  # [Users]
            all_rate_per_user.append(rate_per_user_avg.numpy())
            
            sinr_per_user = tf.reduce_mean(sinr, axis=[1, 2, 4])  # [Batch, Users]
            sinr_per_user_avg = tf.reduce_mean(sinr_per_user, axis=0)  # [Users]
            all_sinr_per_user.append(sinr_per_user_avg.numpy())
            
            sum_rate = tf.reduce_sum(rate_per_user_avg)
            rates.append(float(sum_rate))
        
        # Aggregate metrics
        avg_rate = np.mean(rates)
        avg_ber = np.mean(bers)
        avg_sinr_db = 10 * np.log10(np.mean(sinrs) + 1e-12)
        
        rate_per_user_all = np.mean(all_rate_per_user, axis=0)
        sinr_per_user_all = np.mean(all_sinr_per_user, axis=0)
        sinr_per_user_db = 10 * np.log10(sinr_per_user_all + 1e-12)
        
        # Fairness metrics
        min_rate = np.min(rate_per_user_all)
        K = len(rate_per_user_all)
        jain = np.sum(rate_per_user_all)**2 / (K * np.sum(rate_per_user_all**2))
        
        print(f"\n  SNR={ebno_db:2.0f}dB:")
        print(f"    Sum Rate: {avg_rate:5.2f} bps/Hz, BER: {avg_ber:.2e}, Avg SINR: {avg_sinr_db:5.2f}dB")
        print(f"    Per-User Rates: {rate_per_user_all}")
        print(f"      └─ Min: {min_rate:.2f}, Jain: {jain:.3f}")
        print(f"    Per-User SINR (dB): {sinr_per_user_db}")
        
        results['sum_rate'].append(avg_rate)
        results['ber'].append(avg_ber)
        results['sinr'].append(avg_sinr_db)
        results['sinr_per_user'].append(sinr_per_user_db)
        results['rate_per_user'].append(rate_per_user_all)
        results['jain_index'].append(jain)
        results['min_rate'].append(min_rate)
    
    return results


def plot_comprehensive_comparison(results_all, ebno_range, save_dir='./results'):
    """Generate comprehensive comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    colors = {
        'RZF': 'blue',
        'WMMSE': 'green',
        'Transformer (RB)': 'red',
        'Transformer (Full)': 'purple',
        'Transformer (RB-Sup)': 'orange',
        'Transformer (Full-Sup)': 'brown'
    }
    markers = {
        'RZF': 'o',
        'WMMSE': 's',
        'Transformer (RB)': '^',
        'Transformer (Full)': 'v',
        'Transformer (RB-Sup)': 'D',
        'Transformer (Full-Sup)': 'p'
    }
    
    # Figure 1: Main Performance Metrics
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    
    for name, res in results_all.items():
        color = colors.get(name, 'black')
        marker = markers.get(name, 'x')
        
        # Sum Rate
        axes1[0, 0].plot(ebno_range, res['sum_rate'], label=name,
                        linewidth=2.5, marker=marker, markersize=8, color=color)
        
        # BER
        axes1[0, 1].semilogy(ebno_range, res['ber'], label=name,
                            linewidth=2.5, marker=marker, markersize=8, color=color)
        
        # Jain Index
        axes1[1, 0].plot(ebno_range, res['jain_index'], label=name,
                        linewidth=2.5, marker=marker, markersize=8, color=color)
        
        # Min Rate
        axes1[1, 1].plot(ebno_range, res['min_rate'], label=name,
                        linewidth=2.5, marker=marker, markersize=8, color=color)
    
    axes1[0, 0].set(xlabel='SNR (dB)', ylabel='Sum Rate (bps/Hz)', title='Spectral Efficiency')
    axes1[0, 1].set(xlabel='SNR (dB)', ylabel='BER', title='Bit Error Rate')
    axes1[1, 0].set(xlabel='SNR (dB)', ylabel='Jain Index', title='Fairness (Jain Index)')
    axes1[1, 1].set(xlabel='SNR (dB)', ylabel='Min Rate (bps/Hz)', title='Worst-User Rate')
    
    for ax in axes1.flat:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_main_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Main plot saved: comparison_main_{timestamp}.png")
    
    # Figure 2: Complexity vs Performance @ 20 dB
    idx_20 = np.where(ebno_range == 20)[0][0]
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    complexity_map = {
        'RZF': 1,
        'WMMSE': 10000,
        'Transformer (RB)': 36,
        'Transformer (Full)': 5184,
        'Transformer (RB-Sup)': 36,
        'Transformer (Full-Sup)': 5184
    }
    
    for name, res in results_all.items():
        rate_20 = res['sum_rate'][idx_20]
        complexity = complexity_map.get(name, 100)
        
        ax2.scatter(complexity, rate_20, s=200, marker=markers[name],
                   color=colors[name], label=name, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Computational Complexity (relative to RZF)', fontsize=12)
    ax2.set_ylabel('Sum Rate @ 20 dB (bps/Hz)', fontsize=12)
    ax2.set_title('Complexity vs Performance Trade-off', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'complexity_tradeoff_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Complexity plot saved: complexity_tradeoff_{timestamp}.png")
    
    plt.close('all')


def print_comparison_table(results_all, ebno_range, save_dir='./results'):
    """Print detailed comparison table"""
    idx_20 = np.where(ebno_range == 20)[0][0]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_file = os.path.join(save_dir, f'comparison_table_{timestamp}.txt')
    
    with open(table_file, 'w') as f:
        header = f"\n{'='*140}\n"
        header += f"{'Method':>25} | {'Sum Rate':>10} | {'BER':>10} | {'Jain':>6} | {'Min Rate':>9} | {'Complexity':>12} | {'Rate/Comp':>10}\n"
        header += f"{'='*140}\n"
        
        print(header)
        f.write(header)
        
        complexity_map = {
            'RZF': 1,
            'WMMSE': 10000,
            'Transformer (RB)': 36,
            'Transformer (Full)': 5184,
            'Transformer (RB-Sup)': 36,
            'Transformer (Full-Sup)': 5184
        }
        
        for name, res in results_all.items():
            rate = res['sum_rate'][idx_20]
            ber = res['ber'][idx_20]
            jain = res['jain_index'][idx_20]
            min_rate = res['min_rate'][idx_20]
            comp = complexity_map.get(name, 0)
            efficiency = rate / comp if comp > 0 else 0
            
            line = f"{name:>25} | {rate:10.2f} | {ber:10.2e} | {jain:6.3f} | {min_rate:9.2f} | {comp:12.0f} | {efficiency:10.4f}\n"
            print(line, end='')
            f.write(line)
        
        footer = f"{'='*140}\n"
        print(footer)
        f.write(footer)
    
    print(f"\n✅ Table saved: {table_file}")


def plot_gradient_magnitudes(diagnostic_results):
    """Visualize gradient statistics"""
    import matplotlib.pyplot as plt
    
    param_stats = diagnostic_results['param_stats']
    
    # Extract data
    names = []
    means = []
    maxs = []
    colors = []
    
    for stat in param_stats:
        if stat.get('status') == 'None':
            continue
        
        names.append(stat['name'].split('/')[-1][:20])  # Truncate names
        means.append(stat.get('mean', 0))
        maxs.append(stat.get('max', 0))
        
        if stat.get('has_nan'):
            colors.append('red')
        elif stat.get('is_zero'):
            colors.append('lightgray')
        else:
            colors.append('green')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Mean gradients
    ax1.bar(range(len(means)), means, color=colors, alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Mean |Gradient|', fontsize=12)
    ax1.set_title('Gradient Magnitudes per Parameter', fontsize=14, fontweight='bold')
    ax1.axhline(y=1e-12, color='r', linestyle='--', label='Near-zero threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Max gradients
    ax2.bar(range(len(maxs)), maxs, color=colors, alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_ylabel('Max |Gradient|', fontsize=12)
    ax2.set_xlabel('Parameter Index', fontsize=12)
    ax2.axhline(y=1e-12, color='r', linestyle='--', label='Near-zero threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Gradient plot saved: gradient_analysis.png")
    
    return fig


def compare_sinr_methods(system, h_freq_batch, no, h_eff):
    """Compare Sionna SINR vs manual computation"""
    print("\n" + "="*80)
    print("  SINR COMPUTATION COMPARISON")
    print("="*80)
    
    # Method 1: Sionna's LMMSE
    with tf.GradientTape() as tape1:
        tape1.watch(h_eff)
        sinr_sionna = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        loss1 = tf.reduce_mean(sinr_sionna)
    
    grad1 = tape1.gradient(loss1, h_eff)
    
    print(f"\n1️⃣  Sionna LMMSE SINR:")
    print(f"   Mean SINR: {tf.reduce_mean(sinr_sionna):.4f}")
    print(f"   Gradient exists: {grad1 is not None}")
    if grad1 is not None:
        print(f"   Gradient mean: {tf.reduce_mean(tf.abs(grad1)):.4e}")
    
    # Method 2: Manual computation
    with tf.GradientTape() as tape2:
        tape2.watch(h_eff)
        
        # Squeeze and compute power
        h_sq = tf.squeeze(h_eff, axis=[2, 4])  # [B, rx, tx, ofdm, sc]
        signal_power = tf.abs(h_sq)**2
        
        # Simple SINR: signal / (interference + noise)
        total_power = tf.reduce_sum(signal_power, axis=2, keepdims=True)
        interference = total_power - signal_power
        
        no_val = tf.cast(no, signal_power.dtype)
        sinr_manual = signal_power / (interference + no_val + 1e-12)
        
        loss2 = tf.reduce_mean(sinr_manual)
    
    grad2 = tape2.gradient(loss2, h_eff)
    
    print(f"\n2️⃣  Manual SINR:")
    print(f"   Mean SINR: {tf.reduce_mean(sinr_manual):.4f}")
    print(f"   Gradient exists: {grad2 is not None}")
    if grad2 is not None:
        print(f"   Gradient mean: {tf.reduce_mean(tf.abs(grad2)):.4e}")
    
    print(f"\n💡 Conclusion:")
    if grad1 is None and grad2 is not None:
        print(f"   ✅ Sionna's LMMSE SINR blocks gradients!")
        print(f"   🔧 Solution: Use manual SINR computation")
    elif grad1 is not None and grad2 is not None:
        print(f"   ✅ Both methods allow gradient flow")
        print(f"   ❓ Issue is elsewhere in the graph")
    else:
        print(f"   ❌ Both methods fail - deeper issue")
    
    print("="*80 + "\n")

# ==============================================================================
# ENERGY EFFICIENCY FUNCTIONS
# ==============================================================================

def compute_energy_efficiency(system, snr_db, batch_size=256):
    """
    Compute energy efficiency metrics for a system
    
    Args:
        system: MU_MIMO_System instance
        snr_db: SNR in dB
        batch_size: Batch size
    
    Returns:
        Dictionary with power and energy efficiency metrics
    """
    # Run system
    b, b_hat, _, _, h_eff, no, _, _, _, g = system(
        tf.constant(batch_size), tf.constant(snr_db)
    )
    
    # 1. POWER CONSUMPTION
    # Total transmit power (sum over all dimensions)
    power_per_sample = tf.reduce_sum(tf.abs(g)**2, axis=[1, 2, 3, 4, 5])  # [B]
    avg_power = tf.reduce_mean(power_per_sample)
    power_per_user = avg_power / system.num_rx
    
    # 2. ACHIEVABLE RATE
    sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
    rate_per_element = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
    rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])  # [B, users]
    sum_rate_per_sample = tf.reduce_sum(rate_per_user, axis=1)  # [B]
    avg_sum_rate = tf.reduce_mean(sum_rate_per_sample)
    
    # 3. ENERGY EFFICIENCY (bits/Hz/Joule)
    # EE = Sum Rate / Total Power
    energy_efficiency = avg_sum_rate / (avg_power + 1e-12)  # bps/Hz/W
    
    # 4. BER
    ber = compute_ber(b, b_hat)
    
    # 5. SINR (in dB)
    avg_sinr_db = tf.reduce_mean(10 * tf.math.log(sinr + 1e-12) / tf.math.log(10.0))
    
    return {
        'total_power': float(avg_power),
        'power_per_user': float(power_per_user),
        'sum_rate': float(avg_sum_rate),
        'energy_efficiency': float(energy_efficiency),
        'ber': float(ber),
        'avg_sinr_db': float(avg_sinr_db)
    }


def compute_relative_energy_efficiency(ee_transformer, ee_baseline):
    """
    Compute relative energy efficiency gains vs baseline
    
    Args:
        ee_transformer: Energy efficiency dict for transformer
        ee_baseline: Energy efficiency dict for baseline (RZF/WMMSE)
    
    Returns:
        Dictionary of improvement metrics
    """
    # 1. Energy Efficiency Gain
    ee_gain = ((ee_transformer['energy_efficiency'] / 
                (ee_baseline['energy_efficiency'] + 1e-12)) - 1.0) * 100
    
    # 2. Rate Improvement (at same power)
    rate_gain = ((ee_transformer['sum_rate'] / 
                  (ee_baseline['sum_rate'] + 1e-12)) - 1.0) * 100
    
    # 3. Power Reduction (for same rate)
    # Normalize power for equal rate
    rate_ratio = ee_transformer['sum_rate'] / (ee_baseline['sum_rate'] + 1e-12)
    power_ratio = ee_transformer['total_power'] / (ee_baseline['total_power'] + 1e-12)
    power_reduction = (1.0 - power_ratio / rate_ratio) * 100
    
    # 4. Absolute improvements
    absolute_ee_gain = ee_transformer['energy_efficiency'] - ee_baseline['energy_efficiency']
    absolute_rate_gain = ee_transformer['sum_rate'] - ee_baseline['sum_rate']
    absolute_power_reduction = ee_baseline['total_power'] - ee_transformer['total_power']
    
    return {
        'ee_gain_pct': ee_gain,
        'rate_gain_pct': rate_gain,
        'power_reduction_pct': power_reduction,
        'absolute_ee_gain': absolute_ee_gain,
        'absolute_rate_gain': absolute_rate_gain,
        'absolute_power_reduction': absolute_power_reduction
    }


def jain_fairness_index(rates):
    """Compute Jain's fairness index"""
    rates = np.array(rates)
    n = len(rates)
    if n == 0:
        return 1.0
    numerator = np.sum(rates) ** 2
    denominator = n * np.sum(rates ** 2)
    return numerator / (denominator + 1e-12)


def compute_ber(b, b_hat):
    """Compute bit error rate"""
    return tf.reduce_mean(tf.cast(tf.not_equal(b, b_hat), tf.float32))
# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate_system_detailed_with_ee(system, snr_range, num_batches, batch_size, name):
    """
    Comprehensive evaluation including energy efficiency
    
    Args:
        system: MU_MIMO_System instance
        snr_range: Array of SNR values in dB
        num_batches: Number of batches to average over
        batch_size: Batch size
        name: System name for display
    
    Returns:
        Dictionary with all metrics vs SNR
    """
    print(f"\n📊 Evaluating {name}:")
    
    results = {
        'snr': [],
        'sum_rate': [],
        'ber': [],
        'avg_sinr': [],
        'jain_index': [],
        'per_user_rates': [],
        'total_power': [],
        'power_per_user': [],
        'energy_efficiency': []
    }
    
    for snr in snr_range:
        sum_rates, bers, sinrs, jains, per_user = [], [], [], [], []
        powers, power_users, ees = [], [], []
        
        for _ in range(num_batches):
            system.new_topology(batch_size)
            
            # Get system outputs
            b, b_hat, _, _, h_eff, no, _, _, _, g = system(
                tf.constant(batch_size), tf.constant(snr)
            )
            
            # Compute rate metrics
            sinr_vals = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate_per_element = tf.math.log(1.0 + sinr_vals) / tf.math.log(2.0)
            rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])
            sum_rate_per_sample = tf.reduce_sum(rate_per_user, axis=1)
            
            sum_rates.append(float(tf.reduce_mean(sum_rate_per_sample)))
            bers.append(float(compute_ber(b, b_hat)))
            sinrs.append(float(tf.reduce_mean(10 * tf.math.log(sinr_vals + 1e-12) / tf.math.log(10.0))))
            
            # Power metrics
            power_per_sample = tf.reduce_sum(tf.abs(g)**2, axis=[1, 2, 3, 4, 5])
            total_power = float(tf.reduce_mean(power_per_sample))
            powers.append(total_power)
            power_users.append(total_power / system.num_rx)
            
            # Energy efficiency
            avg_sum_rate = sum_rates[-1]
            ees.append(avg_sum_rate / (total_power + 1e-12))
            
            # Fairness
            rates_np = rate_per_user.numpy()
            jain_vals = [jain_fairness_index(rates) for rates in rates_np]
            jains.append(np.mean(jain_vals))
            per_user.append(np.mean(rates_np, axis=0))
        
        # Store results
        results['snr'].append(snr)
        results['sum_rate'].append(np.mean(sum_rates))
        results['ber'].append(np.mean(bers))
        results['avg_sinr'].append(np.mean(sinrs))
        results['jain_index'].append(np.mean(jains))
        results['per_user_rates'].append(np.mean(per_user, axis=0))
        results['total_power'].append(np.mean(powers))
        results['power_per_user'].append(np.mean(power_users))
        results['energy_efficiency'].append(np.mean(ees))
        
        print(f"\n  SNR={snr:2.0f}dB:")
        print(f"    Sum Rate: {results['sum_rate'][-1]:5.2f} bps/Hz, BER: {results['ber'][-1]:.2e}, Avg SINR: {results['avg_sinr'][-1]:5.2f}dB")
        print(f"    Power: {results['total_power'][-1]:.2f} W, EE: {results['energy_efficiency'][-1]:.4f} bps/Hz/W")
        print(f"    Per-User Rates: {results['per_user_rates'][-1]}")
        print(f"      └─ Min: {np.min(results['per_user_rates'][-1]):.2f}, Jain: {results['jain_index'][-1]:.3f}")
    
    return results

# ==============================================================================
# SUMMARY PRINTING FUNCTIONS
# ==============================================================================

def print_energy_efficiency_summary(results_all, eval_snr_idx, eval_snr):
    """Print detailed energy efficiency comparison"""
    
    print("\n" + "="*80)
    print("  ⚡ ENERGY EFFICIENCY ANALYSIS")
    print("="*80)
    
    if 'RZF' not in results_all:
        print("❌ RZF baseline not found")
        return
    
    rzf_ee = results_all['RZF']['energy_efficiency'][eval_snr_idx]
    rzf_power = results_all['RZF']['total_power'][eval_snr_idx]
    rzf_rate = results_all['RZF']['sum_rate'][eval_snr_idx]
    
    print(f"\n📊 @ {eval_snr:.0f} dB SNR:")
    print(f"\n{'Method':<30} {'Rate':>10} {'Power':>10} {'EE':>12} {'vs RZF'}")
    print(f"{'':30} {'(bps/Hz)':>10} {'(W)':>10} {'(bps/Hz/W)':>12} {'(gain)'}")
    print("-"*80)
    
    print(f"{'RZF (Baseline)':<30} {rzf_rate:>10.2f} {rzf_power:>10.2f} {rzf_ee:>12.4f} {'—':>10}")
    
    for name in ['WMMSE', 'Transformer (RB-Sup)', 'Transformer (Full-Sup)']:
        if name in results_all:
            ee = results_all[name]['energy_efficiency'][eval_snr_idx]
            power = results_all[name]['total_power'][eval_snr_idx]
            rate = results_all[name]['sum_rate'][eval_snr_idx]
            
            ee_gain = ((ee / (rzf_ee + 1e-12)) - 1.0) * 100
            
            print(f"{name:<30} {rate:>10.2f} {power:>10.2f} {ee:>12.4f} {ee_gain:>9.1f}%")
    
    print("="*80)
    
    # Detailed breakdown
    print("\n🎯 Detailed Energy Efficiency Gains:")
    
    for name in ['WMMSE', 'Transformer (RB-Sup)', 'Transformer (Full-Sup)']:
        if name in results_all:
            ee_dict = {
                'sum_rate': results_all[name]['sum_rate'][eval_snr_idx],
                'total_power': results_all[name]['total_power'][eval_snr_idx],
                'energy_efficiency': results_all[name]['energy_efficiency'][eval_snr_idx]
            }
            
            baseline_dict = {
                'sum_rate': rzf_rate,
                'total_power': rzf_power,
                'energy_efficiency': rzf_ee
            }
            
            gains = compute_relative_energy_efficiency(ee_dict, baseline_dict)
            
            print(f"\n  {name}:")
            print(f"    • EE Improvement:     {gains['ee_gain_pct']:>6.1f}%  ({gains['absolute_ee_gain']:>+.4f} bps/Hz/W)")
            print(f"    • Rate Gain:          {gains['rate_gain_pct']:>6.1f}%  ({gains['absolute_rate_gain']:>+.2f} bps/Hz)")
            print(f"    • Power Reduction:    {gains['power_reduction_pct']:>6.1f}%  ({gains['absolute_power_reduction']:>+.2f} W)")

def plot_comprehensive_comparison_with_ee(results_all, snr_range, save_dir='./results'):
    """Generate comprehensive plots including BER and Energy Efficiency"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        'RZF': '#1f77b4',
        'WMMSE': '#ff7f0e',
        'Transformer (RB-Sup)': '#2ca02c',
        'Transformer (Full-Sup)': '#d62728'
    }
    
    markers = {
        'RZF': 'o',
        'WMMSE': 's',
        'Transformer (RB-Sup)': '^',
        'Transformer (Full-Sup)': 'v'
    }
    
    # ✅ 1. Sum Rate
    ax = axes[0, 0]
    for name, res in results_all.items():
        ax.plot(snr_range, res['sum_rate'], 
                marker=markers.get(name, 'x'),
                color=colors.get(name, 'gray'),
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Sum Rate vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ✅ 2. BER (LOG SCALE)
    ax = axes[0, 1]
    for name, res in results_all.items():
        ber_values = np.array(res['ber'])
        ber_values = np.maximum(ber_values, 1e-7)  # Avoid log(0)
        ax.semilogy(snr_range, ber_values,
                   marker=markers.get(name, 'x'),
                   color=colors.get(name, 'gray'),
                   linewidth=2, markersize=8, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax.set_title('BER vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    # ✅ 3. Energy Efficiency
    ax = axes[0, 2]
    for name, res in results_all.items():
        if 'energy_efficiency' in res:
            ax.plot(snr_range, res['energy_efficiency'],
                   marker=markers.get(name, 'x'),
                   color=colors.get(name, 'gray'),
                   linewidth=2, markersize=8, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (bps/Hz/W)', fontsize=12)
    ax.set_title('Energy Efficiency vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ✅ 4. Total Power
    ax = axes[1, 0]
    for name, res in results_all.items():
        if 'total_power' in res:
            ax.plot(snr_range, res['total_power'],
                   marker=markers.get(name, 'x'),
                   color=colors.get(name, 'gray'),
                   linewidth=2, markersize=8, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Total TX Power', fontsize=12)
    ax.set_title('Transmit Power vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ✅ 5. Fairness (Jain Index)
    ax = axes[1, 1]
    for name, res in results_all.items():
        ax.plot(snr_range, res['jain_index'],
               marker=markers.get(name, 'x'),
               color=colors.get(name, 'gray'),
               linewidth=2, markersize=8, label=name)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Jain Fairness Index', fontsize=12)
    ax.set_title('User Fairness vs SNR', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0.5, 1.05])
    
    # ✅ 6. EE Gain over RZF (%)
    ax = axes[1, 2]
    if 'RZF' in results_all:
        rzf_ee = np.array(results_all['RZF'].get('energy_efficiency', [1]*len(snr_range)))
        for name, res in results_all.items():
            if name != 'RZF' and 'energy_efficiency' in res:
                ee_gain = ((np.array(res['energy_efficiency']) / (rzf_ee + 1e-12)) - 1.0) * 100
                ax.plot(snr_range, ee_gain,
                       marker=markers.get(name, 'x'),
                       color=colors.get(name, 'gray'),
                       linewidth=2, markersize=8, label=name)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('EE Gain over RZF (%)', fontsize=12)
    ax.set_title('Energy Efficiency Improvement', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_comparison_with_ee.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/comprehensive_comparison_with_ee.pdf', bbox_inches='tight')
    print(f"✅ Saved comprehensive plots to {save_dir}/")
    plt.close()

def print_comparison_table(results_all, snr_range, save_dir='./results'):
    """Print and save comparison table"""
    
    print("\n" + "="*80)
    print("  📊 PERFORMANCE COMPARISON TABLE")
    print("="*80)
    
    # Save to file
    with open(f'{save_dir}/comparison_table.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        
        for snr in snr_range:
            idx = list(snr_range).index(snr)
            
            line = f"\n📊 SNR = {snr:.0f} dB:\n"
            print(line)
            f.write(line + "\n")
            
            header = f"{'Method':<30} {'Rate':>10} {'BER':>12} {'SINR':>10} {'Jain':>8} {'EE':>12}"
            print(header)
            f.write(header + "\n")
            
            separator = "-" * 90
            print(separator)
            f.write(separator + "\n")
            
            for name, res in results_all.items():
                line = (f"{name:<30} "
                       f"{res['sum_rate'][idx]:>10.2f} "
                       f"{res['ber'][idx]:>12.2e} "
                       f"{res['avg_sinr'][idx]:>10.2f} "
                       f"{res['jain_index'][idx]:>8.3f} "
                       f"{res['energy_efficiency'][idx]:>12.4f}")
                print(line)
                f.write(line + "\n")
    
    print(f"\n✅ Saved comparison table to {save_dir}/comparison_table.txt")


def test_transformer_shapes():
    system = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="transformer_rb", rb_size=12)
    
    # Dummy input
    h_freq = tf.random.normal([32, 4, 1, 1, 8, 14, 72], dtype=tf.complex64)
    
    # Forward pass
    g = system.precoder(h_freq, training=False)
    
    print(f"✅ Precoder output shape: {g.shape}")
    print(f"   Expected: [32, 1, 14, 72, 8, 4]")
    
    # Vérification: [B, num_tx=1, ofdm, fft, num_tx_ant, num_streams]
    assert g.shape == [32, 1, 14, 72, 8, 4], "❌ SHAPE MISMATCH!"
    
    # Vérification puissance
    power_per_ant = tf.reduce_sum(tf.abs(g)**2, axis=-1)  # Sum over streams
    print(f"✅ Power per antenna (should be ~4): {tf.reduce_mean(power_per_ant):.2f}")

# ==============================================================================
# MAIN FUNCTION - UPDATED FOR SAGE-HB AUGMENTATION
# ==============================================================================
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    print("\n" + "="*80)
    print("  🚀 TRANSFORMER PRECODING — V4 vs V5 COMPARISON")
    print("="*80)
    print(f"  Config: {NUM_TX}×{NUM_RX} MIMO | SNR: {TRAINING_SNR} dB | "
          f"Dataset: {DATASET_SIZE:,} samples")
    print("="*80 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    dummy_system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX,
                                   precoder_type="rzf")
    cached_dataset = CachedSionnaDataset(
        dummy_system,
        dataset_size=DATASET_SIZE,
        batch_size=512,
        cache_file=f'/export/tmp/sala/sionna_base_{DATASET_SIZE//1000}k'
                   f'_{NUM_TX}x{NUM_RX}.npz',
        augmentation_multiplier=50,
        seed=SEED)

    # ── Modèles à entraîner ───────────────────────────────────────────────
    # Ajouter/retirer des entrées ici pour changer ce qui est entraîné
    models_to_train = [
        {
            'name'        : 'V5 (per-RB shared)',
            'type'        : 'transformer_v5',
            'system_kwargs': {'num_intra_layers': 2, 'num_inter_layers': 1,
                              'embed_dim': 128, 'num_heads': 4},
        },
        {
            'name'        : 'V4 (1 Tok/RB)',
            'type'        : 'transformer_rb',
            'system_kwargs': {'rb_size': 12, 'tokens_per_rb': 1},
        },
        {
            'name'        : 'V4 (2 Tok/RB)',
            'type'        : 'transformer_rb',
            'system_kwargs': {'rb_size': 12, 'tokens_per_rb': 2},
        },
        {
            'name'        : 'V4 (3 Tok/RB)',
            'type'        : 'transformer_rb',
            'system_kwargs': {'rb_size': 12, 'tokens_per_rb': 3},
        },
    ]

    config_rb    = TRAINING_CONFIG['rb_grouped']
    trained_models = {}

    print("\n" + "="*80)
    print("  🚂 TRAINING")
    print("="*80 + "\n")

    for cfg in models_to_train:
        print("-" * 80)
        print(f"  Training : {cfg['name']}")
        print("-" * 80)

        system = MU_MIMO_System(
            num_tx=NUM_TX, num_rx=NUM_RX,
            precoder_type=cfg['type'],
            **cfg['system_kwargs'])

        trainer = SupervisedTrainer(
            system, cached_dataset,
            snr_db=TRAINING_SNR,
            warmup_epochs=config_rb['warmup_epochs'],
            finetune_epochs=config_rb['finetune_epochs'],
            batch_size=config_rb['batch_size'],
            learning_rate=config_rb['learning_rate'])

        trainer.train(print_every=250, patience=8)
        trained_models[cfg['name']] = (system,
                                        trainer.checkpoint.best_checkpoint_path)

    # ── Évaluation ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  📊 EVALUATION")
    print("="*80)

    results_all = {}

    # Baselines
    for name, ptype in [('RZF', 'rzf'), ('WMMSE', 'wmmse')]:
        print(f"\n  Evaluating {name}...")
        sys_ = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX,
                               precoder_type=ptype)
        results_all[name] = evaluate_system_detailed_with_ee(
            sys_, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, name)

    # Transformers entraînés
    for name, (system, weights_path) in trained_models.items():
        print(f"\n  Evaluating {name}...")
        if weights_path:
            system.load_precoder_weights(weights_path)
        results_all[name] = evaluate_system_detailed_with_ee(
            system, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, name)

    # ── Figures & tableau ─────────────────────────────────────────────────
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    print_comparison_table(results_all, EVALUATION_SNR_RANGE, save_dir)
    plot_comprehensive_comparison_with_ee(
        results_all, EVALUATION_SNR_RANGE, save_dir)

    # ── Résumé Pareto ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  📋 PARETO SUMMARY")
    print("="*80)

    eval_snr_idx = np.argmin(np.abs(EVALUATION_SNR_RANGE - TRAINING_SNR))
    eval_snr     = EVALUATION_SNR_RANGE[eval_snr_idx]
    wmmse_rate   = results_all['WMMSE']['sum_rate'][eval_snr_idx]

    print(f"\n@ {eval_snr:.0f} dB  (WMMSE = {wmmse_rate:.2f} bps/Hz)")
    print(f"{'Architecture':<25} | {'Rate':>10} | {'Gap':>8} | {'Attention O(N²)':>16}")
    print("-" * 70)

    complexity = {
    'RZF'               : 'linear',
    'WMMSE'             : 'O(M³) iter',
    'V4 (1 Tok/RB)'    : f'O({(6*1)**2}) = O(36)',    # ← manquant
    'V4 (2 Tok/RB)'    : f'O({(6*2)**2}) = O(144)',   # ← manquant
    'V4 (3 Tok/RB)'    : f'O({(6*3)**2}) = O(324)',
    'V5 (per-RB shared)': f'O(6×{12**2}+{6**2}) = O(900)',
    }

    for name, res in sorted(results_all.items(),
                             key=lambda x: x[1]['sum_rate'][eval_snr_idx],
                             reverse=True):
        rate = res['sum_rate'][eval_snr_idx]
        gap  = rate - wmmse_rate
        cplx = complexity.get(name, '—')
        print(f"{name:<25} | {rate:>10.2f} | {gap:>+8.2f} | {cplx:>16}")

    print("-" * 70)
# =============================================================================
# TRANSFORMER VALIDATION TESTS
# =============================================================================

def test_transformer_shapes():
    """Test 1: Validate transformer output shapes"""
    system = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="transformer_rb", rb_size=12)
    
    # ✅ FIX: Generate complex random values correctly
    h_real = tf.random.normal([32, 4, 1, 1, 8, 14, 72], dtype=tf.float32)
    h_imag = tf.random.normal([32, 4, 1, 1, 8, 14, 72], dtype=tf.float32)
    h_freq = tf.complex(h_real, h_imag)
    
    g = system.precoder(h_freq, training=False)
    
    print(f"   Output shape: {g.shape}")
    print(f"   Expected:     [32, 1, 14, 72, 8, 4]")
    
    try:
        assert list(g.shape) == [32, 1, 14, 72, 8, 4]
        print(f"   ✅ Shape CORRECT!")
    except AssertionError:
        print(f"   ❌ Shape MISMATCH!")
        return False
    
    power_per_ant = tf.reduce_sum(tf.abs(g)**2, axis=-1)
    avg_power = float(tf.reduce_mean(power_per_ant))
    print(f"   Power/antenna: {avg_power:.2f} (expected ~4.0)")
    
    if 3.5 < avg_power < 4.5:
        print(f"   ✅ Power normalization CORRECT!")
    else:
        print(f"   ⚠️ Power normalization OFF (got {avg_power:.2f})")
    
    return True


def test_transformer_vs_rzf(system_rzf, h_freq_input, no_input):
    """Test 2: Compare untrained transformer with RZF"""
    
    batch_size = tf.shape(h_freq_input)[0]
    
    # Get RZF rate
    _, _, _, _, h_eff_rzf, _, _, _, _, g_rzf = system_rzf(
        batch_size, tf.constant(15.0)
    )
    sinr_rzf = system_rzf.lmmse_sinr(h_eff_rzf, no=no_input, interference_whitening=True)
    rate_per_element = tf.math.log(1.0 + sinr_rzf) / tf.math.log(2.0)
    rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])
    sum_rate_per_sample = tf.reduce_sum(rate_per_user, axis=1)
    rate_rzf = float(tf.reduce_mean(sum_rate_per_sample))
    
    # Transformer (untrained)
    system_tf = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="transformer_rb", rb_size=12)
    g_tf = system_tf.precoder(h_freq_input, training=False)
    h_eff_tf = system_tf.precoded_channel_helper.compute_effective_channel(h_freq_input, g_tf)
    
    sinr_tf = system_tf.lmmse_sinr(h_eff_tf, no=no_input, interference_whitening=True)
    rate_per_element = tf.math.log(1.0 + sinr_tf) / tf.math.log(2.0)
    rate_per_user = tf.reduce_mean(rate_per_element, axis=[1, 2, 4])
    sum_rate_per_sample = tf.reduce_sum(rate_per_user, axis=1)
    rate_tf = float(tf.reduce_mean(sum_rate_per_sample))
    
    print(f"   RZF:         {rate_rzf:6.2f} bps/Hz")
    print(f"   Transformer: {rate_tf:6.2f} bps/Hz (untrained, random init)")
    print(f"   Gap:         {rate_rzf - rate_tf:6.2f} bps/Hz")

    
    
    if rate_tf < 0.5:
        print(f"   ❌ BROKEN! Model output is worse than noise.")
        return False
    elif rate_tf < 5:
        print(f"   ⚠️ Random initialization → WMMSE warmup will fix this ✅")
        return True  # ✅ AUTORISE LE TRAINING!
    elif rate_tf < 15:
        print(f"   ✅ Decent initialization")
        return True
    else:
        print(f"   ✅ Strong initialization")
        return True




def test_upsampling_smoothness():
    """Test 3: Check upsampling smoothness"""
    system = MU_MIMO_System(num_tx=8, num_rx=4, precoder_type="transformer_rb", rb_size=12)
    
    # ✅ FIX: Generate complex random values correctly
    h_real = tf.random.normal([1, 4, 1, 1, 8, 14, 72], dtype=tf.float32)
    h_imag = tf.random.normal([1, 4, 1, 1, 8, 14, 72], dtype=tf.float32)
    h_freq = tf.complex(h_real, h_imag)
    
    g = system.precoder(h_freq, training=False)
    
    g_sample = tf.abs(g[0, 0, 0, :, 0, 0])
    diff = tf.abs(g_sample[1:] - g_sample[:-1])
    avg_diff = float(tf.reduce_mean(diff))
    
    print(f"   Avg adjacent variation: {avg_diff:.4f}")
    
    if avg_diff < 0.1:
        print(f"   ✅ Smooth upsampling")
    elif avg_diff < 0.3:
        print(f"   ⚠️ Moderate smoothness")
    else:
        print(f"   ❌ Rough upsampling")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs('./results', exist_ok=True)
        plt.figure(figsize=(12, 4))
        plt.plot(g_sample.numpy(), marker='o', markersize=3)
        plt.title("Precoder Magnitude (Untrained)")
        plt.xlabel("Subcarrier")
        plt.ylabel("|g|")
        plt.grid(True, alpha=0.3)
        plt.savefig("./results/precoder_smoothness.png", dpi=150, bbox_inches='tight')
        print(f"   📊 Plot saved: ./results/precoder_smoothness.png")
    except:
        pass


if __name__ == "__main__":
    main()