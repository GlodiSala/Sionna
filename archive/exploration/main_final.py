"""
main.py — V4 vs V5 Transformer Precoding: Training + Evaluation + Energy Analysis
===================================================================================
Pipeline complet :
  1. Dataset SAGE-HB (5k samples)
  2. Entraînement V4 (1/2/3 tok/RB) + V5 (per-RB shared)
  3. Évaluation sum-rate sur SNR range [0,5,10,15,20] dB
  4. Calcul d'énergie automatique via FLOPs réels (lab energy model)
  5. Figures publication + tableau Pareto
"""

import os
import json
import pickle
import logging
from datetime import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import sionna
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.channel import (ApplyOFDMChannel, cir_to_ofdm_channel,
                                 subcarrier_frequencies)
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LMMSEEqualizer,
                               LMMSEPostEqualizationSINR,
                               RemoveNulledSubcarriers, RZFPrecoder)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import compute_ber, ebnodb2no

from datasets  import CachedSionnaDataset
from precoders_w import (rzf_precoder, wmmse_precoder,
                          TransformerPrecoderV4, TransformerPrecoderV5)

# =============================================================================
# CONFIGURATION — un seul endroit à modifier
# =============================================================================

SEED         = 42
NUM_TX       = 8
NUM_RX       = 4
BATCH_SIZE   = 256
DATASET_SIZE = 5000

TRAINING_SNR       = 15.0
EVALUATION_SNR_RANGE = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], dtype=np.float32)

# =============================================================================
# ► TRAINING CONFIG — modifier ici
# =============================================================================

SEED          = 42
NUM_TX        = 8
NUM_RX        = 4
BATCH_SIZE    = 256
DATASET_SIZE  = 5000
TRAINING_SNR  = 15.0

EVALUATION_SNR_RANGE = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
                                  dtype=np.float32)

TRAINING_CONFIG = {
    'warmup_epochs'  : 5,
    'finetune_epochs': 10,
    'learning_rate'  : 2e-3,
}

# =============================================================================
# ► MODELS TO TRAIN — ajouter/retirer des entrées ici
# =============================================================================

MODELS_TO_TRAIN = [
    # ── V4.3 SNR-aware — même ablation ──────────────────────────────────
    {
        'name'      : 'V4.3_3tok-RB',
        'type'      : 'transformer_rb',
        'sys_kwargs': {'tokens_per_rb': 3, 'version': 'v4.3',
                       'embed_dim': 128, 'num_heads': 4},
        'batch_size': 256,
    },
    # ── V5 corrigé ───────────────────────────────────────────────────────
    {
        'name'      : 'V5_2L_128d',
        'type'      : 'transformer_v5',
        'sys_kwargs': {'embed_dim': 128, 'num_heads': 4,
                       'num_intra_layers': 2, 'num_inter_layers': 2},
        'batch_size': 128,
    },
    # ── V4.3 SNR-aware — même ablation ──────────────────────────────────

    {
        'name'      : 'V4.3_2tok-RB',
        'type'      : 'transformer_rb',
        'sys_kwargs': {'tokens_per_rb': 2, 'version': 'v4.3',
                       'embed_dim': 128, 'num_heads': 4},
        'batch_size': 256,
    },
    {
        'name'      : 'V4.3_6tok-RB',
        'type'      : 'transformer_rb',
        'sys_kwargs': {'tokens_per_rb': 6, 'version': 'v4.3',
                       'embed_dim': 128, 'num_heads': 4},
        'batch_size': 156,
    },
    
]

# Bit-width pour le modèle d'énergie (lab Energy.py)
Q_W = 16   # poids FP16
Q_A = 16   # activations FP16

# =============================================================================
# LAB ENERGY MODEL — identique à Energy.py du labo
# =============================================================================

def energy_constants(Q):
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    EM   = 2.0 * EMAC
    EL   = EMAC
    return EMAC, EM, EL


def compute_energy_uJ(FLOPs, Weights, Activations, Q_W=32, Q_A=32):
    """
    Énergie totale en µJ via le modèle lab.
    MACs = FLOPs / 2  (1 real MAC = multiply + accumulate = 2 FLOPs)
    """
    import math
    MACs = FLOPs / 2.0

    EMAC, EM,   EL   = energy_constants(Q_W)
    _,    EM_A, EL_A = energy_constants(Q_A)

    sqrt_p_W = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A = math.sqrt(64.0 * (Q_A / 16.0))

    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)

    return (EC + EW + EA) / 1e9   # fJ → µJ


# =============================================================================
# COMPTAGE DE FLOPS — automatique, s'adapte à V4 et V5
# =============================================================================

# =============================================================================
# COMPLEXITÉ — classiques seulement (les neuronaux se décrivent eux-mêmes)
# =============================================================================

def compute_classical_complexity(method, M, K, N_SC, N_OFDM, I_wmmse=10):
    """FLOPs / poids / activations pour ZF, RZF, WMMSE."""
    def zf_per_sc():
        return 7.0 * (2.0/3.0 * K**3 + 2.0 * K**2 * M)

    def wmmse_per_sc():
        per_iter = (
              (14.0/3.0) * K * M**3
            + 12.0 * K**2 * M**2
            + 12.0 * K**2 * M
            +  9.0 * K   * M**2
            +  8.0 * K   * M
            +  5.0 * K**2
            + (68.0/3.0) * K
        )
        return 7.0 * I_wmmse * per_iter

    acts = 2.0 * (K*M + 2.0*K**2 + M*K)
    f_sc = wmmse_per_sc() if 'WMMSE' in method else zf_per_sc()
    if method == 'RZF':
        f_sc += K
    return f_sc * N_SC * N_OFDM, 0.0, acts * N_SC * N_OFDM


def get_complexity(name, system):
    """
    Interface unifiée.
    - Précoder neural : délègue à precoder.complexity(num_ofdm).
    - Classiques      : calcul analytique.
    """
    precoder = system.precoder
    N_OFDM   = system.rg.num_ofdm_symbols
    M        = system.num_bs_antennas
    K        = system.num_users
    N_SC     = system.rg.fft_size

    if precoder is not None and hasattr(precoder, 'complexity'):
        return precoder.complexity(N_OFDM)

    name_up = name.upper()
    if 'WMMSE' in name_up:
        return compute_classical_complexity('WMMSE', M, K, N_SC, N_OFDM)
    if 'RZF'   in name_up:
        return compute_classical_complexity('RZF',   M, K, N_SC, N_OFDM)
    if 'ZF'    in name_up:
        return compute_classical_complexity('ZF',    M, K, N_SC, N_OFDM)

    raise ValueError(
        f"Impossible de calculer la complexité pour '{name}'. "
        f"Ajoute une méthode .complexity() au précoder.")


# =============================================================================
# MU-MIMO SYSTEM
# =============================================================================

class MU_MIMO_System(tf.keras.Model):

    def __init__(self, num_tx=8, num_rx=4, precoder_type="rzf",
                 rb_size=12, batch_size=32, weights_path=None,
                 tokens_per_rb=1,
                 num_intra_layers=2, num_inter_layers=2,
                 embed_dim=128, num_heads=4, version='v4.0'):
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
        self.version = version
        
        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization="single",
            polarization_type="V", antenna_pattern="omni",
            carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx / 2), polarization="dual",
            polarization_type="cross", antenna_pattern="38.901",
            carrier_frequency=2.6e9)

        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        self.binary_source = BinarySource()
        self.encoder       = LDPC5GEncoder(
            int(self.rg.num_data_symbols),
            int(self.rg.num_data_symbols * 2))
        self.mapper        = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper     = ResourceGridMapper(self.rg)
        self.frequencies   = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.channel_freq  = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ     = LMMSEEqualizer(self.rg, self.sm)
        self.demapper      = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder       = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr    = LMMSEPostEqualizationSINR(
            resource_grid=self.rg, stream_management=self.sm)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        from sionna.phy.ofdm import RZFPrecodedChannel
        self.precoded_channel_helper = RZFPrecodedChannel(
            resource_grid=self.rg, stream_management=self.sm)

        self._initialize_precoder(precoder_type, rb_size, weights_path)
    
    def _initialize_precoder(self, precoder_type, rb_size, weights_path):
        if precoder_type == "transformer_rb":
            print(f"✅ Using TransformerPrecoderV4 ({self.tokens_per_rb} Tok/RB, {self.version})")
            self.precoder = TransformerPrecoderV4(
                num_tx=self.num_bs_antennas, num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols, fft_size=self.rg.fft_size,
                rb_size=rb_size, tokens_per_rb=self.tokens_per_rb,
                embed_dim=self.embed_dim, num_heads=self.num_heads, num_layers=4,
                version=self.version) 
            if weights_path:
                self.load_precoder_weights(weights_path)

        elif precoder_type == "transformer_v5":
            print(f"✅ Using TransformerPrecoderV5 (per-RB shared)")
            self.precoder = TransformerPrecoderV5(
                num_tx=self.num_bs_antennas, num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols, fft_size=self.rg.fft_size,
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                num_intra_layers=self.num_intra_layers,
                num_inter_layers=self.num_inter_layers,
                snr_aware=True)
            if weights_path:
                self.load_precoder_weights(weights_path)
        elif precoder_type in ("rzf", "wmmse"):
            print(f"✅ Using {precoder_type.upper()} Precoder")
            self.precoder = None

        else:
            raise ValueError(f"Unknown precoder type: {precoder_type}")

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, self.num_users, "umi")
        self.channel_model.set_topology(*topology)

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
            g = rzf_precoder(h_freq, stream_management=self.sm, no=no)
        elif self.precoder_type == "wmmse":
            g = wmmse_precoder(h_freq, no, num_iterations=10,
                               stream_management=self.sm)
        else:
            g = self.precoder(h_freq, training=training)

        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                         perm=[0, 3, 1, 2]), axis=1)

        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y     = self.channel_freq(x_precoded, h_freq, no)
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

        if self.precoder_type in ("transformer_rb", "transformer_v5"):
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
            g = rzf_precoder(h_freq, stream_management=self.sm, no=no)
        else:
            raise ValueError(f"Unknown precoder: {self.precoder_type}")

        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                         perm=[0, 3, 1, 2]), axis=1)

        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y     = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr   = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_precoded, g
    def load_precoder_weights(self, weights_path):
        if os.path.isdir(weights_path):
            weights_path = os.path.join(weights_path, 'weights.pkl')
        if not os.path.exists(weights_path):
            print(f"❌ Weights not found: {weights_path}")
            return False
        with open(weights_path, 'rb') as f:
            trained_weights = pickle.load(f)
        for var, w in zip(self.precoder.trainable_variables, trained_weights):
            var.assign(w)
        print(f"✅ Loaded {len(trained_weights)} weights from {weights_path}")
        return True


# =============================================================================
# CHECKPOINT — identifiable par run_name
# =============================================================================

class SimpleCheckpoint:
    def __init__(self, run_name: str, base_dir: str = './weights'):
        """
        Sauvegarde dans : {base_dir}/{run_name}/best_{timestamp}/
        Chaque config a son propre sous-dossier, timestamp identifie la run.
        """
        self.run_dir        = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_ckpt_path = None

    @staticmethod
    def _json_safe(obj):
        """Convertit les types numpy non-sérialisables en types Python natifs."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def save_best(self, precoder, config_dict: dict, metrics: dict) -> str:
        import shutil

        # Supprimer l'ancien best pour ne garder que le meilleur
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            shutil.rmtree(self.best_ckpt_path)

        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_dir = os.path.join(self.run_dir, f'best_{ts}')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Poids
        with open(os.path.join(ckpt_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump([v.numpy() for v in precoder.trainable_variables], f)

        # Config complète — robuste contre tout type numpy
        with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
            json.dump(
                {**config_dict,
                 'run_name' : os.path.basename(self.run_dir),
                 'saved_at' : datetime.now().isoformat()},
                f, indent=2, default=self._json_safe)

        # Métriques
        with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as f:
            json.dump(
                {k: float(v) for k, v in metrics.items()},
                f, indent=2, default=self._json_safe)

        self.best_ckpt_path = ckpt_dir
        print(f"  💾 Saved → {ckpt_dir}")
        return ckpt_dir

# =============================================================================
# SUPERVISED TRAINER
# =============================================================================

class SupervisedTrainer:

    def __init__(self, system, cached_dataset,
                 run_name: str = 'unnamed',
                 snr_db=15.0, warmup_epochs=5, finetune_epochs=10,
                 learning_rate=2e-3, batch_size=256, num_iters=None):

        self.system          = system
        self.cached_dataset  = cached_dataset
        self.snr_db          = snr_db
        self.warmup_epochs   = int(warmup_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.total_epochs    = self.warmup_epochs + self.finetune_epochs
        self.rate_norm       = float(system.num_users) * 9.0
        self.batch_size      = batch_size

        if num_iters is None:
            if hasattr(cached_dataset, 'effective_dataset_size'):
                self.num_iters = cached_dataset.effective_dataset_size // batch_size
            else:
                self.num_iters = len(cached_dataset.h_freq_all) // batch_size
        else:
            self.num_iters = int(num_iters)

        warmup_steps   = self.warmup_epochs   * self.num_iters
        finetune_steps = self.finetune_epochs * self.num_iters
        self.warmup_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(warmup_steps, 1), alpha=0.1)
        self.finetune_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate * 0.1,
            decay_steps=max(finetune_steps, 1), alpha=0.01)

        self.optimizer = tf.keras.optimizers.Adam(self.warmup_lr, clipnorm=5.0)

        # Build modèle
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

        self.run_config = {
            'run_name'               : run_name,
            'precoder_type'          : system.precoder_type,
            'num_tx'                 : system.num_bs_antennas,
            'num_rx'                 : system.num_users,
            'fft_size'               : system.rg.fft_size,
            'num_ofdm'               : system.rg.num_ofdm_symbols,
            'embed_dim'              : getattr(system, 'embed_dim', None),
            'num_heads'              : getattr(system, 'num_heads', None),
            'num_intra_layers'       : getattr(system, 'num_intra_layers', None),
            'num_inter_layers'       : getattr(system, 'num_inter_layers', None),
            'tokens_per_rb'          : getattr(system, 'tokens_per_rb', None),
            'total_params'           : int(total_params),   # ← le coupable principal
            'version'  : getattr(system, 'version', 'v4.0'),   # ← NOUVEAU
            'snr_db_train'           : snr_db,
            'warmup_epochs'          : warmup_epochs,
            'finetune_epochs'        : finetune_epochs,
            'total_epochs'           : warmup_epochs + finetune_epochs,
            'batch_size'             : batch_size,
            'learning_rate_warmup'   : learning_rate,
            'learning_rate_finetune' : learning_rate * 0.5,
            'grad_clip_norm'         : 5.0,
            'rate_norm'              : self.rate_norm,
            'num_iters_per_epoch'    : self.num_iters,
            'dataset_size'           : getattr(cached_dataset, 'effective_dataset_size',
                                        len(getattr(cached_dataset, 'h_freq_all', []))),
        }

        self.checkpoint    = SimpleCheckpoint(run_name=run_name)
        self.best_sum_rate = -np.inf
        self.best_epoch    = 0
        self.history       = []

        print(f"\n{'='*60}")
        print(f"  [{run_name}]")
        print(f"  Params: {total_params:,} | Iters/ep: {self.num_iters}")
        print(f"  Warmup: {warmup_epochs}ep LR={learning_rate:.1e} → "
              f"Finetune: {finetune_epochs}ep LR={learning_rate*0.5:.1e}")
        print(f"{'='*60}\n")
    def _call_precoder(self, h_freq, no, training=False):
        precoder = self.system.precoder
        is_snr_aware = (
            getattr(precoder, 'version', '') == 'v4.3' or
            isinstance(precoder, TransformerPrecoderV5)
        )
        if is_snr_aware:
            if training:
                g_re, g_im = precoder(h_freq, no=no, training=True,
                                    return_real_imag=True)
                return tf.complex(g_re, g_im)
            return precoder(h_freq, no=no, training=False)
        else:
            if training:
                g_re, g_im = precoder(h_freq, training=True,
                                    return_real_imag=True)
                return tf.complex(g_re, g_im)
            return precoder(h_freq, training=False)

    @tf.function
    def warmup_step(self, h_freq_batch, snr_db):
        snr_db = tf.random.uniform(shape=[], minval=10.0, maxval=25.0)
        no     = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                        0.5, self.system.rg)
        g_teacher = tf.stop_gradient(
            rzf_precoder(h_freq_batch,
                        stream_management=self.system.sm, no=no))

        with tf.GradientTape() as tape:
            # ── passe no si le précoder le supporte ──────────────────────
            g_pred = self._call_precoder(h_freq_batch, no, training=True)

            mse_loss = 2.0 * tf.reduce_mean(tf.abs(g_pred - g_teacher)**2)

            h_eff = self.system.precoded_channel_helper \
                        .compute_effective_channel(h_freq_batch, g_pred)
            sinr  = self.system.lmmse_sinr(
                h_eff, no=no, interference_whitening=True)
            rate  = tf.reduce_sum(tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
                / tf.math.log(2.0), axis=[0, 1, 2, 4]))

            T        = tf.constant(
                float(self.warmup_epochs * self.num_iters), dtype=tf.float32)
            step     = tf.cast(self.optimizer.iterations, tf.float32)
            progress = tf.minimum(step / T, 1.0)
            mse_w    = tf.maximum(1.0 - progress, 0.05)
            loss     = mse_w * mse_loss - progress * (rate / self.rate_norm)

        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return loss, tf.linalg.global_norm(grads), mse_w, progress

    @tf.function
    def finetune_step(self, h_freq_batch, snr_db):
        with tf.GradientTape() as tape:
            snr_db = tf.random.uniform(shape=[], minval=10.0, maxval=25.0)
            no     = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                            0.5, self.system.rg)
            g     = self._call_precoder(h_freq_batch, no, training=True)
            h_eff = self.system.precoded_channel_helper \
                        .compute_effective_channel(h_freq_batch, g)
            sinr  = self.system.lmmse_sinr(
                h_eff, no=no, interference_whitening=True)
            rate  = tf.reduce_sum(tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
                / tf.math.log(2.0), axis=[0, 1, 2, 4]))
            loss  = -tf.where(tf.math.is_finite(rate),
                            rate / self.rate_norm, tf.constant(0.0))

        grads = tape.gradient(loss, self.trainable_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return loss, tf.linalg.global_norm(grads)

    @tf.function
    def eval_step_light(self, h_freq_batch, snr_db):
        no    = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                        0.5, self.system.rg)
        g     = self._call_precoder(h_freq_batch, no, training=False)
        h_eff = self.system.precoded_channel_helper \
                    .compute_effective_channel(h_freq_batch, g)
        sinr  = self.system.lmmse_sinr(
            h_eff, no=no, interference_whitening=True)
        rates = tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
            / tf.math.log(2.0), axis=[0, 1, 2, 4])
        return tf.reduce_sum(rates), rates

    # ─────────────────────────────────────────────────────────────────
    # TRAIN LOOP
    def train(self, print_every=200, patience=8):
        import time

        # ── Timing diagnostic ────────────────────────────────────────────────
        print(f"⏱  Timing diagnostic...")
        h_test = self.cached_dataset.get_batch(self.batch_size)
        no_test = ebnodb2no(tf.constant(15.0), self.system.num_bits_per_symbol,
                            0.5, self.system.rg)
        _ = self.warmup_step(h_test, tf.constant(15.0))  # chauffe JIT

        t0 = time.time()
        for _ in range(5):
            h = self.cached_dataset.get_batch(self.batch_size)
            _ = self.warmup_step(h, tf.constant(15.0))
        secs = (time.time() - t0) / 5

        print(f"⏱  {secs:.2f}s/iter | "
            f"Epoch: {secs * self.num_iters / 60:.1f} min | "
            f"Total: {secs * self.num_iters * self.total_epochs / 3600:.1f}h\n")

        print(f"🚀 Training @ {self.snr_db} dB  |  "
            f"warmup={self.warmup_epochs}ep  finetune={self.finetune_epochs}ep  "
            f"patience={patience}\n")

        no_improve = 0

        for epoch in range(self.total_epochs):
            is_warmup = epoch < self.warmup_epochs
            mode_str  = "RZF-WU" if is_warmup else "SUM-RATE"

            # ── Switch warmup → finetune ─────────────────────────────────────
            if epoch == self.warmup_epochs:
                # Muter le LR en place — préserve le momentum Adam
                self.optimizer.learning_rate = self.finetune_lr
                no_improve = 0
                print(f"\n🔄 Fine-tune  LR={self.finetune_lr.initial_learning_rate:.1e}"
                    f"  (cosine decay, momentum préservé)\n")

            losses, grads_l, eval_rates, rates_list = [], [], [], []
            t_epoch = time.time()

            for i in range(self.num_iters):
                h      = self.cached_dataset.get_batch(self.batch_size)
                snr_t  = tf.constant(self.snr_db, tf.float32)

                if is_warmup:
                    loss, gnorm, mse_w, progress = self.warmup_step(h, snr_t)
                else:
                    loss, gnorm = self.finetune_step(h, snr_t)

                losses.append(float(loss))
                grads_l.append(float(gnorm))

                # ── Print intermédiaire ──────────────────────────────────────
                if (i + 1) % print_every == 0:
                    snr_eval = tf.constant(self.snr_db, tf.float32)
                    rate_eval, per_user = self.eval_step_light(h, snr_eval)
                    eval_rates.append(float(rate_eval))
                    rates_list.append(per_user.numpy())

                    g_m = float(np.mean(grads_l[-print_every:]))
                    l_m = float(np.mean(losses[-print_every:]))
                    rs  = per_user.numpy()

                    elapsed = time.time() - t_epoch
                    eta_min = elapsed / (i + 1) * (self.num_iters - i - 1) / 60

                    if is_warmup:
                        print(f"    ... {i+1}/{self.num_iters} | "
                            f"Sum: {float(rate_eval):.2f} | "
                            f"Usr: [{' '.join(f'{v:.1f}' for v in rs)}] | "
                            f"MSE→R: {float(mse_w):.2f}→{float(progress):.2f} | "
                            f"Grad: {g_m:.2e} | Loss: {l_m:.3f} | "
                            f"ETA: {eta_min:.1f}min")
                    else:
                        print(f"    ... {i+1}/{self.num_iters} | "
                            f"Sum: {float(rate_eval):.2f} | "
                            f"Usr: [{' '.join(f'{v:.1f}' for v in rs)}] | "
                            f"Grad: {g_m:.2e} | Loss: {l_m:.3f} | "
                            f"ETA: {eta_min:.1f}min")

            # ── Résumé epoch ─────────────────────────────────────────────────
            epoch_time = (time.time() - t_epoch) / 60
            avg_rate   = float(np.mean(eval_rates)) if eval_rates else 0.0
            avg_rates  = np.mean(rates_list, axis=0) if rates_list \
                        else np.zeros(self.system.num_users)
            avg_loss   = float(np.mean(losses))
            avg_grad   = float(np.mean(grads_l))
            rs_str     = "[" + ", ".join(f"{v:.2f}" for v in avg_rates) + "]"

            # ── Checkpoint si meilleur ────────────────────────────────────────
            star = ""
            if avg_rate > self.best_sum_rate:
                self.best_sum_rate = avg_rate
                self.best_epoch    = epoch + 1
                self.checkpoint.save_best(
                    self.system.precoder,
                    {**self.run_config, 'best_epoch': epoch + 1},
                    {'sum_rate': avg_rate})
                star = " ⭐"
                no_improve = 0
            else:
                if not is_warmup:
                    no_improve += 1

            epochs_left = self.total_epochs - (epoch + 1)
            eta_total   = epoch_time * epochs_left

            print(f"\n{epoch+1:3d} | {mode_str:>8} | "
                f"{avg_rate:>10.2f} | {rs_str} | "
                f"{avg_grad:.2e} | {avg_loss:.3f} | "
                f"{epoch_time:.1f}min | ETA:{eta_total:.0f}min"
                f"{star}")

            self.history.append({
                'epoch'      : epoch + 1,
                'mode'       : mode_str,
                'sum_rate'   : avg_rate,
                'per_user'   : avg_rates.tolist(),
                'loss'       : avg_loss,
                'grad_norm'  : avg_grad,
                'epoch_time' : epoch_time,
            })

            # ── Early stopping — finetune seulement ──────────────────────────
            if not is_warmup and no_improve >= patience:
                print(f"\n⏹  Early stopping — {patience} epochs sans amélioration")
                print(f"   Best: {self.best_sum_rate:.2f} @ epoch {self.best_epoch}")
                break

        print(f"\n✅ Best: {self.best_sum_rate:.2f} bps/Hz @ epoch {self.best_epoch}")
        print(f"   → {self.checkpoint.best_ckpt_path}")
        return self.history
# =============================================================================
# ÉVALUATION — Monte Carlo correct + énergie FP16/INT8
# =============================================================================

def evaluate_system(system, snr_range, num_batches=50,
                    batch_size=256, name="System"):
    """
    Monte Carlo correct :
    - BER  : accumulé sur tous les bits (pas moyenne de moyennes)
    - Rate : moyenne sur les batches
    - Énergie : FP32 pour classiques, FP16 pour neuronaux
    """
    print(f"\n📊 Evaluating {name}...")

    flops, weights, acts = get_complexity(name, system)
    is_classical = system.precoder is None

    energy_fp32 = compute_energy_uJ(flops, weights, acts, Q_W=32, Q_A=32)
    energy_fp16 = compute_energy_uJ(flops, weights, acts, Q_W=16, Q_A=16)
    energy_int8 = compute_energy_uJ(flops, weights, acts, Q_W=8,  Q_A=8)
    energy_plot = energy_fp32 if is_classical else energy_fp16

    results = {
        'sum_rate'    : [],
        'ber'         : [],
        'per_user'    : [],
        'flops_M'     : flops   / 1e6,
        'params_K'    : weights / 1e3,
        'energy_fp32' : energy_fp32,
        'energy_fp16' : energy_fp16,
        'energy_int8' : energy_int8,
        'energy_plot' : energy_plot,
        'is_classical': is_classical,
    }

    for snr in snr_range:
        rates_b     = []
        per_user_b  = []
        total_bits  = 0
        total_errors = 0

        for _ in range(num_batches):
            system.new_topology(batch_size)
            b, b_hat, _, _, h_eff, no, _, _, _, _ = system(
                tf.constant(batch_size, dtype=tf.int32),
                tf.constant(float(snr),  dtype=tf.float32),
                training=False)

            # BER Monte Carlo — accumulation correcte
            total_errors += int(tf.reduce_sum(
                tf.cast(b != b_hat, tf.int32)).numpy())
            total_bits   += int(b.numpy().size)

            sinr     = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate_sc  = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            per_user = tf.reduce_mean(rate_sc, axis=[1, 2, 4]).numpy()
            rates_b.append(float(tf.reduce_sum(
                tf.reduce_mean(tf.constant(per_user), axis=0))))
            per_user_b.append(per_user.mean(axis=0))

        ber_mc = total_errors / max(total_bits, 1)
        results['sum_rate'].append(float(np.mean(rates_b)))
        results['ber'].append(float(ber_mc))
        results['per_user'].append(np.mean(per_user_b, axis=0))

        prec = "FP32" if is_classical else "FP16"
        print(f"  SNR={snr:3.0f} dB | "
              f"Rate: {results['sum_rate'][-1]:6.2f} | "
              f"BER: {ber_mc:.2e} ({total_errors}/{total_bits}) | "
              f"Energy({prec}): {energy_plot:.4f} µJ")

    return results


# =============================================================================
# STYLES — partagés entre les deux figures
# =============================================================================

def _build_styles(names):
    """
    Génère couleurs, markers et labels lisibles automatiquement.
    Gère RZF, WMMSE, V4.x Ntok, V5.
    """
    BASE_COLORS = {
        'RZF'   : '#1f77b4',
        'WMMSE' : '#000000',
    }
    TOK_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    TOK_MARKERS = ['v', '^', 'D', 'p', '*', 'h']

    colors  = {}
    markers = {}
    labels  = {}

    tok_idx = 0
    for name in names:
        if name == 'RZF':
            colors[name]  = BASE_COLORS['RZF']
            markers[name] = 'o'
            labels[name]  = 'RZF (per-SC, FP32)'
        elif name == 'WMMSE':
            colors[name]  = BASE_COLORS['WMMSE']
            markers[name] = 's'
            labels[name]  = 'WMMSE (per-SC, FP32)'
        elif 'tok' in name.lower() or 'tok' in name:
            # V4.x_Ntok-RB → extraire N
            import re
            m = re.search(r'(\d+)tok', name)
            n = m.group(1) if m else '?'
            sc = 12 // int(n) if n.isdigit() else '?'
            colors[name]  = TOK_COLORS[tok_idx % len(TOK_COLORS)]
            markers[name] = TOK_MARKERS[tok_idx % len(TOK_MARKERS)]
            labels[name]  = f'{n} tok/RB  ({sc} SC/tok)'
            tok_idx += 1
        else:
            colors[name]  = TOK_COLORS[tok_idx % len(TOK_COLORS)]
            markers[name] = TOK_MARKERS[tok_idx % len(TOK_MARKERS)]
            labels[name]  = name
            tok_idx += 1

    return colors, markers, labels


# =============================================================================
# FIGURES — deux figures distinctes pour François
# =============================================================================

def plot_all(results_all, snr_range, save_dir='./results'):
    """
    Figure 1 (fig1_main) — Transformers + WMMSE + RZF per-SC
        (a) Sum Rate     (b) BER Monte Carlo
        (c) Pareto énergie    (d) Tableau récap

    Figure 2 (fig2_rzf_context) — Contexte complet avec RZF groupé
        (a) Sum Rate tout ensemble
        (b) Pareto FLOPs (log) avec RZF
    """
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    idx_15 = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))

    all_names   = list(results_all.keys())
    colors, markers, labels = _build_styles(all_names)

    # Séparer les modèles
    neural_keys   = [k for k in all_names
                     if k not in ('RZF', 'WMMSE')]
    main_keys     = ['WMMSE', 'RZF'] + neural_keys

    def _lw(n):  return 2.2 if n in ('RZF', 'WMMSE') else 1.8
    def _ls(n):  return '--' if n == 'RZF' else '-.' if n == 'WMMSE' else '-'

    wmmse_rate = results_all.get('WMMSE', {}).get(
        'sum_rate', [0]*10)[idx_15]
    wmmse_e_fp32 = results_all.get('WMMSE', {}).get('energy_fp32', 3.48)

    # ================================================================
    # FIGURE 1 — résultats principaux
    # ================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))
    fig1.suptitle(
        f'V4.2 Transformer Precoder — {NUM_TX}×{NUM_RX} MU-MIMO  '
        f'(8 TX, 4 UE, 72 SC, SNR train = {TRAINING_SNR:.0f} dB)\n'
        f'Ablation study: tokens per Resource Block (RB = 12 SC)',
        fontsize=11, fontweight='bold')

    # (a) Sum Rate
    ax = axes1[0, 0]
    for name in main_keys:
        if name not in results_all:
            continue
        res = results_all[name]
        ax.plot(snr_range, res['sum_rate'],
                marker=markers[name], color=colors[name],
                label=labels[name],
                linewidth=_lw(name), linestyle=_ls(name), markersize=7)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('(a) Spectral Efficiency')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])

    # (b) BER — Monte Carlo
    ax = axes1[0, 1]
    for name in main_keys:
        if name not in results_all:
            continue
        res = results_all[name]
        ber = np.maximum(res['ber'], 5e-6)
        ax.semilogy(snr_range, ber,
                    marker=markers[name], color=colors[name],
                    label=labels[name],
                    linewidth=_lw(name), linestyle=_ls(name), markersize=7)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('BER')
    ax.set_title('(b) Bit Error Rate (Monte Carlo accumulation)')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])

    # (c) Pareto énergie — FP16 neuronaux, FP32 classiques
    # RZF exclu (trop à gauche, distord l'échelle)
    ax = axes1[1, 0]
    pareto_keys = [k for k in main_keys if k != 'RZF']
    for name in pareto_keys:
        if name not in results_all:
            continue
        res  = results_all[name]
        rate = res['sum_rate'][idx_15]
        e    = res['energy_plot']
        ax.scatter(e, rate, s=220, marker=markers[name],
                   color=colors[name], label=labels[name],
                   zorder=4, edgecolors='k', linewidth=0.8)
        ax.annotate(labels[name].split('(')[0].strip(),
                    (e, rate), textcoords='offset points',
                    xytext=(5, 4), fontsize=7)
    ax.set_xlabel('Inference Energy (µJ)\n'
                  '[Transformers → FP16 | WMMSE → FP32]')
    ax.set_ylabel(f'Sum Rate @ {TRAINING_SNR:.0f} dB (bps/Hz)')
    ax.set_title('(c) Energy–Performance Pareto\n(RZF excluded — see Fig. 2)')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)

    # (d) Tableau récapitulatif
    ax = axes1[1, 1]
    ax.axis('off')

    col_headers = ['Method', 'Rate\n(bps/Hz)', 'Gap vs\nWMMSE',
                   'FLOPs\n(M)', 'Energy\nFP16 (µJ)', 'Gain vs\nWMMSE\n(FP32)']
    rows_data = []
    for name in main_keys:
        if name not in results_all:
            continue
        res  = results_all[name]
        rate = res['sum_rate'][idx_15]
        gap  = rate - wmmse_rate
        e16  = res['energy_fp16']
        gain = wmmse_e_fp32 / e16 if (e16 > 0 and not res['is_classical']) \
               else None
        rows_data.append([
            labels[name],
            f"{rate:.2f}",
            f"{gap:+.2f}",
            f"{res['flops_M']:.0f}",
            f"{e16:.4f}" if not res['is_classical'] else "N/A",
            f"{gain:.1f}×" if gain else "ref",
        ])

    rows_data.sort(key=lambda x: float(x[1]), reverse=True)

    tbl = ax.table(
        cellText=rows_data, colLabels=col_headers,
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(rows_data, start=1):
        bg = '#ecf0f1' if ('WMMSE' in row[0] or 'RZF' in row[0]) else 'white'
        for j in range(len(col_headers)):
            tbl[(i, j)].set_facecolor(bg)

    ax.set_title(
        f'(d) Pareto Summary @ {TRAINING_SNR:.0f} dB\n'
        f'Energy gain = WMMSE_FP32 / model_FP16',
        fontweight='bold', fontsize=9)

    plt.tight_layout()
    p1 = f'{save_dir}/fig1_main_{ts}.png'
    fig1.savefig(p1, dpi=300, bbox_inches='tight')
    fig1.savefig(p1.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Figure 1 saved: {p1}")
    plt.close(fig1)

    # ================================================================
    # FIGURE 2 — contexte RZF groupé
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f'Transformer vs RZF with RB Grouping — {NUM_TX}×{NUM_RX} MU-MIMO',
        fontsize=11, fontweight='bold')

    # Ajouter des styles pour les RZF groupés s'ils sont présents
    rzf_grouped = [k for k in all_names if 'RZF' in k and k != 'RZF']
    rzf_colors  = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(rzf_grouped), 1)))

    def _col2(n):
        if n in colors:
            return colors[n]
        for i, rk in enumerate(rzf_grouped):
            if rk == n:
                return rzf_colors[i]
        return 'gray'

    def _lbl2(n):
        if n in labels:
            return labels[n]
        # RZF RB=X → extraire X pour le label
        import re
        m = re.search(r'RB=(\d+)', n)
        if m:
            rb = int(m.group(1))
            n_tok = 72 // rb
            return f'RZF — {n_tok} groups ({rb} SC/group)'
        return n

    full_keys_fig2 = ['WMMSE', 'RZF'] + rzf_grouped + neural_keys

    # (a) Sum Rate — tout ensemble
    ax = axes2[0]
    for name in full_keys_fig2:
        if name not in results_all:
            continue
        res = results_all[name]
        lw_ = 2.2 if name in ('RZF', 'WMMSE') else \
              1.5 if name in rzf_grouped else 1.8
        ls_ = '--' if name == 'RZF' else \
              ':' if name in rzf_grouped else \
              '-.' if name == 'WMMSE' else '-'
        ax.plot(snr_range, res['sum_rate'],
                marker=markers.get(name, 'x'),
                color=_col2(name),
                label=_lbl2(name),
                linewidth=lw_, linestyle=ls_, markersize=6)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('(a) Full Comparison — Transformer vs RZF Grouping')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9,
              ncol=2 if len(full_keys_fig2) > 6 else 1)
    ax.grid(alpha=0.3)

    # (b) Pareto FLOPs — avec RZF (log scale)
    ax = axes2[1]
    for name in full_keys_fig2:
        if name not in results_all:
            continue
        res  = results_all[name]
        rate = res['sum_rate'][idx_15]
        f_m  = res['flops_M']
        ax.scatter(f_m, rate,
                   s=200, marker=markers.get(name, 'x'),
                   color=_col2(name), label=_lbl2(name),
                   zorder=4, edgecolors='k', linewidth=0.7)
        ax.annotate(_lbl2(name).split('(')[0].strip(),
                    (f_m, rate), textcoords='offset points',
                    xytext=(5, 3), fontsize=6.5)
    ax.set_xscale('log')
    ax.set_xlabel('FLOPs (M, log scale)')
    ax.set_ylabel(f'Sum Rate @ {TRAINING_SNR:.0f} dB (bps/Hz)')
    ax.set_title('(b) Complexity–Performance Pareto\n(FP32 for all)')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p2 = f'{save_dir}/fig2_rzf_context_{ts}.png'
    fig2.savefig(p2, dpi=300, bbox_inches='tight')
    fig2.savefig(p2.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Figure 2 saved: {p2}")
    plt.close(fig2)


# =============================================================================
# TABLEAU RÉSUMÉ CONSOLE
# =============================================================================

def print_summary_table(results_all, snr_range):
    idx = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))
    wmmse_rate   = results_all['WMMSE']['sum_rate'][idx]
    wmmse_flops  = results_all['WMMSE']['flops_M']
    wmmse_e_fp32 = results_all['WMMSE']['energy_fp32']

    print(f"\n{'='*100}")
    print(f"  PARETO SUMMARY @ {TRAINING_SNR:.0f} dB  "
          f"(WMMSE = {wmmse_rate:.2f} bps/Hz | {wmmse_e_fp32:.4f} µJ FP32)")
    print(f"{'='*100}")
    print(f"{'Architecture':<24} | {'Rate':>7} | {'Gap':>6} | "
          f"{'FLOPs(M)':>9} | {'FP32(µJ)':>9} | "
          f"{'FP16(µJ)':>9} | {'INT8(µJ)':>9} | {'Gain FP16':>10}")
    print("─" * 100)

    for name, res in sorted(results_all.items(),
                             key=lambda x: x[1]['sum_rate'][idx], reverse=True):
        rate  = res['sum_rate'][idx]
        gap   = rate - wmmse_rate
        gain  = wmmse_e_fp32 / res['energy_fp16'] \
                if (res['energy_fp16'] > 0 and not res['is_classical']) else None
        fp16s = f"{res['energy_fp16']:.4f}" if not res['is_classical'] else "  N/A  "
        i8s   = f"{res['energy_int8']:.4f}"  if not res['is_classical'] else "  N/A  "
        gs    = f"{gain:.1f}×" if gain else "  ref  "
        print(f"{name:<24} | {rate:>7.2f} | {gap:>+6.2f} | "
              f"{res['flops_M']:>9.1f} | {res['energy_fp32']:>9.4f} | "
              f"{fp16s:>9} | {i8s:>9} | {gs:>10}")

    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✅ GPU: {gpus[0]}')

    print('\n' + '='*70)
    print(f'  TRANSFORMER PRECODING — V4.2 ablation + V5.2')
    print(f'  {NUM_TX}×{NUM_RX} MIMO | SNR train: {TRAINING_SNR} dB | '
          f'Dataset: {DATASET_SIZE:,}')
    print('='*70 + '\n')

    # ── 1. Dataset ───────────────────────────────────────────────────────
    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy,
        dataset_size=DATASET_SIZE,
        batch_size=512,
        cache_file=f'/export/tmp/sala/sionna_base_'
                   f'{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        augmentation_multiplier=50,
        seed=SEED)

    # ── 2. Entraînement ──────────────────────────────────────────────────
    print('\n' + '='*70)
    print('  TRAINING')
    print('='*70 + '\n')

    cfg           = TRAINING_CONFIG
    trained_models = {}

    for m_cfg in MODELS_TO_TRAIN:
        run_name = (m_cfg['name']
                    .replace(' ', '_').replace('/', '-')
                    .replace('(', '').replace(')', ''))

        print('─' * 70)
        print(f"  Training : {m_cfg['name']}  →  weights/{run_name}/")
        print('─' * 70)

        system = MU_MIMO_System(
            num_tx=NUM_TX, num_rx=NUM_RX,
            precoder_type=m_cfg['type'],
            **m_cfg['sys_kwargs'])

        trainer = SupervisedTrainer(
            system, dataset,
            run_name=run_name,
            snr_db=TRAINING_SNR,
            warmup_epochs=cfg['warmup_epochs'],
            finetune_epochs=cfg['finetune_epochs'],
            batch_size=m_cfg.get('batch_size', BATCH_SIZE),
            learning_rate=cfg['learning_rate'])

        trainer.train(print_every=250, patience=8)
        trained_models[m_cfg['name']] = (system, trainer.checkpoint.best_ckpt_path)

    # ── 3. Évaluation ────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('  EVALUATION')
    print('='*70 + '\n')

    results_all = {}

    # Baselines classiques
    for bname, ptype in [('RZF', 'rzf'), ('WMMSE', 'wmmse')]:
        sys_ = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ptype)
        results_all[bname] = evaluate_system(
            sys_, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, bname)

    # Modèles entraînés
    for name, (system, weights_path) in trained_models.items():
        if not weights_path:
            print(f'⚠️  No weights for {name} — skipping')
            continue
        system.load_precoder_weights(weights_path)
        results_all[name] = evaluate_system(
            system, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, name)

    # ── 4. Figures & tableau ─────────────────────────────────────────────
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    plot_all(results_all, EVALUATION_SNR_RANGE, save_dir)
    print_summary_table(results_all, EVALUATION_SNR_RANGE)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'{save_dir}/results_{ts}.npy', results_all)
    print(f'\n✅ Results saved: {save_dir}/results_{ts}.npy')


if __name__ == '__main__':
    main()