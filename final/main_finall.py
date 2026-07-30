# =============================================================================
# main.py — IntraRB Transformer Precoding : Training + Evaluation
# =============================================================================
# Pipeline :
#   1. Dataset SAGE-HB (canaux single-user → combinaisons multi-user on-the-fly)
#   2. Entraînement : warmup MSE-RZF → finetune sum-rate (SNR aléatoire partout)
#   3. Évaluation : sum-rate vs SNR, BER Monte Carlo
#   4. Figures publication + tableau Pareto énergie
# =============================================================================

import os, json, pickle, logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.channel import (ApplyOFDMChannel, cir_to_ofdm_channel,
                                 subcarrier_frequencies)
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper,
                               LMMSEEqualizer, LMMSEPostEqualizationSINR,
                               RemoveNulledSubcarriers)
from sionna.phy.ofdm import RZFPrecodedChannel
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import compute_ber, ebnodb2no

from datasets import CachedSionnaDataset
from precoder_intra_rb import IntraRBTransformerPrecoder
from precoders_w import rzf_precoder, wmmse_precoder, TransformerPrecoderV4, TransformerPrecoderV5

# =============================================================================
# CONFIGURATION GLOBALE — un seul endroit à modifier
# =============================================================================

SEED         = 42

# Locked Stage 3/4 channel config -- see channel_config.py for the full
# validation trail (mechanism: forced LOS + 15deg azimuth window + 50%
# loading ratio, validated consistent from standard to massive-MIMO scale).
# Switch CHOSEN_CONFIG to retarget training between the two locked scales --
# this is the one place to change it.
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG, FFT_SIZE
CHOSEN_CONFIG = STANDARD_CONFIG   # or MASSIVE_CONFIG
NUM_TX       = CHOSEN_CONFIG['NUM_TX']
NUM_RX       = CHOSEN_CONFIG['NUM_RX']
BATCH_SIZE   = 128
DATASET_SIZE = 5000

TRAINING_SNR         = 15.0
SNR_MIN_TRAIN        = 5.0    # range SNR pendant l'entraînement
SNR_MAX_TRAIN        = 25.0
EVALUATION_SNR_RANGE = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
                                  dtype=np.float32)

TRAINING_CONFIG = {
    'warmup_epochs'  : 5,
    'finetune_epochs': 20,
    'learning_rate'  : 1e-3,
}

# Énergie : bit-widths pour le modèle lab
Q_W = 16   # poids FP16
Q_A = 16   # activations FP16

# =============================================================================
# MODÈLES À ENTRAÎNER — ajouter/retirer ici
# =============================================================================

MODELS_TO_TRAIN = [

    # ── 1. IntraRB — contribution principale ─────────────────────────────────
    # Poids partagés entre RBs → fréquence-agnostique → 1 circuit HLS pour N_RB
    {
        'name'      : 'IntraRB_4L_128d',
        'type'      : 'intra_rb',
        'sys_kwargs': {'embed_dim': 128, 'num_heads': 4, 'num_layers': 4},
        'batch_size': 256,   # réduit vs M=8 (feat_dim 41→321 = plus lourd)
    },

    # ── 2. V4.2 3tok — point Pareto-optimal V4 (référence de comparaison) ────
    # 95.8% WMMSE à M=8 → voir si M=64 améliore le gap
    {
        'name'      : 'V4_3tok_4L_128d',
        'type'      : 'transformer_rb',
        'sys_kwargs': {
            'embed_dim'    : 128,
            'num_heads'    : 4,
            'num_layers'   : 4,
            'tokens_per_rb': 3,      # point Pareto-optimal des slides
            'version'      : 'v4.2', # rich features + learned upsample
        },
        'batch_size': 256,
    },

    # ── 3. V4.2 6tok — point haute performance V4 ────────────────────────────
    # 97.8% WMMSE à M=8 → complémente l'ablation tok/RB
    {
        'name'      : 'V4_6tok_4L_128d',
        'type'      : 'transformer_rb',
        'sys_kwargs': {
            'embed_dim'    : 128,
            'num_heads'    : 4,
            'num_layers'   : 4,
            'tokens_per_rb': 6,
            'version'      : 'v4.2',
        },
        'batch_size': 256,
    },
]
# =============================================================================
# MODÈLE D'ÉNERGIE — identique au labo Energy.py
# =============================================================================

def energy_constants(Q):
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    EM   = 2.0 * EMAC
    EL   = EMAC
    return EMAC, EM, EL


def compute_energy_uJ(FLOPs, Weights, Activations, Q_W=32, Q_A=32):
    """Énergie totale µJ via modèle lab. MACs = FLOPs/2."""
    import math
    MACs               = FLOPs / 2.0
    EMAC, EM,   EL    = energy_constants(Q_W)
    _,    EM_A, EL_A  = energy_constants(Q_A)
    sqrt_p_W          = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A          = math.sqrt(64.0 * (Q_A / 16.0))
    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)
    return (EC + EW + EA) / 1e9


def compute_classical_complexity(method, M, K, N_SC, N_OFDM, I_wmmse=10):
    def zf_per_sc():
        return 7.0 * (2.0/3.0 * K**3 + 2.0 * K**2 * M)
    def wmmse_per_sc():
        per_iter = (
              (14.0/3.0) * K * M**3 + 12.0 * K**2 * M**2
            + 12.0 * K**2 * M + 9.0 * K * M**2
            + 8.0 * K * M + 5.0 * K**2 + (68.0/3.0) * K
        )
        return 7.0 * I_wmmse * per_iter
    acts = 2.0 * (K*M + 2.0*K**2 + M*K)
    f_sc = wmmse_per_sc() if 'WMMSE' in method else zf_per_sc()
    if method == 'RZF':
        f_sc += K
    return f_sc * N_SC * N_OFDM, 0.0, acts * N_SC * N_OFDM


def get_complexity(name, system):
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
    raise ValueError(f"Impossible de calculer complexité pour '{name}'")


# =============================================================================
# MU-MIMO SYSTEM
# =============================================================================

class MU_MIMO_System(tf.keras.Model):
    """
    Système MU-MIMO complet avec Sionna.
    Supporte : rzf, wmmse, intra_rb, transformer_rb, transformer_v5
    """

    def __init__(self, num_tx=8, num_rx=4, precoder_type='rzf',
                 rb_size=12, embed_dim=128, num_heads=4,
                 num_layers=4, tokens_per_rb=1,
                 num_intra_layers=2, num_inter_layers=2,
                 version='v4.0', weights_path=None):
        super().__init__()

        self.num_bs_antennas     = num_tx
        self.num_users           = num_rx
        self.precoder_type       = precoder_type
        self.num_bits_per_symbol = 2
        self.embed_dim           = embed_dim
        self.num_heads           = num_heads
        self.num_layers          = num_layers
        self.tokens_per_rb       = tokens_per_rb
        self.num_intra_layers    = num_intra_layers
        self.num_inter_layers    = num_inter_layers
        self.version             = version

        # Stream management
        self.sm = StreamManagement(
            np.ones([num_rx, 1]), num_streams_per_tx=num_rx)

        # Resource grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=FFT_SIZE, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern='kronecker', pilot_ofdm_symbol_indices=[2, 11])

        # Antennes
        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization='single',
            polarization_type='V', antenna_pattern='omni',
            carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=num_tx // 2, polarization='dual',
            polarization_type='cross', antenna_pattern='38.901',
            carrier_frequency=2.6e9)

        # Canal
        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        # PHY
        self.binary_source = BinarySource()
        self.encoder       = LDPC5GEncoder(
            int(self.rg.num_data_symbols),
            int(self.rg.num_data_symbols * 2))
        self.mapper        = Mapper('qam', self.num_bits_per_symbol)
        self.rg_mapper     = ResourceGridMapper(self.rg)
        self.frequencies   = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.channel_freq  = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ     = LMMSEEqualizer(self.rg, self.sm)
        self.demapper      = Demapper('app', 'qam', self.num_bits_per_symbol)
        self.decoder       = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr    = LMMSEPostEqualizationSINR(
            resource_grid=self.rg, stream_management=self.sm)
        self.remove_nulled = RemoveNulledSubcarriers(self.rg)

        # Helper Sionna pour canal effectif
        self.ch_helper = RZFPrecodedChannel(
            resource_grid=self.rg, stream_management=self.sm)

        self._init_precoder(precoder_type, rb_size, weights_path)

    # ── Initialisation précoder ───────────────────────────────────────────────
    def _init_precoder(self, precoder_type, rb_size, weights_path):
        if precoder_type == 'intra_rb':
            print(f'✅ IntraRBTransformerPrecoder | '
                  f'D={self.embed_dim} L={self.num_layers} H={self.num_heads}')
            self.precoder = IntraRBTransformerPrecoder(
                num_tx=self.num_bs_antennas, num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols, fft_size=self.rg.fft_size,
                rb_size=rb_size, embed_dim=self.embed_dim,
                num_heads=self.num_heads, num_layers=self.num_layers,
                snr_aware=True)

        elif precoder_type == 'transformer_rb':
            print(f'✅ TransformerPrecoderV4 [{self.version}] | '
                  f'{self.tokens_per_rb} tok/RB')
            self.precoder = TransformerPrecoderV4(
                num_tx=self.num_bs_antennas, num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols, fft_size=self.rg.fft_size,
                rb_size=rb_size, tokens_per_rb=self.tokens_per_rb,
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                num_layers=self.num_layers, version=self.version)

        elif precoder_type == 'transformer_v5':
            print(f'✅ TransformerPrecoderV5')
            self.precoder = TransformerPrecoderV5(
                num_tx=self.num_bs_antennas, num_rx=self.num_users,
                num_ofdm=self.rg.num_ofdm_symbols, fft_size=self.rg.fft_size,
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                num_intra_layers=self.num_intra_layers,
                num_inter_layers=self.num_inter_layers, snr_aware=True)

        elif precoder_type in ('rzf', 'wmmse'):
            print(f'✅ Précoder classique : {precoder_type.upper()}')
            self.precoder = None

        else:
            raise ValueError(f'Précoder inconnu : {precoder_type}')

        if weights_path and self.precoder is not None:
            self.load_weights_from(weights_path)

    # ── Interface précoder uniforme ───────────────────────────────────────────
    def _call_precoder(self, h_freq, no, training=False,
                       return_real_imag=False):
        """
        Interface unique pour tous les précodeurs neuraux.
        Tous acceptent (h_freq, no=no) — no est ignoré si non utilisé.
        """
        p = self.precoder
        if training and return_real_imag:
            g_re, g_im = p(h_freq, no=no, training=True, return_real_imag=True)
            return tf.complex(g_re, g_im)
        return p(h_freq, no=no, training=False)

    # ── Topologie ─────────────────────────────────────────────────────────────
    def new_topology(self, batch_size):
        # Locked channel config (narrow azimuth window + forced LOS) -- see
        # channel_config.py for why. Falls back to plain 3GPP topology if
        # channel_config isn't importable, so this class stays usable
        # standalone at other M/K.
        try:
            from channel_config import set_locked_topology
        except ImportError:
            topology = gen_topology(batch_size, self.num_users, 'umi')
            self.channel_model.set_topology(*topology)
            return
        set_locked_topology(self.channel_model, batch_size, self.num_users,
                             CHOSEN_CONFIG['HALF_ANGLE_DEG'],
                             CHOSEN_CONFIG['FORCE_LOS'],
                             CHOSEN_CONFIG['INDOOR_PROBABILITY'])

    # ── Forward pass complet (génération online) ──────────────────────────────
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
        h_freq = self.remove_nulled(h_freq)

        if self.precoder_type == 'rzf':
            g = rzf_precoder(h_freq, self.sm, no=no)
        elif self.precoder_type == 'wmmse':
            g = wmmse_precoder(h_freq, no, self.sm)
        else:
            g = self._call_precoder(h_freq, no, training=training)

        return self._forward_from_precoder(b, c, x_rg, h_freq, g, no)

    # ── Forward pass avec canal caché (entraînement) ──────────────────────────
    @tf.function
    def call_cached(self, batch_size, ebno_db, h_freq, training=False):
        no   = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        b    = self.binary_source([batch_size, 1, self.num_users,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)

        if self.precoder_type == 'rzf':
            g = rzf_precoder(h_freq, self.sm, no=no)
        elif self.precoder_type == 'wmmse':
            g = wmmse_precoder(h_freq, no, self.sm)
        else:
            g = self._call_precoder(h_freq, no, training=training,
                                    return_real_imag=training)

        return self._forward_from_precoder(b, c, x_rg, h_freq, g, no)

    def _forward_from_precoder(self, b, c, x_rg, h_freq, g, no):
        """Applique le précoder et calcule le forward Sionna complet."""
        W = tf.squeeze(g, axis=1)   # [B, ofdm, fft, M, K]

        # Précoder le signal
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_pre = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff        = self.ch_helper.compute_effective_channel(h_freq, g)
        y            = self.channel_freq(x_pre, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr          = self.demapper(x_hat, no_eff)
        b_hat        = self.decoder(llr)
        return b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_pre, g

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def load_weights_from(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, 'weights.pkl')
        if not os.path.exists(path):
            print(f'❌ Poids introuvables : {path}')
            return False
        with open(path, 'rb') as f:
            ws = pickle.load(f)
        for v, w in zip(self.precoder.trainable_variables, ws):
            v.assign(w)
        print(f'✅ {len(ws)} poids chargés depuis {path}')
        return True


# =============================================================================
# CHECKPOINT
# =============================================================================

class SimpleCheckpoint:
    def __init__(self, run_name, base_dir='./weights'):
        self.run_dir        = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_ckpt_path = None

    @staticmethod
    def _safe(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(type(obj))

    def save(self, precoder, cfg, metrics):
        import shutil
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            shutil.rmtree(self.best_ckpt_path)
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        d   = os.path.join(self.run_dir, f'best_{ts}')
        os.makedirs(d)
        with open(os.path.join(d, 'weights.pkl'), 'wb') as f:
            pickle.dump([v.numpy() for v in precoder.trainable_variables], f)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump({**cfg, 'saved_at': datetime.now().isoformat()},
                      f, indent=2, default=self._safe)
        with open(os.path.join(d, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        self.best_ckpt_path = d
        print(f'  💾 {d}')
        return d


# =============================================================================
# TRAINER
# =============================================================================

class SupervisedTrainer:
    """
    Entraînement deux phases :
      Phase 1 (warmup)  : MSE pure vs RZF — SNR aléatoire [SNR_MIN, SNR_MAX]
      Phase 2 (finetune): sum-rate maximisation — SNR aléatoire idem
    Interface précoder uniforme : precoder(h, no=no, training=..., return_real_imag=...)
    """

    def __init__(self, system, dataset, run_name='run',
                 warmup_epochs=5, finetune_epochs=15,
                 learning_rate=2e-3, batch_size=256):
        self.system          = system
        self.dataset         = dataset
        self.warmup_epochs   = warmup_epochs
        self.finetune_epochs = finetune_epochs
        self.total_epochs    = warmup_epochs + finetune_epochs
        self.batch_size      = batch_size
        self.rate_norm       = float(system.num_users) * 9.0

        eff_size       = getattr(dataset, 'effective_dataset_size',
                         len(getattr(dataset, 'h_freq_all', [0])))
        self.num_iters = max(eff_size // batch_size, 1)

        # LR schedules (cosine decay)
        wu_steps = max(warmup_epochs   * self.num_iters, 1)
        ft_steps = max(finetune_epochs * self.num_iters, 1)
        self.lr_warmup   = tf.keras.optimizers.schedules.CosineDecay(
            learning_rate, wu_steps, alpha=0.05)
        self.lr_finetune = tf.keras.optimizers.schedules.CosineDecay(
            learning_rate * 0.5, ft_steps, alpha=0.01)

        self.opt = tf.keras.optimizers.Adam(self.lr_warmup, clipnorm=5.0)

        # Build le modèle
        dummy_h = tf.zeros(
            [1, system.num_users, 1, 1, system.num_bs_antennas,
             system.rg.num_ofdm_symbols, system.rg.fft_size],
            dtype=tf.complex64)
        try:
            system.precoder(dummy_h, no=tf.constant(1e-3), training=False)
        except Exception:
            pass
        self.vars = system.precoder.trainable_variables

        n_params = sum(tf.size(v).numpy() for v in self.vars)
        self.ckpt = SimpleCheckpoint(run_name)
        self.best = -np.inf
        self.history = []

        print(f'\n{"="*60}')
        print(f'  [{run_name}]  params={n_params:,}  iters/ep={self.num_iters}')
        print(f'  Warmup  {warmup_epochs}ep  LR={learning_rate:.1e}')
        print(f'  Finetune {finetune_epochs}ep  LR={learning_rate*0.5:.1e}')
        print(f'  SNR range [{SNR_MIN_TRAIN},{SNR_MAX_TRAIN}] dB (random chaque step)')
        print(f'{"="*60}\n')

    # ── Warmup : MSE pure vs RZF ──────────────────────────────────────────────
    @tf.function
    def _warmup_step(self, h_freq):
        snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
        no     = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                           0.5, self.system.rg)

        g_rzf = tf.stop_gradient(
            rzf_precoder(h_freq, self.system.sm, no=no))

        with tf.GradientTape() as tape:
            g_pred = self.system._call_precoder(h_freq, no,
                                                training=True,
                                                return_real_imag=True)
            loss = tf.reduce_mean(tf.abs(g_pred - g_rzf) ** 2)

        grads = tape.gradient(loss, self.vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                 for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.opt.apply_gradients(zip(grads, self.vars))
        return loss, tf.linalg.global_norm(grads)

    # ── Finetune : sum-rate maximisation ──────────────────────────────────────
    @tf.function
    def _finetune_step(self, h_freq):
        snr_db = tf.random.uniform([], SNR_MIN_TRAIN, SNR_MAX_TRAIN)
        no     = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                           0.5, self.system.rg)

        with tf.GradientTape() as tape:
            g     = self.system._call_precoder(h_freq, no,
                                               training=True,
                                               return_real_imag=True)
            h_eff = self.system.ch_helper.compute_effective_channel(h_freq, g)
            sinr  = self.system.lmmse_sinr(
                h_eff, no=no, interference_whitening=True)
            rate  = tf.reduce_sum(
                tf.reduce_mean(
                    tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
                    / tf.math.log(2.0),
                    axis=[0, 1, 2, 4]))
            loss  = -tf.where(tf.math.is_finite(rate),
                              rate / self.rate_norm,
                              tf.constant(0.0))

        grads = tape.gradient(loss, self.vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                 for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.opt.apply_gradients(zip(grads, self.vars))
        return loss, tf.linalg.global_norm(grads)

    # ── Eval légère ───────────────────────────────────────────────────────────
    @tf.function
    def _eval_step(self, h_freq, snr_db):
        no    = ebnodb2no(snr_db, self.system.num_bits_per_symbol,
                          0.5, self.system.rg)
        g     = self.system._call_precoder(h_freq, no, training=False)
        h_eff = self.system.ch_helper.compute_effective_channel(h_freq, g)
        sinr  = self.system.lmmse_sinr(
            h_eff, no=no, interference_whitening=True)
        per_user = tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
            / tf.math.log(2.0), axis=[0, 1, 2, 4])
        return tf.reduce_sum(per_user), per_user

    # ── Boucle principale ─────────────────────────────────────────────────────
    def train(self, print_every=200, patience=8):
        import time
        snr_eval = tf.constant(TRAINING_SNR, tf.float32)
        no_improve = 0

        for epoch in range(self.total_epochs):
            is_warmup = epoch < self.warmup_epochs

            # Switch de phase
            if epoch == self.warmup_epochs:
                self.opt.learning_rate = self.lr_finetune
                no_improve = 0
                print(f'\n🔄 → Finetune  '
                      f'LR={self.lr_finetune.initial_learning_rate:.1e}\n')

            losses, gnorms, rates, per_users = [], [], [], []
            t0 = time.time()

            for i in range(self.num_iters):
                h = self.dataset.get_batch(self.batch_size)

                if is_warmup:
                    loss, gn = self._warmup_step(h)
                else:
                    loss, gn = self._finetune_step(h)

                losses.append(float(loss))
                gnorms.append(float(gn))

                if (i + 1) % print_every == 0:
                    rate, pu = self._eval_step(h, snr_eval)
                    rates.append(float(rate))
                    per_users.append(pu.numpy())
                    eta = (time.time()-t0)/(i+1)*(self.num_iters-i-1)/60
                    mode = 'WU-MSE' if is_warmup else 'FINETUNE'
                    print(f'    {i+1:4d}/{self.num_iters} [{mode}] '
                          f'sum={float(rate):.2f} | '
                          f'usr=[{" ".join(f"{v:.1f}" for v in pu.numpy())}] | '
                          f'loss={np.mean(losses[-print_every:]):.3f} | '
                          f'gn={np.mean(gnorms[-print_every:]):.2e} | '
                          f'ETA={eta:.1f}min')

            ep_time  = (time.time() - t0) / 60
            avg_rate = float(np.mean(rates)) if rates else 0.0
            avg_pu   = np.mean(per_users, axis=0) if per_users \
                       else np.zeros(self.system.num_users)

            star = ''
            if avg_rate > self.best:
                self.best = avg_rate
                self.ckpt.save(self.system.precoder,
                               {'epoch': epoch+1, 'sum_rate': avg_rate},
                               {'sum_rate': avg_rate})
                star = ' ⭐'
                no_improve = 0
            elif not is_warmup:
                no_improve += 1

            mode = 'WU-MSE' if is_warmup else 'FINETUNE'
            print(f'\n{epoch+1:3d}/{self.total_epochs} [{mode}] '
                  f'rate={avg_rate:.2f} | '
                  f'usr=[{", ".join(f"{v:.2f}" for v in avg_pu)}] | '
                  f'{ep_time:.1f}min{star}')

            self.history.append({
                'epoch': epoch+1, 'mode': mode,
                'sum_rate': avg_rate, 'per_user': avg_pu.tolist(),
                'loss': float(np.mean(losses)),
            })

            if not is_warmup and no_improve >= patience:
                print(f'\n⏹ Early stop — best={self.best:.2f}')
                break

        print(f'\n✅ Meilleur : {self.best:.2f} → {self.ckpt.best_ckpt_path}')
        return self.history


# =============================================================================
# ÉVALUATION — Monte Carlo correct
# =============================================================================

def evaluate_system(system, snr_range, num_batches=50,
                    batch_size=256, name='System'):
    print(f'\n📊 Évaluation : {name}')

    flops, weights, acts = get_complexity(name, system)
    is_classical         = (system.precoder is None)

    results = {
        'sum_rate'    : [],
        'ber'         : [],
        'per_user'    : [],
        'flops_M'     : flops   / 1e6,
        'params_K'    : weights / 1e3,
        'energy_fp32' : compute_energy_uJ(flops, weights, acts, 32, 32),
        'energy_fp16' : compute_energy_uJ(flops, weights, acts, 16, 16),
        'energy_int8' : compute_energy_uJ(flops, weights, acts, 8,  8),
        'is_classical': is_classical,
    }
    results['energy_plot'] = (results['energy_fp32'] if is_classical
                              else results['energy_fp16'])

    for snr in snr_range:
        rates_b, per_b = [], []
        tot_bits = tot_err = 0

        for _ in range(num_batches):
            system.new_topology(batch_size)
            b, b_hat, _, _, h_eff, no, _, _, _, _ = system(
                tf.constant(batch_size, tf.int32),
                tf.constant(float(snr),  tf.float32),
                training=False)

            tot_err  += int(tf.reduce_sum(
                tf.cast(b != b_hat, tf.int32)).numpy())
            tot_bits += int(b.numpy().size)

            sinr    = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate_sc = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            pu      = tf.reduce_mean(rate_sc, axis=[1, 2, 4]).numpy()
            rates_b.append(float(tf.reduce_sum(
                tf.reduce_mean(rate_sc, axis=[1, 2, 4]))))
            per_b.append(pu.mean(axis=0))

        ber_mc = tot_err / max(tot_bits, 1)
        results['sum_rate'].append(float(np.mean(rates_b)))
        results['ber'].append(float(ber_mc))
        results['per_user'].append(np.mean(per_b, axis=0))

        prec = 'FP32' if is_classical else 'FP16'
        print(f'  SNR={snr:5.1f} dB | '
              f'Rate={results["sum_rate"][-1]:6.2f} | '
              f'BER={ber_mc:.2e} | '
              f'E({prec})={results["energy_plot"]:.4f}µJ')

    return results


# =============================================================================
# FIGURES
# =============================================================================

def _styles(names):
    FIXED = {'RZF':   ('#1f77b4', 'o', 'RZF (FP32)'),
             'WMMSE': ('#000000', 's', 'WMMSE (FP32)')}
    COLS = ['#d62728','#ff7f0e','#2ca02c','#9467bd','#8c564b','#e377c2']
    MKRS = ['v','^','D','p','*','h']
    colors, markers, labels = {}, {}, {}
    ci = 0
    for n in names:
        if n in FIXED:
            colors[n], markers[n], labels[n] = FIXED[n]
        else:
            colors[n]  = COLS[ci % len(COLS)]
            markers[n] = MKRS[ci % len(MKRS)]
            labels[n]  = n
            ci += 1
    return colors, markers, labels


def plot_results(results_all, snr_range, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    idx_15  = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))
    all_n   = list(results_all.keys())
    C, M, L = _styles(all_n)

    wmmse_rate = results_all.get('WMMSE', {}).get('sum_rate', [0]*10)[idx_15]
    wmmse_e32  = results_all.get('WMMSE', {}).get('energy_fp32', 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'IntraRB Transformer Precoder — {NUM_TX}×{NUM_RX} MU-MIMO  '
                 f'(8TX, 4UE, 72SC, SNR train={TRAINING_SNR:.0f}dB)',
                 fontsize=11, fontweight='bold')

    def lw(n): return 2.2 if n in ('RZF','WMMSE') else 1.8
    def ls(n): return '--' if n=='RZF' else '-.' if n=='WMMSE' else '-'

    # (a) Sum Rate
    ax = axes[0, 0]
    for n in all_n:
        res = results_all[n]
        ax.plot(snr_range, res['sum_rate'], marker=M[n], color=C[n],
                label=L[n], linewidth=lw(n), linestyle=ls(n), markersize=7)
    ax.set(xlabel='SNR (dB)', ylabel='Sum Rate (bps/Hz)',
           title='(a) Efficacité spectrale')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) BER
    ax = axes[0, 1]
    for n in all_n:
        ber = np.maximum(results_all[n]['ber'], 5e-6)
        ax.semilogy(snr_range, ber, marker=M[n], color=C[n],
                    label=L[n], linewidth=lw(n), linestyle=ls(n), markersize=7)
    ax.set(xlabel='SNR (dB)', ylabel='BER', title='(b) BER Monte Carlo')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # (c) Pareto énergie
    ax = axes[1, 0]
    for n in [k for k in all_n if k != 'RZF']:
        res  = results_all[n]
        rate = res['sum_rate'][idx_15]
        e    = res['energy_plot']
        ax.scatter(e, rate, s=220, marker=M[n], color=C[n],
                   label=L[n], zorder=4, edgecolors='k', linewidth=0.8)
        ax.annotate(L[n].split('(')[0].strip(), (e, rate),
                    textcoords='offset points', xytext=(5, 4), fontsize=7)
    ax.set(xlabel='Énergie inférence (µJ)\n[NN→FP16 | WMMSE→FP32]',
           ylabel=f'Sum Rate @ {TRAINING_SNR:.0f}dB (bps/Hz)',
           title='(c) Pareto Énergie–Performance')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (d) Tableau
    ax = axes[1, 1]
    ax.axis('off')
    hdrs = ['Méthode', 'Rate\n(bps/Hz)', 'Gap vs\nWMMSE',
            'FLOPs\n(M)', 'E FP16\n(µJ)', 'Gain\nFP16']
    rows = []
    for n, res in results_all.items():
        rate = res['sum_rate'][idx_15]
        gap  = rate - wmmse_rate
        e16  = res['energy_fp16']
        gain = wmmse_e32 / e16 if (e16 > 0 and not res['is_classical']) else None
        rows.append([L[n], f'{rate:.2f}', f'{gap:+.2f}',
                     f'{res["flops_M"]:.0f}',
                     f'{e16:.4f}' if not res['is_classical'] else 'N/A',
                     f'{gain:.1f}×' if gain else 'ref'])
    rows.sort(key=lambda r: float(r[1]), reverse=True)
    tbl = ax.table(cellText=rows, colLabels=hdrs,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for j in range(len(hdrs)):
        tbl[(0,j)].set_facecolor('#2c3e50')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold')
    ax.set_title(f'(d) Résumé @ {TRAINING_SNR:.0f}dB', fontweight='bold')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = f'{save_dir}/results_{ts}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'✅ Figure sauvée : {p}')
    plt.close(fig)


def print_table(results_all, snr_range):
    idx        = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))
    wmmse_rate = results_all.get('WMMSE', {}).get('sum_rate', [0]*10)[idx]
    wmmse_e32  = results_all.get('WMMSE', {}).get('energy_fp32', 1.0)
    print(f'\n{"="*90}')
    print(f'  PARETO @ {TRAINING_SNR:.0f}dB  (WMMSE={wmmse_rate:.2f} bps/Hz | {wmmse_e32:.4f}µJ FP32)')
    print(f'{"="*90}')
    print(f'{"Architecture":<24} | {"Rate":>7} | {"Gap":>6} | '
          f'{"FLOPs(M)":>8} | {"FP32(µJ)":>9} | {"FP16(µJ)":>9} | {"Gain FP16":>10}')
    print('─' * 90)
    for n, res in sorted(results_all.items(),
                         key=lambda x: x[1]['sum_rate'][idx], reverse=True):
        rate = res['sum_rate'][idx]
        gap  = rate - wmmse_rate
        gain = wmmse_e32 / res['energy_fp16'] \
               if (res['energy_fp16'] > 0 and not res['is_classical']) else None
        fp16 = f"{res['energy_fp16']:.4f}" if not res['is_classical'] else '  N/A  '
        gs   = f'{gain:.1f}×' if gain else '  ref  '
        print(f'{n:<24} | {rate:>7.2f} | {gap:>+6.2f} | '
              f'{res["flops_M"]:>8.1f} | {res["energy_fp32"]:>9.4f} | '
              f'{fp16:>9} | {gs:>10}')
    print('=' * 90)


# =============================================================================
# MAIN
# =============================================================================

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f'✅ GPU : {gpus[0]}')

    print('\n' + '='*70)
    print(f'  INTRA-RB TRANSFORMER PRECODING')
    print(f'  {NUM_TX}×{NUM_RX} MIMO | SNR train [{SNR_MIN_TRAIN},{SNR_MAX_TRAIN}]dB | '
          f'Dataset {DATASET_SIZE:,}')
    print('='*70 + '\n')

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    print('='*60 + '\n  DATASET\n' + '='*60)
    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy,
        dataset_size=DATASET_SIZE,
        batch_size=512,
        cache_file=f'/export/tmp/sala/sionna_base_'
                   f'{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        augmentation_multiplier=50,
        seed=SEED)

    # ── 2. Entraînement ──────────────────────────────────────────────────────
    print('\n' + '='*60 + '\n  TRAINING\n' + '='*60)
    cfg            = TRAINING_CONFIG
    trained_models = {}

    for m_cfg in MODELS_TO_TRAIN:
        run_name = (m_cfg['name']
                    .replace(' ', '_').replace('/', '-')
                    .replace('(', '').replace(')', ''))
        print(f'\n{"─"*60}\n  {m_cfg["name"]} → weights/{run_name}/\n{"─"*60}')

        system = MU_MIMO_System(
            num_tx=NUM_TX, num_rx=NUM_RX,
            precoder_type=m_cfg['type'],
            **m_cfg['sys_kwargs'])

        trainer = SupervisedTrainer(
            system, dataset,
            run_name=run_name,
            warmup_epochs=cfg['warmup_epochs'],
            finetune_epochs=cfg['finetune_epochs'],
            batch_size=m_cfg.get('batch_size', BATCH_SIZE),
            learning_rate=cfg['learning_rate'])

        trainer.train(print_every=200, patience=8)
        trained_models[m_cfg['name']] = (system, trainer.ckpt.best_ckpt_path)

    # ── 3. Évaluation ────────────────────────────────────────────────────────
    print('\n' + '='*60 + '\n  EVALUATION\n' + '='*60)
    results_all = {}

    for bname, ptype in [('RZF', 'rzf'), ('WMMSE', 'wmmse')]:
        sys_ = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ptype)
        results_all[bname] = evaluate_system(
            sys_, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, bname)

    for name, (system, wpath) in trained_models.items():
        if not wpath:
            print(f'⚠️  Pas de poids pour {name} — ignoré')
            continue
        system.load_weights_from(wpath)
        results_all[name] = evaluate_system(
            system, EVALUATION_SNR_RANGE, 50, BATCH_SIZE, name)

    # ── 4. Figures & tableau ─────────────────────────────────────────────────
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    plot_results(results_all, EVALUATION_SNR_RANGE, save_dir)
    print_table(results_all, EVALUATION_SNR_RANGE)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'{save_dir}/results_{ts}.npy', results_all)
    print(f'\n✅ Résultats sauvés : {save_dir}/results_{ts}.npy')


if __name__ == '__main__':
    main()