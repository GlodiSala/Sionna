"""
eval_only.py — Évaluation standalone V4.2 + V5 + Baselines
=============================================================
CONFIGURATION CENTRALE : modifier uniquement les sections
  ► SYSTEM CONFIG
  ► TRAINED_MODELS_V4
  ► TRAINED_MODELS_V5
  ► RZF_GROUP_CONFIGS
pour ajouter/retirer des modèles à évaluer.
"""

import os
import pickle
import logging
import math
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import (ApplyOFDMChannel, cir_to_ofdm_channel,
                                 subcarrier_frequencies)
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LMMSEEqualizer,
                               LMMSEPostEqualizationSINR,
                               RemoveNulledSubcarriers)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no

from precoders_w import rzf_precoder, wmmse_precoder, TransformerPrecoderV4, TransformerPrecoderV5

# =============================================================================
# ► SYSTEM CONFIG — modifier ici
# =============================================================================

SEED         = 42
NUM_TX       = 4
NUM_RX       = 4
BATCH_SIZE   = 256
NUM_BATCHES  = 50
TRAINING_SNR = 15.0   # dB — utilisé pour le Pareto

EVALUATION_SNR_RANGE = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20],
                                  dtype=np.float32)

# =============================================================================
# ► TRAINED_MODELS_V4 — ajouter/retirer des entrées ici
# =============================================================================

TRAINED_MODELS_V4 = [
    {
        'name'         : 'V4.2 — 1 tok/RB',
        'tokens_per_rb': 1,
        'version'      : 'v4.2',
        'embed_dim'    : 128,
        'num_heads'    : 4,
        'weights_path' : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V4.2_1tok-RB/best_20260318_205000/weights.pkl',
    },
    {
        'name'         : 'V4.2 — 2 tok/RB',
        'tokens_per_rb': 2,
        'version'      : 'v4.2',
        'embed_dim'    : 128,
        'num_heads'    : 4,
        'weights_path' : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V4.2_2tok-RB/best_20260319_002146/weights.pkl',
    },{

        'name'         : 'V4.2 — 3 tok/RB',
        'tokens_per_rb': 3,
        'version'      : 'v4.2',
        'embed_dim'    : 128,
        'num_heads'    : 4,
        'weights_path' : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V4.2_3tok-RB/best_20260319_035709/weights.pkl',
    },
    {
        'name'         : 'V4.2 — 4 tok/RB',
        'tokens_per_rb': 4,
        'version'      : 'v4.2',
        'embed_dim'    : 128,
        'num_heads'    : 4,
        'weights_path' : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V4.2_4tok-RB/best_20260319_092852/weights.pkl',
        
    },
    {
        'name'         : 'V4.2 — 6 tok/RB',
        'tokens_per_rb': 6,
        'version'      : 'v4.2',
        'embed_dim'    : 128,
        'num_heads'    : 4,
        'weights_path' : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V4.2_6tok-RB/best_20260319_162214/weights.pkl',
        
    },
]

# =============================================================================
# ► TRAINED_MODELS_V5 — ajouter/retirer des entrées ici
# =============================================================================

TRAINED_MODELS_V5 = [
    {
        'name'             : 'V5 — 2L/128d',
        'num_intra_layers' : 2,
        'num_inter_layers' : 2,
        'embed_dim'        : 128,
        'num_heads'        : 4,
        'weights_path'     : '/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights/V5.2_2L_128d/best_20260319_230824/weights.pkl',
    },
    # Exemple pour en ajouter un autre :
    # {
    #     'name'             : 'V5 — 4L/256d',
    #     'num_intra_layers' : 4,
    #     'num_inter_layers' : 4,
    #     'embed_dim'        : 256,
    #     'num_heads'        : 8,
    #     'weights_path'     : './weights/V5-4L/best_.../weights.pkl',
    # },
]

# =============================================================================
# ► RZF_GROUP_CONFIGS — groupes à évaluer (n_groups, label)
#   Comparaison avec V4 uniquement (pas V5)
# =============================================================================

RZF_GROUP_CONFIGS = [
    (6,  'RZF — 6 gr. (12 SC/gr.)'),
    (12, 'RZF — 12 gr. (6 SC/gr.)'),
    (18, 'RZF — 18 gr. (4 SC/gr.)'),
    (36, 'RZF — 36 gr. (2 SC/gr.)'),
]

# =============================================================================
# ÉNERGIE — modèle lab
# =============================================================================

def energy_constants(Q):
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    return EMAC, 2.0 * EMAC, EMAC   # EMAC, EM, EL

def compute_energy_uJ(FLOPs, Weights, Activations, Q_W=32, Q_A=32):
    MACs            = FLOPs / 2.0
    EMAC, EM, EL    = energy_constants(Q_W)
    _,  EM_A, EL_A  = energy_constants(Q_A)
    sqrt_p_W        = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A        = math.sqrt(64.0 * (Q_A / 16.0))
    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)
    return (EC + EW + EA) / 1e9

# =============================================================================
# COMPLEXITÉ — classiques
# =============================================================================

def classical_complexity(method, M=8, K=4, N_SC=72, N_OFDM=14, I=10):
    if 'WMMSE' in method.upper():
        per_iter = ((14.0/3.0)*K*M**3 + 12.0*K**2*M**2 + 12.0*K**2*M
                    + 9.0*K*M**2 + 8.0*K*M + 5.0*K**2 + (68.0/3.0)*K)
        f = 7.0 * I * per_iter * N_SC * N_OFDM
    else:   # RZF
        f = (7.0 * (2.0/3.0*K**3 + 2.0*K**2*M) + K) * N_SC * N_OFDM
    acts = 2.0 * (K*M + 2.0*K**2 + M*K) * N_SC * N_OFDM
    return f, 0.0, acts

def rzf_grouped_complexity(n_groups, M=8, K=4, N_SC=72, N_OFDM=14):
    f    = (7.0 * (2.0/3.0*K**3 + 2.0*K**2*M) + K) * n_groups * N_OFDM
    acts = 2.0 * (K*M + 2.0*K**2 + M*K) * n_groups * N_OFDM
    return f, 0.0, acts

# =============================================================================
# SYSTÈME MU-MIMO
# =============================================================================

class MU_MIMO_System(tf.keras.Model):

    def __init__(self, num_tx=8, num_rx=4, precoder_type='rzf',
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, version='v4.2',
                 num_intra_layers=2, num_inter_layers=2,
                 rzf_n_groups=None):
        super().__init__()

        self.num_bs_antennas     = num_tx
        self.num_users           = num_rx
        self.precoder_type       = precoder_type
        self.num_bits_per_symbol = 2
        self.rzf_n_groups        = rzf_n_groups

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern='kronecker', pilot_ofdm_symbol_indices=[2, 11])

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization='single',
            polarization_type='V', antenna_pattern='omni',
            carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx / 2), polarization='dual',
            polarization_type='cross', antenna_pattern='38.901',
            carrier_frequency=2.6e9)

        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        self.binary_source     = BinarySource()
        self.encoder           = LDPC5GEncoder(int(self.rg.num_data_symbols),
                                               int(self.rg.num_data_symbols * 2))
        self.mapper            = Mapper('qam', self.num_bits_per_symbol)
        self.rg_mapper         = ResourceGridMapper(self.rg)
        self.frequencies       = subcarrier_frequencies(self.rg.fft_size,
                                                        self.rg.subcarrier_spacing)
        self.channel_freq      = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ         = LMMSEEqualizer(self.rg, self.sm)
        self.demapper          = Demapper('app', 'qam', self.num_bits_per_symbol)
        self.decoder           = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr        = LMMSEPostEqualizationSINR(
            resource_grid=self.rg, stream_management=self.sm)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        from sionna.phy.ofdm import RZFPrecodedChannel
        self.precoded_channel_helper = RZFPrecodedChannel(
            resource_grid=self.rg, stream_management=self.sm)

        self.precoder = None
        if precoder_type == 'transformer_v4':
            print(f'✅ TransformerPrecoderV4 — {tokens_per_rb} tok/RB ({version})')
            self.precoder = TransformerPrecoderV4(
                num_tx=num_tx, num_rx=num_rx, num_ofdm=14, fft_size=72,
                rb_size=rb_size, tokens_per_rb=tokens_per_rb,
                embed_dim=embed_dim, num_heads=num_heads,
                num_layers=4, version=version)

        elif precoder_type == 'transformer_v5':
            print(f'✅ TransformerPrecoderV5 — {num_intra_layers}L intra / '
                  f'{num_inter_layers}L inter / d={embed_dim}')
            self.precoder = TransformerPrecoderV5(
                num_tx=num_tx, num_rx=num_rx, num_ofdm=14, fft_size=72,
                embed_dim=embed_dim, num_heads=num_heads,
                num_intra_layers=num_intra_layers,
                num_inter_layers=num_inter_layers)
        else:
            print(f'✅ {precoder_type.upper()} Precoder')

    def load_weights_from_pkl(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, 'weights.pkl')
        with open(path, 'rb') as f:
            ws = pickle.load(f)
        for var, w in zip(self.precoder.trainable_variables, ws):
            var.assign(w)
        print(f'  ✅ Loaded {len(ws)} weights')

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, self.num_users, 'umi')
        self.channel_model.set_topology(*topology)

    def _rzf_grouped_precoder(self, h_freq, n_groups, alpha=0.1):
        fft_size = 72
        sc_per_group = fft_size // n_groups
        B        = tf.shape(h_freq)[0]
        num_rx   = tf.shape(h_freq)[1]
        num_tx   = tf.shape(h_freq)[4]
        num_ofdm = tf.shape(h_freq)[5]

        h_sq  = tf.squeeze(h_freq, axis=[2, 3])
        h_grp = tf.reshape(h_sq,
            [B, num_rx, num_tx, num_ofdm, n_groups, sc_per_group])
        h_avg = tf.reduce_mean(h_grp, axis=-1)

        h_avg_p = tf.transpose(h_avg, [0, 3, 4, 1, 2])
        h_flat  = tf.reshape(h_avg_p, [-1, num_rx, num_tx])
        HHH     = tf.matmul(h_flat, h_flat, adjoint_b=True)
        eye     = tf.eye(num_rx, dtype=HHH.dtype)
        HHH_inv = tf.linalg.inv(HHH + alpha * eye)
        W_grp   = tf.matmul(h_flat, HHH_inv, adjoint_a=True)

        W_grp  = tf.reshape(W_grp, [B, num_ofdm, n_groups, num_tx, num_rx])
        W_full = tf.reshape(
            tf.repeat(tf.expand_dims(W_grp, axis=3), sc_per_group, axis=3),
            [B, num_ofdm, fft_size, num_tx, num_rx])

        pwr   = tf.reduce_sum(tf.abs(W_full)**2, axis=[-2, -1], keepdims=True)
        scale = tf.cast(
            tf.sqrt(tf.cast(num_rx, tf.float32) / (tf.cast(pwr, tf.float32) + 1e-12)),
            W_full.dtype)
        return tf.expand_dims(W_full * scale, axis=1)

    @tf.function
    def call(self, batch_size, ebno_db, training=False):
        no   = ebnodb2no(ebno_db, self.num_bits_per_symbol, 0.5, self.rg)
        b    = self.binary_source([batch_size, 1, self.num_users,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)

        cir    = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                    1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled_scs(h_freq)

        if self.precoder_type == 'rzf':
            g = rzf_precoder(h_freq, stream_management=self.sm, alpha=0.1)
        elif self.precoder_type == 'rzf_grouped':
            g = self._rzf_grouped_precoder(h_freq, self.rzf_n_groups)
        elif self.precoder_type == 'wmmse':
            g = wmmse_precoder(h_freq, no, stream_management=self.sm,
                               num_iterations=10)
        else:
            g = self.precoder(h_freq, training=training)

        W         = tf.squeeze(g, axis=1)
        x_vec     = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                         perm=[0, 3, 1, 2]), axis=1)

        h_eff         = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y             = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr           = self.demapper(x_hat, no_eff)
        b_hat         = self.decoder(llr)

        return b, b_hat, h_eff, no

# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate(system, snr_range, name, flops, weights_c, acts,
             is_classical=True, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):

    energy_fp32 = compute_energy_uJ(flops, weights_c, acts, 32, 32)
    energy_fp16 = compute_energy_uJ(flops, weights_c, acts, 16, 16)
    energy_plot = energy_fp32 if is_classical else energy_fp16

    print(f'\n📊 {name}  |  FLOPs: {flops/1e6:.1f}M  |  '
          f'Energy ({"FP32" if is_classical else "FP16"}): {energy_plot:.4f} µJ')

    results = {
        'sum_rate'    : [],
        'per_user'    : [],
        'flops_M'     : flops     / 1e6,
        'params_K'    : weights_c / 1e3,
        'energy_fp32' : energy_fp32,
        'energy_fp16' : energy_fp16,
        'energy_plot' : energy_plot,
        'is_classical': is_classical,
    }

    for snr in snr_range:
        rates_b    = []
        per_user_b = []
        for _ in range(num_batches):
            system.new_topology(batch_size)
            _, _, h_eff, no = system(
                tf.constant(batch_size, dtype=tf.int32),
                tf.constant(float(snr),  dtype=tf.float32))

            sinr = tf.squeeze(
                system.lmmse_sinr(h_eff, no=no, interference_whitening=True),
                axis=-1)
            rate_sc   = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
            rate_user = tf.reduce_mean(rate_sc, axis=[0, 1, 2])
            rates_b.append(float(tf.reduce_sum(rate_user).numpy()))
            per_user_b.append(rate_user.numpy())

        results['sum_rate'].append(float(np.mean(rates_b)))
        results['per_user'].append(np.mean(per_user_b, axis=0))
        print(f'  Eb/N0={snr:4.1f} dB | Rate: {results["sum_rate"][-1]:6.2f} bps/Hz')

    return results

# =============================================================================
# STYLES VISUELS
# =============================================================================

# PalettePublication-quality
STYLE = {
    'WMMSE' : dict(color='#000000', marker='s', ls='-.', lw=2.4, ms=7,
                   label='WMMSE (upper bound)'),
    'RZF'   : dict(color='#1f77b4', marker='o', ls='--', lw=2.2, ms=7,
                   label='RZF (per-SC, α=σ²)'),
}

# V4 : nuances de rouge/orange/vert/violet/brun — séquentiels
_V4_PALETTE = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
_V4_MARKERS = ['v', '^', 'D', 'p', '*']

# V5 : nuances de cyan/teal
_V5_PALETTE = ['#17becf', '#1f9e9e', '#006d6d']
_V5_MARKERS = ['h', 'H', '8']

# RZF groupé : gradient de bleus clairs
_RZF_GRP_PALETTE = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
_RZF_GRP_MARKERS = ['x', 'x', 'x', 'x']


def get_style(name, v4_names, v5_names, rzf_grp_names):
    if name in STYLE:
        return STYLE[name]
    if name in v4_names:
        i = v4_names.index(name)
        import re
        m  = re.search(r'(\d+) tok/RB', name)
        n  = m.group(1) if m else str(i+1)
        sc = 12 // int(n) if n.isdigit() else '?'
        return dict(color=_V4_PALETTE[i % len(_V4_PALETTE)],
                    marker=_V4_MARKERS[i % len(_V4_MARKERS)],
                    ls='-', lw=1.8, ms=7,
                    label=f'V4.2  {n} tok/RB  ({sc} SC/tok)')
    if name in v5_names:
        i = v5_names.index(name)
        return dict(color=_V5_PALETTE[i % len(_V5_PALETTE)],
                    marker=_V5_MARKERS[i % len(_V5_MARKERS)],
                    ls='-', lw=1.8, ms=7,
                    label=name)
    if name in rzf_grp_names:
        i = rzf_grp_names.index(name)
        return dict(color=_RZF_GRP_PALETTE[i % len(_RZF_GRP_PALETTE)],
                    marker='x', ls=':', lw=1.4, ms=7,
                    label=name)
    return dict(color='gray', marker='.', ls='-', lw=1.5, ms=6, label=name)

# =============================================================================
# FIGURES
# =============================================================================

def plot_results(results_all, snr_range, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    idx_15 = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))

    v4_names      = [c['name'] for c in TRAINED_MODELS_V4 if c['name'] in results_all]
    v5_names      = [c['name'] for c in TRAINED_MODELS_V5 if c['name'] in results_all]
    rzf_grp_names = [label for _, label in RZF_GROUP_CONFIGS if label in results_all]

    def sty(n): return get_style(n, v4_names, v5_names, rzf_grp_names)

    wmmse_rate   = results_all.get('WMMSE', {}).get('sum_rate', [0]*20)[idx_15]
    wmmse_e_fp32 = results_all.get('WMMSE', {}).get('energy_fp32', 3.48)

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 1 — Sum Rate : V4 ablation + WMMSE + RZF
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        f'TransformerPrecoderV4.2 — Ablation tok/RB\n'
        f'{NUM_TX}×{NUM_RX} MU-MIMO | 8 TX, 4 UE, 72 SC, RB=12SC',
        fontsize=11, fontweight='bold')

    for n in ['WMMSE', 'RZF'] + v4_names:
        if n not in results_all: continue
        s = sty(n)
        ax.plot(snr_range, results_all[n]['sum_rate'],
                color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], ms=s['ms'], label=s['label'])

    ax.set_xlabel('Eb/N0 (dB)', fontsize=11)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.92)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(snr_range[0] - 0.3, snr_range[-1] + 0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    p = f'{save_dir}/fig1_v4_sumrate_{ts}.png'
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Fig1: {p}')

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 2 — Sum Rate : V5 + WMMSE + RZF
    # ──────────────────────────────────────────────────────────────────────────
    if v5_names:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.suptitle(
            f'TransformerPrecoderV5 vs Baselines\n'
            f'{NUM_TX}×{NUM_RX} MU-MIMO | 8 TX, 4 UE, 72 SC',
            fontsize=11, fontweight='bold')

        for n in ['WMMSE', 'RZF'] + v5_names:
            if n not in results_all: continue
            s = sty(n)
            ax.plot(snr_range, results_all[n]['sum_rate'],
                    color=s['color'], marker=s['marker'], ls=s['ls'],
                    lw=s['lw'], ms=s['ms'], label=s['label'])

        ax.set_xlabel('Eb/N0 (dB)', fontsize=11)
        ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.92)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(snr_range[0] - 0.3, snr_range[-1] + 0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        p = f'{save_dir}/fig2_v5_sumrate_{ts}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ Fig2: {p}')

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 3 — Sum Rate : V4 + V5 ensemble (comparaison directe)
    # ──────────────────────────────────────────────────────────────────────────
    if v5_names:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.suptitle(
            f'V4.2 vs V5 vs Baselines — {NUM_TX}×{NUM_RX} MU-MIMO',
            fontsize=11, fontweight='bold')

        for n in ['WMMSE', 'RZF'] + v4_names + v5_names:
            if n not in results_all: continue
            s = sty(n)
            ax.plot(snr_range, results_all[n]['sum_rate'],
                    color=s['color'], marker=s['marker'], ls=s['ls'],
                    lw=s['lw'], ms=s['ms'], label=s['label'])

        ax.set_xlabel('Eb/N0 (dB)', fontsize=11)
        ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.92,
                  ncol=2 if len(v4_names) + len(v5_names) > 5 else 1)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(snr_range[0] - 0.3, snr_range[-1] + 0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        p = f'{save_dir}/fig3_v4v5_sumrate_{ts}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ Fig3: {p}')

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 4 — Sum Rate : V4 + RZF groupé (contexte complexité)
    # ──────────────────────────────────────────────────────────────────────────
    if rzf_grp_names:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.suptitle(
            f'V4.2 vs RZF Grouping — {NUM_TX}×{NUM_RX} MU-MIMO',
            fontsize=11, fontweight='bold')

        for n in ['WMMSE', 'RZF'] + rzf_grp_names + v4_names:
            if n not in results_all: continue
            s = sty(n)
            ax.plot(snr_range, results_all[n]['sum_rate'],
                    color=s['color'], marker=s['marker'], ls=s['ls'],
                    lw=s['lw'], ms=s['ms'], label=s['label'])

        ax.set_xlabel('Eb/N0 (dB)', fontsize=11)
        ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.92, ncol=2)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(snr_range[0] - 0.3, snr_range[-1] + 0.3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        p = f'{save_dir}/fig4_v4_rzf_sumrate_{ts}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ Fig4: {p}')

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 5 — Pareto Énergie @ TRAINING_SNR (V4 + V5 + WMMSE, pas RZF)
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        f'Energy–Performance Pareto @ {TRAINING_SNR:.0f} dB Eb/N0\n'
        f'{NUM_TX}×{NUM_RX} MU-MIMO  [Transformers: FP16 | WMMSE: FP32]',
        fontsize=11, fontweight='bold')

    pareto_keys = ['WMMSE'] + v4_names + v5_names   # RZF exclu
    for n in pareto_keys:
        if n not in results_all: continue
        res  = results_all[n]
        rate = res['sum_rate'][idx_15]
        e    = res['energy_plot']
        s    = sty(n)
        ax.scatter(e, rate, s=260, marker=s['marker'], color=s['color'],
                   label=s['label'], zorder=4, edgecolors='k', linewidths=0.8)
        ax.annotate(s['label'].split('(')[0].strip(),
                    (e, rate), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)

    ax.set_xlabel('Inference Energy (µJ)', fontsize=11)
    ax.set_ylabel(f'Sum Rate @ {TRAINING_SNR:.0f} dB (bps/Hz)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.92)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    p = f'{save_dir}/fig5_pareto_energy_{ts}.png'
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Fig5: {p}')

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 6 — Pareto FLOPs @ TRAINING_SNR (V4 + RZF groupé, log scale)
    # ──────────────────────────────────────────────────────────────────────────
    if rzf_grp_names:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.suptitle(
            f'Complexity–Performance Pareto @ {TRAINING_SNR:.0f} dB Eb/N0\n'
            f'V4.2 vs RZF Grouping — {NUM_TX}×{NUM_RX} MU-MIMO',
            fontsize=11, fontweight='bold')

        flop_keys = ['WMMSE', 'RZF'] + rzf_grp_names + v4_names
        for n in flop_keys:
            if n not in results_all: continue
            res  = results_all[n]
            rate = res['sum_rate'][idx_15]
            f_m  = res['flops_M']
            s    = sty(n)
            ax.scatter(f_m, rate, s=220, marker=s['marker'], color=s['color'],
                       label=s['label'], zorder=4,
                       edgecolors='k', linewidths=0.7)
            short = s['label'].split('(')[0].strip()
            ax.annotate(short, (f_m, rate), textcoords='offset points',
                        xytext=(5, 3), fontsize=7.5)

        ax.set_xscale('log')
        ax.set_xlabel('FLOPs (M, log scale)', fontsize=11)
        ax.set_ylabel(f'Sum Rate @ {TRAINING_SNR:.0f} dB (bps/Hz)', fontsize=11)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.92)
        ax.grid(alpha=0.3, linestyle='--', which='both')
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        p = f'{save_dir}/fig6_pareto_flops_{ts}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'✅ Fig6: {p}')

# =============================================================================
# TABLEAU CONSOLE
# =============================================================================

def print_table(results_all, snr_range):
    idx          = int(np.argmin(np.abs(snr_range - TRAINING_SNR)))
    wmmse_rate   = results_all.get('WMMSE', {}).get('sum_rate', [0]*20)[idx]
    wmmse_e_fp32 = results_all.get('WMMSE', {}).get('energy_fp32', 3.48)

    print(f"\n{'='*105}")
    print(f"  PARETO @ Eb/N0={TRAINING_SNR:.0f} dB  |  WMMSE = {wmmse_rate:.2f} bps/Hz")
    print(f"{'='*105}")
    print(f"{'Method':<32} | {'Rate':>7} | {'% WMMSE':>8} | "
          f"{'FLOPs(M)':>9} | {'FP32(µJ)':>9} | "
          f"{'FP16(µJ)':>9} | {'Gain vs WMMSE':>14}")
    print('─' * 105)

    for name, res in sorted(results_all.items(),
                             key=lambda x: x[1]['sum_rate'][idx], reverse=True):
        rate = res['sum_rate'][idx]
        pct  = 100.0 * rate / wmmse_rate if wmmse_rate > 0 else 0.0
        gain = (wmmse_e_fp32 / res['energy_fp16']
                if (res['energy_fp16'] > 0 and not res['is_classical']) else None)
        fp16 = f"{res['energy_fp16']:.4f}" if not res['is_classical'] else '  N/A  '
        gs   = f'{gain:.1f}×' if gain else '  ref  '
        print(f"{name:<32} | {rate:>7.2f} | {pct:>7.1f}% | "
              f"{res['flops_M']:>9.1f} | {res['energy_fp32']:>9.4f} | "
              f"{fp16:>9} | {gs:>14}")
    print('=' * 105)

# =============================================================================
# MAIN
# =============================================================================

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✅ GPU: {gpus[0]}\n')

    print('=' * 70)
    print(f'  EVALUATION ONLY — V4.2 + V5 + Baselines')
    print(f'  {NUM_TX}×{NUM_RX} MIMO | {NUM_BATCHES} batches × {BATCH_SIZE} samples')
    print(f'  Convention : Eb/N0 via ebnodb2no')
    print('=' * 70)

    results_all = {}

    # ── 1. RZF per-SC ────────────────────────────────────────────────────
    print('\n── RZF (per-SC) ─────────────────────────────────────')
    sys_ = MU_MIMO_System(precoder_type='rzf')
    f, w, a = classical_complexity('RZF')
    results_all['RZF'] = evaluate(sys_, EVALUATION_SNR_RANGE, 'RZF',
                                   f, w, a, is_classical=True)

    # ── 2. WMMSE ─────────────────────────────────────────────────────────
    print('\n── WMMSE ────────────────────────────────────────────')
    sys_ = MU_MIMO_System(precoder_type='wmmse')
    f, w, a = classical_complexity('WMMSE')
    results_all['WMMSE'] = evaluate(sys_, EVALUATION_SNR_RANGE, 'WMMSE',
                                     f, w, a, is_classical=True)

    # ── 3. RZF groupé ────────────────────────────────────────────────────
    print('\n── RZF Groupé ───────────────────────────────────────')
    for n_groups, label in RZF_GROUP_CONFIGS:
        sys_ = MU_MIMO_System(precoder_type='rzf_grouped', rzf_n_groups=n_groups)
        f, w, a = rzf_grouped_complexity(n_groups)
        results_all[label] = evaluate(sys_, EVALUATION_SNR_RANGE, label,
                                       f, w, a, is_classical=True)

    # ── 4. V4.2 ──────────────────────────────────────────────────────────
    print('\n── Transformers V4.2 ────────────────────────────────')
    for cfg in TRAINED_MODELS_V4:
        print(f'\n  → {cfg["name"]}')
        sys_ = MU_MIMO_System(
            precoder_type='transformer_v4',
            rb_size=12,
            tokens_per_rb=cfg['tokens_per_rb'],
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            version=cfg['version'])

        dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 72], dtype=tf.complex64)
        _ = sys_.precoder(dummy, training=False)
        sys_.load_weights_from_pkl(cfg['weights_path'])

        f, w, a = sys_.precoder.complexity(num_ofdm=14)
        results_all[cfg['name']] = evaluate(
            sys_, EVALUATION_SNR_RANGE, cfg['name'],
            f, w, a, is_classical=False)

    # ── 5. V5 ────────────────────────────────────────────────────────────
    print('\n── Transformers V5 ──────────────────────────────────')
    for cfg in TRAINED_MODELS_V5:
        print(f'\n  → {cfg["name"]}')
        sys_ = MU_MIMO_System(
            precoder_type='transformer_v5',
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_intra_layers=cfg['num_intra_layers'],
            num_inter_layers=cfg['num_inter_layers'])

        dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, 72], dtype=tf.complex64)
        _ = sys_.precoder(dummy, training=False)
        sys_.load_weights_from_pkl(cfg['weights_path'])

        f, w, a = sys_.precoder.complexity(num_ofdm=14)
        results_all[cfg['name']] = evaluate(
            sys_, EVALUATION_SNR_RANGE, cfg['name'],
            f, w, a, is_classical=False)

    # ── 6. Figures & tableau ─────────────────────────────────────────────
    plot_results(results_all, EVALUATION_SNR_RANGE, save_dir='./results')
    print_table(results_all, EVALUATION_SNR_RANGE)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'./results/eval_results_{ts}.npy', results_all)
    print(f'\n✅ Results saved: ./results/eval_results_{ts}.npy')


if __name__ == '__main__':
    main()