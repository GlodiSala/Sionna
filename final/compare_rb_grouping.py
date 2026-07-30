

#weights_path = '/users/sala/test_projet/Trans/freq/Sionna/training_weights_transformer_rb/best_20260115_023858/weights.pkl'
#weights_path = '/users/sala/test_projet/Trans/freq/Sionna/training_weights_transformer_rb/best_20260205_075840/weights.pkl'
#weights_path = '/users/sala/test_projet/Trans/freq/Sionna/training_weights_transformer_rb/best_20260205_082803/weights.pkl'
#weights_path ='/export/tmp/sala/test_projet/Trans/freq/Sionna/training_weights_transformer_rb/best_20260316_122106/weights.pkl'

"""
Comparison: RB Grouping Impact on RZF and Trained Transformer
SNR sweep: 0-25 dB + Energy analysis (FP32 / FP16 / INT8)
"""

import os
import math
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from functools import partial

from precoders_w import rzf_precoder, wmmse_precoder, TransformerPrecoderV4
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, RemoveNulledSubcarriers,
                              ResourceGridMapper, LMMSEEqualizer,
                              LMMSEPostEqualizationSINR)
from sionna.phy.channel import (ApplyOFDMChannel, subcarrier_frequencies,
                                 cir_to_ofdm_channel)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED        = 42
NUM_TX      = 8
NUM_RX      = 4
FFT_SIZE    = 72
NUM_OFDM    = 14
BATCH_SIZE  = 256
NUM_BATCHES = 30
SNR_RANGE   = np.arange(0, 26, 5)
RB_SIZES    = [2, 4, 6, 12, 24, 72]

WEIGHTS_PATH = ('/export/tmp/sala/test_projet/Trans/freq/Sionna/training_weights_transformer_rb/best_20260316_163253/weights.pkl')

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# ENERGY MODEL
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
    Défaut FP32 — appeler avec Q_W=16/Q_A=16 pour FP16, Q_W=8/Q_A=8 pour INT8.
    """
    MACs = FLOPs / 2.0
    EMAC, EM,   EL   = energy_constants(Q_W)
    _,    EM_A, EL_A = energy_constants(Q_A)
    sqrt_p_W = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A = math.sqrt(64.0 * (Q_A / 16.0))
    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)
    return (EC + EW + EA) / 1e9


# =============================================================================
# COMPLEXITÉ
# =============================================================================

def compute_rzf_flops(M, K, N_SC, N_OFDM):
    """FLOPs RZF full frequency — formule labo."""
    f_sc = 7.0 * (2.0/3.0 * K**3 + 2.0 * K**2 * M) + K
    acts = 2.0 * (K*M + 2.0*K**2 + M*K)
    return f_sc * N_SC * N_OFDM, 0.0, acts * N_SC * N_OFDM


def compute_rzf_rb_flops(M, K, N_SC, N_OFDM, rb_size):
    """
    FLOPs RZF avec RB grouping.
    num_rb inversions au lieu de N_SC — tf.repeat = zéro FLOPs.
    """
    num_rb = N_SC // rb_size
    f_rb   = (7.0 * (2.0/3.0 * K**3 + 2.0 * K**2 * M) + K) * num_rb
    acts   = 2.0 * (K*M + 2.0*K**2 + M*K) * num_rb
    return f_rb * N_OFDM, 0.0, acts * N_OFDM


def compute_wmmse_flops(M, K, N_SC, N_OFDM, I=10):
    """FLOPs WMMSE — formule labo."""
    per_iter = (
          (14.0/3.0) * K * M**3
        + 12.0 * K**2 * M**2
        + 12.0 * K**2 * M
        +  9.0 * K   * M**2
        +  8.0 * K   * M
        +  5.0 * K**2
        + (68.0/3.0) * K
    )
    f_sc = 7.0 * I * per_iter
    acts = 2.0 * (K*M + 2.0*K**2 + M*K)
    return f_sc * N_SC * N_OFDM, 0.0, acts * N_SC * N_OFDM


def get_complexity(name, transformer=None):
    """Interface unifiée — retourne (FLOPs, Weights, Activations)."""
    M, K = NUM_TX, NUM_RX

    if name == 'RZF Full (72 SC)':
        return compute_rzf_flops(M, K, FFT_SIZE, NUM_OFDM)

    if name.startswith('RZF RB='):
        rb = int(name.split('=')[1])
        return compute_rzf_rb_flops(M, K, FFT_SIZE, NUM_OFDM, rb)

    if name == 'WMMSE':
        return compute_wmmse_flops(M, K, FFT_SIZE, NUM_OFDM)

    if 'Transformer' in name and transformer is not None:
        if hasattr(transformer, 'complexity'):
            return transformer.complexity(NUM_OFDM)

    return None


# =============================================================================
# RZF WITH RB GROUPING
# =============================================================================

def rzf_precoder_with_rb_grouping(h_freq, stream_management, rb_size=12, alpha=0.1):
    """RZF avec précoder constant par RB (canal moyenné intra-RB)."""
    if rb_size >= FFT_SIZE:
        return rzf_precoder(h_freq, stream_management=stream_management, alpha=alpha)

    B        = tf.shape(h_freq)[0]
    num_rx   = tf.shape(h_freq)[1]
    num_tx   = tf.shape(h_freq)[4]
    num_ofdm = tf.shape(h_freq)[5]
    num_rb   = FFT_SIZE // rb_size

    h_sq   = tf.squeeze(h_freq, axis=[2, 3])
    h_rb   = tf.reshape(h_sq, [B, num_rx, num_tx, num_ofdm, num_rb, rb_size])
    h_avg  = tf.reduce_mean(h_rb, axis=-1)
    h_avg  = tf.transpose(h_avg, [0, 3, 4, 1, 2])
    h_flat = tf.reshape(h_avg, [-1, num_rx, num_tx])

    HHH     = tf.matmul(h_flat, h_flat, adjoint_b=True)
    eye     = tf.eye(num_rx, dtype=HHH.dtype)
    HHH_inv = tf.linalg.inv(HHH + alpha * eye)
    W_rb    = tf.matmul(h_flat, HHH_inv, adjoint_a=True)
    W_rb    = tf.reshape(W_rb, [B, num_ofdm, num_rb, num_tx, num_rx])

    W_full = tf.reshape(
        tf.repeat(tf.expand_dims(W_rb, axis=3), repeats=rb_size, axis=3),
        [B, num_ofdm, FFT_SIZE, num_tx, num_rx])

    pwr   = tf.reduce_sum(tf.abs(W_full)**2, axis=[-2, -1], keepdims=True)
    scale = tf.cast(
        tf.sqrt(tf.cast(num_rx, tf.float32) / (tf.cast(pwr, tf.float32) + 1e-12)),
        W_full.dtype)

    return tf.expand_dims(W_full * scale, axis=1)


# =============================================================================
# SYSTÈME MU-MIMO MINIMAL
# =============================================================================

class MinimalMIMOSystem(tf.keras.Model):

    def __init__(self, num_tx=8, num_rx=4):
        super().__init__()
        self.num_tx              = num_tx
        self.num_rx              = num_rx
        self.num_bits_per_symbol = 2

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
            subcarrier_spacing=30e3, num_tx=1,
            num_streams_per_tx=num_rx, cyclic_prefix_length=6,
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
        self.lmmse_sinr    = LMMSEPostEqualizationSINR(self.rg, self.sm)
        self.remove_nulled = RemoveNulledSubcarriers(self.rg)

        from sionna.phy.ofdm import RZFPrecodedChannel
        self.precoded_channel_helper = RZFPrecodedChannel(self.rg, self.sm)

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, self.num_rx, "umi")
        self.channel_model.set_topology(*topology)

    def _forward(self, batch_size, snr_db, g, h_freq, no):
        """Partie commune forward après le précoder."""
        W          = tf.squeeze(g, axis=1)
        x_vec      = tf.transpose(
            self.rg_mapper(self.mapper(self.encoder(
                self.binary_source([batch_size, 1, self.num_rx,
                                    int(self.rg.num_data_symbols)])))),
            perm=[0, 3, 4, 2, 1])

        b    = self.binary_source([batch_size, 1, self.num_rx,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)
        x_v  = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])

        x_precoded = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_v), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff         = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y             = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr           = self.demapper(x_hat, no_eff)
        b_hat         = self.decoder(llr)

        ber      = compute_ber(b, b_hat)
        sinr     = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sum_rate = tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(
                tf.math.log(1.0 + sinr) / tf.math.log(2.0),
                axis=[1, 2, 4]),
            axis=0))
        return sum_rate, ber

    @tf.function
    def evaluate_step(self, batch_size, snr_db, precoder_fn):
        """Évaluation générique — RZF et Transformer."""
        no   = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)
        b    = self.binary_source([batch_size, 1, self.num_rx,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)

        cir    = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                    1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled(h_freq)

        g = precoder_fn(h_freq)

        W          = tf.squeeze(g, axis=1)
        x_vec      = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff         = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y             = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr           = self.demapper(x_hat, no_eff)
        b_hat         = self.decoder(llr)

        ber      = compute_ber(b, b_hat)
        sinr     = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sum_rate = tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(
                tf.math.log(1.0 + sinr) / tf.math.log(2.0),
                axis=[1, 2, 4]),
            axis=0))
        return sum_rate, ber

    @tf.function
    def evaluate_step_wmmse(self, batch_size, snr_db):
        """
        Évaluation WMMSE avec no calculé dynamiquement au bon SNR.
        Séparé de evaluate_step car WMMSE a besoin de no en entrée.
        """
        no   = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)
        b    = self.binary_source([batch_size, 1, self.num_rx,
                                   int(self.rg.num_data_symbols)])
        c    = self.encoder(b)
        x    = self.mapper(c)
        x_rg = self.rg_mapper(x)

        cir    = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                    1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled(h_freq)

        # WMMSE reçoit le no correspondant au SNR courant
        g = wmmse_precoder(h_freq, no=no,
                           stream_management=self.sm,
                           num_iterations=10)

        W          = tf.squeeze(g, axis=1)
        x_vec      = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_precoded = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff         = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        y             = self.channel_freq(x_precoded, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr           = self.demapper(x_hat, no_eff)
        b_hat         = self.decoder(llr)

        ber      = compute_ber(b, b_hat)
        sinr     = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sum_rate = tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(
                tf.math.log(1.0 + sinr) / tf.math.log(2.0),
                axis=[1, 2, 4]),
            axis=0))
        return sum_rate, ber


# =============================================================================
# ÉVALUATION SNR SWEEP
# =============================================================================

def evaluate_snr_sweep(system, precoder_fn, name, snr_range,
                       num_batches=NUM_BATCHES):
    """Évaluation générique — RZF et Transformer."""
    print(f"\n📊 {name}")
    print("-" * 60)

    results = {'snr': [], 'rate': [], 'ber': []}
    bs = tf.constant(BATCH_SIZE, dtype=tf.int32)

    for snr in snr_range:
        rates, bers = [], []
        snr_t = tf.constant(float(snr), dtype=tf.float32)

        for _ in range(num_batches):
            system.new_topology(BATCH_SIZE)
            r, b = system.evaluate_step(bs, snr_t, precoder_fn)
            rates.append(float(r))
            bers.append(float(b))

        avg_r = float(np.mean(rates))
        avg_b = float(np.mean(bers))
        results['snr'].append(snr)
        results['rate'].append(avg_r)
        results['ber'].append(avg_b)
        print(f"  SNR={snr:2.0f} dB | Rate={avg_r:6.2f} bps/Hz | BER={avg_b:.2e}")

    return results


def evaluate_snr_sweep_wmmse(system, name, snr_range,
                              num_batches=NUM_BATCHES):
    """
    Évaluation WMMSE — no calculé dynamiquement par SNR.
    WMMSE utilise no pour optimiser le précoder → crucial d'avoir
    le bon no à chaque SNR, sinon les résultats à bas SNR sont faux.
    """
    print(f"\n📊 {name}")
    print("-" * 60)

    results = {'snr': [], 'rate': [], 'ber': []}
    bs = tf.constant(BATCH_SIZE, dtype=tf.int32)

    for snr in snr_range:
        rates, bers = [], []
        snr_t = tf.constant(float(snr), dtype=tf.float32)

        for _ in range(num_batches):
            system.new_topology(BATCH_SIZE)
            r, b = system.evaluate_step_wmmse(bs, snr_t)
            rates.append(float(r))
            bers.append(float(b))

        avg_r = float(np.mean(rates))
        avg_b = float(np.mean(bers))
        results['snr'].append(snr)
        results['rate'].append(avg_r)
        results['ber'].append(avg_b)
        print(f"  SNR={snr:2.0f} dB | Rate={avg_r:6.2f} bps/Hz | BER={avg_b:.2e}")

    return results


# =============================================================================
# FIGURES
# =============================================================================

def plot_results(all_results, complexity_map, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    idx = list(SNR_RANGE).index(15)

    n       = len(all_results)
    colors  = plt.cm.tab10(np.linspace(0, 1, n))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<']

    def style(name):
        if 'Transformer' in name: return '-',  2.5
        if 'WMMSE'       in name: return '-.', 2.5
        if 'Full'        in name: return '--', 2.0
        return ':', 1.5

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'RB Grouping + Transformer — {NUM_TX}×{NUM_RX} MIMO',
                 fontsize=14, fontweight='bold')

    # (a) Sum rate
    ax = axes[0]
    for (name, res), color, mk in zip(all_results.items(), colors, markers):
        ls, lw = style(name)
        ax.plot(res['snr'], res['rate'], marker=mk, color=color,
                label=name, linewidth=lw, linestyle=ls, markersize=6)
    ax.set(xlabel='SNR (dB)', ylabel='Sum Rate (bps/Hz)',
           title='(a) Spectral Efficiency')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) BER
    ax = axes[1]
    for (name, res), color, mk in zip(all_results.items(), colors, markers):
        ls, lw = style(name)
        ax.semilogy(res['snr'], [max(b, 1e-7) for b in res['ber']],
                    marker=mk, color=color, label=name,
                    linewidth=lw, linestyle=ls, markersize=6)
    ax.set(xlabel='SNR (dB)', ylabel='BER', title='(b) BER')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')

    # (c) Pareto FLOPs vs Rate @ 15 dB
    ax = axes[2]
    for (name, res), color, mk in zip(all_results.items(), colors, markers):
        c = complexity_map.get(name)
        if c is None:
            continue
        f, w, a = c
        rate    = res['rate'][idx]
        ax.scatter(f / 1e6, rate, s=200, marker=mk, color=color,
                   label=name, zorder=3, edgecolors='k', linewidth=0.8)
        ax.annotate(name, (f / 1e6, rate),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)
    ax.set(xlabel='FLOPs (M)', ylabel='Sum Rate @ 15 dB (bps/Hz)',
           title='(c) Complexity–Performance Pareto', xscale='log')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{save_dir}/rb_grouping_{ts}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✅ Figure: {path}")
    plt.close()


# =============================================================================
# TABLEAUX RÉSUMÉ
# =============================================================================

def print_summary(all_results, complexity_map):
    idx      = list(SNR_RANGE).index(15)
    rzf_rate = all_results.get('RZF Full (72 SC)', {}).get(
        'rate', [0]*len(SNR_RANGE))[idx]

    # ── Performance ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  PERFORMANCE @ 15 dB  ({NUM_TX}×{NUM_RX} MIMO)")
    print(f"{'='*90}")
    print(f"{'Method':<35} {'Rate':>8} {'vs RZF Full':>12} {'BER':>12}")
    print("-" * 70)
    for name, res in all_results.items():
        rate = res['rate'][idx]
        ber  = res['ber'][idx]
        gap  = rate - rzf_rate
        print(f"{name:<35} {rate:>8.2f} {gap:>+12.2f} {ber:>12.2e}")

    # ── Énergie ───────────────────────────────────────────────────────
    #
    # Politique de quantification :
    #   RZF / WMMSE  : inversion matricielle → FP32 obligatoire
    #                  FP16/INT8 = N/A (divergence numérique garantie)
    #   Transformer  : poids appris → quantifiable
    #                  FP32 (training référence)
    #                  FP16 (inférence GPU/NPU standard)
    #                  INT8 (déploiement FPGA/ASIC embarqué)
    #
    print(f"\n{'='*115}")
    print(f"  ENERGY TABLE  ({NUM_TX}×{NUM_RX} MIMO)")
    print(f"  RZF/WMMSE : FP32 only (matrix inversion cannot be quantized)")
    print(f"  Transformer: FP32 / FP16 / INT8 (learned weights)")
    print(f"{'='*115}")
    print(f"{'Method':<35} {'FLOPs(M)':>10} {'Params(K)':>10} {'Rate@15':>8} "
          f"{'FP32(µJ)':>10} {'FP16(µJ)':>10} {'INT8(µJ)':>10}")
    print("-" * 115)

    for name, res in all_results.items():
        c = complexity_map.get(name)
        if c is None:
            continue
        f, w, a  = c
        rate     = res['rate'][idx]
        is_class = ('RZF' in name) or ('WMMSE' in name)

        e_fp32 = compute_energy_uJ(f, w, a, 32, 32)

        if is_class:
            fp16_s = "   N/A   "
            int8_s = "   N/A   "
        else:
            fp16_s = f"{compute_energy_uJ(f, w, a, 16, 16):>10.4f}"
            int8_s = f"{compute_energy_uJ(f, w, a,  8,  8):>10.4f}"

        print(f"{name:<35} {f/1e6:>10.1f} {w/1e3:>10.1f} {rate:>8.2f} "
              f"{e_fp32:>10.4f} {fp16_s:>10} {int8_s:>10}")

    # ── Pareto énergie/performance ─────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ENERGY–PERFORMANCE PARETO @ 15 dB")
    print(f"  Transformers → FP16  |  RZF/WMMSE → FP32")
    print(f"{'='*90}")
    print(f"{'Method':<35} {'Rate':>8} {'Energy(µJ)':>12} {'Precision':>10} {'µJ/bps':>10}")
    print("-" * 78)

    rows = []
    for name, res in all_results.items():
        c = complexity_map.get(name)
        if c is None:
            continue
        f, w, a  = c
        rate     = res['rate'][idx]
        is_class = ('RZF' in name) or ('WMMSE' in name)

        if is_class:
            e    = compute_energy_uJ(f, w, a, 32, 32)
            prec = 'FP32'
        else:
            e    = compute_energy_uJ(f, w, a, 16, 16)
            prec = 'FP16'

        rows.append((name, rate, e, prec, f / 1e6))

    for name, rate, e, prec, flops_m in sorted(rows, key=lambda x: -x[1]):
        eff = e / rate if rate > 0 else float('inf')
        print(f"{name:<35} {rate:>8.2f} {e:>12.4f} {prec:>10} {eff:>10.4f}")

    # ── Impact du grouping RB ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  RB GROUPING IMPACT @ 15 dB  (baseline RZF Full = {rzf_rate:.2f} bps/Hz)")
    print(f"{'='*80}")
    f_full, _, _ = compute_rzf_flops(NUM_TX, NUM_RX, FFT_SIZE, NUM_OFDM)

    for rb in [2, 4, 6, 12, 24]:
        key = f'RZF RB={rb}'
        if key not in all_results:
            continue
        rate    = all_results[key]['rate'][idx]
        gap     = rate - rzf_rate
        pct     = gap / rzf_rate * 100
        f, _, _ = complexity_map[key]
        sx      = f_full / f if f > 0 else float('inf')
        status  = "✅" if abs(pct) < 2 else "⚠️" if abs(pct) < 5 else "❌"
        print(f"  RB={rb:2d} | {rate:5.2f} bps/Hz ({gap:+.2f}, {pct:+.1f}%) "
              f"| FLOPs ÷{sx:.1f}× {status}")

    print(f"{'='*80}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("  🔬 RB GROUPING + ENERGY ANALYSIS")
    print(f"  {NUM_TX}×{NUM_RX} MIMO | {FFT_SIZE} SC | "
          f"SNR {SNR_RANGE[0]}–{SNR_RANGE[-1]} dB")
    print("="*80 + "\n")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU: {gpus[0]}\n")

    system = MinimalMIMOSystem(num_tx=NUM_TX, num_rx=NUM_RX)

    all_results    = {}
    complexity_map = {}
    transformer    = None

    # ── 1. RZF avec différents RB sizes ──────────────────────────────
    print("="*80)
    print("  PART 1 : RZF with RB Grouping")
    print("="*80)

    for rb_size in RB_SIZES:
        if rb_size >= FFT_SIZE:
            name        = 'RZF Full (72 SC)'
            precoder_fn = partial(rzf_precoder,
                                  stream_management=system.sm, alpha=0.1)
        else:
            name        = f'RZF RB={rb_size}'
            precoder_fn = partial(rzf_precoder_with_rb_grouping,
                                  stream_management=system.sm,
                                  rb_size=rb_size, alpha=0.1)

        all_results[name]    = evaluate_snr_sweep(
            system, precoder_fn, name, SNR_RANGE)
        complexity_map[name] = get_complexity(name)

    # ── 2. WMMSE ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  PART 2 : WMMSE (no dynamique par SNR)")
    print("="*80)

    all_results['WMMSE']    = evaluate_snr_sweep_wmmse(
        system, 'WMMSE', SNR_RANGE)
    complexity_map['WMMSE'] = get_complexity('WMMSE')

    # ── 3. Transformer entraîné ───────────────────────────────────────
    print("\n" + "="*80)
    print("  PART 3 : Trained Transformer")
    print("="*80)

    if os.path.exists(WEIGHTS_PATH):
        print(f"\n📂 Loading: {WEIGHTS_PATH}")

        transformer = TransformerPrecoderV4(
            num_tx=NUM_TX, num_rx=NUM_RX,
            num_ofdm=NUM_OFDM, fft_size=FFT_SIZE,
            rb_size=12, tokens_per_rb=3,
            embed_dim=128, num_heads=4, num_layers=4)

        dummy = tf.zeros(
            [1, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE],
            dtype=tf.complex64)
        _ = transformer(dummy, training=False)

        try:
            with open(WEIGHTS_PATH, 'rb') as fh:
                saved = pickle.load(fh)
            n_ok = 0
            for var, wt in zip(transformer.trainable_variables, saved):
                if var.shape == wt.shape:
                    var.assign(wt); n_ok += 1
                else:
                    print(f"  ⚠️  {var.name} {var.shape} vs {wt.shape}")
            print(f"  ✅ Loaded {n_ok}/{len(saved)} weights")

            t_name      = 'Transformer (RB=12, 1Tok)'
            precoder_fn = partial(transformer, training=False)
            all_results[t_name]    = evaluate_snr_sweep(
                system, precoder_fn, t_name, SNR_RANGE)
            complexity_map[t_name] = get_complexity(t_name, transformer)

        except Exception as e:
            print(f"  ❌ {e}")
    else:
        print(f"\n⚠️  Weights not found: {WEIGHTS_PATH}")

    # ── 4. Résumé & figures ───────────────────────────────────────────
    print_summary(all_results, complexity_map)
    plot_results(all_results, complexity_map)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('./results', exist_ok=True)
    np.save(f'./results/rb_grouping_{ts}.npy', all_results)
    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()