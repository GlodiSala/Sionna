"""
plot_sumrate_final.py
Evaluates and plots sum-rate for:
  - RZF Full (all 72 subcarriers)
  - RZF RB-12 (averaged per 12-SC resource block)
  - WMMSE (near-optimal upper bound)
  - Transformer V2, 1/2/3 Tok/RB
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------------------------------------------------------------------
# Adjust to your paths
# ---------------------------------------------------------------------------
sys.path.insert(0, '/users/sala/test_projet/Trans/freq/Sionna/final')

from precoders_w import rzf_precoder, wmmse_precoder, TransformerPrecoderV2
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, RemoveNulledSubcarriers,
                              ResourceGridMapper, LMMSEEqualizer,
                              LMMSEPostEqualizationSINR, RZFPrecodedChannel)
from sionna.phy.channel import (ApplyOFDMChannel, subcarrier_frequencies,
                                 cir_to_ofdm_channel)
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber

# ===========================================================================
# CONFIG
# ===========================================================================
NUM_TX     = 8
NUM_RX     = 4
FFT_SIZE   = 72
NUM_OFDM   = 14
BATCH_SIZE = 512
NUM_BATCHES = 80       # 80 × 512 = 40,960 samples per SNR point
SNR_RANGE  = np.array([0, 5, 10, 15, 20], dtype=np.float32)

WEIGHT_PATHS = {
    1: '/users/sala/test_projet/Trans/freq/Sionna/'
       'training_weights_transformer_rb/best_20260220_022337/weights.pkl',
    2: '/users/sala/test_projet/Trans/freq/Sionna/'
       'training_weights_transformer_rb/best_20260220_041732/weights.pkl',
    3: '/users/sala/test_projet/Trans/freq/Sionna/'
       'training_weights_transformer_rb/best_20260220_064057/weights.pkl',
}

OUTPUT_DIR = './presentation_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# MINIMAL SYSTEM  (same as your existing code, 8×4)
# ===========================================================================
class MinimalMIMOSystem(tf.keras.Model):
    def __init__(self, num_tx=8, num_rx=4):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_bits_per_symbol = 2

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association,
                                   num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
            subcarrier_spacing=30e3, num_tx=1,
            num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11])

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
        self.lmmse_sinr     = LMMSEPostEqualizationSINR(self.rg, self.sm)
        self.remove_nulled  = RemoveNulledSubcarriers(self.rg)
        self.precoded_ch    = RZFPrecodedChannel(self.rg, self.sm)

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, self.num_rx, "umi")
        self.channel_model.set_topology(*topology)

    @tf.function
    def _run(self, batch_size, snr_db, precoder_fn):
        no = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)

        cir    = self.channel_model(
            batch_size, self.rg.num_ofdm_symbols,
            1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled(h_freq)

        g = precoder_fn(h_freq)

        W               = tf.squeeze(g, axis=1)
        x_vec           = tf.transpose(
            self.rg_mapper(
                self.mapper(
                    self.encoder(
                        self.binary_source(
                            [batch_size, 1, self.num_rx,
                             int(self.rg.num_data_symbols)])))),
            perm=[0, 3, 4, 2, 1])
        x_pre           = tf.matmul(W, x_vec)
        x_pre           = tf.squeeze(x_pre, axis=-1)
        x_pre           = tf.expand_dims(
            tf.transpose(x_pre, perm=[0, 3, 1, 2]), axis=1)

        h_eff           = self.precoded_ch.compute_effective_channel(h_freq, g)
        y               = self.channel_freq(x_pre, h_freq, no)
        x_hat, no_eff   = self.lmmse_equ(y, h_eff, 0.0, no)

        sinr            = self.lmmse_sinr(
            h_eff, no=no, interference_whitening=True)
        rate_per_user   = tf.reduce_mean(
            tf.math.log(1.0 + sinr) / tf.math.log(2.0),
            axis=[1, 2, 4])                            # [B, users]
        sum_rate        = tf.reduce_sum(
            tf.reduce_mean(rate_per_user, axis=0))     # scalar
        return sum_rate

    def evaluate_snr_sweep(self, precoder_fn, label, snr_range,
                           num_batches, batch_size):
        print(f"\n  [{label}]")
        rates = []
        for snr in snr_range:
            batch_rates = []
            for _ in range(num_batches):
                self.new_topology(batch_size)
                r = self._run(
                    tf.constant(batch_size, dtype=tf.int32),
                    tf.constant(float(snr), dtype=tf.float32),
                    precoder_fn)
                batch_rates.append(float(r))
            mean_r = float(np.mean(batch_rates))
            rates.append(mean_r)
            print(f"    SNR={snr:2.0f} dB  →  {mean_r:.2f} bps/Hz")
        return np.array(rates)


# ===========================================================================
# RZF-RB12 PRECODER  (average channel within each 12-SC RB)
# ===========================================================================
def rzf_rb12_precoder(h_freq, stream_management, rb_size=12, alpha=0.1):
    """RZF applied on the per-RB averaged channel, then upsampled."""
    B          = tf.shape(h_freq)[0]
    num_rx     = NUM_RX
    num_tx_ant = NUM_TX
    num_ofdm   = NUM_OFDM
    num_rbs    = FFT_SIZE // rb_size          # 72 // 12 = 6

    h_sq  = tf.squeeze(h_freq, axis=[2, 3])  # [B, rx, tx, ofdm, fft]
    h_rb  = tf.reshape(h_sq,
                       [B, num_rx, num_tx_ant, num_ofdm, num_rbs, rb_size])
    h_avg = tf.reduce_mean(h_rb, axis=-1)    # [B, rx, tx, ofdm, num_rbs]

    # Flatten (B, ofdm, rb) → single "batch" axis for inversion
    h_perm = tf.transpose(h_avg, [0, 3, 4, 1, 2])   # [B, ofdm, rbs, rx, tx]
    h_flat = tf.reshape(h_perm, [-1, num_rx, num_tx_ant])

    HHH   = tf.matmul(h_flat, h_flat, adjoint_b=True)
    eye   = tf.eye(num_rx, dtype=HHH.dtype)
    W     = tf.matmul(h_flat,
                      tf.linalg.inv(HHH + alpha * eye),
                      adjoint_a=True)         # [B*ofdm*rbs, tx, rx]

    W_rb  = tf.reshape(W, [B, num_ofdm, num_rbs, num_tx_ant, num_rx])

    # Upsample: repeat each RB precoder rb_size times
    W_up  = tf.repeat(W_rb, rb_size, axis=2) # [B, ofdm, fft, tx, rx]

    # Power normalise: unit norm per column (per stream)
    power = tf.reduce_sum(tf.abs(W_up) ** 2, axis=3, keepdims=True)
    W_up  = W_up / tf.cast(
        tf.sqrt(power + 1e-12), W_up.dtype)

    return tf.expand_dims(W_up, axis=1)      # [B, 1, ofdm, fft, tx, rx]


# ===========================================================================
# LOAD TRANSFORMER WEIGHTS
# ===========================================================================
def load_transformer(tokens_per_rb, weights_path):
    model = TransformerPrecoderV2(
        num_tx=NUM_TX, num_rx=NUM_RX,
        num_ofdm=NUM_OFDM, fft_size=FFT_SIZE,
        rb_size=12, tokens_per_rb=tokens_per_rb,
        embed_dim=128, num_heads=4, num_layers=4)

    # Build
    dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE],
                     dtype=tf.complex64)
    _ = model(dummy, training=False)

    with open(weights_path, 'rb') as f:
        saved = pickle.load(f)

    assigned = 0
    for var, w in zip(model.trainable_variables, saved):
        if var.shape == w.shape:
            var.assign(w)
            assigned += 1
        else:
            print(f"  ⚠ shape mismatch: {var.name}  {var.shape} vs {w.shape}")

    print(f"  Loaded {assigned}/{len(saved)} weights  ← {weights_path}")
    return model


# ===========================================================================
# EVALUATE ALL METHODS
# ===========================================================================
system  = MinimalMIMOSystem(num_tx=NUM_TX, num_rx=NUM_RX)
results = {}

print("\n" + "="*70)
print("  Evaluating baselines")
print("="*70)

# RZF Full
results['rzf_full'] = system.evaluate_snr_sweep(
    partial(rzf_precoder, stream_management=system.sm, alpha=0.1),
    'RZF — all 72 subcarriers (full CSI)',
    SNR_RANGE, NUM_BATCHES, BATCH_SIZE)

# RZF RB-12
results['rzf_rb12'] = system.evaluate_snr_sweep(
    partial(rzf_rb12_precoder, stream_management=system.sm,
            rb_size=12, alpha=0.1),
    'RZF — 12-SC resource block averaging',
    SNR_RANGE, NUM_BATCHES, BATCH_SIZE)

# WMMSE
results['wmmse'] = system.evaluate_snr_sweep(
    partial(wmmse_precoder, no=ebnodb2no(
        tf.constant(15.0), 2, 0.5, system.rg),
            stream_management=system.sm, num_iterations=10),
    'WMMSE — near-optimal upper bound',
    SNR_RANGE, NUM_BATCHES, BATCH_SIZE)

print("\n" + "="*70)
print("  Evaluating Transformer V2 models")
print("="*70)

for tok, wpath in WEIGHT_PATHS.items():
    key   = f'tf_{tok}tok'
    label = f'Transformer V2 — {tok} tok/RB  (rb_size=12)'
    print(f"\n  Loading token={tok}")
    model = load_transformer(tok, wpath)
    results[key] = system.evaluate_snr_sweep(
        partial(model, training=False),
        label, SNR_RANGE, NUM_BATCHES, BATCH_SIZE)


# ===========================================================================
# PLOT
# ===========================================================================
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10.5,
    'lines.linewidth': 2.3,
    'lines.markersize': 8,
    'grid.alpha': 0.35,
})

# ---- curve definitions ----------------------------------------------------
curves = [
    # key,         legend label (clean, unambiguous),              color       marker  ls       lw
    ('wmmse',    'WMMSE  (near-optimal, iterative)',              '#2ca02c',  's',    '-',     2.8),
    ('rzf_full', 'RZF  (full 72-SC CSI per subcarrier)',         '#1f77b4',  'o',    '--',    2.1),
    ('rzf_rb12', 'RZF-RB12  (one precoder per 12-SC RB)',        '#17becf',  'P',    '--',    2.1),
    ('tf_1tok',  'Transformer V2  — 1 tok/RB  (6 tokens total)', '#d62728',  '^',    '-',     2.3),
    ('tf_2tok',  'Transformer V2  — 2 tok/RB  (12 tokens total)','#ff7f0e',  'D',    '-',     2.3),
    ('tf_3tok',  'Transformer V2  — 3 tok/RB  (18 tokens total)','#9467bd',  'v',    '-',     2.3),
]

fig, ax = plt.subplots(figsize=(9, 6.2))

for key, label, color, marker, ls, lw in curves:
    if key not in results:
        continue
    ax.plot(SNR_RANGE, results[key],
            label=label, color=color, marker=marker,
            linestyle=ls, linewidth=lw,
            markeredgecolor='white', markeredgewidth=0.7)

# Shade WMMSE ceiling
if 'wmmse' in results:
    ax.fill_between(SNR_RANGE,
                    results['wmmse'] - 0.1, results['wmmse'] + 0.1,
                    color='#2ca02c', alpha=0.12, label='_nolegend_')

ax.set_xlabel('SNR (dB)', fontweight='bold')
ax.set_ylabel('Sum Rate (bps/Hz)', fontweight='bold')
ax.set_title('8×4 MU-MIMO Downlink  —  Sum Rate vs SNR\n'
             'RB-Aggregated Transformer Precoder Ablation  (rb\\_size = 12 SC)',
             fontweight='bold', pad=12)

ax.set_xlim([-0.5, 20.5])
ax.set_ylim([0, 38])
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True, linestyle='--', alpha=0.4)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', alpha=0.2)

# Legend — single column, ordered best → worst
ax.legend(loc='upper left', framealpha=0.93, edgecolor='gray',
          borderpad=0.8, labelspacing=0.55, handlelength=2.4)

# Annotate gap to WMMSE at 20 dB for the best transformer
if 'wmmse' in results and 'tf_3tok' in results:
    snr20_idx  = np.where(SNR_RANGE == 20)[0][0]
    wmmse_top  = results['wmmse'][snr20_idx]
    tf3_top    = results['tf_3tok'][snr20_idx]
    ax.annotate(
        f'Gap to WMMSE\n@ 20 dB: {wmmse_top - tf3_top:.1f} bps/Hz',
        xy=(20, tf3_top),
        xytext=(15.5, tf3_top - 5.5),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  alpha=0.85, edgecolor='gray'))

fig.tight_layout()

png = os.path.join(OUTPUT_DIR, 'sumrate_v2_token_ablation.png')
pdf = os.path.join(OUTPUT_DIR, 'sumrate_v2_token_ablation.pdf')
fig.savefig(png, dpi=300, bbox_inches='tight')
fig.savefig(pdf,           bbox_inches='tight')
print(f"\n  PNG → {png}")
print(f"  PDF → {pdf}")


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
print("\n" + "="*70)
print("  SUMMARY TABLE")
print("="*70)

wmmse_arr = results.get('wmmse', np.ones_like(SNR_RANGE))

for snr_target in [15, 20]:
    idx = np.where(SNR_RANGE == snr_target)[0][0]
    print(f"\n  SNR = {snr_target} dB:")
    print(f"  {'Method':<52} {'bps/Hz':>8}  {'% of WMMSE':>11}")
    print("  " + "-"*74)
    for key, label, *_ in curves:
        if key not in results:
            continue
        r   = results[key][idx]
        pct = 100.0 * r / (wmmse_arr[idx] + 1e-9)
        print(f"  {label:<52} {r:>8.2f}  {pct:>10.1f}%")