"""
compare_models.py — Comparaison IntraRB vs V4 tok/RB vs baselines
Charge les poids existants et évalue sur SNR [0-25 dB]
"""

import os, math, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from functools import partial

from precoders_w import rzf_precoder, wmmse_precoder, TransformerPrecoderV4
from precoder_intra_rb import IntraRBTransformerPrecoder
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
NUM_BATCHES = 50
SNR_RANGE   = np.arange(0, 15, 2.5, dtype=np.float32)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Modèles à comparer ────────────────────────────────────────────────────────
MODELS = [
    {
        'name'       : 'V4.2 — 3 tok/RB',
        'type'       : 'v4',
        'weights'    : '/users/sala/Documents/test_projet/Trans/freq/Sionna/'
                       'training_weights_transformer_rb/'
                       'best_20260316_163253/weights.pkl',
        'model_kwargs': {
            'tokens_per_rb': 3, 'version': 'v4.2',
            'embed_dim': 128,   'num_heads': 4, 'num_layers': 4,
        },
        'color' : '#d62728',
        'marker': 'v',
        'ls'    : '-',
    },
    {
        'name'       : 'V4.3 — 4 tok/RB',
        'type'       : 'v4',
        'weights'    : '/users/sala/Documents/test_projet/Trans/freq/Sionna/'
                       'training_weights_transformer_rb/'
                       'best_20260316_122106/weights.pkl',
        'model_kwargs': {
            'tokens_per_rb': 4, 'version': 'v4.3',
            'embed_dim': 128,   'num_heads': 4, 'num_layers': 4,
        },
        'color' : '#ff7f0e',
        'marker': '^',
        'ls'    : '-',
    },
    {
        'name'       : 'IntraRB — 4L 128d (new)',
        'type'       : 'intra_rb',
        'weights'    : '/users/sala/Documents/test_projet/Trans/freq/Sionna/'
                       'final/weights/IntraRB_4L_128d/'
                       'best_20260615_132353/weights.pkl',
        'model_kwargs': {
            'embed_dim': 128, 'num_heads': 4, 'num_layers': 4,
        },
        'color' : '#2ca02c',
        'marker': 'D',
        'ls'    : '-',
    },
]

# =============================================================================
# ÉNERGIE
# =============================================================================

def energy_constants(Q):
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    EM   = 2.0 * EMAC
    EL   = EMAC
    return EMAC, EM, EL

def compute_energy_uJ(FLOPs, Weights, Activations, Q_W=32, Q_A=32):
    MACs = FLOPs / 2.0
    EMAC, EM, EL     = energy_constants(Q_W)
    _,    EM_A, EL_A = energy_constants(Q_A)
    sqrt_p_W = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A = math.sqrt(64.0 * (Q_A / 16.0))
    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)
    return (EC + EW + EA) / 1e9

def compute_classical_flops(method, M, K, N_SC, N_OFDM, I=10):
    if method == 'wmmse':
        per_iter = (
              (14.0/3.0) * K * M**3 + 12.0 * K**2 * M**2
            + 12.0 * K**2 * M + 9.0 * K * M**2
            + 8.0 * K * M + 5.0 * K**2 + (68.0/3.0) * K)
        f_sc = 7.0 * I * per_iter
    else:  # rzf
        f_sc = 7.0 * (2.0/3.0 * K**3 + 2.0 * K**2 * M) + K
    acts = 2.0 * (K*M + 2.0*K**2 + M*K)
    return f_sc * N_SC * N_OFDM, 0.0, acts * N_SC * N_OFDM

# =============================================================================
# SYSTÈME SIONNA
# =============================================================================

class EvalSystem(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.num_bits_per_symbol = 2

        self.sm = StreamManagement(
            np.ones([NUM_RX, 1]), num_streams_per_tx=NUM_RX)

        self.rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
            subcarrier_spacing=30e3, num_tx=1,
            num_streams_per_tx=NUM_RX, cyclic_prefix_length=6,
            pilot_pattern='kronecker',
            pilot_ofdm_symbol_indices=[2, 11])

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization='single',
            polarization_type='V', antenna_pattern='omni',
            carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=NUM_TX // 2, polarization='dual',
            polarization_type='cross', antenna_pattern='38.901',
            carrier_frequency=2.6e9)
        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        self.binary_source = BinarySource()
        self.encoder    = LDPC5GEncoder(
            int(self.rg.num_data_symbols),
            int(self.rg.num_data_symbols * 2))
        self.mapper     = Mapper('qam', self.num_bits_per_symbol)
        self.rg_mapper  = ResourceGridMapper(self.rg)
        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ  = LMMSEEqualizer(self.rg, self.sm)
        self.demapper   = Demapper('app', 'qam', self.num_bits_per_symbol)
        self.decoder    = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(self.rg, self.sm)
        self.remove_nulled = RemoveNulledSubcarriers(self.rg)

        from sionna.phy.ofdm import RZFPrecodedChannel
        self.ch_helper = RZFPrecodedChannel(self.rg, self.sm)

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size, NUM_RX, 'umi')
        self.channel_model.set_topology(*topology)

    @tf.function
    def eval_step(self, batch_size, snr_db, precoder_fn, needs_no=False):
        no    = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)
        b     = self.binary_source([batch_size, 1, NUM_RX,
                                    int(self.rg.num_data_symbols)])
        c     = self.encoder(b)
        x     = self.mapper(c)
        x_rg  = self.rg_mapper(x)

        cir    = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                    1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        h_freq = self.remove_nulled(h_freq)

        if needs_no:
            g = precoder_fn(h_freq, no)
        else:
            g = precoder_fn(h_freq)

        W = tf.squeeze(g, axis=1)
        x_vec = tf.transpose(x_rg, perm=[0, 3, 4, 2, 1])
        x_pre = tf.expand_dims(
            tf.transpose(
                tf.squeeze(tf.matmul(W, x_vec), axis=-1),
                perm=[0, 3, 1, 2]),
            axis=1)

        h_eff         = self.ch_helper.compute_effective_channel(h_freq, g)
        y             = self.channel_freq(x_pre, h_freq, no)
        x_hat, no_eff = self.lmmse_equ(y, h_eff, 0.0, no)
        llr           = self.demapper(x_hat, no_eff)
        b_hat         = self.decoder(llr)

        ber  = compute_ber(b, b_hat)
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4))
            / tf.math.log(2.0), axis=[0, 1, 2, 4]))
        return rate, ber

# =============================================================================
# HELPERS
# =============================================================================

def load_model(cfg, system):
    """Charge un modèle en lisant d'abord le config.json du checkpoint."""
    wpath = cfg['weights']
    if not os.path.exists(wpath):
        print(f'  ❌ Poids introuvables : {wpath}')
        return None

    # Lire le config.json du checkpoint pour connaître les vrais params
    ckpt_dir  = os.path.dirname(wpath)
    cfg_path  = os.path.join(ckpt_dir, 'config.json')

    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            saved_cfg = json.load(f)
        print(f'  📋 Config trouvé : {cfg_path}')
        print(f'     version={saved_cfg.get("version")} | '
              f'tokens_per_rb={saved_cfg.get("tokens_per_rb")} | '
              f'embed_dim={saved_cfg.get("embed_dim")} | '
              f'total_params={saved_cfg.get("total_params")}')
    else:
        saved_cfg = {}
        print(f'  ⚠️  Pas de config.json — utilise les params du dict')

    # Params réels depuis le config sauvegardé
    tok    = saved_cfg.get('tokens_per_rb',
                            cfg['model_kwargs'].get('tokens_per_rb', 3))
    ver    = saved_cfg.get('version',
                            cfg['model_kwargs'].get('version', 'v4.2'))
    edim   = saved_cfg.get('embed_dim',
                            cfg['model_kwargs'].get('embed_dim', 128))
    nheads = saved_cfg.get('num_heads',
                            cfg['model_kwargs'].get('num_heads', 4))

    if cfg['type'] == 'v4':
        model = TransformerPrecoderV4(
            num_tx=NUM_TX, num_rx=NUM_RX,
            num_ofdm=NUM_OFDM, fft_size=FFT_SIZE,
            rb_size=12, tokens_per_rb=tok,
            embed_dim=edim, num_heads=nheads,
            num_layers=4, version=ver)

        dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE],
                         dtype=tf.complex64)
        if ver == 'v4.3':
            no_d = tf.constant(1e-2, tf.float32)
            _ = model(dummy, no=no_d, training=False)
        else:
            _ = model(dummy, training=False)

    else:  # intra_rb
        model = IntraRBTransformerPrecoder(
            num_tx=NUM_TX, num_rx=NUM_RX,
            num_ofdm=NUM_OFDM, fft_size=FFT_SIZE,
            rb_size=12, snr_aware=True,
            embed_dim=edim, num_heads=nheads,
            num_layers=cfg['model_kwargs'].get('num_layers', 4))
        dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE],
                         dtype=tf.complex64)
        _ = model(dummy, no=tf.constant(1e-2, tf.float32), training=False)

    # Chargement des poids — strict cette fois
    with open(wpath, 'rb') as f:
        saved = pickle.load(f)

    n_ok = n_bad = 0
    for var, w in zip(model.trainable_variables, saved):
        if var.shape == w.shape:
            var.assign(w)
            n_ok += 1
        else:
            print(f'  ⚠️  Shape mismatch : {var.name} {var.shape} vs {w.shape}')
            n_bad += 1

    if n_bad > 0:
        print(f'  ❌ {n_bad} poids non chargés — mauvais checkpoint pour ce modèle')
        return None

    print(f'  ✅ {cfg["name"]} — {n_ok}/{len(saved)} poids chargés parfaitement')
    return model

def make_precoder_fn(model, cfg, system):
    """Retourne une fonction précoder compatible eval_step."""
    if cfg['type'] == 'intra_rb':
        # IntraRB a besoin de no → needs_no=True dans eval_step
        def fn(h_freq, no):
            return model(h_freq, no=no, training=False)
        return fn, True
    else:
        # V4 sans SNR (v4.2) ou avec SNR (v4.3)
        if cfg['model_kwargs'].get('version') == 'v4.3':
            def fn(h_freq, no):
                return model(h_freq, no=no, training=False)
            return fn, True
        else:
            def fn(h_freq):
                return model(h_freq, training=False)
            return fn, False


def evaluate(system, precoder_fn, needs_no, name, snr_range):
    """Boucle d'évaluation Monte Carlo."""
    print(f'\n📊 {name}')
    rates, bers = [], []
    bs = tf.constant(BATCH_SIZE, tf.int32)

    for snr in snr_range:
        r_b, b_b = [], []
        snr_t = tf.constant(float(snr), tf.float32)
        for _ in range(NUM_BATCHES):
            system.new_topology(BATCH_SIZE)
            r, b = system.eval_step(bs, snr_t, precoder_fn, needs_no)
            r_b.append(float(r))
            b_b.append(float(b))
        rates.append(float(np.mean(r_b)))
        bers.append(float(np.mean(b_b)))
        print(f'  SNR={snr:5.1f} dB | Rate={rates[-1]:6.2f} | BER={bers[-1]:.2e}')

    return {'snr': list(snr_range), 'rate': rates, 'ber': bers}

# =============================================================================
# FIGURES
# =============================================================================

def plot_all(results, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    snr = SNR_RANGE

    # Styles fixes pour baselines
    BASE_STYLES = {
        'RZF'  : {'color': '#1f77b4', 'marker': 'o', 'ls': '--', 'lw': 2.2},
        'WMMSE': {'color': '#000000', 'marker': 's', 'ls': '-.', 'lw': 2.2},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'IntraRB vs V4 Transformer — {NUM_TX}×{NUM_RX} MU-MIMO\n'
        f'(8 TX, 4 UE, 72 SC, UMi 2.6 GHz)',
        fontsize=12, fontweight='bold')

    # ── (a) Sum Rate ─────────────────────────────────────────────────────────
    ax = axes[0]
    for entry in results:
        name = entry['name']
        res  = entry['result']
        if name in BASE_STYLES:
            s = BASE_STYLES[name]
            ax.plot(res['snr'], res['rate'],
                    color=s['color'], marker=s['marker'],
                    ls=s['ls'], lw=s['lw'], label=name, markersize=7)
        else:
            cfg = entry['cfg']
            ax.plot(res['snr'], res['rate'],
                    color=cfg['color'], marker=cfg['marker'],
                    ls=cfg['ls'], lw=2.0, label=name, markersize=7)

    ax.set(xlabel='SNR (dB)', ylabel='Sum Rate (bps/Hz)',
           title='(a) Efficacité spectrale')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    # ── (b) BER ──────────────────────────────────────────────────────────────
    ax = axes[1]
    for entry in results:
        name = entry['name']
        res  = entry['result']
        ber  = [max(b, 1e-7) for b in res['ber']]
        if name in BASE_STYLES:
            s = BASE_STYLES[name]
            ax.semilogy(res['snr'], ber,
                        color=s['color'], marker=s['marker'],
                        ls=s['ls'], lw=s['lw'], label=name, markersize=7)
        else:
            cfg = entry['cfg']
            ax.semilogy(res['snr'], ber,
                        color=cfg['color'], marker=cfg['marker'],
                        ls=cfg['ls'], lw=2.0, label=name, markersize=7)

    ax.set(xlabel='SNR (dB)', ylabel='BER', title='(b) BER Monte Carlo')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # ── (c) Pareto énergie @ 15 dB ───────────────────────────────────────────
    ax = axes[2]
    idx_15 = int(np.argmin(np.abs(snr - 15.0)))

    for entry in results:
        name    = entry['name']
        res     = entry['result']
        flops   = entry.get('flops')
        weights = entry.get('weights_count')
        acts    = entry.get('acts')
        if flops is None:
            continue

        rate        = res['rate'][idx_15]
        is_classical = name in ('RZF', 'WMMSE')
        e = compute_energy_uJ(flops, weights, acts,
                              32 if is_classical else 16,
                              32 if is_classical else 16)

        if name in BASE_STYLES:
            s = BASE_STYLES[name]
            ax.scatter(e, rate, s=250, marker=s['marker'],
                       color=s['color'], zorder=4,
                       edgecolors='k', linewidth=0.8)
        else:
            cfg = entry['cfg']
            ax.scatter(e, rate, s=250, marker=cfg['marker'],
                       color=cfg['color'], zorder=4,
                       edgecolors='k', linewidth=0.8)

        ax.annotate(name, (e, rate),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

    ax.set(xlabel='Énergie inférence (µJ)\n[NN→FP16 | Classiques→FP32]',
           ylabel='Sum Rate @ 15 dB (bps/Hz)',
           title='(c) Pareto Énergie–Performance')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = f'{save_dir}/compare_{ts}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'✅ Figure : {p}')
    plt.close()

# =============================================================================
# TABLEAU CONSOLE
# =============================================================================

def print_table(results, snr_range):
    idx_15 = int(np.argmin(np.abs(np.array(snr_range) - 15.0)))
    wmmse_rate = next(
        (e['result']['rate'][idx_15] for e in results if e['name'] == 'WMMSE'),
        None)
    rzf_rate = next(
        (e['result']['rate'][idx_15] for e in results if e['name'] == 'RZF'),
        None)

    print(f'\n{"="*95}')
    print(f'  RÉSUMÉ @ 15 dB  —  {NUM_TX}×{NUM_RX} MU-MIMO')
    if wmmse_rate:
        print(f'  WMMSE = {wmmse_rate:.2f} bps/Hz | RZF = {rzf_rate:.2f} bps/Hz')
    print(f'{"="*95}')
    print(f'{"Modèle":<35} | {"Rate":>7} | {"vs RZF":>7} | {"vs WMMSE":>9} | '
          f'{"FLOPs(M)":>9} | {"Params(K)":>9} | {"E FP16(µJ)":>11}')
    print('─' * 95)

    for entry in results:
        name  = entry['name']
        res   = entry['result']
        rate  = res['rate'][idx_15]
        drzf  = f'{rate - rzf_rate:+.2f}'   if rzf_rate  else 'N/A'
        dwmm  = f'{rate - wmmse_rate:+.2f}' if wmmse_rate else 'N/A'
        flops = entry.get('flops')
        w_cnt = entry.get('weights_count')
        acts  = entry.get('acts')

        if flops:
            is_cl = name in ('RZF', 'WMMSE')
            e16   = compute_energy_uJ(flops, w_cnt, acts,
                                      32 if is_cl else 16,
                                      32 if is_cl else 16)
            e_str = f'N/A (FP32)' if is_cl else f'{e16:.4f}'
            f_str = f'{flops/1e6:.1f}'
            p_str = f'{w_cnt/1e3:.1f}'
        else:
            e_str = f_str = p_str = 'N/A'

        print(f'{name:<35} | {rate:>7.2f} | {drzf:>7} | {dwmm:>9} | '
              f'{f_str:>9} | {p_str:>9} | {e_str:>11}')

    print('=' * 95)

# =============================================================================
# MAIN
# =============================================================================

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f'✅ GPU : {gpus[0]}')

    system  = EvalSystem()
    results = []   # liste ordonnée pour les figures

    # ── 1. Baselines classiques ───────────────────────────────────────────────
    print('\n' + '='*60 + '\n  BASELINES\n' + '='*60)

    # RZF
    rzf_fn = partial(rzf_precoder, stream_management=system.sm, no=None)
    def rzf_with_no(h_freq, no):
        return rzf_precoder(h_freq, system.sm, no=no)
    rzf_res = evaluate(system, rzf_with_no, True, 'RZF', SNR_RANGE)
    f, w, a = compute_classical_flops('rzf', NUM_TX, NUM_RX, FFT_SIZE, NUM_OFDM)
    results.append({'name': 'RZF', 'result': rzf_res, 'cfg': None,
                    'flops': f, 'weights_count': w, 'acts': a})

    # WMMSE
    def wmmse_fn(h_freq, no):
        return wmmse_precoder(h_freq, no, stream_management=system.sm,
                              num_iterations=10)
    wmmse_res = evaluate(system, wmmse_fn, True, 'WMMSE', SNR_RANGE)
    f, w, a = compute_classical_flops('wmmse', NUM_TX, NUM_RX, FFT_SIZE, NUM_OFDM)
    results.append({'name': 'WMMSE', 'result': wmmse_res, 'cfg': None,
                    'flops': f, 'weights_count': w, 'acts': a})

    # ── 2. Modèles neuraux ────────────────────────────────────────────────────
    print('\n' + '='*60 + '\n  MODÈLES NEURAUX\n' + '='*60)

    for cfg in MODELS:
        print(f'\n── {cfg["name"]} ──')
        model = load_model(cfg, system)
        if model is None:
            continue

        precoder_fn, needs_no = make_precoder_fn(model, cfg, system)
        res = evaluate(system, precoder_fn, needs_no, cfg['name'], SNR_RANGE)

        # Complexité
        if hasattr(model, 'complexity'):
            f, w, a = model.complexity(NUM_OFDM)
        else:
            f = w = a = None

        results.append({
            'name'          : cfg['name'],
            'result'        : res,
            'cfg'           : cfg,
            'flops'         : f,
            'weights_count' : w,
            'acts'          : a,
        })

    # ── 3. Figures & tableau ──────────────────────────────────────────────────
    print_table(results, SNR_RANGE)
    plot_all(results, save_dir='./results')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'./results/compare_{ts}.npy', results)
    print(f'\n✅ Terminé.')


if __name__ == '__main__':
    main()