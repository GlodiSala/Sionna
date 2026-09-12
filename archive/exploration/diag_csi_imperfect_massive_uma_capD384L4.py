"""
diag_csi_imperfect_massive_uma_capD384L4.py — réplique de
diag_csi_imperfect_massive_capD384L4.py (UMi, session du 12/08) pour
le canal UMa MASSIVE (M=64,K=8), avec les 3 checkpoints D=384,L=4
réentraînés cette nuit (chantier UMa, cf. RAPPORT_MASSIVE_UMA_GAP_
CLOSING). Méthodologie strictement identique (bruit LS gaussien sur
canal estimé, canal vrai pour SINR/détection, RZF/WMMSE recalculés
sur les mêmes tirages) -- seul le channel_model (UMa au lieu de UMi)
et les chemins de checkpoints changent.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_csi_imperfect_massive_uma_capD384L4.py
"""
import os, sys, json, pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import UMa
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from precoders_w import rzf_precoder, wmmse_precoder
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
D, L = 384, 4
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNRS_DB = [20.0, 10.0]
NUM_DRAWS = 12
BATCH = 16
SEED = 2026

RESULT_JSONS = {
    'SingleSC': 'results/diag_massive_gap_fullbudget_uma_single_sc_cap_d384l4.json',
    'IntraRB':  'results/diag_massive_gap_fullbudget_uma_intra_rb_cap_d384l4.json',
    'TA_RB_residual': 'results/diag_massive_gap_fullbudget_uma_ta_rb_residual_cap_d384l4.json',
}


def resolve_ckpt(name):
    with open(RESULT_JSONS[name]) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
    return ckpt


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=D, num_heads=4, num_layers=L)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    raise ValueError(kind)


class LockedSystemUMa(ConfigurableMIMOSystem):
    cluster_radius_m = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

    def __init__(self, num_tx, num_rx):
        super().__init__(num_tx, num_rx)
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=MASSIVE_TRUE_CONFIG['FORCE_LOS'],
                             indoor_probability=self.indoor_probability)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    system = LockedSystemUMa(M, K)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models = {}
    for name in RESULT_JSONS:
        ckpt = resolve_ckpt(name)
        no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        m = build_neural(name)
        load_weights(m, ckpt, dummy_h, no0)
        neural_models[name] = m
        print(f'✅ {name} (D={D}, UMa) chargé depuis {ckpt}', flush=True)

    rng = np.random.RandomState(SEED)
    methods = ['RZF', 'WMMSE'] + list(RESULT_JSONS.keys())
    results = {pilot: {m: {} for m in methods} for pilot in PILOT_SNRS_DB}
    perfect_cache = {}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        print(f'\n=== data SNR={data_snr}dB ===', flush=True)

        r_perfect = {m: [] for m in methods}
        r_pilot = {pilot: {m: [] for m in methods} for pilot in PILOT_SNRS_DB}
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))

            g_rzf_p = rzf_precoder(h_true, stream_management=system.sm, no=no)
            r_perfect['RZF'].append(float(system._sum_rate(h_true, g_rzf_p, no)))
            g_wmmse_p = wmmse_precoder(h_true, no=no, stream_management=system.sm, num_iterations=10)
            r_perfect['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse_p, no)))
            for name, model in neural_models.items():
                g_p = model(h_true, no=no, training=False)
                r_perfect[name].append(float(system._sum_rate(h_true, g_p, no)))

            for pilot in PILOT_SNRS_DB:
                h_est = noisy_channel(h_true, pilot, rng)
                g_rzf_n = rzf_precoder(h_est, stream_management=system.sm, no=no)
                r_pilot[pilot]['RZF'].append(float(system._sum_rate(h_true, g_rzf_n, no)))
                g_wmmse_n = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
                r_pilot[pilot]['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse_n, no)))
                for name, model in neural_models.items():
                    g_n = model(h_est, no=no, training=False)
                    r_pilot[pilot][name].append(float(system._sum_rate(h_true, g_n, no)))

        perf = {m: float(np.mean(r_perfect[m])) for m in methods}
        perfect_cache[str(data_snr)] = perf
        for pilot in PILOT_SNRS_DB:
            for m in methods:
                pil = float(np.mean(r_pilot[pilot][m]))
                pct = 100.0 * pil / perf[m] if perf[m] > 0 else float('nan')
                results[pilot][m][str(data_snr)] = {'perfect': perf[m], 'pilot': pil, 'pct_retained': pct}
                print(f'  pilot={pilot:4.0f}dB {m:<16} perfect={perf[m]:7.2f} noisy={pil:7.2f} ({pct:5.1f}%)', flush=True)

        with open('results/diag_csi_imperfect_massive_uma_capD384L4.json', 'w') as f:
            json.dump({'pilot_results': results, 'perfect': perfect_cache,
                       'M': M, 'K': K, 'D': D, 'L': L}, f, indent=2)

    print('\n✅ Sauvé -> results/diag_csi_imperfect_massive_uma_capD384L4.json')
