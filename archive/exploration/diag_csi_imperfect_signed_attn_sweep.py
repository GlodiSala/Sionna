"""
diag_csi_imperfect_signed_attn_sweep.py — Priorité 2 (demande utilisateur) :
CSI imparfait sur toute la plage SNR data ∈ {0,5,10,15,20}dB, pilote fixé
à 20dB (le plus informatif), pour les 3 architectures signed_attn --
étend diag_csi_imperfect_signed_attn.py (qui ne mesurait qu'à data
SNR=15dB fixe) à une vraie courbe.

Méthodologie identique (poids figés, pas de réentraînement, bruit LS
gaussien sur le canal, 20 tirages x batch 32, seed 2026), juste DATA_SNR_
DB balayé au lieu de fixe.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_csi_imperfect_signed_attn_sweep.py
"""
import os, sys, json, pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, set_locked_topology
from precoders_w import rzf_precoder, wmmse_precoder
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_DRAWS = 20
BATCH = 32
SEED = 2026

RESULT_JSONS = {
    'SingleSC': 'results/diag_front_a_signed_attn.json',
    'IntraRB':  'results/diag_front_a_signed_attn_intra_rb.json',
    'TA_RB_residual': 'results/diag_front_a_signed_attn_ta_rb_residual.json',
}


def resolve_ckpt(name):
    with open(RESULT_JSONS[name]) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
    return ckpt


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=128, num_heads=4, num_layers=4)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kwargs)
    raise ValueError(kind)


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


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
    system = LockedSystem(M, K)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models = {}
    for name in RESULT_JSONS:
        ckpt = resolve_ckpt(name)
        no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        m = build_neural(name)
        load_weights(m, ckpt, dummy_h, no0)
        neural_models[name] = m
        print(f'✅ {name} chargé depuis {ckpt}')

    rng = np.random.RandomState(SEED)
    methods = ['RZF', 'WMMSE'] + list(RESULT_JSONS.keys())
    results = {m: {} for m in methods}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        print(f'\n=== data SNR={data_snr}dB (pilote={PILOT_SNR_DB}dB) ===', flush=True)

        r_perfect = {m: [] for m in methods}
        r_pilot = {m: [] for m in methods}
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_rzf_p = rzf_precoder(h_true, stream_management=system.sm, no=no)
            r_perfect['RZF'].append(float(system._sum_rate(h_true, g_rzf_p, no)))
            g_rzf_n = rzf_precoder(h_est, stream_management=system.sm, no=no)
            r_pilot['RZF'].append(float(system._sum_rate(h_true, g_rzf_n, no)))

            g_wmmse_p = wmmse_precoder(h_true, no=no, stream_management=system.sm, num_iterations=10)
            r_perfect['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse_p, no)))
            g_wmmse_n = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            r_pilot['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse_n, no)))

            for name, model in neural_models.items():
                g_p = model(h_true, no=no, training=False)
                r_perfect[name].append(float(system._sum_rate(h_true, g_p, no)))
                g_n = model(h_est, no=no, training=False)
                r_pilot[name].append(float(system._sum_rate(h_true, g_n, no)))

        for m in methods:
            perf = float(np.mean(r_perfect[m]))
            pil = float(np.mean(r_pilot[m]))
            pct = 100.0 * pil / perf if perf > 0 else float('nan')
            results[m][str(data_snr)] = {'perfect': perf, 'pilot20dB': pil, 'pct_retained': pct}
            print(f'  {m:<16} perfect={perf:6.2f} pilot20dB={pil:6.2f} ({pct:5.1f}%)', flush=True)

        with open('results/diag_csi_imperfect_signed_attn_sweep.json', 'w') as f:
            json.dump(results, f, indent=2)   # sauvé après chaque SNR -- reprenable

    print('\n✅ Sauvé -> results/diag_csi_imperfect_signed_attn_sweep.json')
