"""
diag_csi_imperfect_signed_attn.py — Priorité 3 (demande utilisateur) : les
3 architectures signed_attn (SingleSC/IntraRB/TA-RB résiduel, Priorités 1-2)
résistent-elles différemment au CSI imparfait ? Poids figés, pas de
réentraînement -- réplique EXACTE de la méthodologie diag_csi_imperfect.py
(UMi, data SNR=15dB, pilote {parfait,20dB,10dB}, 20 tirages x batch 32,
seed 2026) pour comparabilité directe avec les résultats déjà en main
(SingleSC/IntraRB/TA-RB non-signed_attn).

Checkpoints résolus automatiquement depuis les JSON de résultats P1/P2
(champ 'best_ckpt'), pas hardcodés -- évite une erreur de timestamp.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_csi_imperfect_signed_attn.py
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
DATA_SNR_DB   = 15.0
PILOT_SNRS_DB = [None, 20.0, 10.0]
NUM_DRAWS     = 20
BATCH         = 32
SEED          = 2026

# ckpt résolu depuis le JSON produit par le run correspondant (P1/P2)
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
    if pilot_snr_db is None:
        return h_freq
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2  = 1.0 / snr_lin
    shape   = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise = tf.complex(noise_re, noise_im)
    return h_freq + tf.cast(noise, h_freq.dtype)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    system = LockedSystem(M, K)
    no = ebnodb2no(tf.constant(DATA_SNR_DB, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models = {}
    for name in RESULT_JSONS:
        ckpt = resolve_ckpt(name)
        m = build_neural(name)
        load_weights(m, ckpt, dummy_h, no)
        neural_models[name] = m
        print(f'✅ {name} chargé depuis {ckpt}')

    rng = np.random.RandomState(SEED)
    results = {}
    methods = ['RZF', 'WMMSE'] + list(RESULT_JSONS.keys())
    for meth in methods:
        results[meth] = {}

    for draw in range(NUM_DRAWS):
        system.new_topology(BATCH)
        h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(DATA_SNR_DB, tf.float32))

        for pilot_snr in PILOT_SNRS_DB:
            cond = 'perfect' if pilot_snr is None else f'pilot{int(pilot_snr)}dB'
            h_est = noisy_channel(h_true, pilot_snr, rng)

            g_rzf = rzf_precoder(h_est, stream_management=system.sm, no=no)
            results['RZF'].setdefault(cond, []).append(float(system._sum_rate(h_true, g_rzf, no)))

            g_wmmse = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            results['WMMSE'].setdefault(cond, []).append(float(system._sum_rate(h_true, g_wmmse, no)))

            for name, model in neural_models.items():
                g = model(h_est, no=no, training=False)
                results[name].setdefault(cond, []).append(float(system._sum_rate(h_true, g, no)))

        if (draw + 1) % 5 == 0:
            print(f'  tirage {draw+1}/{NUM_DRAWS} fait', flush=True)

    print(f'\n{"="*90}\nCSI imparfait -- signed_attn, STANDARD (M8K4, R=20m), '
          f'data SNR={DATA_SNR_DB}dB, {NUM_DRAWS} tirages x {BATCH}\n{"="*90}')
    print(f'{"Méthode":<16} {"perfect":>10} {"pilot20dB":>12} {"%vs perfect":>12} {"pilot10dB":>12} {"%vs perfect":>12}')
    summary = {}
    for meth in methods:
        r_perf = np.mean(results[meth]['perfect'])
        r_20   = np.mean(results[meth]['pilot20dB'])
        r_10   = np.mean(results[meth]['pilot10dB'])
        pct20 = 100.0 * r_20 / r_perf
        pct10 = 100.0 * r_10 / r_perf
        print(f'{meth:<16} {r_perf:>10.2f} {r_20:>12.2f} {pct20:>11.1f}% {r_10:>12.2f} {pct10:>11.1f}%')
        summary[meth] = {'perfect': r_perf, 'pilot20dB': r_20, 'pilot10dB': r_10,
                          'pct_retained_20dB': pct20, 'pct_retained_10dB': pct10}

    print(f'\n{"="*90}\nComparaison rétention relative (SingleSC = référence 0) -- '
          f'et vs les versions non-signed_attn déjà en main\n{"="*90}')
    base20 = summary['SingleSC']['pct_retained_20dB']
    base10 = summary['SingleSC']['pct_retained_10dB']
    for name in ['IntraRB', 'TA_RB_residual']:
        d20 = summary[name]['pct_retained_20dB'] - base20
        d10 = summary[name]['pct_retained_10dB'] - base10
        tag20 = ' <-- résiste MIEUX' if d20 > 0.5 else (' <-- résiste MOINS bien' if d20 < -0.5 else ' <-- ~égal')
        tag10 = ' <-- résiste MIEUX' if d10 > 0.5 else (' <-- résiste MOINS bien' if d10 < -0.5 else ' <-- ~égal')
        print(f'{name}: pilot20dB {d20:+.2f}pts{tag20} | pilot10dB {d10:+.2f}pts{tag10}')

    with open('results/diag_csi_imperfect_signed_attn.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSauvé -> results/diag_csi_imperfect_signed_attn.json')
