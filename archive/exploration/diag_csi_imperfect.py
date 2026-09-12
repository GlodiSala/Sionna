"""
diag_csi_imperfect.py — PRIORITÉ 1 (demande utilisateur) : les checkpoints
STANDARD déjà entraînés (poids figés, PAS de réentraînement) résistent-ils
différemment à un CSI imparfait selon qu'ils exploitent ou non la
corrélation cross-SC (IntraRB/TA-RB vs SingleSC) ?

Hypothèse : sous CSI parfait, RZF/WMMSE n'ont besoin QUE du canal à leur
propre SC (aucun rôle pour la corrélation cross-SC) -- cohérent avec
Partie 0/1 (single_sc ≈ intra_rb ≈ ta_rb). Sous CSI imparfait (bruit
d'estimation pilote), la corrélation entre SC voisines pourrait permettre
à IntraRB/TA-RB de débruiter implicitement leur estimée -- un rôle que
RZF/WMMSE ne jouent structurellement pas (ils ne voient qu'1 SC à la
fois). Si l'écart SingleSC vs IntraRB/TA-RB se creuse en faveur
d'IntraRB/TA-RB sous CSI imparfait (dégradation relative moindre) : piste
confirmée. Sinon : la piste ne tient pas non plus.

Méthodologie CSI imparfaite (reconstruite -- le script original
"avant-hier" n'a pas survécu, seuls les résultats `results/eval_csi_
noise_*.npy` (ancien canal, pré-clustering) en restent, utilisés ici
uniquement comme référence de FORME, pas de valeurs) : bruit LS gaussien
complexe sur le canal, variance = puissance_canal / SNR_pilote_linéaire
(canal normalisé à puissance unité, confirmé dans tous les logs de cette
session). Le PRÉCODEUR voit le canal BRUITÉ (décision), le SINR/rate
réel est calculé avec le VRAI canal (impact réel de l'erreur d'estimation).

Inference seule (poids figés chargés depuis les checkpoints STANDARD du
run complet ÉTAPE 4 v2) -- rapide, un seul job GPU, pas de concurrence
(GPU0 confirmé libre avant lancement).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_csi_imperfect.py
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
from precoder_intra_rb import SingleSCTransformerPrecoder, IntraRBTransformerPrecoder
from precoders_v2 import TransformerPrecoderClean
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
DATA_SNR_DB   = 15.0          # SNR data fixe (métrique cible ÉTAPE 3/4)
PILOT_SNRS_DB = [None, 20.0, 10.0]   # None = CSI parfait
NUM_DRAWS     = 20
BATCH         = 32
SEED          = 2026

CHECKPOINTS = {
    'SingleSC': ('single_sc', 'weights/SingleSC_4L_128d/best_20260807_125302'),
    'IntraRB':  ('intra_rb',  'weights/IntraRB_4L_128d/best_20260807_141928'),
    'TA_RB_T6': ('ta_rb',     'weights/TA_RB_6tok_4L_128d/best_20260807_153511'),
}


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=128, num_heads=4, num_layers=4)
    if kind == 'single_sc':
        return SingleSCTransformerPrecoder(**kwargs)
    if kind == 'intra_rb':
        return IntraRBTransformerPrecoder(rb_size=12, **kwargs)
    if kind == 'ta_rb':
        return TransformerPrecoderClean(rb_size=12, tokens_per_rb=6, **kwargs)
    raise ValueError(kind)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    """h_freq complex64 [..]. Bruit LS gaussien complexe, var = 1/SNR_lin
    (canal normalisé puissance unité)."""
    if pilot_snr_db is None:
        return h_freq
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2  = 1.0 / snr_lin
    shape   = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise = tf.complex(noise_re, noise_im)
    return h_freq + tf.cast(noise, h_freq.dtype)


def sum_rate_from(h_true, g, sm_helper, lmmse_sinr, no):
    """Rate RÉEL (canal vrai) pour un précodage g décidé sur canal (parfait
    ou bruité selon ce qui a été passé au précodeur en amont)."""
    h_eff = sm_helper.compute_effective_channel(h_true, g)
    sinr  = lmmse_sinr(h_eff, no=no, interference_whitening=True)
    return tf.reduce_sum(tf.reduce_mean(
        tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
        axis=[0, 1, 2, 4]))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    system = LockedSystem(M, K)
    no = ebnodb2no(tf.constant(DATA_SNR_DB, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models = {}
    for name, (kind, ckpt) in CHECKPOINTS.items():
        m = build_neural(kind)
        load_weights(m, ckpt, dummy_h, no)
        neural_models[name] = m
        print(f'✅ {name} chargé depuis {ckpt}')

    rng = np.random.RandomState(SEED)
    results = {}   # method -> pilot_cond -> list of rates

    methods = ['RZF', 'WMMSE'] + list(CHECKPOINTS.keys())
    for meth in methods:
        results[meth] = {}

    for draw in range(NUM_DRAWS):
        system.new_topology(BATCH)
        h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(DATA_SNR_DB, tf.float32))

        for pilot_snr in PILOT_SNRS_DB:
            cond = 'perfect' if pilot_snr is None else f'pilot{int(pilot_snr)}dB'
            h_est = noisy_channel(h_true, pilot_snr, rng)

            g_rzf = rzf_precoder(h_est, stream_management=system.sm, no=no)
            results['RZF'].setdefault(cond, []).append(
                float(system._sum_rate(h_true, g_rzf, no)))

            g_wmmse = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            results['WMMSE'].setdefault(cond, []).append(
                float(system._sum_rate(h_true, g_wmmse, no)))

            for name, model in neural_models.items():
                g = model(h_est, no=no, training=False)
                results[name].setdefault(cond, []).append(
                    float(system._sum_rate(h_true, g, no)))

        if (draw + 1) % 5 == 0:
            print(f'  tirage {draw+1}/{NUM_DRAWS} fait', flush=True)

    print(f'\n{"="*90}\nCSI imparfait -- STANDARD (M8K4, R=20m), data SNR={DATA_SNR_DB}dB, {NUM_DRAWS} tirages x {BATCH}\n{"="*90}')
    print(f'{"Méthode":<12} {"perfect":>10} {"pilot20dB":>12} {"%vs perfect":>12} {"pilot10dB":>12} {"%vs perfect":>12}')
    summary = {}
    for meth in methods:
        r_perf = np.mean(results[meth]['perfect'])
        r_20   = np.mean(results[meth]['pilot20dB'])
        r_10   = np.mean(results[meth]['pilot10dB'])
        pct20 = 100.0 * r_20 / r_perf
        pct10 = 100.0 * r_10 / r_perf
        print(f'{meth:<12} {r_perf:>10.2f} {r_20:>12.2f} {pct20:>11.1f}% {r_10:>12.2f} {pct10:>11.1f}%')
        summary[meth] = {'perfect': r_perf, 'pilot20dB': r_20, 'pilot10dB': r_10,
                          'pct_retained_20dB': pct20, 'pct_retained_10dB': pct10}

    print(f'\n{"="*90}\nComparaison rétention relative (SingleSC = référence 0)\n{"="*90}')
    base20 = summary['SingleSC']['pct_retained_20dB']
    base10 = summary['SingleSC']['pct_retained_10dB']
    for name in ['IntraRB', 'TA_RB_T6']:
        d20 = summary[name]['pct_retained_20dB'] - base20
        d10 = summary[name]['pct_retained_10dB'] - base10
        tag20 = ' <-- résiste MIEUX que SingleSC' if d20 > 0.5 else (' <-- résiste MOINS bien' if d20 < -0.5 else ' <-- ~égal')
        tag10 = ' <-- résiste MIEUX que SingleSC' if d10 > 0.5 else (' <-- résiste MOINS bien' if d10 < -0.5 else ' <-- ~égal')
        print(f'{name}: pilot20dB {d20:+.2f}pts{tag20} | pilot10dB {d10:+.2f}pts{tag10}')

    with open('results/diag_csi_imperfect.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSauvé -> results/diag_csi_imperfect.json')
