"""
diag_high_snr_disadvantage.py — pourquoi IntraRB/TA-RB sont ACTIVEMENT
moins bons que SingleSC à haut SNR (pas juste "pas mieux"), écart qui
grandit avec le SNR ? (demande utilisateur, prioritaire avant UMa)

Confirmé sans calcul (résultats déjà sauvés) : à 0dB, IntraRB bat même
légèrement SingleSC (single-intra=-0.33) ; à 20dB, SingleSC devance
IntraRB de 0.53 et TA-RB de 2.48 -- désavantage croissant, monotone,
pas un artefact de bruit de mesure isolé.

Teste (poids figés, PAS de réentraînement) :
1. Écart pool-entraînement vs canaux frais (généralisation) à chaque SNR
   -- si IntraRB/TA-RB "sur-apprennent" plus que SingleSC, l'écart
   pool-vs-frais doit grandir avec le SNR PLUS VITE pour eux.
2. Norme du gradient (backward sur loss sum-rate réelle) à bas vs haut
   SNR, par architecture -- optimisation plus dure à haut SNR pour les
   réseaux plus larges/profonds ?
3. Écart IntraRB-vs-TA-RB (compression T=6) à bas vs haut SNR -- doit
   se creuser avec le SNR si la perte de compression devient le facteur
   limitant dominant en régime MUI-limité (haut SNR).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_high_snr_disadvantage.py
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
from datasets import CachedSionnaDataset
from main_finall import MU_MIMO_System, DATASET_SIZE, NUM_TX, NUM_RX, CHOSEN_CONFIG
from precoder_intra_rb import SingleSCTransformerPrecoder, IntraRBTransformerPrecoder
from precoders_v2 import TransformerPrecoderClean
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
SNR_POINTS = [5.0, 10.0, 15.0, 20.0]
NUM_DRAWS  = 15
BATCH      = 32

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


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # -- systèmes : frais (fresh, hors pool) + pool d'entraînement (cache) --
    fresh_system = LockedSystem(M, K)
    dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    pool_dataset = CachedSionnaDataset(
        dummy_sys, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=1234)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    dummy_no = ebnodb2no(tf.constant(15.0, tf.float32), 2, 0.5, fresh_system.rg)
    models = {}
    for name, (kind, ckpt) in CHECKPOINTS.items():
        m = build_neural(kind)
        load_weights(m, ckpt, dummy_h, dummy_no)
        models[name] = m
        n_params = sum(int(tf.size(v)) for v in m.trainable_variables)
        print(f'✅ {name} ({n_params:,} params) chargé depuis {ckpt}')

    # ============================================================
    # 1. Généralisation : rate pool-entraînement vs frais, par SNR
    # ============================================================
    print(f'\n{"="*100}\n1. GÉNÉRALISATION -- rate (pool entraînement) - rate (canaux frais), par SNR\n{"="*100}')
    gen_gap = {name: {} for name in models}
    for snr in SNR_POINTS:
        no = ebnodb2no(tf.constant(snr, tf.float32), 2, 0.5, fresh_system.rg)
        for name, model in models.items():
            r_fresh, r_pool = [], []
            for _ in range(NUM_DRAWS):
                fresh_system.new_topology(BATCH)
                h_fresh, _ = fresh_system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(snr, tf.float32))
                g_fresh = model(h_fresh, no=no, training=False)
                r_fresh.append(float(fresh_system._sum_rate(h_fresh, g_fresh, no)))

                h_pool = pool_dataset.get_batch(BATCH)
                g_pool = model(h_pool, no=no, training=False)
                r_pool.append(float(fresh_system._sum_rate(h_pool, g_pool, no)))
            gen_gap[name][snr] = {'fresh': float(np.mean(r_fresh)), 'pool': float(np.mean(r_pool))}

    print(f'{"SNR":>6}' + ''.join(f'{name:>28}' for name in models))
    print(f'{"":>6}' + ''.join(f'{"fresh /  pool /  gap":>28}' for _ in models))
    for snr in SNR_POINTS:
        row = f'{snr:>6.1f}'
        for name in models:
            d = gen_gap[name][snr]
            gap = d['pool'] - d['fresh']
            row += f'{d["fresh"]:>8.2f}/{d["pool"]:>7.2f}/{gap:>+7.2f}'
        print(row)

    # ============================================================
    # 2. Norme du gradient à bas vs haut SNR, par architecture
    # ============================================================
    print(f'\n{"="*100}\n2. NORME DU GRADIENT (loss=-sum_rate/rate_norm) par SNR, canaux frais\n{"="*100}')
    rate_norm = float(K) * 9.0
    grad_norms = {name: {} for name in models}
    for snr in SNR_POINTS:
        no = ebnodb2no(tf.constant(snr, tf.float32), 2, 0.5, fresh_system.rg)
        for name, model in models.items():
            gns = []
            for _ in range(NUM_DRAWS):
                fresh_system.new_topology(BATCH)
                h_fresh, _ = fresh_system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(snr, tf.float32))
                with tf.GradientTape() as tape:
                    g_re, g_im = model(h_fresh, no=no, training=True, return_real_imag=True)
                    g = tf.complex(g_re, g_im)
                    h_eff = fresh_system.precoded_channel_helper.compute_effective_channel(h_fresh, g)
                    sinr = fresh_system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
                    rate = tf.reduce_sum(tf.reduce_mean(
                        tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
                        axis=[0, 1, 2, 4]))
                    loss = -rate / rate_norm
                grads = tape.gradient(loss, model.trainable_variables)
                grads = [g for g in grads if g is not None]
                gn = float(tf.linalg.global_norm(grads))
                gns.append(gn)
            grad_norms[name][snr] = float(np.mean(gns))

    print(f'{"SNR":>6}' + ''.join(f'{name:>15}' for name in models))
    for snr in SNR_POINTS:
        row = f'{snr:>6.1f}' + ''.join(f'{grad_norms[name][snr]:>15.4f}' for name in models)
        print(row)
    print('\nRatio norme(20dB)/norme(5dB) -- <1 = gradient qui s\'aplatit à haut SNR :')
    for name in models:
        ratio = grad_norms[name][20.0] / grad_norms[name][5.0]
        print(f'  {name}: {ratio:.3f}')

    # ============================================================
    # 3. Écart IntraRB vs TA-RB (compression) à bas vs haut SNR
    # ============================================================
    print(f'\n{"="*100}\n3. Écart IntraRB - TA-RB (perte de compression T=6) par SNR (pool entraînement, déjà calculé au point 1)\n{"="*100}')
    for snr in SNR_POINTS:
        gap = gen_gap['IntraRB'][snr]['fresh'] - gen_gap['TA_RB_T6'][snr]['fresh']
        print(f'  SNR={snr:5.1f}dB : IntraRB - TA-RB = {gap:+.2f} bps/Hz')

    with open('results/diag_high_snr_disadvantage.json', 'w') as f:
        json.dump({'gen_gap': {n: {str(s): v for s, v in d.items()} for n, d in gen_gap.items()},
                   'grad_norms': {n: {str(s): v for s, v in d.items()} for n, d in grad_norms.items()}},
                  f, indent=2)
    print('\nSauvé -> results/diag_high_snr_disadvantage.json')
