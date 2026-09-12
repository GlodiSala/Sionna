"""
diag_csi_imperfect_snr_sweep.py — suite de diag_csi_imperfect.py (demande
utilisateur) : le test précédent était à SNR DONNÉES FIXE = 15dB, pilote
variable {parfait, 20dB, 10dB}. Ici : SNR données variable [0,5,10,15,20]
dB, PILOTE FIXE (20dB, modérément bruité) -- l'avantage de TA-RB
(débruitage par moyennage) persiste-t-il à haut SNR données (lien déjà
propre, bruit d'estimation relativement moins important), ou s'estompe-t-il ?

Même méthodologie exacte que diag_csi_imperfect.py (bruit LS gaussien
complexe sur le canal, variance=1/SNR_pilote_lin, précodeur décide sur
canal bruité, rate calculé sur le vrai canal), mêmes checkpoints figés.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_csi_imperfect_snr_sweep.py
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
DATA_SNRS_DB  = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB  = 20.0          # fixe, modérément bruité (comme demandé)
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
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2  = 1.0 / snr_lin
    shape   = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    system = LockedSystem(M, K)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    dummy_no = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    neural_models = {}
    for name, (kind, ckpt) in CHECKPOINTS.items():
        m = build_neural(kind)
        load_weights(m, ckpt, dummy_h, dummy_no)
        neural_models[name] = m
        print(f'✅ {name} chargé depuis {ckpt}')

    rng = np.random.RandomState(SEED)
    methods = ['RZF', 'WMMSE'] + list(CHECKPOINTS.keys())
    # results[method][data_snr] = {'perfect': [...], 'imperfect': [...]}
    results = {m: {snr: {'perfect': [], 'imperfect': []} for snr in DATA_SNRS_DB} for m in methods}

    for draw in range(NUM_DRAWS):
        system.new_topology(BATCH)
        for data_snr in DATA_SNRS_DB:
            no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            for cond, h_in in [('perfect', h_true), ('imperfect', h_est)]:
                g_rzf = rzf_precoder(h_in, stream_management=system.sm, no=no)
                results['RZF'][data_snr][cond].append(float(system._sum_rate(h_true, g_rzf, no)))

                g_wmmse = wmmse_precoder(h_in, no=no, stream_management=system.sm, num_iterations=10)
                results['WMMSE'][data_snr][cond].append(float(system._sum_rate(h_true, g_wmmse, no)))

                for name, model in neural_models.items():
                    g = model(h_in, no=no, training=False)
                    results[name][data_snr][cond].append(float(system._sum_rate(h_true, g, no)))

        if (draw + 1) % 5 == 0:
            print(f'  tirage {draw+1}/{NUM_DRAWS} fait', flush=True)

    # -- tableau perfect --
    print(f'\n{"="*100}\nSNR-sweep, CSI PARFAIT (référence) -- pilote={PILOT_SNR_DB}dB pour la colonne imparfaite ci-dessous\n{"="*100}')
    header = f'{"SNR data":>10}' + ''.join(f'{m:>12}' for m in methods)
    print(header)
    for snr in DATA_SNRS_DB:
        row = f'{snr:>10.1f}' + ''.join(f'{np.mean(results[m][snr]["perfect"]):>12.2f}' for m in methods)
        print(row)

    print(f'\n{"="*100}\nSNR-sweep, CSI IMPARFAIT (pilote={PILOT_SNR_DB}dB fixe)\n{"="*100}')
    print(header)
    for snr in DATA_SNRS_DB:
        row = f'{snr:>10.1f}' + ''.join(f'{np.mean(results[m][snr]["imperfect"]):>12.2f}' for m in methods)
        print(row)

    print(f'\n{"="*100}\n%% retenu (imparfait/parfait) -- où l\'avantage TA-RB persiste-t-il ?\n{"="*100}')
    print(header)
    pct_table = {}
    for snr in DATA_SNRS_DB:
        pct_table[snr] = {}
        row = f'{snr:>10.1f}'
        for m in methods:
            p = np.mean(results[m][snr]['perfect'])
            imp = np.mean(results[m][snr]['imperfect'])
            pct = 100.0 * imp / p if p > 0 else float('nan')
            pct_table[snr][m] = pct
            row += f'{pct:>11.1f}%'
        print(row)

    print(f'\n{"="*100}\nAvantage TA-RB vs SingleSC (points de %% retenu, TA-RB - SingleSC)\n{"="*100}')
    for snr in DATA_SNRS_DB:
        d = pct_table[snr]['TA_RB_T6'] - pct_table[snr]['SingleSC']
        d_intra = pct_table[snr]['IntraRB'] - pct_table[snr]['SingleSC']
        print(f'  SNR={snr:5.1f}dB : TA-RB {d:+.2f}pts | IntraRB {d_intra:+.2f}pts')

    out = {'pilot_snr_db': PILOT_SNR_DB,
           'perfect': {str(snr): {m: float(np.mean(results[m][snr]['perfect'])) for m in methods} for snr in DATA_SNRS_DB},
           'imperfect': {str(snr): {m: float(np.mean(results[m][snr]['imperfect'])) for m in methods} for snr in DATA_SNRS_DB},
           'pct_retained': {str(snr): pct_table[snr] for snr in DATA_SNRS_DB}}
    with open('results/diag_csi_imperfect_snr_sweep.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_csi_imperfect_snr_sweep.json')
