"""
diag_csi_imperfect_tarb_T4.py — Figure A (double panneau, demande
utilisateur) : T=4 devient la référence TA-RB -- recalcule UNIQUEMENT
la courbe CSI imparfait TA-RB avec le checkpoint T=4 (RZF/WMMSE/SC/IB
restent ceux déjà calculés dans diag_csi_imperfect_signed_attn_sweep.json,
inchangés). Même méthodologie (poids figés, bruit LS gaussien, pilote
20dB, DATA_SNR ∈ {0,5,10,15,20}dB, 20 tirages x batch 32, seed 2026).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_csi_imperfect_tarb_T4.py
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
from precoder_experimental import TransformerPrecoderCleanResidualSignedAttn
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_DRAWS = 20
BATCH = 32
SEED = 2026
JSON_PATH = 'results/diag_csi_imperfect_signed_attn_sweep.json'
T4_RESULT_JSON = 'results/diag_tarb_residual_signed_attn_T4_train.json'


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

    with open(T4_RESULT_JSON) as f:
        ckpt = json.load(f)['best_ckpt']
    assert os.path.isdir(ckpt), f"checkpoint introuvable: {ckpt}"
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    model = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12, tokens_per_rb=4,
        embed_dim=128, num_heads=4, num_layers=4)
    load_weights(model, ckpt, dummy_h, no0)
    print(f'✅ TA-RB T=4 chargé depuis {ckpt}', flush=True)

    rng = np.random.RandomState(SEED)
    results = {}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_perfect, r_pilot = [], []
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_p = model(h_true, no=no, training=False)
            r_perfect.append(float(system._sum_rate(h_true, g_p, no)))
            g_n = model(h_est, no=no, training=False)
            r_pilot.append(float(system._sum_rate(h_true, g_n, no)))

        perf = float(np.mean(r_perfect))
        pil = float(np.mean(r_pilot))
        pct = 100.0 * pil / perf if perf > 0 else float('nan')
        results[str(data_snr)] = {'perfect': perf, 'pilot20dB': pil, 'pct_retained': pct}
        print(f'  data_snr={data_snr}dB perfect={perf:6.2f} pilot20dB={pil:6.2f} ({pct:5.1f}%)', flush=True)

    with open(JSON_PATH) as f:
        out = json.load(f)
    out['TA_RB_residual'] = results
    out['TA_RB_residual']['_T'] = 4
    with open(JSON_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Fusionné (TA-RB T=4) -> {JSON_PATH}')
