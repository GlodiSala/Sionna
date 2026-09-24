"""
experiments/eval_csi_pilot_snr_sweep_umi_standard.py -- sensibilite du
debit conserve (regime validation UMi, M=8,K=4) au SNR pilote, avec la
meme methode appariee que eval_csi_conserve_4regimes.py::umi_standard
(memes checkpoints, meme metrique).

Pour chaque (SNR de donnees, tirage) : UN SEUL tirage de canal h_true,
partage par :
  - la reference CSI parfait (precodeur calcule sur h_true)
  - les 5 valeurs de SNR pilote (0,5,10,15,20 dB), chacune produisant un
    h_est = noisy_channel(h_true, pilote) distinct
  - les 5 methodes (RZF, WMMSE, SingleSC, IntraRB, TA_RB_residual)

Metrique : manual_sinr uniquement (SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no),
eq 2.2). Checkpoints TA-RB=T4, comme experiments/eval_csi_conserve_4regimes.py
::REGIMES['umi_standard'].

N'ecrase aucun fichier existant.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/eval_csi_pilot_snr_sweep_umi_standard.py
"""
import os, sys, json, pickle, time, hashlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 2026
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from sionna.phy.utils import ebnodb2no
from eval_system import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, set_locked_topology
from precoders.classical import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
D, L, T = 128, 4, 4
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_DRAWS = 20
BATCH = 32
METHODS = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']
NAME_MAP = {'SingleSC': 'SC', 'IntraRB': 'IB', 'TA_RB_residual': 'TA-RB'}

CHECKPOINTS = {
    'SingleSC': 'weights/front_a_signed_attn/best_20260808_103631',
    'IntraRB': 'weights/IntraRB_signed_attn_4L_128d/best_20260808_113308',
    'TA_RB_residual': 'weights/TA_RB_residual_signed_attn_T4_4L_128d/best_20260808_181436',
}


class LockedSystemUMi(ConfigurableMIMOSystem):
    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=STANDARD_CONFIG['CLUSTER_RADIUS_M'],
                             force_los=STANDARD_CONFIG.get('FORCE_LOS', False),
                             indoor_probability=STANDARD_CONFIG['INDOOR_PROBABILITY'])


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=D, num_heads=4, num_layers=L)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=T, **kwargs)
    raise ValueError(kind)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    sigma2 = 1.0 / 10.0 ** (pilot_snr_db / 10.0)
    nre = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    nim = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(nre, nim), h_freq.dtype)


def manual_sinr_rate(sm, h_true, g, no):
    if len(g.shape) == 5:
        g = tf.expand_dims(g, axis=1)
    h_pc = _get_desired_channels(h_true, sm)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal_pwr = tf.abs(tf.linalg.diag_part(HW)) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    sinr = signal_pwr / (interference_pwr + tf.cast(tf.reshape(no, []), tf.float64))
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    return float(tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(rate, axis=[1, 2]), axis=0)))


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    system = LockedSystemUMi(M, K)

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    neural_models = {}
    for nm, ckpt in CHECKPOINTS.items():
        assert os.path.isfile(os.path.join(ckpt, 'weights.pkl')), f"checkpoint introuvable: {ckpt}"
        model = build_neural(nm)
        load_weights(model, ckpt, dummy_h, no0)
        neural_models[nm] = model
        print(f'OK {nm} <- {ckpt}', flush=True)

    rng = np.random.RandomState(SEED)
    perfect = {m: {s: [] for s in DATA_SNRS_DB} for m in METHODS}
    imperfect = {p: {m: {s: [] for s in DATA_SNRS_DB} for m in METHODS} for p in PILOT_SNRS_DB}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))

            g_p = rzf_precoder(h_true, stream_management=system.sm, no=no)
            perfect['RZF'][data_snr].append(manual_sinr_rate(system.sm, h_true, g_p, no))
            g_p = wmmse_precoder(h_true, no=no, stream_management=system.sm, num_iterations=10)
            perfect['WMMSE'][data_snr].append(manual_sinr_rate(system.sm, h_true, g_p, no))
            for nm, model in neural_models.items():
                g_p = model(h_true, no=no, training=False)
                perfect[nm][data_snr].append(manual_sinr_rate(system.sm, h_true, g_p, no))

            for pilot_snr in PILOT_SNRS_DB:
                h_est = noisy_channel(h_true, pilot_snr, rng)
                g_i = rzf_precoder(h_est, stream_management=system.sm, no=no)
                imperfect[pilot_snr]['RZF'][data_snr].append(manual_sinr_rate(system.sm, h_true, g_i, no))
                g_i = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
                imperfect[pilot_snr]['WMMSE'][data_snr].append(manual_sinr_rate(system.sm, h_true, g_i, no))
                for nm, model in neural_models.items():
                    g_i = model(h_est, no=no, training=False)
                    imperfect[pilot_snr][nm][data_snr].append(manual_sinr_rate(system.sm, h_true, g_i, no))

        print(f"SNR={data_snr:5.1f} done | perfect " + " ".join(
            f"{NAME_MAP.get(m,m)}={np.mean(perfect[m][data_snr]):.3f}" for m in METHODS), flush=True)
        for pilot_snr in PILOT_SNRS_DB:
            print(f"    pilot={pilot_snr:5.1f} | " + " ".join(
                f"{NAME_MAP.get(m,m)}={np.mean(imperfect[pilot_snr][m][data_snr]):.3f}" for m in METHODS), flush=True)

    perfect_mean = {m: {s: float(np.mean(perfect[m][s])) for s in DATA_SNRS_DB} for m in METHODS}
    imperfect_mean = {p: {m: {s: float(np.mean(imperfect[p][m][s])) for s in DATA_SNRS_DB} for m in METHODS}
                       for p in PILOT_SNRS_DB}
    conserved = {p: {m: {s: 100.0 * imperfect_mean[p][m][s] / perfect_mean[m][s] for s in DATA_SNRS_DB}
                      for m in METHODS} for p in PILOT_SNRS_DB}

    meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': ('h_true tire UNE fois par (data_snr, draw), reutilise pour la reference '
                                 'CSI parfait, pour LES 5 valeurs de SNR pilote, et pour les 5 methodes.'),
        'metric': 'manual_sinr (SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no), eq 2.2, precoders/classical.py)',
        'regime': 'umi_standard_pilot_snr_sweep', 'embed_dim': D, 'num_layers': L, 'tokens_per_rb': T,
        'checkpoints': CHECKPOINTS,
        'batch_size': BATCH, 'num_draws': NUM_DRAWS,
        'pilot_snrs_db': PILOT_SNRS_DB, 'data_snrs_db': DATA_SNRS_DB,
        'config': STANDARD_CONFIG,
        'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
        'precoders_classical_sha256_16': sha256_of('precoders/classical.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    tag = time.strftime('%Y%m%d_%H%M%S')
    out = {
        'metadata': meta,
        'table_debit_parfait_bps_hz': {NAME_MAP.get(m, m): [perfect_mean[m][s] for s in DATA_SNRS_DB] for m in METHODS},
        'table_debit_imparfait_bps_hz_par_pilote': {
            str(p): {NAME_MAP.get(m, m): [imperfect_mean[p][m][s] for s in DATA_SNRS_DB] for m in METHODS}
            for p in PILOT_SNRS_DB},
        'table_debit_conserve_pct_par_pilote': {
            str(p): {NAME_MAP.get(m, m): [conserved[p][m][s] for s in DATA_SNRS_DB] for m in METHODS}
            for p in PILOT_SNRS_DB},
        'data_snr_db': DATA_SNRS_DB, 'pilot_snr_db': PILOT_SNRS_DB,
    }
    path = f'results/eval_csi_pilot_snr_sweep_umi_standard_seedfix_{tag}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSauvegarde -> {path}")
    print(f"Temps: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
