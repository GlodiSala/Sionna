"""
tsweep_seedfix_v2_2560samples.py -- meme methodologie EXACTE que
tsweep_seedfix_v2.py (seed corrige, canal partage, SINR manuel + Sionna,
CSI parfait ET imparfait pilote 20dB, T=1,2,3,4,6,12), mais avec le
PROTOCOLE D'ECHANTILLONNAGE D'ORIGINE : 20 batches x 128 = 2560
echantillons/point (au lieu de 10x32=320 dans tsweep_seedfix_v2.py),
pour verifier si le Bug 4 (sous-echantillonnage) explique un ecart
significatif ou si 320 echantillons etait deja dans le bruit attendu.

Ne modifie aucun fichier existant -- copie de tsweep_seedfix_v2.py avec
uniquement NUM_BATCHES=20 et BATCH_SIZE_OVERRIDE=128.
"""
import os, sys, json, time, hashlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from classical_comparison import LockedClusterSystem, RB_SIZE, WMMSE_ITERS
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from channel_config import STANDARD_CONFIG
from precoder_experimental import TransformerPrecoderCleanResidualSignedAttn
from sionna.phy.utils import ebnodb2no
import pickle

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
T_VALUES = [1, 2, 3, 4, 6, 12]
CLASSICAL = ['rzf_sc', 'rzf_rb12', 'wmmse']
METHODS = CLASSICAL + [f'T{t}' for t in T_VALUES]
BATCH_SIZE = 128          # protocole d'origine (diag_tarb_residual_signed_attn_T_train.py: EVAL_BATCHES=20, batch_size=128)
NUM_BATCHES = 20          # -> 20*128 = 2560 echantillons/point
PILOT_SNR_DB = 20.0
PILOT_SEED = 2026

CKPT_JSON = {t: f'results/diag_tarb_residual_signed_attn_T{t}_train.json' for t in T_VALUES}


def build_precoder(T):
    with open(CKPT_JSON[T]) as f:
        ckpt = json.load(f)['best_ckpt']
    model = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12, tokens_per_rb=T,
        embed_dim=128, num_heads=4, num_layers=4)
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    no0 = tf.constant(0.01, tf.float32)
    _ = model(dummy_h, no=no0, training=False)
    with open(os.path.join(ckpt, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    print(f"OK poids charges T={T} depuis {ckpt}", flush=True)
    return model


def normalize_g(g):
    return tf.expand_dims(g, axis=1) if len(g.shape) == 5 else g


def manual_sinr_rate(sm, h_true, g, no):
    g = normalize_g(g)
    h_pc = _get_desired_channels(h_true, sm)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    Y = tf.reduce_mean(rate, axis=[1, 2])
    return float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


t0 = time.time()
system = LockedClusterSystem(M, K)
system.cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

precoders = {t: build_precoder(t) for t in T_VALUES}
pilot_rng = np.random.RandomState(PILOT_SEED)

results_perfect_sionna = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}
results_perfect_manual = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}
results_pilot_sionna = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}
results_pilot_manual = {m: {s: [] for s in REPORT_SNRS} for m in METHODS}

for snr in REPORT_SNRS:
    snr_t = tf.constant(snr, tf.float32)
    for b in range(NUM_BATCHES):
        system.new_topology(BATCH_SIZE)
        h_true, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
        h_est = noisy_channel(h_true, PILOT_SNR_DB, pilot_rng)

        g_perfect, g_pilot = {}, {}
        g_perfect['rzf_sc'] = rzf_precoder(h_true, stream_management=system.sm, no=no)
        g_perfect['rzf_rb12'] = normalize_g(rzf_precoder_with_rb_grouping(h_true, stream_management=system.sm, rb_size=RB_SIZE))
        g_perfect['wmmse'] = wmmse_precoder(h_true, no, system.sm, num_iterations=WMMSE_ITERS)
        g_pilot['rzf_sc'] = rzf_precoder(h_est, stream_management=system.sm, no=no)
        g_pilot['rzf_rb12'] = normalize_g(rzf_precoder_with_rb_grouping(h_est, stream_management=system.sm, rb_size=RB_SIZE))
        g_pilot['wmmse'] = wmmse_precoder(h_est, no, system.sm, num_iterations=WMMSE_ITERS)
        for t in T_VALUES:
            g_perfect[f'T{t}'] = precoders[t](h_true, no=no, training=False)
            g_pilot[f'T{t}'] = precoders[t](h_est, no=no, training=False)

        for m in METHODS:
            results_perfect_sionna[m][snr].append(float(system._sum_rate(h_true, g_perfect[m], no)))
            results_perfect_manual[m][snr].append(manual_sinr_rate(system.sm, h_true, g_perfect[m], no))
            results_pilot_sionna[m][snr].append(float(system._sum_rate(h_true, g_pilot[m], no)))
            results_pilot_manual[m][snr].append(manual_sinr_rate(system.sm, h_true, g_pilot[m], no))

    print(f"SNR={snr:5.1f} done | perfect(sionna): " +
          " ".join(f"{m}={np.mean(results_perfect_sionna[m][snr]):.2f}" for m in METHODS), flush=True)
    print(f"SNR={snr:5.1f} done | pilot20(sionna): " +
          " ".join(f"{m}={np.mean(results_pilot_sionna[m][snr]):.2f}" for m in METHODS), flush=True)


def build_table(out):
    t = {'snr': REPORT_SNRS}
    for m in METHODS:
        t[m] = [float(np.mean(out[m][s])) for s in REPORT_SNRS]
    return t


def build_std(out):
    t = {'snr': REPORT_SNRS}
    for m in METHODS:
        t[m] = [float(np.std(out[m][s])) for s in REPORT_SNRS]
    return t


base_meta = {
    'seed': SEED, 'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
    'shared_channel': True,
    'shared_channel_note': 'h_true/no tires UNE fois par batch, reutilises identiques pour RZF-SC/RZF-RB12/WMMSE et les 6 T (CSI parfait) ; h_est derive du meme h_true (bruit pilote 20dB) reutilise pareillement pour CSI imparfait',
    'pilot_snr_db': PILOT_SNR_DB, 'pilot_seed': PILOT_SEED,
    'sampling_note': 'protocole original (diag_tarb_residual_signed_attn_T_train.py: EVAL_BATCHES=20, batch_size=128) -- reprend a l identique le nombre d echantillons/point du run d entrainement, en remplacement du run intermediaire 10x32=320 (tsweep_seedfix_v2.py)',
    'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
    'precoders_w_sha256_16': sha256_of('precoders_w.py'),
    'channel_config_sha256_16': sha256_of('channel_config.py'),
    'checkpoints': {f'T{t}': json.load(open(CKPT_JSON[t]))['best_ckpt'] for t in T_VALUES},
    'batch_size': BATCH_SIZE, 'num_batches': NUM_BATCHES, 'rb_size': RB_SIZE, 'wmmse_iters': WMMSE_ITERS,
    'config': STANDARD_CONFIG, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
}

date_tag = time.strftime('%Y%m%d_%H%M%S')
outputs = {
    'tsweep_seedfix_perfect_sionna_2560': {'metric': 'sionna', 'csi': 'perfect', 'table': build_table(results_perfect_sionna), 'std_per_batch_mean': build_std(results_perfect_sionna), 'metadata': base_meta},
    'tsweep_seedfix_perfect_manual_2560': {'metric': 'manual', 'csi': 'perfect', 'table': build_table(results_perfect_manual), 'std_per_batch_mean': build_std(results_perfect_manual), 'metadata': base_meta},
    'tsweep_seedfix_pilot20_sionna_2560': {'metric': 'sionna', 'csi': 'pilot20dB', 'table': build_table(results_pilot_sionna), 'std_per_batch_mean': build_std(results_pilot_sionna), 'metadata': base_meta},
    'tsweep_seedfix_pilot20_manual_2560': {'metric': 'manual', 'csi': 'pilot20dB', 'table': build_table(results_pilot_manual), 'std_per_batch_mean': build_std(results_pilot_manual), 'metadata': base_meta},
}
for name, out in outputs.items():
    path = f'results/{name}_{date_tag}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Sauvegarde -> {path}")

print("\n=== CSI PARFAIT -- Sionna ===")
print(f"{'SNR':>6} " + " ".join(f"{m:>8}" for m in METHODS))
tb = build_table(results_perfect_sionna)
for i, s in enumerate(REPORT_SNRS):
    print(f"{s:6.1f} " + " ".join(f"{tb[m][i]:8.3f}" for m in METHODS))

print("\n=== CSI PARFAIT -- Manuel ===")
tb = build_table(results_perfect_manual)
for i, s in enumerate(REPORT_SNRS):
    print(f"{s:6.1f} " + " ".join(f"{tb[m][i]:8.3f}" for m in METHODS))

print("\n=== CSI IMPARFAIT pilote20dB -- Sionna ===")
tb = build_table(results_pilot_sionna)
for i, s in enumerate(REPORT_SNRS):
    print(f"{s:6.1f} " + " ".join(f"{tb[m][i]:8.3f}" for m in METHODS))

print("\n=== CSI IMPARFAIT pilote20dB -- Manuel ===")
tb = build_table(results_pilot_manual)
for i, s in enumerate(REPORT_SNRS):
    print(f"{s:6.1f} " + " ".join(f"{tb[m][i]:8.3f}" for m in METHODS))

print(f"\nTotal time: {time.time()-t0:.0f}s")
