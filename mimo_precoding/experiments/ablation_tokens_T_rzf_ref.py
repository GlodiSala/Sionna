"""
experiments/ablation_tokens_T_rzf_ref.py -- reproduit les courbes
"RZF groupe a compression equivalente" de l'ancienne figC_sumrate_snr_T.py
(diag_rzf_grouped_for_tsweep.py, rb_size=RB_SIZE//T) sur EXACTEMENT le
meme canal que le run T-sweep seedfix a 2560 echantillons/point
(experiments/ablation_tokens_T.py) : meme seed=42, meme
LockedClusterSystem/STANDARD_CONFIG, meme BATCH_SIZE=128/NUM_BATCHES=20,
meme boucle SNR->batch->new_topology()->channel_and_no() DANS LE MEME
ORDRE (aucun autre appel ne consomme le flux RNG global config.tf_rng
entre-temps, donc les tirages de canal sont bit-identiques -- verifie
par recoupement direct : rb_size=12 et rb_size=1 ci-dessous DOIVENT
reproduire exactement rzf_rb12/rzf_sc du fichier
tsweep_seedfix_perfect_manual_2560_20260820_153545.json).

Ne recalcule PAS WMMSE ni les 6 transformateurs TA-RB (deterministes,
ne consomment pas config.tf_rng, donc inutiles a rejouer pour garantir
la reproductibilite du canal) -- seulement les 6 RZF groupes.

Mapping (RB_SIZE=12) : T=1->rb_size=12 (= RZF-RB12), T=2->rb_size=6,
T=3->rb_size=4, T=4->rb_size=3, T=6->rb_size=2, T=12->rb_size=1 (= RZF-SC).
"""
import os, sys, json, time, hashlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from classical_reference import LockedClusterSystem, RB_SIZE
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from precoders.classical import _get_desired_channels
from channel_config import STANDARD_CONFIG

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
T_VALUES = [1, 2, 3, 4, 6, 12]
GROUP_SIZE = {T: RB_SIZE // T for T in T_VALUES}   # {1:12, 2:6, 3:4, 4:3, 6:2, 12:1}
BATCH_SIZE = 128
NUM_BATCHES = 20


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


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


t0 = time.time()
system = LockedClusterSystem(M, K)
system.cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

results = {T: {s: [] for s in REPORT_SNRS} for T in T_VALUES}

# -- boucle IDENTIQUE (ordre des appels RNG) a experiments/ablation_tokens_T.py --
for snr in REPORT_SNRS:
    snr_t = tf.constant(snr, tf.float32)
    for b in range(NUM_BATCHES):
        system.new_topology(BATCH_SIZE)
        h_true, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)

        for T in T_VALUES:
            g = normalize_g(rzf_precoder_with_rb_grouping(h_true, stream_management=system.sm, rb_size=GROUP_SIZE[T]))
            results[T][snr].append(manual_sinr_rate(system.sm, h_true, g, no))

    print(f"SNR={snr:5.1f} done | " + " ".join(f"T{T}(rb{GROUP_SIZE[T]})={np.mean(results[T][snr]):.3f}" for T in T_VALUES), flush=True)

table = {'snr': REPORT_SNRS}
for T in T_VALUES:
    table[f'T{T}'] = [float(np.mean(results[T][s])) for s in REPORT_SNRS]

# -- verification de reproductibilite : rb_size=12 (T=1) et rb_size=1 (T=12)
# doivent reproduire EXACTEMENT rzf_rb12 / rzf_sc du run 2560 deja sauvegarde --
ref = json.load(open('results/tsweep_seedfix_perfect_manual_2560_20260820_153545.json'))['table']
repro_check = {
    'T1_vs_rzf_rb12': {'nouveau': table['T1'], 'reference': ref['rzf_rb12'],
                        'max_abs_diff': max(abs(a - b) for a, b in zip(table['T1'], ref['rzf_rb12']))},
    'T12_vs_rzf_sc': {'nouveau': table['T12'], 'reference': ref['rzf_sc'],
                       'max_abs_diff': max(abs(a - b) for a, b in zip(table['T12'], ref['rzf_sc']))},
}
print("\n=== VERIFICATION REPRODUCTIBILITE (doit etre ~0) ===")
print(f"T1 (rb_size=12) vs rzf_rb12 du run 2560 : max|diff| = {repro_check['T1_vs_rzf_rb12']['max_abs_diff']:.2e}")
print(f"T12 (rb_size=1) vs rzf_sc du run 2560   : max|diff| = {repro_check['T12_vs_rzf_sc']['max_abs_diff']:.2e}")

out = {
    'description': 'RZF groupe a compression equivalente (rb_size=RB_SIZE//T) sur le meme canal que le run T-sweep 2560 echantillons/point',
    'table': table,
    'group_size_mapping': GROUP_SIZE,
    'repro_check_vs_2560run': repro_check,
    'metadata': {
        'seed': SEED, 'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': 'meme sequence exacte de system.new_topology()/channel_and_no() que experiments/ablation_tokens_T.py (WMMSE/T1-T12 neuronaux non rejoues, deterministes, ne consomment pas config.tf_rng) -- verifie par repro_check_vs_2560run',
        'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders/classical.py'),
        'compare_rb_grouping_sha256_16': sha256_of('precoders/rb_grouping.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'batch_size': BATCH_SIZE, 'num_batches': NUM_BATCHES, 'rb_size': RB_SIZE,
        'config': STANDARD_CONFIG, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'reference_tsweep_run': 'results/tsweep_seedfix_perfect_manual_2560_20260820_153545.json',
    },
}
date_tag = time.strftime('%Y%m%d_%H%M%S')
path = f'results/diag_rzf_grouped_for_tsweep_2560seedfix_{date_tag}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegarde -> {path}")
print(f"Total time: {time.time()-t0:.0f}s")
