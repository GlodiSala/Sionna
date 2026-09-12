"""
experiments/mechanism_rzf_grouping_degradation.py -- montre la degradation progressive du
SINR de RZF quand on regroupe T=1,2,3,4,6,12 sous-porteuses consecutives
sous un meme precodeur (moyenne du canal sur le groupe, precodeur RZF
calcule sur cette moyenne, applique tel quel a chaque SC du groupe).

Reutilise rzf_precoder_with_rb_grouping(h_freq, sm, rb_size=T) tel quel
(precoders/rb_grouping.py) -- pour T=1, aucune moyenne (groupe de taille 1),
equivaut au RZF par sous-porteuse standard ; pour T=12, equivaut a
RZF-RB12 (Table 3.9).

Meme canal/seed/mecanisme que tab:results_umi (classical_comparison_
seedfix.py -- sionna_config.seed=42, LockedClusterSystem, STANDARD_CONFIG).

Ne modifie aucun fichier existant.
"""
import os, sys, json, time
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

from sionna.phy.utils import ebnodb2no
from classical_reference import LockedClusterSystem, BATCH_SIZE
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from precoders.classical import _get_desired_channels
from channel_config import STANDARD_CONFIG

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
T_VALUES = [1, 2, 3, 4, 6, 12]
SNR_DB = 20.0
NUM_BATCHES = 10

t0 = time.time()
system = LockedClusterSystem(M, K)
system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']


def manual_sinr(stream_management, h_freq, g, no):
    """SINR_k = |h_k^H w_k|^2 / (sum_j!=k |h_k^H w_j|^2 + no) -- meme
    formule que precoders/classical.py:103-110. Retourne (sinr [B,ofdm,fft,K],
    rate_total scalaire, matrice HW [B,ofdm,fft,K,K])."""
    g5 = g if len(g.shape) == 6 else tf.expand_dims(g, axis=1)
    h_pc = _get_desired_channels(h_freq, stream_management)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g5, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate_per = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    Y = tf.reduce_mean(rate_per, axis=[1, 2])
    rate_total = float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))
    return sinr.numpy(), rate_total, HW.numpy()


results_per_T = {}
example_HW = {}   # T=1 et T=12, 1ere realisation/ofdm/sc, pour illustration

snr_t = tf.constant(SNR_DB, dtype=tf.float32)
for T in T_VALUES:
    all_sinr = []
    all_rates = []
    for b in range(NUM_BATCHES):
        system.new_topology(BATCH_SIZE)
        h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
        g = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=T)
        sinr, rate_total, HW = manual_sinr(system.sm, h_freq, g, no)
        all_sinr.append(sinr.reshape(-1))
        all_rates.append(rate_total)
        if b == 0 and T in (1, 12):
            example_HW[T] = HW[0, 2, 0].tolist()  # [realisation 0, ofdm=2, sc=0] -> KxK complexe (liste [re,im] via json plus bas)

    all_sinr = np.concatenate(all_sinr)
    sinr_db = 10 * np.log10(np.maximum(all_sinr, 1e-12))
    results_per_T[T] = {
        'sinr_linear_mean': float(all_sinr.mean()),
        'sinr_linear_median': float(np.median(all_sinr)),
        'sinr_dB_mean': float(sinr_db.mean()),
        'sinr_dB_std': float(sinr_db.std()),
        'sum_rate_mean': float(np.mean(all_rates)),
        'sum_rate_std': float(np.std(all_rates)),
        'n_sinr_samples': int(all_sinr.size),
    }
    print(f"T={T:3d} | SINR moyen = {results_per_T[T]['sinr_dB_mean']:6.2f} dB "
          f"(std={results_per_T[T]['sinr_dB_std']:.2f}) | "
          f"debit-somme = {results_per_T[T]['sum_rate_mean']:.3f} bps/Hz "
          f"(std={results_per_T[T]['sum_rate_std']:.3f})", flush=True)

# ---- 3. Exemple concret HW pour T=1 vs T=12 (1ere realisation, ofdm=2, sc=0) ----
print("\n=== Exemple H@W (realisation 0, ofdm=2, sc=0) ===")
for T in (1, 12):
    HW0 = np.array(example_HW[T])
    print(f"T={T} :")
    print("  |HW| (module) =")
    for row in np.abs(HW0):
        print("   ", " ".join(f"{v:6.3f}" for v in row))
    diag_mean = float(np.mean(np.abs(np.diag(HW0))))
    offdiag = HW0 - np.diag(np.diag(HW0))
    offdiag_mean = float(np.mean(np.abs(offdiag)[np.abs(offdiag) > 0]))
    print(f"  moyenne |diagonale| = {diag_mean:.4f} | moyenne |hors-diagonale| = {offdiag_mean:.4f}")

# ---- 4. Monotonie ----
sinr_seq = [results_per_T[T]['sinr_dB_mean'] for T in T_VALUES]
rate_seq = [results_per_T[T]['sum_rate_mean'] for T in T_VALUES]
is_monotone_sinr = all(sinr_seq[i] >= sinr_seq[i+1] - 1e-9 for i in range(len(sinr_seq)-1))
is_monotone_rate = all(rate_seq[i] >= rate_seq[i+1] - 1e-9 for i in range(len(rate_seq)-1))
print(f"\n=== Monotonie ===")
print(f"SINR (dB) par T : {[round(v,2) for v in sinr_seq]} -> monotone decroissant = {is_monotone_sinr}")
print(f"Debit-somme par T : {[round(v,3) for v in rate_seq]} -> monotone decroissant = {is_monotone_rate}")

out = {
    'config': {'M': M, 'K': K, 'scenario': 'umi', 'snr_db': SNR_DB,
               'num_batches': NUM_BATCHES, 'batch_size': BATCH_SIZE, 'seed': SEED,
               'T_values': T_VALUES},
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    'results_per_T': results_per_T,
    'example_HW_T1': example_HW[1],
    'example_HW_T12': example_HW[12],
    'monotone_sinr_decreasing': is_monotone_sinr,
    'monotone_rate_decreasing': is_monotone_rate,
}

class ComplexEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return {'re': o.real, 'im': o.imag}
        return super().default(o)

with open('results/verify_rzf_grouping_degradation_result.json', 'w') as f:
    json.dump(out, f, indent=2, cls=ComplexEncoder)
print(f"\nSauvegarde -> results/verify_rzf_grouping_degradation_result.json")
print(f"Total time: {time.time()-t0:.0f}s")
