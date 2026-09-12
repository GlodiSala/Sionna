"""
experiments/mechanism_interference_floor.py -- verification numerique de l'hypothese :
l'ecart RZF-RB vs RZF-SC vient d'un terme d'interference residuelle
Delta_sc @ pinv(H_avg) (Delta_sc = H_sc - H_avg), constant en puissance
(ne depend que du canal, pas du bruit), qui domine specifiquement a haut
SNR (plancher d'interference).

Config : M8K4, canal UMi, meme mecanisme/seed que tab:results_umi
(classical_comparison_seedfix.py -- sionna_config.seed=42, LockedClusterSystem,
STANDARD_CONFIG).

Methodologie :
  1. UN SEUL tirage de canal (h_freq), independant du SNR (_gen_channel).
  2. Pour chaque realisation b, chaque RB r (8 RBs de 12 SC), 1 symbole
     OFDM (pilot_idx=2, canal statique confirme ailleurs dans le depot) :
       H_avg[b,r]   = moyenne des 12 H_sc du RB          (K x M)
       Delta_sc      = H_sc - H_avg[b,r]                  (K x M), pour
                        chacune des 12 SC
       P_pred        = Delta_sc @ pinv(H_avg[b,r])         (K x K) --
                        terme d'interference residuelle predite (H_sc @
                        pinv(H_avg) = I + P_pred puisque H_avg @
                        pinv(H_avg) = I par construction du pseudo-inverse
                        de Moore-Penrose, K<M, rang plein)
     -> ||P_pred||_F, ||offdiag(P_pred)||_F, ||Delta_sc||_F/||H_avg||_F
        (P_pred ne depend QUE du canal h_freq, jamais de `no` -- donc
        constant en SNR par construction ; verifie explicitement au
        point 3 en re-derivant le meme h_freq pour chaque SNR).
  3. Sur ce MEME h_freq, a 0/5/10/15/20dB : RZF-SC (rzf_precoder, alpha=
     no/2) vs RZF-RB (rzf_precoder_with_rb_grouping, alpha=0.1 fixe) --
     debit Sionna (_sum_rate) ET SINR manuel (meme formule que
     precoders/classical.py:103-110), par realisation, pour observer si l'ecart
     converge vers un plancher a haut SNR (comme predit) plutot que de
     continuer a croitre.
  4. Comparaison a rho(12)=0.89 (coherence intra-RB deja etablie ailleurs
     dans le memoire) via ||Delta_sc||_F/||H_avg||_F.

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
from classical_reference import LockedClusterSystem, BATCH_SIZE, RB_SIZE
from precoders.rb_grouping import rzf_precoder_with_rb_grouping
from precoders.classical import rzf_precoder, _get_desired_channels
from channel_config import STANDARD_CONFIG

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_IDX = 2
FFT_SIZE = 96
NUM_RB = FFT_SIZE // RB_SIZE   # 8


def manual_sinr_rate_and_per_user(stream_management, h_freq, g, no):
    """Meme formule que precoders/classical.py:103-110 (SINR_k = |h_k^H w_k|^2 /
    (sum_j!=k |h_k^H w_j|^2 + no)), avec retour du debit total ET du
    vecteur SINR par (batch, ofdm, sc, user) pour analyse fine."""
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
    total = float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))
    return total


t0 = time.time()
system = LockedClusterSystem(M, K)
system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

# ---- 1 seul tirage de canal, independant du SNR ----
system.new_topology(BATCH_SIZE)
h_freq = system._gen_channel(BATCH_SIZE)
print(f"h_freq shape: {h_freq.shape}", flush=True)

# ---- H_sc, H_avg, Delta_sc, P_pred (independant de no) ----
h_pc = _get_desired_channels(h_freq, system.sm)                 # [B,1,ofdm,fft,K,M]
H_full = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)        # [B,ofdm,fft,K,M]
H = H_full[:, PILOT_IDX, :, :, :].numpy()                        # [B,fft,K,M]
B = H.shape[0]
H_rb = H.reshape(B, NUM_RB, RB_SIZE, K, M)
H_avg = H_rb.mean(axis=2)                                        # [B,NUM_RB,K,M]

per_realization = []   # une entree par (b, r) avec les 12 P_pred (b,r,sc)
delta_ratios = []      # ||Delta||_F/||H_avg||_F, toutes realisations/RB/SC
p_norms = []           # ||P_pred||_F
p_offdiag_norms = []   # ||offdiag(P_pred)||_F

for b in range(B):
    for r in range(NUM_RB):
        Havg_br = H_avg[b, r]                      # K x M
        pinv_Havg = np.linalg.pinv(Havg_br)         # M x K
        # sanity check : Havg @ pinv(Havg) ~= I_K
        eye_check = Havg_br @ pinv_Havg
        entry = {'b': b, 'rb': r,
                 'eye_check_offdiag_norm': float(np.linalg.norm(eye_check - np.eye(K), ord='fro')),
                 'sc_entries': []}
        for sc in range(RB_SIZE):
            Hsc = H_rb[b, r, sc]                     # K x M
            Delta = Hsc - Havg_br
            P = Delta @ pinv_Havg                    # K x K
            offdiag = P - np.diag(np.diag(P))
            p_norm = float(np.linalg.norm(P, ord='fro'))
            p_off_norm = float(np.linalg.norm(offdiag, ord='fro'))
            delta_ratio = float(np.linalg.norm(Delta, ord='fro') / np.linalg.norm(Havg_br, ord='fro'))
            entry['sc_entries'].append({'sc': sc, 'P_norm': p_norm,
                                         'P_offdiag_norm': p_off_norm,
                                         'delta_over_havg': delta_ratio})
            p_norms.append(p_norm)
            p_offdiag_norms.append(p_off_norm)
            delta_ratios.append(delta_ratio)
        per_realization.append(entry)

p_norms = np.array(p_norms)
p_offdiag_norms = np.array(p_offdiag_norms)
delta_ratios = np.array(delta_ratios)

print(f"\n=== TERME PREDIT Delta_sc @ pinv(H_avg) -- {len(p_norms)} (realisation,RB,SC) ===")
print(f"||P_pred||_F        : mean={p_norms.mean():.4f} std={p_norms.std():.4f} "
      f"min={p_norms.min():.4f} max={p_norms.max():.4f}")
print(f"||offdiag(P_pred)||_F: mean={p_offdiag_norms.mean():.4f} std={p_offdiag_norms.std():.4f} "
      f"min={p_offdiag_norms.min():.4f} max={p_offdiag_norms.max():.4f}")
print(f"||Delta_sc||_F/||H_avg||_F : mean={delta_ratios.mean():.4f} std={delta_ratios.std():.4f}")

# ---- 3. Sur ce MEME h_freq, ecart RZF-SC vs RZF-RB a 5 SNR (Sionna + manuel) ----
snr_results = {}
for snr in REPORT_SNRS:
    no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)

    g_sc = rzf_precoder(h_freq, stream_management=system.sm, no=no)
    g_rb = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=RB_SIZE)

    rate_sc_sionna = float(system._sum_rate(h_freq, g_sc, no))
    rate_rb_sionna = float(system._sum_rate(h_freq, g_rb, no))
    rate_sc_manual = manual_sinr_rate_and_per_user(system.sm, h_freq, g_sc, no)
    rate_rb_manual = manual_sinr_rate_and_per_user(system.sm, h_freq, g_rb, no)

    snr_results[snr] = {
        'rzf_sc_sionna': rate_sc_sionna, 'rzf_rb_sionna': rate_rb_sionna,
        'gap_sionna': rate_sc_sionna - rate_rb_sionna,
        'rzf_sc_manual': rate_sc_manual, 'rzf_rb_manual': rate_rb_manual,
        'gap_manual': rate_sc_manual - rate_rb_manual,
    }
    print(f"SNR={snr:5.1f} | Sionna: SC={rate_sc_sionna:.3f} RB={rate_rb_sionna:.3f} "
          f"gap={rate_sc_sionna-rate_rb_sionna:.3f} | "
          f"Manuel: SC={rate_sc_manual:.3f} RB={rate_rb_manual:.3f} "
          f"gap={rate_sc_manual-rate_rb_manual:.3f}", flush=True)

print("\n=== VERIFICATION PLANCHER (ecart SC-RB vs SNR) ===")
gaps_sionna = [snr_results[s]['gap_sionna'] for s in REPORT_SNRS]
gaps_manual = [snr_results[s]['gap_manual'] for s in REPORT_SNRS]
print("gaps (Sionna):", [f"{g:.3f}" for g in gaps_sionna])
print("gaps (manuel):", [f"{g:.3f}" for g in gaps_manual])

# ---- 4. Comparaison a rho(12)=0.89 ----
RHO_12 = 0.89
print(f"\n=== COMPARAISON A rho(12)={RHO_12} ===")
print(f"||Delta_sc||_F/||H_avg||_F moyen = {delta_ratios.mean():.4f} "
      f"(equivalent a une decorrelation ~{1-delta_ratios.mean():.4f} si on assimile a 1-rho)")

# ---- 5. Sauvegarde des donnees brutes ----
out = {
    'config': {'M': M, 'K': K, 'scenario': 'umi', 'pilot_idx': PILOT_IDX,
               'rb_size': RB_SIZE, 'num_rb': NUM_RB, 'batch_size': BATCH_SIZE,
               'seed': SEED, 'rho_12_reference': RHO_12},
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    'P_pred_stats': {
        'norm_mean': float(p_norms.mean()), 'norm_std': float(p_norms.std()),
        'norm_min': float(p_norms.min()), 'norm_max': float(p_norms.max()),
        'offdiag_norm_mean': float(p_offdiag_norms.mean()), 'offdiag_norm_std': float(p_offdiag_norms.std()),
        'delta_over_havg_mean': float(delta_ratios.mean()), 'delta_over_havg_std': float(delta_ratios.std()),
    },
    'P_pred_raw_norms': p_norms.tolist(),
    'P_pred_raw_offdiag_norms': p_offdiag_norms.tolist(),
    'delta_over_havg_raw': delta_ratios.tolist(),
    'per_realization_detail': per_realization,   # b, rb, eye_check, 12 sc entries chacun
    'snr_sweep_same_channel': snr_results,
    'gaps_vs_snr': {'sionna': gaps_sionna, 'manual': gaps_manual, 'snrs': REPORT_SNRS},
}
with open('results/verify_interference_floor_result.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegarde -> results/verify_interference_floor_result.json")
print(f"Total time: {time.time()-t0:.0f}s")
