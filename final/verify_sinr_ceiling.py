"""
verify_sinr_ceiling.py -- derive un plafond de SINR/debit theorique a partir
de P_pred = Delta_sc @ pinv(H_avg) (meme canal/mecanisme que
verify_interference_floor.py : M8K4, UMi, seed=42, LockedClusterSystem,
STANDARD_CONFIG, 32 realisations), et le compare au plafond REELLEMENT
mesure en simulant RZF-RB a tres haut SNR (30/40/50/60dB).

Formule (donnee par l'utilisateur) : pour chaque (realisation b, RB r),
utilisateur k :
    SINR_ceiling[k] = |1+P_pred[k,k]|^2 / sum_{j!=k} |P_pred[k,j]|^2
    debit-somme plafond du RB = sum_k log2(1+SINR_ceiling[k])

P_pred varie par sous-porteuse (12 SC/RB, chacune a son propre Delta_sc) --
on agrege les 12 P_pred du RB en moyennant les puissances (signal et
interference) AVANT de prendre le ratio, ce qui correspond a ce que
mesurerait une simulation moyennee sur les 12 SC du RB (meme precodeur
RZF-RB applique aux 12 SC, chacune avec sa propre erreur residuelle) :
    signal_power[k]      = mean_sc |1+P_pred_sc[k,k]|^2
    interference_power[k]= mean_sc sum_{j!=k} |P_pred_sc[k,j]|^2
    SINR_ceiling[k]       = signal_power[k] / interference_power[k]

Ne modifie aucun fichier existant.
"""
import os, sys, json, time
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

from sionna.phy.utils import ebnodb2no
from classical_comparison import LockedClusterSystem, BATCH_SIZE, RB_SIZE
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import _get_desired_channels
from channel_config import STANDARD_CONFIG

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
PILOT_IDX = 2
FFT_SIZE = 96
NUM_RB = FFT_SIZE // RB_SIZE
HIGH_SNRS = [30.0, 40.0, 50.0, 60.0]

t0 = time.time()
system = LockedClusterSystem(M, K)
system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

# ---- meme tirage que verify_interference_floor.py (meme seed/mecanisme/ordre d'appel) ----
system.new_topology(BATCH_SIZE)
h_freq = system._gen_channel(BATCH_SIZE)

h_pc = _get_desired_channels(h_freq, system.sm)
H_full = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
H = H_full[:, PILOT_IDX, :, :, :].numpy()          # [B,fft,K,M]
B = H.shape[0]
H_rb = H.reshape(B, NUM_RB, RB_SIZE, K, M)
H_avg = H_rb.mean(axis=2)                           # [B,NUM_RB,K,M]

# ---- 1. Plafond PREDIT (P_pred complet, complexe, par SC, agrege par RB) ----
per_rb_ceiling_rate = []      # une valeur par (b,r) : sum_k log2(1+SINR_ceiling[k])
per_rb_sinr_ceiling = []      # [K] par (b,r)
for b in range(B):
    for r in range(NUM_RB):
        Havg_br = H_avg[b, r]
        pinv_Havg = np.linalg.pinv(Havg_br)
        signal_pow_acc = np.zeros(K)
        interf_pow_acc = np.zeros(K)
        for sc in range(RB_SIZE):
            Hsc = H_rb[b, r, sc]
            Delta = Hsc - Havg_br
            P = Delta @ pinv_Havg                    # K x K complexe
            M_eff = np.eye(K) + P                     # H_sc @ pinv(H_avg)
            for k in range(K):
                signal_pow_acc[k]  += np.abs(M_eff[k, k]) ** 2
                interf_pow_acc[k]  += np.sum(np.abs(M_eff[k, :]) ** 2) - np.abs(M_eff[k, k]) ** 2
        signal_pow = signal_pow_acc / RB_SIZE
        interf_pow = interf_pow_acc / RB_SIZE
        sinr_ceiling = signal_pow / np.maximum(interf_pow, 1e-30)
        rate_ceiling = float(np.sum(np.log2(1.0 + sinr_ceiling)))
        per_rb_ceiling_rate.append(rate_ceiling)
        per_rb_sinr_ceiling.append(sinr_ceiling.tolist())

per_rb_ceiling_rate = np.array(per_rb_ceiling_rate)
predicted_ceiling_mean = float(per_rb_ceiling_rate.mean())
predicted_ceiling_std = float(per_rb_ceiling_rate.std())
print(f"=== 1. PLAFOND PREDIT (P_pred) ===")
print(f"debit-somme plafond predit (moyenne sur {len(per_rb_ceiling_rate)} (b,RB)) = "
      f"{predicted_ceiling_mean:.4f} bps/Hz (std={predicted_ceiling_std:.4f})", flush=True)

# ---- 2. Plafond MESURE : simulation RZF-RB a 30/40/50/60dB, meme canal ----
print(f"\n=== 2. PLAFOND MESURE (simulation RZF-RB, SNR eleve) ===")
measured = {}
for snr in HIGH_SNRS:
    no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    g_rb = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=RB_SIZE)
    rate_sionna = float(system._sum_rate(h_freq, g_rb, no))

    # SINR manuel (meme formule que precoders_w.py:103-110)
    g5 = tf.expand_dims(g_rb, axis=1) if len(g_rb.shape) == 5 else g_rb
    Hc = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    Wc = tf.cast(tf.squeeze(g5, axis=1), tf.complex128)
    HW = tf.matmul(Hc, Wc)
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate_per = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    Y = tf.reduce_mean(rate_per, axis=[1, 2])
    rate_manual = float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))

    # rapport signal/bruit effectif : puissance moyenne du signal recu (|h_k^H w_k|^2)
    # vs no, et vs la puissance d'interference residuelle moyenne
    mean_signal_pwr = float(tf.reduce_mean(signal_pwr))
    mean_interf_pwr = float(tf.reduce_mean(interference_pwr))
    eff_snr_vs_noise = mean_signal_pwr / float(no_val)
    interf_vs_noise = mean_interf_pwr / float(no_val)

    measured[snr] = {'rate_sionna': rate_sionna, 'rate_manual': rate_manual,
                      'no': float(no_val), 'mean_signal_pwr': mean_signal_pwr,
                      'mean_interference_pwr': mean_interf_pwr,
                      'eff_signal_to_noise': eff_snr_vs_noise,
                      'interference_to_noise_ratio': interf_vs_noise}
    print(f"SNR={snr:5.1f} | Sionna={rate_sionna:.4f} Manuel={rate_manual:.4f} | "
          f"no={float(no_val):.3e} | signal/no={eff_snr_vs_noise:.2e} | "
          f"interference/no={interf_vs_noise:.2e}", flush=True)

# ---- 3. Comparaison directe ----
print(f"\n=== 3. COMPARAISON PLAFOND PREDIT vs MESURE ===")
last_snr = HIGH_SNRS[-1]
measured_asymptote_sionna = measured[last_snr]['rate_sionna']
measured_asymptote_manual = measured[last_snr]['rate_manual']
diff_sionna_pct = 100 * (measured_asymptote_sionna - predicted_ceiling_mean) / predicted_ceiling_mean
diff_manual_pct = 100 * (measured_asymptote_manual - predicted_ceiling_mean) / predicted_ceiling_mean
print(f"Predit                        : {predicted_ceiling_mean:.4f} bps/Hz")
print(f"Mesure a {last_snr}dB (Sionna)    : {measured_asymptote_sionna:.4f} bps/Hz "
      f"(ecart {diff_sionna_pct:+.2f}%, {measured_asymptote_sionna-predicted_ceiling_mean:+.4f} bps/Hz)")
print(f"Mesure a {last_snr}dB (manuel)    : {measured_asymptote_manual:.4f} bps/Hz "
      f"(ecart {diff_manual_pct:+.2f}%, {measured_asymptote_manual-predicted_ceiling_mean:+.4f} bps/Hz)")

def categorize(pct):
    a = abs(pct)
    if a < 10: return "proche (<10%)"
    if a < 30: return "moyennement proche (10-30%)"
    return "tres different (>30%)"

print(f"Categorie (Sionna) : {categorize(diff_sionna_pct)}")
print(f"Categorie (manuel) : {categorize(diff_manual_pct)}")

# ---- 4. Verification bruit negligeable a 60dB ----
print(f"\n=== 4. BRUIT NEGLIGEABLE A {last_snr}dB ? ===")
print(f"signal/no = {measured[last_snr]['eff_signal_to_noise']:.3e} "
      f"({10*np.log10(measured[last_snr]['eff_signal_to_noise']):.1f} dB effectif)")
print(f"interference/no = {measured[last_snr]['interference_to_noise_ratio']:.3e} "
      f"({10*np.log10(measured[last_snr]['interference_to_noise_ratio']):.1f} dB)")

# ---- sauvegarde ----
out = {
    'config': {'M': M, 'K': K, 'scenario': 'umi', 'pilot_idx': PILOT_IDX,
               'rb_size': RB_SIZE, 'num_rb': NUM_RB, 'batch_size': BATCH_SIZE, 'seed': SEED},
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    'predicted_ceiling': {'mean': predicted_ceiling_mean, 'std': predicted_ceiling_std,
                            'n_rb_samples': len(per_rb_ceiling_rate)},
    'predicted_ceiling_raw_per_rb': per_rb_ceiling_rate.tolist(),
    'predicted_sinr_ceiling_per_rb_per_user': per_rb_sinr_ceiling,
    'measured_high_snr': measured,
    'comparison': {
        'predicted': predicted_ceiling_mean,
        'measured_60dB_sionna': measured_asymptote_sionna,
        'measured_60dB_manual': measured_asymptote_manual,
        'diff_pct_sionna': diff_sionna_pct,
        'diff_pct_manual': diff_manual_pct,
        'category_sionna': categorize(diff_sionna_pct),
        'category_manual': categorize(diff_manual_pct),
    },
}
with open('results/verify_sinr_ceiling_result.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegarde -> results/verify_sinr_ceiling_result.json")
print(f"Total time: {time.time()-t0:.0f}s")
