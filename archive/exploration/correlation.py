"""
analyze_channel.py — Analyse de la sélectivité fréquentielle du canal
Lancer : python analyze_channel.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib

import numpy as np
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paramètres — adapter si besoin ───────────────────────────────────────────
CACHE_FILE  = '/export/tmp/sala/sionna_base_5k_8x4.npz'
NUM_TX      = 8
NUM_RX      = 4
NUM_OFDM    = 14
FFT_SIZE    = 72
RB_SIZE     = 12
NUM_RB      = FFT_SIZE // RB_SIZE   # = 6
N_SAMPLES   = 1000
SEED        = 42

# =============================================================================
# 1. Charger le dataset de base
# =============================================================================
print(f"\n📂 Loading {CACHE_FILE}...")
data      = np.load(CACHE_FILE)
h_base    = data['h_freq']   # [N_base, 1, 1, 1, tx, ofdm, fft]
N_base    = h_base.shape[0]
print(f"   Base samples : {N_base}")

# Construire N_SAMPLES canaux multi-user par SAGE-HB
rng = np.random.default_rng(SEED)
idx = rng.integers(0, N_base, size=(N_SAMPLES, NUM_RX))
# [N_SAMPLES, rx, 1, 1, tx, ofdm, fft]
h_multi = h_base[idx]
# squeeze → [N_SAMPLES, rx, tx, ofdm, fft]
h_sq = h_multi[:, :, 0, 0, :, :, :]
print(f"   h_sq shape   : {h_sq.shape}")

h_sq = tf.constant(h_sq, dtype=tf.complex64)

# =============================================================================
# 2. Corrélation intra-RB
# =============================================================================
print("\n" + "="*60)
print("  CORRÉLATION INTRA-RB")
print("="*60)

# [N, rx, tx, ofdm, num_rb, rb_size]
h_rb = tf.reshape(h_sq,
    [N_SAMPLES, NUM_RX, NUM_TX, NUM_OFDM, NUM_RB, RB_SIZE])

def sc_corr(h_rb, i, j):
    """Corrélation entre SC i et SC j, moyennée sur tout."""
    hi = h_rb[..., i]   # [N, rx, tx, ofdm, num_rb]
    hj = h_rb[..., j]
    num   = tf.abs(tf.reduce_mean(tf.math.conj(hi) * hj))
    denom = tf.sqrt(tf.reduce_mean(tf.abs(hi)**2) *
                    tf.reduce_mean(tf.abs(hj)**2))
    return float(num / (denom + 1e-12))

# Corrélation pour chaque lag à l'intérieur d'un RB
lags  = list(range(1, RB_SIZE))
corrs = [sc_corr(h_rb, 0, lag) for lag in lags]

print(f"\n  SC0 vs SC_lag  (lag = distance en sous-porteuses) :")
for lag, c in zip(lags, corrs):
    bar = '█' * int(c * 30)
    print(f"  lag={lag:2d}  {c:.4f}  {bar}")

print(f"\n  ► Corr SC0-SC1  (adjacent)    : {corrs[0]:.4f}")
print(f"  ► Corr SC0-SC11 (full RB span): {corrs[-1]:.4f}")

# Interprétation
c_adj  = corrs[0]
c_span = corrs[-1]
if c_span > 0.90:
    verdict = "QUASI-PLAT — mean pooling excellent, weight sharing très justifié"
elif c_span > 0.75:
    verdict = "MODÉRÉMENT SÉLECTIF — mean pooling acceptable, weight sharing justifié"
elif c_span > 0.55:
    verdict = "SÉLECTIF — mean pooling perd de l'info, rb_pos_bias recommandé"
else:
    verdict = "TRÈS SÉLECTIF — weight sharing discutable, envisager rb_size=6"
print(f"\n  Verdict : {verdict}")

# =============================================================================
# 3. Corrélation inter-RB (entre RBs)
# =============================================================================
print("\n" + "="*60)
print("  CORRÉLATION INTER-RB")
print("="*60)

# Token RB = moyenne sur les SC du RB
h_rb_mean = tf.reduce_mean(h_rb, axis=-1)  # [N, rx, tx, ofdm, num_rb]

print(f"\n  RB0 vs RB_k :")
for k in range(1, NUM_RB):
    rb0 = h_rb_mean[..., 0]
    rbk = h_rb_mean[..., k]
    num   = tf.abs(tf.reduce_mean(tf.math.conj(rb0) * rbk))
    denom = tf.sqrt(tf.reduce_mean(tf.abs(rb0)**2) *
                    tf.reduce_mean(tf.abs(rbk)**2))
    c = float(num / (denom + 1e-12))
    bar = '█' * int(c * 30)
    print(f"  RB0 vs RB{k}  {c:.4f}  {bar}")

# =============================================================================
# 4. Corrélation full-band SC0 → SC71
# =============================================================================
print("\n" + "="*60)
print("  PROFIL DE CORRÉLATION FULL-BAND (SC0 → SC71)")
print("="*60)

h_sc = tf.reshape(h_sq, [N_SAMPLES, NUM_RX, NUM_TX, NUM_OFDM, FFT_SIZE])
full_corrs = []
for lag in range(0, FFT_SIZE, 2):   # tous les 2 SC pour aller vite
    hi = h_sc[..., 0]
    hj = h_sc[..., lag]
    num   = tf.abs(tf.reduce_mean(tf.math.conj(hi) * hj))
    denom = tf.sqrt(tf.reduce_mean(tf.abs(hi)**2) *
                    tf.reduce_mean(tf.abs(hj)**2))
    full_corrs.append((lag, float(num / (denom + 1e-12))))

# Cohérence fréquentielle : lag où corrélation tombe sous 0.5
coherence_sc = next((lag for lag, c in full_corrs if c < 0.5), FFT_SIZE)
coherence_hz = coherence_sc * 30e3 / 1e3  # en kHz (SCS = 30 kHz)
print(f"\n  Cohérence fréquentielle (corr < 0.5) :")
print(f"    ≈ {coherence_sc} sous-porteuses = {coherence_hz:.0f} kHz")
print(f"    RB = 12 SC = 360 kHz  →  {'INTRA-RB quasi-plat ✅' if coherence_sc >= 12 else 'INTRA-RB sélectif ⚠️'}")

# =============================================================================
# 5. Figure
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Canal UMi — Analyse sélectivité fréquentielle', fontweight='bold')

# (a) Corrélation intra-RB
ax = axes[0]
ax.plot(lags, corrs, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axhline(0.9, color='green',  linestyle='--', alpha=0.7, label='0.9 (quasi-plat)')
ax.axhline(0.7, color='orange', linestyle='--', alpha=0.7, label='0.7 (sélectif)')
ax.axhline(0.5, color='red',    linestyle='--', alpha=0.7, label='0.5 (très sélectif)')
ax.set_xlabel('Lag (SC)'); ax.set_ylabel('|Corrélation|')
ax.set_title('(a) Corrélation intra-RB')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

# (b) Profil full-band
ax = axes[1]
lags_full, c_full = zip(*full_corrs)
ax.plot(lags_full, c_full, color='purple', linewidth=1.5)
ax.axvline(coherence_sc, color='red', linestyle='--',
           label=f'Cohérence ≈ {coherence_sc} SC')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
# RB boundaries
for rb in range(NUM_RB + 1):
    ax.axvline(rb * RB_SIZE, color='steelblue', alpha=0.2, linewidth=0.8)
ax.set_xlabel('Lag (SC)'); ax.set_ylabel('|Corrélation|')
ax.set_title('(b) Profil full-band')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

# (c) Magnitude canal moyen
ax = axes[2]
h_mag = np.abs(h_sq.numpy())  # [N, rx, tx, ofdm, fft]
h_mean_mag = h_mag.mean(axis=(0, 1, 2, 3))  # [fft]
ax.plot(h_mean_mag, color='teal', linewidth=1.5)
for rb in range(NUM_RB + 1):
    ax.axvline(rb * RB_SIZE, color='gray', alpha=0.3, linewidth=0.8)
ax.set_xlabel('Subcarrier index'); ax.set_ylabel('|H| moyen')
ax.set_title('(c) Magnitude canal (moyenne)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./channel_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Figure sauvegardée : ./channel_analysis.png")

# =============================================================================
# 6. Résumé pour l'architecture
# =============================================================================
print("\n" + "="*60)
print("  RÉSUMÉ ARCHITECTURAL")
print("="*60)
print(f"  Corr SC0-SC1  (adjacent)     : {corrs[0]:.4f}")
print(f"  Corr SC0-SC11 (full RB)      : {corrs[-1]:.4f}")
print(f"  Cohérence fréquentielle      : ~{coherence_sc} SC = {coherence_hz:.0f} kHz")
print(f"  RB size (5G NR)              : 12 SC = 360 kHz")
print()
if c_span > 0.85:
    print("  → Mean pooling : JUSTIFIÉ ✅")
    print("  → Weight sharing intra-RB : JUSTIFIÉ ✅")
    print("  → rb_pos_bias : optionnel, gain marginal attendu")
elif c_span > 0.65:
    print("  → Mean pooling : ACCEPTABLE ⚠️  (perte modérée)")
    print("  → Weight sharing : JUSTIFIÉ mais rb_pos_bias recommandé")
    print("  → rb_pos_bias : RECOMMANDÉ pour capter les bords de bande")
else:
    print("  → Mean pooling : PROBLÉMATIQUE ❌")
    print("  → Weight sharing strict : DISCUTABLE")
    print("  → Considérer rb_size=6 ou learned pooling")
print("="*60)