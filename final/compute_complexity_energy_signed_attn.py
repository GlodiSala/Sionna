"""
compute_complexity_energy_signed_attn.py — Recalcul énergie/complexité
(demande utilisateur, priorité avant toute figure finale) pour EXACTEMENT
les 3 architectures qui ont produit les résultats finaux (signed_attn,
décodeur résiduel pour TA-RB) -- remplace compute_complexity_energy_v3.py
(obsolète : ne connaissait ni signed_attn ni le décodeur résiduel).

MÉTHODOLOGIE (confirmée/appliquée, documentée sans ambiguïté) :
  - RZF, WMMSE  : FP32 (Q_W=Q_A=32) -- calculs de référence exacts, non
    quantifiés. Inchangé vs v3.
  - Les 3 architectures neuronales (SingleSC/IntraRB/TA-RB-résiduel,
    toutes signed_attn) : **INT8 (Q_W=Q_A=8)**, PAS FP16. Le modèle
    d'énergie EXISTANT (compute_complexity_energy_v3.py, via Q_W/Q_A de
    main_finall.py) calculait en FP16 (Q_W=Q_A=16) -- CORRIGÉ ici.
    Note : l'hypothèse "la performance à 8 bits reste proche de la pleine
    précision" n'est PAS validée empiriquement dans ce dépôt (aucun script
    de robustesse à la quantification trouvé, aucune mention dans les
    logs de session) -- si le Chapitre 4 du mémoire l'établit, c'est une
    référence externe à ce code, pas quelque chose de vérifié ici. Le
    calcul d'énergie lui-même est une formule analytique (bit-width dans
    compute_energy_uJ), pas une simulation de quantification réelle des
    poids/activations.

Poids (weights) VALIDÉS exactement contre trainable_variables réel avant
ce calcul (0 écart sur les 3 architectures, cf. session log) -- les
complexity() héritées des classes de base restent correctes pour
SignedGateMHA : même nombre de projections Dense(D,D) que la
MultiHeadAttention standard qu'elle remplace (4 x Dense(D,D)+biais,
formule mha_params(d)=4*(d²+d) inchangée), et même graphe de matmuls pour
les FLOPs (les opérations supplémentaires de SignedGateMHA -- tanh,
multiplication élément-par-élément -- sont ponctuelles, non comptées par
convention dans TOUT ce modèle de complexité, comme le softmax/GELU/
SwiGLU ailleurs).

Usage: python3 compute_complexity_energy_signed_attn.py  (CPU suffit, pas de calcul GPU)
"""
import os, sys, csv, json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from main_finall import compute_energy_uJ, compute_classical_complexity
from channel_config import STANDARD_CONFIG, FFT_SIZE

NUM_OFDM = 14
D, L, H = 128, 4, 4
TOKENS_PER_RB = 6          # standard TA-RB résiduel actuel (pas 3)
Q_CLASSICAL = 32           # FP32, RZF/WMMSE -- référence exacte
Q_NEURAL    = 8            # INT8 -- résolution de déploiement (PAS FP16)

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
cfg_name = f"STANDARD (M{M}K{K}, R={STANDARD_CONFIG['CLUSTER_RADIUS_M']}m)"

rows = []


def add_row(method, flops, weights, acts, q_w, q_a, note=''):
    e = compute_energy_uJ(flops, weights, acts, q_w, q_a)
    rows.append(dict(method=method, params_k=weights / 1e3, flops_m=flops / 1e6,
                      energy_uj=e, q_w=q_w, q_a=q_a, note=note))


print(f'\n{"="*70}\n{cfg_name}\n{"="*70}')
h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)

# ── Classiques (FP32) ────────────────────────────────────────────────────
f_rzf, w_rzf, a_rzf = compute_classical_complexity('RZF', M, K, FFT_SIZE, NUM_OFDM)
add_row('RZF', f_rzf, w_rzf, a_rzf, Q_CLASSICAL, Q_CLASSICAL, 'FP32, référence exacte')

f_wmmse, w_wmmse, a_wmmse = compute_classical_complexity('WMMSE', M, K, FFT_SIZE, NUM_OFDM, I_wmmse=10)
add_row('WMMSE (I=10)', f_wmmse, w_wmmse, a_wmmse, Q_CLASSICAL, Q_CLASSICAL, 'FP32, référence exacte')

# ── SingleSC-signed_attn ─────────────────────────────────────────────────
m_single = SingleSCTransformerPrecoderSignedAttn(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                                  fft_size=FFT_SIZE, embed_dim=D,
                                                  num_heads=H, num_layers=L)
_ = m_single(h_dummy, no=tf.constant(1e-3))
real_w = int(sum(tf.size(v).numpy() for v in m_single.trainable_variables))
f_r, w_s, a_r = m_single.complexity(NUM_OFDM, convention_x_ofdm=False)
assert int(w_s) == real_w, f"SingleSC: weights mismatch {int(w_s)} vs {real_w}"
add_row('SingleSC-signed_attn', f_r, w_s, a_r, Q_NEURAL, Q_NEURAL,
        f'INT8, poids vérifiés exacts vs trainable_variables ({real_w:,})')

# ── IntraRB-signed_attn ──────────────────────────────────────────────────
m_intra = IntraRBTransformerPrecoderSignedAttn(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                                fft_size=FFT_SIZE, rb_size=12,
                                                embed_dim=D, num_heads=H, num_layers=L)
_ = m_intra(h_dummy, no=tf.constant(1e-3))
real_w = int(sum(tf.size(v).numpy() for v in m_intra.trainable_variables))
f_r, w_i, a_r = m_intra.complexity(NUM_OFDM, convention_x_ofdm=False)
assert int(w_i) == real_w, f"IntraRB: weights mismatch {int(w_i)} vs {real_w}"
add_row('IntraRB-signed_attn', f_r, w_i, a_r, Q_NEURAL, Q_NEURAL,
        f'INT8, poids vérifiés exacts vs trainable_variables ({real_w:,})')

# ── TA-RB résiduel-signed_attn (standard actuel, T=6) ────────────────────
m_tarb = TransformerPrecoderCleanResidualSignedAttn(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                                     fft_size=FFT_SIZE, rb_size=12,
                                                     tokens_per_rb=TOKENS_PER_RB,
                                                     embed_dim=D, num_heads=H, num_layers=L)
_ = m_tarb(h_dummy, no=tf.constant(1e-3))
real_w = int(sum(tf.size(v).numpy() for v in m_tarb.trainable_variables))
f_r, w_t, a_r = m_tarb.complexity(NUM_OFDM, convention_x_ofdm=False)
assert int(w_t) == real_w, f"TA-RB résiduel: weights mismatch {int(w_t)} vs {real_w}"
add_row('TA-RB-résiduel-signed_attn (T=6)', f_r, w_t, a_r, Q_NEURAL, Q_NEURAL,
        f'INT8, poids vérifiés exacts vs trainable_variables ({real_w:,})')

tf.keras.backend.clear_session()

# ── Tableau ───────────────────────────────────────────────────────────────
print(f'\n{"Méthode":<36} {"Précision":>10} {"Params(K)":>10} {"FLOPs réel(M)":>14} '
      f'{"Énergie(µJ)":>13} {"vs WMMSE FLOPs":>15} {"vs WMMSE Énergie":>17}')
wmmse_row = next(r for r in rows if r['method'] == 'WMMSE (I=10)')
for r in rows:
    vs_f = wmmse_row['flops_m'] / r['flops_m'] if r['flops_m'] > 0 else float('nan')
    vs_e = wmmse_row['energy_uj'] / r['energy_uj'] if r['energy_uj'] > 0 else float('nan')
    prec = f"FP{r['q_w']}" if r['q_w'] == 32 else f"INT{r['q_w']}"
    print(f'{r["method"]:<36} {prec:>10} {r["params_k"]:>10.1f} {r["flops_m"]:>14.3f} '
          f'{r["energy_uj"]:>13.4f} {vs_f:>14.2f}× {vs_e:>16.2f}×')
    r['vs_wmmse_flops'] = vs_f
    r['vs_wmmse_energy'] = vs_e

with open('results/complexity_energy_signed_attn.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['Method', 'Precision', 'Params (K)', 'FLOPs réel (M)', 'Énergie (µJ)',
                 'vs WMMSE FLOPs', 'vs WMMSE Énergie', 'Note'])
    for r in rows:
        prec = f"FP{r['q_w']}" if r['q_w'] == 32 else f"INT{r['q_w']}"
        wr.writerow([r['method'], prec, f"{r['params_k']:.1f}", f"{r['flops_m']:.3f}",
                     f"{r['energy_uj']:.4f}", f"{r['vs_wmmse_flops']:.2f}",
                     f"{r['vs_wmmse_energy']:.2f}", r['note']])

with open('results/complexity_energy_signed_attn.json', 'w') as f:
    json.dump(rows, f, indent=2)

print(f'\n✅ Sauvé -> results/complexity_energy_signed_attn.{{csv,json}}')
