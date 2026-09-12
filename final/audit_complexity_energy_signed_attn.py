"""
audit_complexity_energy_signed_attn.py — Priorité 1 (demande utilisateur) :
audit EXHAUSTIF du modèle de complexité/énergie pour les 3 architectures
signed_attn, pas juste un recalcul. Imprime le détail composant par
composant (pas seulement les totaux), avec comparaison AVANT/APRÈS un fix
trouvé pendant cet audit (cf. plus bas), et calcule l'énergie à
INT8 ET INT16.

FIX TROUVÉ PENDANT L'AUDIT (documenté ici, appliqué dans precoder_intra_
rb.py) : `mha_flops(seq,d)` dans SingleSCTransformerPrecoder.complexity()
et IntraRBTransformerPrecoder.complexity() ne comptait QUE les projections
Q/K/V/out + le calcul des scores QK^T (2*seq²*d) -- PAS la somme pondérée
scores@V (2*seq²*d également), qui est un VRAI matmul de même coût que
QK^T. `precoders_v2.py::TransformerPrecoderClean.complexity()` (TA-RB)
comptait déjà correctement les deux (coefficient 4x, pas 2x) -- les deux
fichiers étaient INCOHÉRENTS entre eux depuis leur revalidation respective
du 7/8 (ÉTAPE 2), jamais croisés. Corrigé : les deux fichiers utilisent
maintenant la même formule complète (8*seq*d² + 4*seq²*d).

QUESTION POSÉE PAR L'UTILISATEUR -- SignedGateMHA introduit-il un coût
propre non compté ? Réponse : NON, structurellement -- SignedGateMHA fait
EXACTEMENT les mêmes matmuls qu'une MultiHeadAttention standard (4
projections Dense(D,D) + QK^T + somme pondérée par V) ; sa SEULE
différence est de multiplier les poids d'attention par tanh(scores) (porte
de signe) au lieu d'appliquer UNIQUEMENT softmax -- une opération
PONCTUELLE (élément par élément, O(seq²) par tête), pas un matmul
supplémentaire. Quantifié plus bas (comparaison au coût matmul total) pour
vérifier que "négligeable" n'est pas juste affirmé mais mesuré.

Usage: python3 audit_complexity_energy_signed_attn.py  (CPU suffit)
"""
import os, sys, json, csv
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
TOKENS_PER_RB = 6
M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
KD = D // H

h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)

print(f'{"="*78}\n1. QUANTIFICATION DU COÛT PONCTUEL (tanh + mult.) DE SignedGateMHA\n{"="*78}')
print(f'D={D}, H={H}, kd={KD}')
for name, seq in [('SC-attention (IntraRB, seq=S=12)', 12), ('user-attention (seq=K=4)', 4)]:
    matmul_flops = 8.0 * seq * D**2 + 4.0 * seq**2 * D
    # tanh(scores) + mag*sign : 2 opérations élément-par-élément sur [H,seq,seq]
    pointwise_flops = 2.0 * H * seq * seq   # 1 tanh + 1 mult, par tête, par entrée de la matrice seq x seq
    pct = 100 * pointwise_flops / matmul_flops
    print(f'  {name:<38} matmul={matmul_flops:>12,.0f}  ponctuel(tanh+mult)={pointwise_flops:>8,.0f}  '
          f'({pct:.3f}% du matmul -- négligeable, confirmé chiffré, pas juste affirmé)')

print(f'\n{"="*78}\n2. FIX APPLIQUÉ -- comparaison AVANT (bug) / APRÈS (corrigé)\n{"="*78}')


def mha_flops_buggy(seq, d):
    return 8.0 * seq * d**2 + 2.0 * seq**2 * d   # ancien (QK^T seul)


def mha_flops_fixed(seq, d):
    return 8.0 * seq * d**2 + 4.0 * seq**2 * d   # corrigé (QK^T + scores@V)


configs = [
    ('SingleSC-signed_attn', SingleSCTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, embed_dim=D, num_heads=H, num_layers=L),
     [('user-attn (seq=K)', K, FFT_SIZE * L)]),   # FFT_SIZE instances par couche, L couches
    ('IntraRB-signed_attn', IntraRBTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12, embed_dim=D, num_heads=H, num_layers=L),
     [('SC-attn (seq=S=12)', 12, (FFT_SIZE // 12) * K * L), ('user-attn (seq=K)', K, (FFT_SIZE // 12) * 12 * L)]),
    ('TA-RB-résiduel-signed_attn', TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12, tokens_per_rb=TOKENS_PER_RB,
        embed_dim=D, num_heads=H, num_layers=L),
     None),  # précoders_v2.py était déjà correct, pas de delta ici
]

results = {}
for name, m, attn_instances in configs:
    _ = m(h_dummy, no=tf.constant(1e-3))
    real_w = int(sum(tf.size(v).numpy() for v in m.trainable_variables))
    f_fixed, w, a = m.complexity(NUM_OFDM, convention_x_ofdm=False)
    assert int(w) == real_w, f"{name}: poids {int(w)} != réel {real_w}"

    if attn_instances is not None:
        delta = sum(n_inst * (mha_flops_fixed(seq, D) - mha_flops_buggy(seq, D)) * NUM_OFDM
                    for _, seq, n_inst in attn_instances)
        f_buggy = f_fixed - delta
        pct_change = 100 * (f_fixed - f_buggy) / f_buggy
    else:
        f_buggy = f_fixed
        pct_change = 0.0

    print(f'{name:<30} poids={real_w:>9,} (✅ exact) | FLOPs AVANT={f_buggy/1e6:>9.2f}M -> '
          f'APRÈS={f_fixed/1e6:>9.2f}M ({pct_change:+.2f}%)')
    results[name] = {'params': real_w, 'flops_real': f_fixed, 'acts': a}

tf.keras.backend.clear_session()

print(f'\n{"="*78}\n3. INVENTAIRE EXHAUSTIF DES OPÉRATIONS COMPTÉES (par architecture)\n{"="*78}')
print("""
SingleSC-signed_attn / IntraRB-signed_attn (precoder_intra_rb.py) :
  - input_embed         : Dense(feat_dim -> D)                    [compté: df]
  - input_norm          : LayerNormalization                      [PAS compté -- convention: normalisations jamais comptées en FLOPs dans ce modèle, seulement en poids (2D, gamma+beta)]
  - pos_enc_sc (IntraRB) : Embedding lookup                       [PAS compté en FLOPs (lookup pur), compté en poids (S*D)]
  - attn_sc (IntraRB)    : MultiHeadAttention standard             [compté: mha_flops, SEQ=S=12]
  - attn_usr             : SignedGateMHA (softmax*tanh)            [compté: mha_flops IDENTIQUE (même matmuls) -- porte tanh ponctuelle non comptée, quantifiée négligeable ci-dessus (section 1)]
  - norm_sc/norm_usr/norm_ffn : LayerNormalization x3/bloc         [PAS comptées en FLOPs, comptées en poids (2D chacune)]
  - ffn (SwiGLU)          : Dense(D->4D) + split + Dense(4D->D)... [compté: ffn_flops -- note: SwiGLU split/sigmoid PAS comptés (ponctuels)]
  - s_sc/s_usr/s_ffn      : scalaires résiduels (multiplication)  [PAS comptés (1 scalaire x D éléments = D FLOPs/bloc, négligeable, jamais compté nulle part dans ce modèle)]
  - output_proj           : Dense(D -> 2M)                        [compté: df]

TA-RB-résiduel-signed_attn (precoders_v2.py) :
  - input_proj            : Dense(feat_dim -> D) + GELU            [compté: df ; GELU ponctuel non compté]
  - input_norm (RMSNorm)  : RMSNormalization                       [PAS compté en FLOPs, poids comptés]
  - snr_proj              : Dense(1 -> D) + GELU                   [compté: dp en poids ; FLOPs négligés, n=1 token -- déjà le cas avant, pas changé]
  - pos_embed              : Embedding lookup                       [PAS compté FLOPs, poids comptés]
  - freq_attn              : MultiHeadAttention standard            [compté: mf (déjà complet, 4x, pas de fix nécessaire)]
  - user_attn              : SignedGateMHA                          [compté: mf IDENTIQUE (mêmes matmuls que freq_attn) -- porte tanh ponctuelle non comptée]
  - ffn (bloc TA-RB)       : Dense(D->4D)+GELU+Dense(4D->D)         [compté: ff]
  - token_to_precoder      : Dense(D -> 2M)                         [compté: df]
  - **interpolation bilinéaire (tf.image.resize)** : AUCUN poids, PAS
    un Dense -- vérifié ligne 490-493 de precoders_v2.py : commentaire
    explicite "PAS de poids, FLOPs négligeables... convention lookup/
    interp = 0 FLOPs" -- CONFIRMÉ non confondu avec un Dense.
  - sc_refine (résiduel)   : Conv1D(rb_size)+Conv1D(1) x K users    [compté: terme dédié, PAS mha_flops ni df générique]
  - alpha                  : scalaire (résidu appris)                [PAS compté, poids=1]
""")

print(f'\n{"="*78}\n4. ÉNERGIE -- INT8 ET INT16 (nouveau, demande utilisateur)\n{"="*78}')

f_rzf, w_rzf, a_rzf = compute_classical_complexity('RZF', M, K, FFT_SIZE, NUM_OFDM)
f_wmmse, w_wmmse, a_wmmse = compute_classical_complexity('WMMSE', M, K, FFT_SIZE, NUM_OFDM, I_wmmse=10)

rows = []
rows.append(dict(method='RZF', precision='FP32', params_k=w_rzf/1e3, flops_m=f_rzf/1e6,
                  energy_uj=compute_energy_uJ(f_rzf, w_rzf, a_rzf, 32, 32)))
rows.append(dict(method='WMMSE (I=10)', precision='FP32', params_k=w_wmmse/1e3, flops_m=f_wmmse/1e6,
                  energy_uj=compute_energy_uJ(f_wmmse, w_wmmse, a_wmmse, 32, 32)))

name_map = {'SingleSC-signed_attn': 'SingleSC-signed_attn',
            'IntraRB-signed_attn': 'IntraRB-signed_attn',
            'TA-RB-résiduel-signed_attn': 'TA-RB-résiduel-signed_attn (T=6)'}
for internal_name, r in results.items():
    for q in (8, 16):
        e = compute_energy_uJ(r['flops_real'], r['params'], r['acts'], q, q)
        rows.append(dict(method=name_map[internal_name], precision=f'INT{q}',
                          params_k=r['params']/1e3, flops_m=r['flops_real']/1e6, energy_uj=e))

wmmse_e = next(r['energy_uj'] for r in rows if r['method'] == 'WMMSE (I=10)')
wmmse_f = next(r['flops_m'] for r in rows if r['method'] == 'WMMSE (I=10)')
print(f'\n{"Méthode":<32} {"Précision":>10} {"Params(K)":>10} {"FLOPs(M)":>10} {"Énergie(µJ)":>13} '
      f'{"vs WMMSE F":>11} {"vs WMMSE E":>11}')
for r in rows:
    vs_f = wmmse_f / r['flops_m'] if r['flops_m'] > 0 else float('nan')
    vs_e = wmmse_e / r['energy_uj'] if r['energy_uj'] > 0 else float('nan')
    print(f'{r["method"]:<32} {r["precision"]:>10} {r["params_k"]:>10.1f} {r["flops_m"]:>10.2f} '
          f'{r["energy_uj"]:>13.4f} {vs_f:>10.2f}× {vs_e:>10.2f}×')
    r['vs_wmmse_flops'], r['vs_wmmse_energy'] = vs_f, vs_e

os.makedirs('results', exist_ok=True)
with open('results/complexity_energy_signed_attn_v2.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['Method', 'Precision', 'Params (K)', 'FLOPs réel (M)', 'Énergie (µJ)',
                 'vs WMMSE FLOPs', 'vs WMMSE Énergie'])
    for r in rows:
        wr.writerow([r['method'], r['precision'], f"{r['params_k']:.1f}", f"{r['flops_m']:.3f}",
                     f"{r['energy_uj']:.4f}", f"{r['vs_wmmse_flops']:.2f}", f"{r['vs_wmmse_energy']:.2f}"])
with open('results/complexity_energy_signed_attn_v2.json', 'w') as f:
    json.dump(rows, f, indent=2)
print(f'\n✅ Sauvé -> results/complexity_energy_signed_attn_v2.{{csv,json}} (v2 = post-fix mha_flops)')
