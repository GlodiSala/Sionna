"""
figC_sumrate_snr_T.py — Figure C : débit somme vs SNR pour TA-RB
(Agrégation par RB, résiduel) pour les 6 T ∈ {1,2,3,4,6,12}, + RZF
groupé à compression équivalente par T + WMMSE en repère.

**Couleurs** (2e révision, demande utilisateur) : le dégradé séquentiel
magenta (1ère révision) rendait les 6 T trop proches visuellement,
notamment les pointillés RZF-groupé difficiles à associer à leur T.
Remplacé par une **palette catégorielle à 6 teintes bien distinctes**
(palette de référence dataviz, sous-ensemble à plus fort contraste
mutuel) -- TA-RB (plein) et son RZF-groupé (pointillé, même teinte)
restent appariés par couleur pour un même T, mais chaque T est
maintenant immédiatement identifiable.

T=4 = référence TA-RB (meilleur %WMMSE moyen du sweep, moins cher que
T=6/T=12, cf. Figure B) -- mis en évidence (marqueur plus grand).

Usage: python3 figC_sumrate_snr_T.py  (depuis results/figures_final/, CPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../results/'

T_VALUES = [1, 2, 3, 4, 6, 12]
T_RECOMMENDED = 4
SNR_NEURAL = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
SNR_RZF = [0.0, 10.0, 20.0]

# 6 teintes catégorielles distinctes (palette de référence dataviz, sous-ensemble
# à fort contraste mutuel -- évite les paires adjacentes faibles yellow/aqua) :
T_COLORS = {1: '#2a78d6', 2: '#eb6834', 3: '#008300', 4: '#8b1a5c',
            6: '#4a3aa7', 12: '#e34948'}

classical = np.load(DATA_DIR + 'classical_comparison_M8K4.npy', allow_pickle=True).item()
snr_classical = classical['snr']

tsweep = {T: json.load(open(DATA_DIR + f'diag_tarb_residual_signed_attn_T{T}_train.json'))['evals']
          for T in T_VALUES}
rzf_grouped = json.load(open(DATA_DIR + 'diag_rzf_grouped_for_tsweep.json'))

fig, ax = plt.subplots(figsize=(9, 6.5))

for T in T_VALUES:
    c = T_COLORS[T]
    is_ref = (T == T_RECOMMENDED)
    y_n = [tsweep[T][f'{s:.1f}'] for s in SNR_NEURAL]
    ax.plot(SNR_NEURAL, y_n, color=c, marker='v', linestyle='-',
            linewidth=2.6 if is_ref else 1.8, markersize=8 if is_ref else 6,
            label=f'T={T}' + (' (réf.)' if is_ref else ''), zorder=4 if is_ref else 3)

    y_r = [rzf_grouped[str(T)]['rates'][f'{s:.1f}'] for s in SNR_RZF]
    ax.plot(SNR_RZF, y_r, color=c, marker='o', linestyle='--',
            linewidth=1.3, markersize=5, alpha=0.6, zorder=2)

ax.plot([], [], color='#888888', marker='o', linestyle='--', linewidth=1.3,
        label='RZF groupé (même teinte)')
ax.plot(snr_classical, classical['wmmse'], color='#2a2a2a', marker='s', linestyle=':',
        linewidth=2, markersize=5, label='WMMSE', zorder=5)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Débit somme (bps/Hz)', fontsize=12)
ax.set_title('Débit somme vs SNR par T — TA-RB, UMi (M=8, K=4)',
              fontsize=13, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('figC_sumrate_snr_T.png', dpi=200, bbox_inches='tight')
fig.savefig('figC_sumrate_snr_T.pdf', bbox_inches='tight')
plt.close(fig)

out = {'neural_by_T': {str(T): tsweep[T] for T in T_VALUES},
       'rzf_grouped_by_T': rzf_grouped,
       'wmmse_ref': {'snr': list(snr_classical), 'rate': list(classical['wmmse'])},
       'T_recommended': T_RECOMMENDED}
with open('figC_sumrate_snr_T.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure C -> figC_sumrate_snr_T.{png,pdf,json}')
