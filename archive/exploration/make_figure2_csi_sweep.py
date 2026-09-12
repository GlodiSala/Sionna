"""
make_figure2_csi_sweep.py — Figure 2 remplacée (Priorité 2, demande
utilisateur) : vraie courbe vs SNR sous CSI imparfait (pilote fixé à
20dB, le plus informatif), au lieu du point unique à SNR données=15dB.
Remplace fig_signed_attn_csi_imperfect.png/pdf.

Légendes en français, terminologie du chapitre (pas de noms de code),
palette validée dataviz (identique fig.1/3/4).

Usage: python3 make_figure2_csi_sweep.py  (CPU, pas de calcul GPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SingleSC': '#1baf7a',
          'IntraRB': '#eda100', 'TA_RB_residual': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SingleSC': '^', 'IntraRB': 'D', 'TA_RB_residual': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SingleSC': 'Précodeur par sous-porteuse',
          'IntraRB': 'Transformer intra-bloc', 'TA_RB_residual': 'Agrégation par RB (décodeur résiduel)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SingleSC': '-', 'IntraRB': '-', 'TA_RB_residual': '-'}

data = json.load(open('results/diag_csi_imperfect_signed_attn_sweep.json'))
snrs = sorted(float(s) for s in next(iter(data.values())).keys())
methods = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# -- gauche : débit sous CSI imparfait (pilote 20dB) vs SNR données --
for m in methods:
    y = [data[m][f'{s:.1f}']['pilot20dB'] for s in snrs]
    axes[0].plot(snrs, y, color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLE[m],
                 label=LABELS[m], linewidth=2, markersize=7)
axes[0].set_xlabel('SNR données (dB)', fontsize=12)
axes[0].set_ylabel('Débit somme (bps/Hz)', fontsize=12)
axes[0].set_title('Débit sous CSI imparfait (pilote 20dB)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# -- droite : % débit conservé (imparfait/parfait) vs SNR données --
for m in methods:
    y = [data[m][f'{s:.1f}']['pct_retained'] for s in snrs]
    axes[1].plot(snrs, y, color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLE[m],
                 label=LABELS[m], linewidth=2, markersize=7)
axes[1].set_xlabel('SNR données (dB)', fontsize=12)
axes[1].set_ylabel('Débit conservé (%, imparfait / parfait)', fontsize=12)
axes[1].set_title('Robustesse relative au bruit pilote (20dB)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9, loc='lower left')

fig.suptitle('Débit sous CSI imparfait vs SNR — canal UMi (M=8, K=4), pilote fixé à 20dB\n'
             'Architectures neuronales à attention à poids signés', fontsize=13, fontweight='bold', y=1.04)
fig.tight_layout()
fig.savefig('results/fig_signed_attn_csi_imperfect_sweep.png', dpi=200, bbox_inches='tight')
fig.savefig('results/fig_signed_attn_csi_imperfect_sweep.pdf', bbox_inches='tight')
plt.close(fig)
print('✅ Figure 2 (sweep) -> results/fig_signed_attn_csi_imperfect_sweep.{png,pdf}')

print(f'\n{"SNR":>6}', *[f'{LABELS[m]:>16}' for m in methods])
for s in snrs:
    row = [f'{data[m][f"{s:.1f}"]["pct_retained"]:>15.1f}%' for m in methods]
    print(f'{s:>6.1f}', *row)
