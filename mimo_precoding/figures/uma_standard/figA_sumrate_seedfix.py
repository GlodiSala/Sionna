"""
figA_sumrate_seedfix.py -- UMa STANDARD (M8K4), panneau CSI parfait
regenere avec les donnees corrigees (seed reel + canal partage + 4 bugs
verifies, cf. cible 1 de la tache de reverification systematique) :
final/results/classical_comparison_M8K4_uma_seedfix_manual_sinr_20260820_125429.json

Panneau CSI imparfait NON touche ici -- ce cote n'a pas encore ete
verifie/corrige pour les 4 bugs (hors perimetre de cible 1, qui ne
couvrait que le CSI parfait). Reutilise tel quel depuis l'ancien script
figA_sumrate_csi_perfect.py (donnees non validees pour ce panneau).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../results/'

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE',
          'SC': 'SC — Précodeur par sous-porteuse',
          'IB': 'IB — Transformer intra-bloc',
          'TA-RB': 'TA-RB — Agrégation par RB (résiduel, T=4)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}

# --- CSI parfait : donnees CORRIGEES (seed reel, canal partage, 4 bugs verifies) ---
fixed = json.load(open(DATA_DIR + 'classical_comparison_M8K4_uma_seedfix_manual_sinr_20260820_125429.json'))
snr_fixed = fixed['table']['snr']

# --- CSI imparfait : donnees NON corrigees (hors perimetre, pas encore verifie) ---
data = {k: json.load(open(DATA_DIR + f'diag_uma_signed_attn_{ARCH_KEY[k]}.json')) for k in ('SC', 'IB', 'TA-RB')}
snr_neural = sorted(float(s) for s in data['SC']['evals_perfect'].keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# === panneau gauche : CSI parfait (CORRIGE) ===
ax1.plot(snr_fixed, fixed['table']['rzf_sc'], color=COLORS['RZF'], marker=MARKERS['RZF'],
         linestyle=LINESTYLE['RZF'], label=LABELS['RZF'], linewidth=2, markersize=6)
ax1.plot(snr_fixed, fixed['table']['wmmse'], color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
         linestyle=LINESTYLE['WMMSE'], label=LABELS['WMMSE'], linewidth=2, markersize=6)
for k in ('SC', 'IB', 'TA-RB'):
    ax1.plot(snr_fixed, fixed['table'][k], color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             label=LABELS[k], linewidth=2, markersize=7)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax1.set_title('CSI parfait', fontsize=12.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.3)

# === panneau droit : CSI imparfait (pilote 20dB) -- donnees NON corrigees ===
for k in ('SC', 'IB', 'TA-RB'):
    y = [data[k]['csi_imperfect'][f'{s:.1f}']['pilot20dB'] for s in snr_neural]
    ax2.plot(snr_neural, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             linewidth=2, markersize=7)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax2.set_title('CSI imparfait (pilote 20dB) -- NON corrige', fontsize=12.5, fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('figA_sumrate_seedfix.png', dpi=200, bbox_inches='tight')
fig.savefig('figA_sumrate_seedfix.pdf', bbox_inches='tight')
plt.close(fig)
print('OK -> figA_sumrate_seedfix.{png,pdf}')
