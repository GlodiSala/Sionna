"""
figA_sumrate_seedfix.py -- fig:sumrate_umi, 2 panneaux, avec correction du
biais LMMSEPostEqualizationSINR :
  - Panneau gauche (CSI parfait) : pas de nouveau calcul, retrace les donnees
    DEJA produites pour Table 3.9 corrigee
    (classical_comparison_M8K4_seedfix_manual_sinr_20260820_023135.json).
  - Panneau droit (CSI imparfait, pilote 20dB) : donnees regenerees par
    experiments/eval_csi_imperfect_umi_standard.py (meme script/mecanisme que
    diag_csi_imperfect_signed_attn_sweep.py original, seed reel +
    double metrique Sionna/SINR manuel).

Ne modifie pas figA_sumrate_csi_perfect.py (fichier separe).
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE',
          'SC': 'SC — Précodeur par sous-porteuse',
          'IB': 'IB — Transformer intra-bloc',
          'TA-RB': 'TA-RB — Agrégation par RB (résiduel)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}
METHOD_KEY_LEFT = {'RZF': 'rzf_sc', 'WMMSE': 'wmmse', 'SC': 'SC', 'IB': 'IB', 'TA-RB': 'TA-RB'}

with open('../../results/classical_comparison_M8K4_seedfix_manual_sinr_20260820_023135.json') as f:
    left = json.load(f)
snr_left = left['table']['snr']

with open('../../results/diag_csi_imperfect_seedfix_manual_sinr_20260820_030418.json') as f:
    right = json.load(f)
snr_right = right['table']['snr']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for k in ('RZF', 'WMMSE', 'SC', 'IB', 'TA-RB'):
    y = left['table'][METHOD_KEY_LEFT[k]]
    ax1.plot(snr_left, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
              label=LABELS[k], linewidth=2, markersize=7 if k in ('SC','IB','TA-RB') else 6)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax1.set_title('CSI parfait', fontsize=12.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.3)

for k in ('RZF', 'WMMSE', 'SC', 'IB', 'TA-RB'):
    y = right['table'][k]
    ax2.plot(snr_right, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
              linewidth=2, markersize=7 if k in ('SC','IB','TA-RB') else 6)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax2.set_title('CSI imparfait (pilote 20dB)', fontsize=12.5, fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('figA_sumrate_seedfix.png', dpi=200, bbox_inches='tight')
fig.savefig('figA_sumrate_seedfix.pdf', bbox_inches='tight')
plt.close(fig)
print('OK -> figA_sumrate_seedfix.{png,pdf}')
