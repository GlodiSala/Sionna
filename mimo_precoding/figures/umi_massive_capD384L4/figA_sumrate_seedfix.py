"""
figA_sumrate_seedfix.py -- UMi MASSIVE (M=64, K=8), D=384, L=4, T=6, 2
panneaux :
  - Panneau gauche (CSI parfait) : INCHANGE, copie exacte du panneau gauche
    de figA_sumrate_csi_perfect.py (fichier separe, non modifie), sources :
    results/classical_comparison_M64K8_true.npy (RZF, WMMSE) +
    results/diag_massive_gap_fullbudget_{single_sc,intra_rb,ta_rb_residual}
    _cap_d384l4.json (SC, IB, TA-RB) -- grille 0-20dB par pas de 2,5dB,
    concorde avec les tableaux du memoire.
  - Panneau droit (CSI imparfait, pilote 20dB) : source =
    results/eval_csi_conserve_umi_massive_D384_seedfix_20260923_233705.json
    (evaluation appariee -- CSI parfait et CSI estime calcules sur les
    memes tirages de canal, meme checkpoint T=6/D=384 ; remplace l'ancien
    diag_csi_imperfect_massive_capD384L4.json, campagne non appariee).

Ne modifie pas figA_sumrate_csi_perfect.py (fichier separe).
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
          'SC': 'SC — Précodeur par sous-porteuse (D=384)',
          'IB': 'IB — Transformer intra-bloc (D=384)',
          'TA-RB': 'TA-RB — Agrégation par RB (résiduel, T=6, D=384)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}

# --- CSI parfait : INCHANGE (copie exacte de figA_sumrate_csi_perfect.py) ---
classical = np.load(DATA_DIR + 'classical_comparison_M64K8_true.npy', allow_pickle=True).item()
snr_classical = classical['snr']

neural_evals = {k: json.load(open(DATA_DIR + f'diag_massive_gap_fullbudget_{ARCH_KEY[k]}_cap_d384l4.json'))['evals']
                for k in ('SC', 'IB', 'TA-RB')}
snr_neural = sorted(float(s) for s in next(iter(neural_evals.values())).keys())

# --- CSI imparfait : evaluation appariee (meme canal, meme checkpoint T=6/D=384) ---
imperfect = json.load(open(DATA_DIR + 'eval_csi_conserve_umi_massive_D384_seedfix_20260923_233705.json'))
snr_imperfect = imperfect['snr_db']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# === panneau gauche : CSI parfait (INCHANGE) ===
ax1.plot(snr_classical, classical['rzf_full'], color=COLORS['RZF'], marker=MARKERS['RZF'],
         linestyle=LINESTYLE['RZF'], label=LABELS['RZF'], linewidth=2, markersize=6)
ax1.plot(snr_classical, classical['wmmse'], color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
         linestyle=LINESTYLE['WMMSE'], label=LABELS['WMMSE'], linewidth=2, markersize=6)
for k in ('SC', 'IB', 'TA-RB'):
    y = [neural_evals[k][f'{s:.1f}'] for s in snr_neural]
    ax1.plot(snr_neural, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             label=LABELS[k], linewidth=2, markersize=7)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax1.set_title('CSI parfait', fontsize=12.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.3)

# === panneau droit : CSI imparfait (pilote 20dB) -- evaluation appariee ===
for k in ('RZF', 'WMMSE', 'SC', 'IB', 'TA-RB'):
    y = imperfect['table_debit_imparfait_bps_hz'][k]
    ax2.plot(snr_imperfect, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             linewidth=2, markersize=6 if k in ('RZF', 'WMMSE') else 7)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax2.set_title('CSI imparfait (pilote 20dB)', fontsize=12.5, fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('figA_sumrate_seedfix.png', dpi=200, bbox_inches='tight')
fig.savefig('figA_sumrate_seedfix.pdf', bbox_inches='tight')
plt.close(fig)
print('OK -> figA_sumrate_seedfix.{png,pdf}')
