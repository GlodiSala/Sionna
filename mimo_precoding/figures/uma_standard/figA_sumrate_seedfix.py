"""
figA_sumrate_seedfix.py -- UMa STANDARD (M8K4), 2 panneaux :
  - Panneau gauche (CSI parfait) : INCHANGE, donnees corrigees (seed reel +
    canal partage + 4 bugs verifies, cf. cible 1 de la tache de
    reverification systematique), source :
    results/classical_comparison_M8K4_uma_seedfix_manual_sinr_20260820_125429.json
    (TA-RB T=4).
  - Panneau droit (CSI imparfait, pilote 20dB) : source =
    results/eval_csi_conserve_uma_standard_seedfix_20260923_231656.json
    (evaluation appariee -- CSI parfait et CSI estime calcules sur les
    memes tirages de canal, meme checkpoint T=4 que le panneau gauche ;
    remplace l'ancien diag_uma_signed_attn_*.json qui ne couvrait que
    SC/IB/TA-RB, sans RZF ni WMMSE, et venait d'une campagne non appariee).
"""
import json
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

# --- CSI parfait : donnees CORRIGEES (seed reel, canal partage, 4 bugs verifies) ---
fixed = json.load(open(DATA_DIR + 'classical_comparison_M8K4_uma_seedfix_manual_sinr_20260820_125429.json'))
snr_fixed = fixed['table']['snr']

# --- CSI imparfait : evaluation appariee (meme canal, meme checkpoint T=4) ---
imperfect = json.load(open(DATA_DIR + 'eval_csi_conserve_uma_standard_seedfix_20260923_231656.json'))
snr_imperfect = imperfect['snr_db']

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
