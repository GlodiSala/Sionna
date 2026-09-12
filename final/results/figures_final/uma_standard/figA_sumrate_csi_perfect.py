"""
figA_sumrate_csi_perfect.py — Figure A, UMa STANDARD (M=8, K=4), double
panneau (gauche : CSI parfait, droite : CSI imparfait pilote 20dB).
RZF, WMMSE, SC, IB, TA-RB (T=4). Budget complet (83ep) pour les 3
architectures.

Usage: python3 figA_sumrate_csi_perfect.py  (depuis son propre dossier, CPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../'

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE',
          'SC': 'SC — Précodeur par sous-porteuse',
          'IB': 'IB — Transformer intra-bloc',
          'TA-RB': 'TA-RB — Agrégation par RB (résiduel, T=4)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}

classical = np.load(DATA_DIR + 'classical_comparison_M8K4_uma.npy', allow_pickle=True).item()
snr_classical = classical['snr']

data = {k: json.load(open(DATA_DIR + f'diag_uma_signed_attn_{ARCH_KEY[k]}.json')) for k in ('SC', 'IB', 'TA-RB')}
snr_neural = sorted(float(s) for s in data['SC']['evals_perfect'].keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# === panneau gauche : CSI parfait ===
ax1.plot(snr_classical, classical['rzf_full'], color=COLORS['RZF'], marker=MARKERS['RZF'],
         linestyle=LINESTYLE['RZF'], label=LABELS['RZF'], linewidth=2, markersize=6)
ax1.plot(snr_classical, classical['wmmse'], color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
         linestyle=LINESTYLE['WMMSE'], label=LABELS['WMMSE'], linewidth=2, markersize=6)
for k in ('SC', 'IB', 'TA-RB'):
    y = [data[k]['evals_perfect'][f'{s:.1f}'] for s in snr_neural]
    ax1.plot(snr_neural, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             label=LABELS[k], linewidth=2, markersize=7)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax1.set_title('CSI parfait', fontsize=12.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.3)

# === panneau droit : CSI imparfait (pilote 20dB) ===
for k in ('SC', 'IB', 'TA-RB'):
    y = [data[k]['csi_imperfect'][f'{s:.1f}']['pilot20dB'] for s in snr_neural]
    ax2.plot(snr_neural, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             linewidth=2, markersize=7)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax2.set_title('CSI imparfait (pilote 20dB)', fontsize=12.5, fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('figA_sumrate_csi_perfect.png', dpi=200, bbox_inches='tight')
fig.savefig('figA_sumrate_csi_perfect.pdf', bbox_inches='tight')
plt.close(fig)

out = {'perfect': {'RZF': {'snr': list(snr_classical), 'rate': list(classical['rzf_full'])},
                    'WMMSE': {'snr': list(snr_classical), 'rate': list(classical['wmmse'])},
                    **{k: {'snr': snr_neural, 'rate': [data[k]['evals_perfect'][f'{s:.1f}'] for s in snr_neural]}
                       for k in ('SC', 'IB', 'TA-RB')}},
       'imperfect_pilot20dB': {k: {'snr': snr_neural,
                                    'rate': [data[k]['csi_imperfect'][f'{s:.1f}']['pilot20dB'] for s in snr_neural]}
                                for k in ('SC', 'IB', 'TA-RB')}}
with open('figA_sumrate_csi_perfect.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure A (UMa STANDARD) -> figA_sumrate_csi_perfect.{png,pdf,json}')
