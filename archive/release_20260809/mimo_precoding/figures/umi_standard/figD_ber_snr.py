"""
figD_ber_snr.py — Figure D, double panneau (demande utilisateur) : BER
vs SNR, UMi (M=8, K=4) -- **gauche : CSI parfait** (0-20dB, 9 points),
**droite : CSI imparfait** (pilote 20dB, 0-20dB, 5 points) -- même
principe que la Figure A/B. RZF, WMMSE, SC, IB, TA-RB (**T=4**).

Usage: python3 figD_ber_snr.py  (depuis results/figures_final/, CPU)
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
KEY_MAP = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SingleSC-signed_attn',
           'IB': 'IntraRB-signed_attn', 'TA-RB': 'TA_RB_residual-signed_attn'}

perfect = json.load(open(DATA_DIR + 'diag_ber_signed_attn_full_range.json'))
snr_perfect = perfect['RZF']['snr']
imperfect = json.load(open(DATA_DIR + 'diag_ber_csi_imperfect.json'))
snr_imperfect = sorted(float(s) for s in imperfect['RZF'].keys())


def plot_panel(ax, get_ber, snr, title):
    for k in KEY_MAP:
        ber = np.maximum(np.array(get_ber(k)), 1e-6)   # plancher d'affichage (0 exacts, échelle log)
        ax.semilogy(snr, ber, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
                    label=LABELS[k], linewidth=2, markersize=6)
    ax.axhline(1e-6, color='#999999', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_title(title, fontsize=12.5, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
plot_panel(ax1, lambda k: perfect[KEY_MAP[k]]['ber'], snr_perfect, 'CSI parfait')
ax1.set_ylabel('BER (échelle log)', fontsize=12)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.text(snr_perfect[0], 1.15e-6, "BER=0 (plancher d'affichage)", fontsize=7.5, color='#777777')

plot_panel(ax2, lambda k: [imperfect[KEY_MAP[k]][f'{s:.1f}'] for s in snr_imperfect],
           snr_imperfect, 'CSI imparfait (pilote 20dB)')

fig.suptitle('BER vs SNR — UMi (M=8, K=4)', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figD_ber_snr.png', dpi=200, bbox_inches='tight')
fig.savefig('figD_ber_snr.pdf', bbox_inches='tight')
plt.close(fig)

out = {'perfect': {k: {'snr': snr_perfect, 'ber': perfect[KEY_MAP[k]]['ber']} for k in KEY_MAP},
       'imperfect_pilot20dB': {k: {'snr': snr_imperfect,
                                    'ber': [imperfect[KEY_MAP[k]][f'{s:.1f}'] for s in snr_imperfect]}
                                for k in KEY_MAP}}
with open('figD_ber_snr.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure D -> figD_ber_snr.{png,pdf,json}')
