"""
figA_sumrate_csi_perfect.py — Figure A, double panneau (demande
utilisateur) : débit somme vs SNR, UMi (M=8, K=4) -- **gauche : CSI
parfait**, **droite : CSI imparfait** (pilote 20dB, bruit LS gaussien
sur le canal, poids figés). RZF, WMMSE, SC (Précodeur par
sous-porteuse), IB (Transformer intra-bloc), TA-RB (Agrégation par RB,
décodeur résiduel, **T=4** -- référence, cf. Figure B : meilleur %WMMSE
moyen du T-sweep, moins cher que T=6/T=12). Légende unique (couleurs
partagées entre les deux panneaux), titres/légendes concis.

Usage: python3 figA_sumrate_csi_perfect.py  (depuis results/figures_final/, CPU)
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

# -- CSI parfait --
classical = np.load(DATA_DIR + 'classical_comparison_M8K4.npy', allow_pickle=True).item()
snr_classical = classical['snr']

# T=4 = référence TA-RB (meilleur %WMMSE moyen du T-sweep, moins cher que
# T=6/T=12 -- cf. Figure B).
neural_files = {'SC': 'diag_front_a_signed_attn.json',
                'IB': 'diag_front_a_signed_attn_intra_rb.json',
                'TA-RB': 'diag_tarb_residual_signed_attn_T4_train.json'}
neural_evals = {k: json.load(open(DATA_DIR + p))['evals'] for k, p in neural_files.items()}
snr_neural = sorted(float(s) for s in next(iter(neural_evals.values())).keys())

# -- CSI imparfait (pilote 20dB, TA-RB T=4 fusionné dans le même JSON) --
csi_imp = json.load(open(DATA_DIR + 'diag_csi_imperfect_signed_attn_sweep.json'))
CSI_KEY_MAP = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SingleSC', 'IB': 'IntraRB', 'TA-RB': 'TA_RB_residual'}
snr_csi = sorted(float(s) for s in csi_imp['RZF'].keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# === panneau gauche : CSI parfait ===
ax1.plot(snr_classical, classical['rzf_full'], color=COLORS['RZF'], marker=MARKERS['RZF'],
         linestyle=LINESTYLE['RZF'], label=LABELS['RZF'], linewidth=2, markersize=6)
ax1.plot(snr_classical, classical['wmmse'], color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
         linestyle=LINESTYLE['WMMSE'], label=LABELS['WMMSE'], linewidth=2, markersize=6)
for k in ('SC', 'IB', 'TA-RB'):
    y = [neural_evals[k][f'{s:.1f}'] for s in snr_neural]
    ax1.plot(snr_neural, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             label=LABELS[k], linewidth=2, markersize=7)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Débit somme (bps/Hz)', fontsize=12)
ax1.set_title('CSI parfait', fontsize=12.5, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.3)

# === panneau droit : CSI imparfait (pilote 20dB) ===
for k in ('RZF', 'WMMSE', 'SC', 'IB', 'TA-RB'):
    jkey = CSI_KEY_MAP[k]
    y = [csi_imp[jkey][f'{s:.1f}']['pilot20dB'] for s in snr_csi]
    ax2.plot(snr_csi, y, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
             linewidth=2, markersize=6 if k in ('RZF', 'WMMSE') else 7)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Débit somme (bps/Hz)', fontsize=12)
ax2.set_title('CSI imparfait (pilote 20dB)', fontsize=12.5, fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.suptitle('Débit somme vs SNR — UMi (M=8, K=4)', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figA_sumrate_csi_perfect.png', dpi=200, bbox_inches='tight')
fig.savefig('figA_sumrate_csi_perfect.pdf', bbox_inches='tight')
plt.close(fig)

out = {'perfect': {'RZF': {'snr': list(snr_classical), 'rate': list(classical['rzf_full'])},
                    'WMMSE': {'snr': list(snr_classical), 'rate': list(classical['wmmse'])},
                    **{k: {'snr': snr_neural, 'rate': [neural_evals[k][f'{s:.1f}'] for s in snr_neural]}
                       for k in ('SC', 'IB', 'TA-RB')}},
       'imperfect_pilot20dB': {k: {'snr': snr_csi,
                                    'rate': [csi_imp[CSI_KEY_MAP[k]][f'{s:.1f}']['pilot20dB'] for s in snr_csi]}
                                for k in ('RZF', 'WMMSE', 'SC', 'IB', 'TA-RB')}}
with open('figA_sumrate_csi_perfect.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure A -> figA_sumrate_csi_perfect.{png,pdf,json}')
