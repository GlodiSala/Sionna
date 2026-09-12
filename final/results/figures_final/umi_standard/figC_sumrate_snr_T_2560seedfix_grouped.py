"""
figC_sumrate_snr_T_2560seedfix_grouped.py -- version finale de la Figure C,
restaure les courbes RZF groupees a compression equivalente (une par T,
meme teinte, pointille) de l'originale figC_sumrate_snr_T.py, calculees
sur le run seedfix 2560 echantillons/point (cf.
diag_rzf_grouped_for_tsweep_2560seedfix.py, reproductibilite du canal
verifiee exactement contre le run T-sweep).

Remplace figC_sumrate_snr_T_2560seedfix.py (version simplifiee sans les
courbes RZF groupees, publiee ce matin par erreur de simplification).

Sources :
  final/results/tsweep_seedfix_perfect_manual_2560_20260820_153545.json
  final/results/diag_rzf_grouped_for_tsweep_2560seedfix_20260820_161421.json
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../'

T_VALUES = [1, 2, 3, 4, 6, 12]
T_RECOMMENDED = 4
T_COLORS = {1: '#2a78d6', 2: '#eb6834', 3: '#008300', 4: '#8b1a5c',
            6: '#4a3aa7', 12: '#e34948'}

tsweep = json.load(open(DATA_DIR + 'tsweep_seedfix_perfect_manual_2560_20260820_153545.json'))['table']
rzf_grouped = json.load(open(DATA_DIR + 'diag_rzf_grouped_for_tsweep_2560seedfix_20260820_161421.json'))['table']
snr = tsweep['snr']
assert snr == rzf_grouped['snr']

fig, ax = plt.subplots(figsize=(9, 6.5))

for T in T_VALUES:
    c = T_COLORS[T]
    is_ref = (T == T_RECOMMENDED)
    key = f'T{T}'
    ax.plot(snr, tsweep[key], color=c, marker='v', linestyle='-',
            linewidth=2.6 if is_ref else 1.8, markersize=8 if is_ref else 6,
            label=f'T={T}' + (' (réf.)' if is_ref else ''), zorder=4 if is_ref else 3)
    ax.plot(snr, rzf_grouped[key], color=c, marker='o', linestyle='--',
            linewidth=1.3, markersize=5, alpha=0.6, zorder=2)

ax.plot([], [], color='#888888', marker='o', linestyle='--', linewidth=1.3,
        label='RZF groupé (même teinte)')
ax.plot(snr, tsweep['wmmse'], color='#2a2a2a', marker='s', linestyle=':',
        linewidth=2, markersize=5, label='WMMSE', zorder=5)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax.legend(fontsize=8.5, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('figC_sumrate_snr_T.png', dpi=200, bbox_inches='tight')
fig.savefig('figC_sumrate_snr_T.pdf', bbox_inches='tight')
plt.close(fig)

out = {'neural_by_T': tsweep, 'rzf_grouped_by_T': rzf_grouped, 'T_recommended': T_RECOMMENDED,
       'sources': ['tsweep_seedfix_perfect_manual_2560_20260820_153545.json',
                   'diag_rzf_grouped_for_tsweep_2560seedfix_20260820_161421.json']}
with open('figC_sumrate_snr_T_2560seedfix_grouped.json', 'w') as f:
    json.dump(out, f, indent=2)
print('OK -> figC_sumrate_snr_T.{png,pdf} (T + RZF groupe, run 2560) + figC_sumrate_snr_T_2560seedfix_grouped.json')
