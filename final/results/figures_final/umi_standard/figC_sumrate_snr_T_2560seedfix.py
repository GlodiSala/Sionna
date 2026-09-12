"""
figC_sumrate_snr_T_2560seedfix.py -- version finale de la Figure C
(remplace figC_tsweep_seedfix.py, run intermediaire a 320 echantillons),
regeneree avec le run 2560 echantillons/point (20x128, protocole
d'entrainement d'origine) :
final/results/tsweep_seedfix_perfect_manual_2560_20260820_153545.json

Ecrit dans figC_sumrate_snr_T.png/.pdf (memes noms que l'original
figC_sumrate_snr_T.py, qui n'est PAS modifie) pour etre copiee telle
quelle dans le depot Overleaf clone (figures/figC_tsweep.png), en
remplacement de la version non corrigee (canal non partage, metrique
Sionna biaisee, 4 bugs non verifies).

Simplifie par rapport a l'original : ne reproduit pas la comparaison
"RZF groupe a compression equivalente par T" (hors perimetre des
donnees seedfix disponibles ici) -- seulement TA-RB par T vs RZF-SC/WMMSE
de reference, desormais sur EXACTEMENT le meme canal pour toutes les
methodes.
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

d = json.load(open(DATA_DIR + 'tsweep_seedfix_perfect_manual_2560_20260820_153545.json'))
t = d['table']
snr = t['snr']

fig, ax = plt.subplots(figsize=(9, 6.5))

ax.plot(snr, t['rzf_sc'], color='#2a2a2a', marker='o', linestyle='--',
        linewidth=2, markersize=6, label='RZF (par SC)', zorder=5)
ax.plot(snr, t['wmmse'], color='#888888', marker='s', linestyle=':',
        linewidth=2, markersize=6, label='WMMSE', zorder=5)

for T in T_VALUES:
    c = T_COLORS[T]
    is_ref = (T == T_RECOMMENDED)
    key = f'T{T}'
    ax.plot(snr, t[key], color=c, marker='v', linestyle='-',
            linewidth=2.6 if is_ref else 1.8, markersize=8 if is_ref else 6,
            label=f'T={T}' + (' (réf.)' if is_ref else ''), zorder=4 if is_ref else 3)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Débit total (bps/Hz)', fontsize=12)
ax.legend(fontsize=8.5, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('figC_sumrate_snr_T.png', dpi=200, bbox_inches='tight')
fig.savefig('figC_sumrate_snr_T.pdf', bbox_inches='tight')
plt.close(fig)

out = {'source': 'tsweep_seedfix_perfect_manual_2560_20260820_153545.json',
       'table': t, 'T_recommended': T_RECOMMENDED,
       'note': 'figure finale seedfix (2560 echantillons/point) -- remplace figC_tsweep_seedfix.py (320 echantillons)'}
with open('figC_sumrate_snr_T_2560seedfix.json', 'w') as f:
    json.dump(out, f, indent=2)
print('OK -> figC_sumrate_snr_T.{png,pdf} (source: run 2560) + figC_sumrate_snr_T_2560seedfix.json')
