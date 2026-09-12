"""
figB_pareto_energy_T.py — Figure B, double panneau (demande utilisateur) :
Pareto débit/énergie, UMi (M=8, K=4) -- **gauche : CSI parfait**,
**droite : CSI imparfait** (pilote 20dB), même structure (RZF, WMMSE,
SC, IB, TA-RB par T ∈ {1,2,3,4,6,12}, débit à 15dB). Énergie par
précodage identique dans les deux panneaux (dépend de l'architecture,
pas du CSI). **T=4 mis en avant** comme point de fonctionnement
recommandé (meilleure moyenne %WMMSE du sweep CSI parfait ET moins
cher que T=6/T=12).

Usage: python3 figB_pareto_energy_T.py  (depuis results/figures_final/, CPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR = '../../results/'

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D'}
T_VALUES = [1, 2, 3, 4, 6, 12]
T_RECOMMENDED = 4
LABEL_OFFSETS = {1: (7, -11), 2: (7, -11), 3: (7, -11), 6: (7, -14), 12: (-8, 14)}

classical = np.load(DATA_DIR + 'classical_comparison_M8K4.npy', allow_pickle=True).item()
snr_classical = classical['snr']

single_evals = json.load(open(DATA_DIR + 'diag_front_a_signed_attn.json'))['evals']
intra_evals = json.load(open(DATA_DIR + 'diag_front_a_signed_attn_intra_rb.json'))['evals']
energy_v2 = json.load(open(DATA_DIR + 'complexity_energy_signed_attn_v2.json'))
energy_by_method = {r['method']: r for r in energy_v2 if r['precision'] == 'INT8'}
energy_fp32 = {r['method']: r for r in energy_v2 if r['precision'] == 'FP32'}
energy_per_T = json.load(open(DATA_DIR + 'energy_per_T_signed_attn.json'))
tsweep = {T: json.load(open(DATA_DIR + f'diag_tarb_residual_signed_attn_T{T}_train.json'))
          for T in T_VALUES}

e_rzf = energy_fp32['RZF']['energy_uj']
e_wmmse = energy_fp32['WMMSE (I=10)']['energy_uj']
e_sc = energy_by_method['SingleSC-signed_attn']['energy_uj']
e_ib = energy_by_method['IntraRB-signed_attn']['energy_uj']
e_T = {T: energy_per_T[str(T)]['energy_int8_uJ'] for T in T_VALUES}

# -- CSI parfait : débit à 15dB --
rate15_rzf = float(np.interp(15.0, snr_classical, classical['rzf_full']))
rate15_wmmse = float(np.interp(15.0, snr_classical, classical['wmmse']))
perfect_rates = {'RZF': rate15_rzf, 'WMMSE': rate15_wmmse,
                  'SC': single_evals['15.0'], 'IB': intra_evals['15.0']}
perfect_tsweep = {T: tsweep[T]['evals']['15.0'] for T in T_VALUES}

# -- CSI imparfait (pilote 20dB) : débit à 15dB -- T=4 déjà dans le sweep
# signed_attn (Figure A), T∈{1,2,3,6,12} calculés séparément (diag_csi_
# imperfect_tsweep.py) pour compléter la courbe Pareto --
csi_imp = json.load(open(DATA_DIR + 'diag_csi_imperfect_signed_attn_sweep.json'))
imperfect_rates = {'RZF': csi_imp['RZF']['15.0']['pilot20dB'],
                    'WMMSE': csi_imp['WMMSE']['15.0']['pilot20dB'],
                    'SC': csi_imp['SingleSC']['15.0']['pilot20dB'],
                    'IB': csi_imp['IntraRB']['15.0']['pilot20dB']}
imperfect_tsweep = {4: csi_imp['TA_RB_residual']['15.0']['pilot20dB']}
for T in T_VALUES:
    if T == 4:
        continue
    d = json.load(open(DATA_DIR + f'diag_csi_imperfect_tsweep_T{T}.json'))
    imperfect_tsweep[T] = d['results']['15.0']['pilot20dB']


def plot_panel(ax, rates, tsweep_rates, title):
    cmap = cm.get_cmap('RdPu')
    shades = [cmap(0.35 + 0.55 * i / (len(T_VALUES) - 1)) for i in range(len(T_VALUES))]

    ax.scatter(e_rzf, rates['RZF'], s=140, color=COLORS['RZF'], marker=MARKERS['RZF'],
               label='RZF', zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate('RZF', (e_rzf, rates['RZF']), textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.scatter(e_wmmse, rates['WMMSE'], s=140, color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
               label='WMMSE', zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate('WMMSE', (e_wmmse, rates['WMMSE']), textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.scatter(e_sc, rates['SC'], s=140, color=COLORS['SC'], marker=MARKERS['SC'],
               label='SC', zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate('SC', (e_sc, rates['SC']), textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.scatter(e_ib, rates['IB'], s=140, color=COLORS['IB'], marker=MARKERS['IB'],
               label='IB', zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate('IB', (e_ib, rates['IB']), textcoords='offset points', xytext=(8, 6), fontsize=9)

    e_line = [e_T[T] for T in T_VALUES]
    r_line = [tsweep_rates[T] for T in T_VALUES]
    ax.plot(e_line, r_line, color='#c94f8f', linestyle='-', linewidth=1.5, zorder=2, alpha=0.6)
    for T, e, r, c in zip(T_VALUES, e_line, r_line, shades):
        if T == T_RECOMMENDED:
            continue
        ax.scatter(e, r, s=130, color=c, marker='v', zorder=4, edgecolors='white', linewidths=0.8)
        ax.annotate(f'T={T}', (e, r), textcoords='offset points', xytext=LABEL_OFFSETS[T], fontsize=8)
    ax.plot([], [], color='#c94f8f', marker='v', linestyle='-', linewidth=1.5, markersize=8,
            label='TA-RB par T')

    i4 = T_VALUES.index(T_RECOMMENDED)
    ax.scatter(e_line[i4], r_line[i4], s=240, color=shades[i4], marker='*', zorder=5,
               edgecolors='#8b1a5c', linewidths=1.2, label=f'T={T_RECOMMENDED} (recommandé)')
    # pas d'étiquette texte ici (le marqueur étoile + la légende suffisent --
    # évite toute collision avec les T voisins, qui se resserrent différemment
    # selon le panneau CSI parfait/imparfait)

    ax.set_xscale('log')
    ax.set_xlabel('Énergie par précodage (µJ, échelle log)', fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
plot_panel(ax1, perfect_rates, perfect_tsweep, 'CSI parfait')
ax1.set_ylabel('Débit somme à 15dB (bps/Hz)', fontsize=12)
ax1.legend(fontsize=8.5, loc='lower right')
plot_panel(ax2, imperfect_rates, imperfect_tsweep, 'CSI imparfait (pilote 20dB)')

fig.suptitle('Compromis débit / énergie — UMi (M=8, K=4)', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figB_pareto_energy_T.png', dpi=200, bbox_inches='tight')
fig.savefig('figB_pareto_energy_T.pdf', bbox_inches='tight')
plt.close(fig)

out = {'perfect': {'RZF': {'energy_uj': e_rzf, 'rate15': perfect_rates['RZF']},
                    'WMMSE': {'energy_uj': e_wmmse, 'rate15': perfect_rates['WMMSE']},
                    'SC': {'energy_uj': e_sc, 'rate15': perfect_rates['SC']},
                    'IB': {'energy_uj': e_ib, 'rate15': perfect_rates['IB']},
                    'TA_RB_by_T': {str(T): {'energy_uj': e_T[T], 'rate15': perfect_tsweep[T],
                                             'recommended': (T == T_RECOMMENDED)} for T in T_VALUES}},
       'imperfect_pilot20dB': {'RZF': {'energy_uj': e_rzf, 'rate15': imperfect_rates['RZF']},
                                'WMMSE': {'energy_uj': e_wmmse, 'rate15': imperfect_rates['WMMSE']},
                                'SC': {'energy_uj': e_sc, 'rate15': imperfect_rates['SC']},
                                'IB': {'energy_uj': e_ib, 'rate15': imperfect_rates['IB']},
                                'TA_RB_by_T': {str(T): {'energy_uj': e_T[T], 'rate15': imperfect_tsweep[T],
                                                         'recommended': (T == T_RECOMMENDED)} for T in T_VALUES}}}
with open('figB_pareto_energy_T.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure B -> figB_pareto_energy_T.{png,pdf,json}')
