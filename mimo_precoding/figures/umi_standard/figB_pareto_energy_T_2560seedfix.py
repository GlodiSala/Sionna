"""
figB_pareto_energy_T_2560seedfix.py -- Figure B (Pareto debit/energie),
regeneree avec le run T-sweep seedfix a 2560 echantillons/point (20x128,
protocole d'entrainement d'origine) pour les points RZF / WMMSE / T={1,
2,3,4,6,12} (TA-RB), aux DEUX panneaux (CSI parfait / CSI imparfait
pilote 20dB), debit a 15dB :
final/results/tsweep_seedfix_perfect_manual_2560_20260820_153545.json
final/results/tsweep_seedfix_pilot20_manual_2560_20260820_153545.json

Les points SC et IB (Transformer par-SC / Intra-bloc) restent inchanges
(hors perimetre de cette re-verification, qui ne porte que sur le
T-sweep TA-RB -- SC/IB n'ont pas ete regeneres avec seed fixe+canal
partage sur ce protocole). L'energie par precodage (architecture,
independante du canal/CSI) est egalement inchangee.

Ecrit dans figB_pareto_energy_T.png/.pdf (memes noms que l'original
figB_pareto_energy_T.py, qui n'est PAS modifie) pour etre copiee dans
le depot Overleaf clone (figures/figB_pareto_energy.png).
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
# Panneau CSI imparfait : T=2/T=3/T=4(etoile) sont beaucoup plus proches les
# uns des autres qu'avec les anciennes donnees (cf. collapse de robustesse a
# T eleve) -> decalages verticaux alternes pour eviter le chevauchement des
# etiquettes T=2/T=3.
LABEL_OFFSETS_IMPERFECT = {1: (7, -13), 2: (7, 9), 3: (7, -18), 6: (7, -14), 12: (-8, 14)}

# -- T-sweep seedfix 2560 echantillons (RZF-SC, WMMSE, T1..T12) --
perfect_tsweep_d = json.load(open(DATA_DIR + 'tsweep_seedfix_perfect_manual_2560_20260820_153545.json'))['table']
pilot_tsweep_d = json.load(open(DATA_DIR + 'tsweep_seedfix_pilot20_manual_2560_20260820_153545.json'))['table']
snr_tsweep = perfect_tsweep_d['snr']
i15 = snr_tsweep.index(15.0)

# -- SC / IB : inchanges (hors perimetre de cette re-verification) --
single_evals = json.load(open(DATA_DIR + 'diag_front_a_signed_attn.json'))['evals']
intra_evals = json.load(open(DATA_DIR + 'diag_front_a_signed_attn_intra_rb.json'))['evals']
csi_imp = json.load(open(DATA_DIR + 'diag_csi_imperfect_signed_attn_sweep.json'))

# -- Energie (architecture, inchangee) --
energy_v2 = json.load(open(DATA_DIR + 'complexity_energy_signed_attn_v2.json'))
energy_by_method = {r['method']: r for r in energy_v2 if r['precision'] == 'INT8'}
energy_fp32 = {r['method']: r for r in energy_v2 if r['precision'] == 'FP32'}
energy_per_T = json.load(open(DATA_DIR + 'energy_per_T_signed_attn.json'))

e_rzf = energy_fp32['RZF']['energy_uj']
e_wmmse = energy_fp32['WMMSE (I=10)']['energy_uj']
e_sc = energy_by_method['SingleSC-signed_attn']['energy_uj']
e_ib = energy_by_method['IntraRB-signed_attn']['energy_uj']
e_T = {T: energy_per_T[str(T)]['energy_int8_uJ'] for T in T_VALUES}

# -- CSI parfait : debit a 15dB (RZF/WMMSE/T du run 2560 ; SC/IB inchanges) --
perfect_rates = {'RZF': perfect_tsweep_d['rzf_sc'][i15], 'WMMSE': perfect_tsweep_d['wmmse'][i15],
                  'SC': single_evals['15.0'], 'IB': intra_evals['15.0']}
perfect_tsweep = {T: perfect_tsweep_d[f'T{T}'][i15] for T in T_VALUES}

# -- CSI imparfait (pilote 20dB) : debit a 15dB --
imperfect_rates = {'RZF': pilot_tsweep_d['rzf_sc'][i15], 'WMMSE': pilot_tsweep_d['wmmse'][i15],
                    'SC': csi_imp['SingleSC']['15.0']['pilot20dB'],
                    'IB': csi_imp['IntraRB']['15.0']['pilot20dB']}
imperfect_tsweep = {T: pilot_tsweep_d[f'T{T}'][i15] for T in T_VALUES}


def plot_panel(ax, rates, tsweep_rates, title, label_offsets=LABEL_OFFSETS):
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
        ax.annotate(f'T={T}', (e, r), textcoords='offset points', xytext=label_offsets[T], fontsize=8)
    ax.plot([], [], color='#c94f8f', marker='v', linestyle='-', linewidth=1.5, markersize=8,
            label='TA-RB par T')

    i4 = T_VALUES.index(T_RECOMMENDED)
    ax.scatter(e_line[i4], r_line[i4], s=240, color=shades[i4], marker='*', zorder=5,
               edgecolors='#8b1a5c', linewidths=1.2, label=f'T={T_RECOMMENDED} (recommandé)')

    ax.set_xscale('log')
    ax.set_xlabel('Énergie par précodage (µJ, échelle log)', fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
plot_panel(ax1, perfect_rates, perfect_tsweep, 'CSI parfait')
ax1.set_ylabel('Débit total à 15dB (bps/Hz)', fontsize=12)
ax1.legend(fontsize=8.5, loc='lower right')
plot_panel(ax2, imperfect_rates, imperfect_tsweep, 'CSI imparfait (pilote 20dB)', label_offsets=LABEL_OFFSETS_IMPERFECT)

fig.tight_layout()
fig.savefig('figB_pareto_energy_T.png', dpi=200, bbox_inches='tight')
fig.savefig('figB_pareto_energy_T.pdf', bbox_inches='tight')
plt.close(fig)

out = {'perfect': {'RZF': {'energy_uj': e_rzf, 'rate15': perfect_rates['RZF']},
                    'WMMSE': {'energy_uj': e_wmmse, 'rate15': perfect_rates['WMMSE']},
                    'SC': {'energy_uj': e_sc, 'rate15': perfect_rates['SC'], 'note': 'inchange (hors perimetre)'},
                    'IB': {'energy_uj': e_ib, 'rate15': perfect_rates['IB'], 'note': 'inchange (hors perimetre)'},
                    'TA_RB_by_T': {str(T): {'energy_uj': e_T[T], 'rate15': perfect_tsweep[T],
                                             'recommended': (T == T_RECOMMENDED)} for T in T_VALUES}},
       'imperfect_pilot20dB': {'RZF': {'energy_uj': e_rzf, 'rate15': imperfect_rates['RZF']},
                                'WMMSE': {'energy_uj': e_wmmse, 'rate15': imperfect_rates['WMMSE']},
                                'SC': {'energy_uj': e_sc, 'rate15': imperfect_rates['SC'], 'note': 'inchange (hors perimetre)'},
                                'IB': {'energy_uj': e_ib, 'rate15': imperfect_rates['IB'], 'note': 'inchange (hors perimetre)'},
                                'TA_RB_by_T': {str(T): {'energy_uj': e_T[T], 'rate15': imperfect_tsweep[T],
                                                         'recommended': (T == T_RECOMMENDED)} for T in T_VALUES}},
       'source_tsweep': {
           'perfect': 'tsweep_seedfix_perfect_manual_2560_20260820_153545.json',
           'pilot20': 'tsweep_seedfix_pilot20_manual_2560_20260820_153545.json'}}
with open('figB_pareto_energy_T_2560seedfix.json', 'w') as f:
    json.dump(out, f, indent=2)
print('OK -> figB_pareto_energy_T.{png,pdf} (source T/RZF/WMMSE: run 2560 ; SC/IB/energie: inchanges) + figB_pareto_energy_T_2560seedfix.json')
