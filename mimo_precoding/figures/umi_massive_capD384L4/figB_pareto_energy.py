"""
figB_pareto_energy.py — Figure B, UMi MASSIVE (M=64, K=8), VARIANTE
D=384,L=4. Pareto débit/énergie, mêmes conventions que la figure de
production (`umi_massive/figB_pareto_energy.py`, D=128, INTACTE) --
purement descriptif, aucune recommandation visuelle.

Usage: python3 figB_pareto_energy.py  (depuis son propre dossier, CPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../results/'

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SC (D=384)', 'IB': 'IB (D=384)', 'TA-RB': 'TA-RB (T=6, D=384)'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}

classical = np.load(DATA_DIR + 'classical_comparison_M64K8_true.npy', allow_pickle=True).item()
snr_classical = classical['snr']
energy = json.load(open(DATA_DIR + 'complexity_energy_massive_capD384L4.json'))
neural_evals = {k: json.load(open(DATA_DIR + f'diag_massive_gap_fullbudget_{ARCH_KEY[k]}_cap_d384l4.json'))['evals']
                for k in ('SC', 'IB', 'TA-RB')}

rate15_rzf = float(np.interp(15.0, snr_classical, classical['rzf_full']))
rate15_wmmse = float(np.interp(15.0, snr_classical, classical['wmmse']))
points = {
    'RZF':   (energy['RZF']['energy_uj'], rate15_rzf),
    'WMMSE': (energy['WMMSE']['energy_uj'], rate15_wmmse),
    'SC':    (energy['SingleSC']['energy_uj_int8'], neural_evals['SC']['15.0']),
    'IB':    (energy['IntraRB']['energy_uj_int8'], neural_evals['IB']['15.0']),
    'TA-RB': (energy['TA_RB_residual']['energy_uj_int8'], neural_evals['TA-RB']['15.0']),
}

fig, ax = plt.subplots(figsize=(7.5, 5.8))
for k, (e, r) in points.items():
    ax.scatter(e, r, s=140, color=COLORS[k], marker=MARKERS[k], label=LABELS[k],
               zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate(k, (e, r), textcoords='offset points', xytext=(8, 6), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel('Énergie par précodage (µJ, échelle log)', fontsize=11.5)
ax.set_ylabel('Débit somme à 15dB (bps/Hz)', fontsize=12)
ax.set_title('Compromis débit / énergie — UMi MASSIVE (M=64, K=8), D=384', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('figB_pareto_energy.png', dpi=200, bbox_inches='tight')
fig.savefig('figB_pareto_energy.pdf', bbox_inches='tight')
plt.close(fig)

out = {k: {'energy_uj': e, 'rate15': r} for k, (e, r) in points.items()}
with open('figB_pareto_energy.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure B (UMi MASSIVE, D=384) -> figB_pareto_energy.{png,pdf,json}')
