"""
figB_pareto_energy.py — Figure B, UMa MASSIVE (M=64, K=8) : Pareto
débit/énergie. Pas de T-sweep -- 5 points, **purement descriptif,
aucune recommandation ni mise en avant visuelle**. Énergie réutilisée
de umi_massive (mêmes FLOPs, indépendants du canal, seulement de M/K
et de l'architecture).

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
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SC', 'IB': 'IB', 'TA-RB': 'TA-RB (T=6)'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}
ENERGY_KEY = {'SC': 'SingleSC', 'IB': 'IntraRB', 'TA-RB': 'TA_RB_residual'}

classical = np.load(DATA_DIR + 'classical_comparison_M64K8_true_uma.npy', allow_pickle=True).item()
snr_classical = classical['snr']
energy = json.load(open(DATA_DIR + 'complexity_energy_massive_true.json'))
data = {k: json.load(open(DATA_DIR + f'diag_uma_massive_{ARCH_KEY[k]}.json')) for k in ('SC', 'IB', 'TA-RB')}

rate15_rzf = float(np.interp(15.0, snr_classical, classical['rzf_full']))
rate15_wmmse = float(np.interp(15.0, snr_classical, classical['wmmse']))
points = {
    'RZF':   (energy['RZF']['energy_uj'], rate15_rzf),
    'WMMSE': (energy['WMMSE']['energy_uj'], rate15_wmmse),
    'SC':    (energy['SingleSC']['energy_uj_int8'], data['SC']['evals_perfect']['15.0']),
    'IB':    (energy['IntraRB']['energy_uj_int8'], data['IB']['evals_perfect']['15.0']),
    'TA-RB': (energy['TA_RB_residual']['energy_uj_int8'], data['TA-RB']['evals_perfect']['15.0']),
}

fig, ax = plt.subplots(figsize=(7.5, 5.8))
for k, (e, r) in points.items():
    ax.scatter(e, r, s=140, color=COLORS[k], marker=MARKERS[k], label=LABELS[k],
               zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate(k, (e, r), textcoords='offset points', xytext=(8, 6), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel('Énergie par précodage (µJ, échelle log)', fontsize=11.5)
ax.set_ylabel('Débit somme à 15dB (bps/Hz)', fontsize=12)
ax.set_title('Compromis débit / énergie — UMa MASSIVE (M=64, K=8)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('figB_pareto_energy.png', dpi=200, bbox_inches='tight')
fig.savefig('figB_pareto_energy.pdf', bbox_inches='tight')
plt.close(fig)

out = {k: {'energy_uj': e, 'rate15': r} for k, (e, r) in points.items()}
with open('figB_pareto_energy.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure B (UMa MASSIVE) -> figB_pareto_energy.{png,pdf,json}')
