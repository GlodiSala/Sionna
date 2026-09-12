"""
figB_pareto_energy.py — Figure B, UMa STANDARD (M=8, K=4) : Pareto
débit/énergie. Pas de T-sweep sur ce régime (TA-RB à T=4 fixe,
référence identifiée sur UMi standard) -- 5 points, **purement
descriptif, aucune recommandation ni mise en avant visuelle**.

Usage: python3 figB_pareto_energy.py  (depuis son propre dossier, CPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../'

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SC', 'IB': 'IB', 'TA-RB': 'TA-RB (T=4)'}
ARCH_KEY = {'SC': 'single_sc', 'IB': 'intra_rb', 'TA-RB': 'ta_rb_residual'}

classical = np.load(DATA_DIR + 'classical_comparison_M8K4_uma.npy', allow_pickle=True).item()
snr_classical = classical['snr']
energy_v2 = json.load(open(DATA_DIR + 'complexity_energy_signed_attn_v2.json'))
energy_by_method = {r['method']: r for r in energy_v2 if r['precision'] == 'INT8'}
energy_fp32 = {r['method']: r for r in energy_v2 if r['precision'] == 'FP32'}
energy_per_T = json.load(open(DATA_DIR + 'energy_per_T_signed_attn.json'))
data = {k: json.load(open(DATA_DIR + f'diag_uma_signed_attn_{ARCH_KEY[k]}.json')) for k in ('SC', 'IB', 'TA-RB')}

rate15_rzf = float(np.interp(15.0, snr_classical, classical['rzf_full']))
rate15_wmmse = float(np.interp(15.0, snr_classical, classical['wmmse']))
points = {
    'RZF':   (energy_fp32['RZF']['energy_uj'], rate15_rzf),
    'WMMSE': (energy_fp32['WMMSE (I=10)']['energy_uj'], rate15_wmmse),
    'SC':    (energy_by_method['SingleSC-signed_attn']['energy_uj'], data['SC']['evals_perfect']['15.0']),
    'IB':    (energy_by_method['IntraRB-signed_attn']['energy_uj'], data['IB']['evals_perfect']['15.0']),
    'TA-RB': (energy_per_T['4']['energy_int8_uJ'], data['TA-RB']['evals_perfect']['15.0']),
}

fig, ax = plt.subplots(figsize=(7.5, 5.8))
for k, (e, r) in points.items():
    ax.scatter(e, r, s=140, color=COLORS[k], marker=MARKERS[k], label=LABELS[k],
               zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate(k, (e, r), textcoords='offset points', xytext=(8, 6), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel('Énergie par précodage (µJ, échelle log)', fontsize=11.5)
ax.set_ylabel('Débit somme à 15dB (bps/Hz)', fontsize=12)
ax.set_title('Compromis débit / énergie — UMa STANDARD (M=8, K=4)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('figB_pareto_energy.png', dpi=200, bbox_inches='tight')
fig.savefig('figB_pareto_energy.pdf', bbox_inches='tight')
plt.close(fig)

out = {k: {'energy_uj': e, 'rate15': r} for k, (e, r) in points.items()}
with open('figB_pareto_energy.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure B (UMa STANDARD) -> figB_pareto_energy.{png,pdf,json}')
