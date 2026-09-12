"""
make_figures_signed_attn.py — figures finales (demande utilisateur) pour
le chapitre : sum-rate vs SNR (CSI parfait), débit conservé (CSI
imparfait), Pareto débit/énergie. Figure 3 (T-sweep) séparée, dans
plot_tsweep_figure.py (données pas encore prêtes au moment de ce script).

Palette : validée colorblind-safe via dataviz skill
(scripts/validate_palette.js, ALL CHECKS PASS light+dark) -- PAS la
convention RZF=bleu/WMMSE=noir/vert+orange déjà utilisée dans les figures
précédentes du dépôt (`main_finall.py::_styles`), qui échoue le test CVD
(vert #2ca02c vs orange #ff7f0e : ΔE=0.7 en protanopie, quasi
indiscernable) -- corrigé pour ces figures finales.

Usage: python3 make_figures_signed_attn.py  (CPU, pas de calcul GPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Palette catégorielle validée (dataviz skill, slots 1-5, light mode)
COLORS = {
    'RZF':            '#2a78d6',   # slot1 blue
    'WMMSE':          '#eb6834',   # slot2 orange
    'SingleSC':       '#1baf7a',   # slot3 aqua
    'IntraRB':        '#eda100',   # slot4 yellow
    'TA_RB_residual': '#e87ba4',   # slot5 magenta
}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SingleSC': '^', 'IntraRB': 'D', 'TA_RB_residual': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SingleSC': 'Précodeur par sous-porteuse',
          'IntraRB': 'Transformer intra-bloc', 'TA_RB_residual': 'Agrégation par RB (décodeur résiduel)'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SingleSC': '-', 'IntraRB': '-', 'TA_RB_residual': '-'}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# FIGURE 1 — Sum-rate vs SNR, CSI parfait
# =============================================================================
classical = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
snr_classical = classical['snr']

neural_files = {
    'SingleSC': 'results/diag_front_a_signed_attn.json',
    'IntraRB': 'results/diag_front_a_signed_attn_intra_rb.json',
    'TA_RB_residual': 'results/diag_front_a_signed_attn_ta_rb_residual.json',
}
neural_evals = {name: load_json(path)['evals'] for name, path in neural_files.items()}
snr_neural = sorted(float(s) for s in next(iter(neural_evals.values())).keys())

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(snr_classical, classical['rzf_full'], color=COLORS['RZF'], marker=MARKERS['RZF'],
        linestyle=LINESTYLE['RZF'], label=LABELS['RZF'], linewidth=2, markersize=6)
ax.plot(snr_classical, classical['wmmse'], color=COLORS['WMMSE'], marker=MARKERS['WMMSE'],
        linestyle=LINESTYLE['WMMSE'], label=LABELS['WMMSE'], linewidth=2, markersize=6)
for name in ('SingleSC', 'IntraRB', 'TA_RB_residual'):
    y = [neural_evals[name][str(s) if s != int(s) or s == 0 else f'{s:.1f}'] for s in snr_neural]
    # clé JSON est "0.0","5.0",... -- reconstruite proprement :
    y = [neural_evals[name][f'{s:.1f}'] for s in snr_neural]
    ax.plot(snr_neural, y, color=COLORS[name], marker=MARKERS[name], linestyle=LINESTYLE[name],
            label=LABELS[name], linewidth=2, markersize=7)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Débit somme (bps/Hz)', fontsize=12)
ax.set_title('Débit somme vs SNR — CSI parfait, canal UMi (M=8, K=4)\n'
              'Architectures neuronales à attention à poids signés', fontsize=13, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/fig_signed_attn_sumrate_vs_snr.png', dpi=200, bbox_inches='tight')
fig.savefig('results/fig_signed_attn_sumrate_vs_snr.pdf', bbox_inches='tight')
plt.close(fig)
print('✅ Figure 1 -> results/fig_signed_attn_sumrate_vs_snr.{png,pdf}')

# Tableau %WMMSE en complément
print('\n--- Tableau %WMMSE (fig.1) ---')
wmmse_by_snr = dict(zip(snr_classical, classical['wmmse']))
print(f'{"SNR":>6}', *[f'{n:>12}' for n in ('RZF', 'SingleSC', 'IntraRB', 'TA_RB_res')])
for s in snr_neural:
    wmmse_v = wmmse_by_snr[s] if s in wmmse_by_snr else np.interp(s, snr_classical, classical['wmmse'])
    row = [100 * neural_evals[n][f'{s:.1f}'] / wmmse_v for n in ('SingleSC', 'IntraRB', 'TA_RB_residual')]
    rzf_v = np.interp(s, snr_classical, classical['rzf_full'])
    print(f'{s:>6.1f} {100*rzf_v/wmmse_v:>11.1f}%', *[f'{v:>11.1f}%' for v in row])


# =============================================================================
# FIGURE 2 — Débit conservé, CSI imparfait (data SNR=15dB fixe, 3 pilotes)
# NOTE méthodologique : PAS une courbe vs SNR -- diag_csi_imperfect_signed_
# attn.py mesure à data SNR=15dB FIXE avec pilote {parfait,20dB,10dB}, pas
# un sweep SNR sous CSI imparfait (jamais mesuré). Barres groupées, seule
# représentation honnête des données disponibles.
# =============================================================================
csi = load_json('results/diag_csi_imperfect_signed_attn.json')
methods = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']
conditions = ['perfect', 'pilot20dB', 'pilot10dB']
cond_labels = ['CSI parfait', 'Pilote 20dB', 'Pilote 10dB']

fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(len(conditions))
width = 0.15
for i, m in enumerate(methods):
    vals = [csi[m][c] for c in conditions]
    positions = x + (i - 2) * width
    bars = ax.bar(positions, vals, width, label=LABELS[m], color=COLORS[m], alpha=0.9)

ax.set_xlabel('Condition CSI (SNR données = 15dB fixe)', fontsize=12)
ax.set_ylabel('Débit somme (bps/Hz)', fontsize=12)
ax.set_title('Débit somme sous CSI imparfait — canal UMi (M=8, K=4), SNR données=15dB\n'
              'Architectures neuronales à attention à poids signés', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cond_labels)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('results/fig_signed_attn_csi_imperfect.png', dpi=200, bbox_inches='tight')
fig.savefig('results/fig_signed_attn_csi_imperfect.pdf', bbox_inches='tight')
plt.close(fig)
print('\n✅ Figure 2 -> results/fig_signed_attn_csi_imperfect.{png,pdf}')

print('\n--- Tableau %retenu (fig.2) ---')
for m in methods:
    p20 = 100 * csi[m]['pilot20dB'] / csi[m]['perfect']
    p10 = 100 * csi[m]['pilot10dB'] / csi[m]['perfect']
    print(f'  {m:<16} perfect={csi[m]["perfect"]:.2f}  pilot20dB={csi[m]["pilot20dB"]:.2f} ({p20:.1f}%)  '
          f'pilot10dB={csi[m]["pilot10dB"]:.2f} ({p10:.1f}%)')


# =============================================================================
# FIGURE 4 — Pareto débit vs énergie (@15dB, CSI parfait)
# =============================================================================
energy = load_json('results/complexity_energy_signed_attn.json')
energy_by_method = {r['method']: r for r in energy}

pareto_map = {
    'RZF':             ('RZF', 'rzf_full', classical),
    'WMMSE':           ('WMMSE (I=10)', 'wmmse', classical),
    'SingleSC':        ('SingleSC-signed_attn', None, neural_evals['SingleSC']),
    'IntraRB':         ('IntraRB-signed_attn', None, neural_evals['IntraRB']),
    'TA_RB_residual':  ('TA-RB-résiduel-signed_attn (T=6)', None, neural_evals['TA_RB_residual']),
}

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for name, (energy_key, classical_key, data) in pareto_map.items():
    e = energy_by_method[energy_key]['energy_uj']
    if classical_key is not None:
        rate15 = float(np.interp(15.0, snr_classical, classical[classical_key]))
    else:
        rate15 = data['15.0']
    ax.scatter(e, rate15, s=140, color=COLORS[name], marker=MARKERS[name],
               label=LABELS[name], zorder=3, edgecolors='white', linewidths=0.8)
    ax.annotate(LABELS[name], (e, rate15), textcoords='offset points',
                xytext=(8, 6), fontsize=8.5, color='#3a3a3a')

ax.set_xscale('log')
ax.set_xlabel('Énergie estimée par précodage (µJ, échelle log)\nRZF/WMMSE : FP32 — architectures neuronales : INT8', fontsize=11.5)
ax.set_ylabel('Débit somme à 15dB (bps/Hz)', fontsize=12)
ax.set_title('Compromis débit / énergie — canal UMi (M=8, K=4), CSI parfait\n'
              'Architectures neuronales à attention à poids signés', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('results/fig_signed_attn_pareto_energy.png', dpi=200, bbox_inches='tight')
fig.savefig('results/fig_signed_attn_pareto_energy.pdf', bbox_inches='tight')
plt.close(fig)
print('\n✅ Figure 4 -> results/fig_signed_attn_pareto_energy.{png,pdf}')

print('\nToutes les figures (sauf T-sweep, séparée) générées.')
