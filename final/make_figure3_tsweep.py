"""
make_figure3_tsweep.py — Figure 3 (mise à jour, demande utilisateur) :
débit vs T pour l'Agrégation par RB (décodeur résiduel), séparée des
figures 1/2/4, avec RZF groupé par sous-porteuses (group_size=rb_size/T)
en comparaison, ET l'énergie par T (Priorité 4, sous-graphique séparé --
PAS de double axe, anti-pattern -- cf. skill dataviz).

NOTE méthodologique (inchangée) : T-sweep COURT et DIRECTIONNEL (500 pas,
~3.3 équiv-époques), PAS le protocole de production complet -- valeurs
absolues non comparables aux figures 1/2/4.

Usage: python3 make_figure3_tsweep.py  (CPU, pas de calcul GPU)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR_NEURAL = '#e87ba4'   # magenta (même famille que "Agrégation par RB" dans fig.1/2/4)
COLOR_RZF    = '#2a78d6'   # blue (même que RZF dans fig.1/2/4)
LABEL_NEURAL = 'Agrégation par RB (décodeur résiduel)'
LABEL_RZF    = 'RZF groupé (compression équivalente)'

tsweep = json.load(open('results/diag_tarb_residual_signed_attn_T_sweep.json'))
rzf_grouped = json.load(open('results/diag_rzf_grouped_for_tsweep.json'))
energy_per_T = json.load(open('results/energy_per_T_signed_attn.json'))

T_VALUES = [1, 2, 3, 4, 6, 12]
SNRS = [0.0, 10.0, 20.0]
x = np.arange(len(T_VALUES))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

for ax, snr in zip(axes[:3], SNRS):
    neural_y = [tsweep[str(T)]['eval_sum_rate'][str(snr)] for T in T_VALUES]
    rzf_y = [rzf_grouped[str(T)]['rates'][str(snr)] for T in T_VALUES]

    ax.plot(x, rzf_y, color=COLOR_RZF, marker='o', linestyle='--', linewidth=2,
            markersize=7, label=LABEL_RZF)
    ax.plot(x, neural_y, color=COLOR_NEURAL, marker='v', linestyle='-', linewidth=2,
            markersize=8, label=LABEL_NEURAL)

    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in T_VALUES])
    ax.set_xlabel('T (tokens par RB)', fontsize=11)
    ax.set_title(f'Débit — SNR = {snr:.0f}dB', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Débit somme (bps/Hz)', fontsize=11)
axes[0].legend(fontsize=8.5, loc='upper left')

# -- 4e sous-graphique : énergie vs T (Priorité 4, pas de double axe) --
ax_e = axes[3]
e_vals = [energy_per_T[str(T)]['energy_int8_uJ'] for T in T_VALUES]
ax_e.plot(x, e_vals, color=COLOR_NEURAL, marker='v', linestyle='-', linewidth=2, markersize=8)
ax_e.set_xticks(x)
ax_e.set_xticklabels([str(T) for T in T_VALUES])
ax_e.set_xlabel('T (tokens par RB)', fontsize=11)
ax_e.set_ylabel('Énergie estimée (µJ, INT8)', fontsize=11)
ax_e.set_title('Énergie par précodage', fontsize=12, fontweight='bold')
ax_e.grid(True, alpha=0.3)

fig.suptitle('Débit et énergie vs T — Agrégation par RB (décodeur résiduel), canal UMi (M=8, K=4)\n'
             'Run court directionnel (500 pas) — comparaison relative entre T, pas le protocole complet',
             fontsize=13, fontweight='bold', y=1.06)
fig.tight_layout()
fig.savefig('results/fig3_tsweep.png', dpi=200, bbox_inches='tight')
fig.savefig('results/fig3_tsweep.pdf', bbox_inches='tight')
plt.close(fig)
print('✅ Figure 3 (+ énergie) -> results/fig3_tsweep.{png,pdf}')

print(f'\n{"T":>3} {"FLOPs(M)":>10} {"Énergie(µJ)":>12} {"Débit@20dB":>12} {"RZF-grp@20dB":>13} {"ratio":>8}')
for T in T_VALUES:
    f = tsweep[str(T)]['flops_real_M']
    e = energy_per_T[str(T)]['energy_int8_uJ']
    n20 = tsweep[str(T)]['eval_sum_rate']['20.0']
    r20 = rzf_grouped[str(T)]['rates']['20.0']
    print(f'{T:>3} {f:>10.1f} {e:>12.5f} {n20:>12.2f} {r20:>13.2f} {100*n20/r20:>7.1f}%')
