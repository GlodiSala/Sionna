"""
figF_system_throughput.py — Figure F : débit réellement livré au niveau
système (ordonnanceur proportionnel équitable + adaptation de lien en boucle
externe + abstraction PHY), UMi standard (M=8, K=4).

Double panneau : CSI parfait | CSI imparfait (pilote 20 dB). L'ordonnée est
l'efficacité spectrale **livrée** : un bloc de transport en échec compte pour
zéro bit (`num_decoded_bits = harq_feedback x num_cb x cb_size`), donc ces
courbes paient déjà le prix des trames perdues -- contrairement au débit-somme
de la figure A, qui est une capacité. Il n'y a pas de retransmission HARQ :
les chiffres sont une borne inférieure.

C'est la conclusion pratique de la figure E : le plancher d'erreur des
précodeurs appris se paie en un cran de MCS, pas en débit.

Données : results/diag_system_scheduler_eval.json et
results/diag_system_scheduler_eval_csi_imperfect_pilot20dB.json
(experiments/eval_system_scheduler.py). Le nombre de slots est déduit de la
longueur de `mcs_history` et reporté dans le titre : à 100 slots les courbes
sont visiblement bruitées, il en faut davantage pour citer un écart.

Usage: python3 figF_system_throughput.py   (depuis figures/umi_standard/, CPU)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../results/'
COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SC — par sous-porteuse',
          'IB': 'IB — intra-bloc', 'TA-RB': 'TA-RB — agrégation par RB'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}

PANELS = [('diag_system_scheduler_eval.json', 'CSI parfait'),
          ('diag_system_scheduler_eval_csi_imperfect_pilot20dB.json', 'CSI imparfait (pilote 20 dB)')]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
meta = {}
for ax, (fname, title) in zip(axes, PANELS):
    path = DATA_DIR + fname
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f'{fname}\nabsent', ha='center', va='center', transform=ax.transAxes)
        continue
    d = json.load(open(path))
    snrs = sorted(float(s) for s in d['RZF'])
    nslots = d.get('metadata', {}).get('num_slots')
    meta[title] = nslots
    for m in LABELS:
        y = [d[m][s]['spectral_efficiency'] for s in sorted(d[m], key=float)]
        ax.plot(snrs, y, color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLE[m],
                linewidth=2, markersize=6, label=LABELS[m])
    ax.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
    ax.set_title(f'{title} — {nslots} slots' if nslots else title,
                 fontsize=12.5, fontweight='bold')
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Efficacité spectrale livrée (bps/Hz)', fontsize=12)
axes[0].legend(fontsize=8.5, loc='upper left')

fig.suptitle('Débit livré — ordonnanceur + MCS adaptatif, UMi (M=8, K=4)',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figF_system_throughput.png', dpi=200, bbox_inches='tight')
fig.savefig('figF_system_throughput.pdf', bbox_inches='tight')
plt.close(fig)

out = {'num_slots': meta, 'note': 'efficacite spectrale livree, blocs en echec comptes a zero, '
                                   'pas de retransmission HARQ'}
for fname, title in PANELS:
    if os.path.exists(DATA_DIR + fname):
        d = json.load(open(DATA_DIR + fname))
        out[title] = {m: {'snr': sorted((float(s) for s in d[m]), key=float),
                          'spectral_efficiency': [d[m][s]['spectral_efficiency'] for s in sorted(d[m], key=float)],
                          'jain_fairness': [d[m][s]['jain_fairness'] for s in sorted(d[m], key=float)]}
                      for m in LABELS}
with open('figF_system_throughput.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure F -> figF_system_throughput.{png,pdf,json} —', meta)
