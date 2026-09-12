"""
figE_se_mean_vs_tail.py — Figure E : pourquoi le BER des précodeurs appris
plafonne alors que leur débit-somme reste à 92-95 % de RZF.

C'est la figure explicative du BER, à lire AVANT figD : elle montre que la
différence entre méthodes n'est pas dans la moyenne mais dans la QUEUE de la
distribution, et que le BER à MCS fixe (QPSK r=1/2) est une statistique de
queue.

  - Panneau gauche : efficacité spectrale MOYENNE (traits pleins) et
    efficacité spectrale du PIRE mot de code (pointillés) en fonction du SNR.
    Les moyennes se superposent ; les pires divergent. La ligne horizontale
    marque le seuil de décodage (~1,2 bps/Hz pour QPSK r=1/2, limite de
    Shannon à 1,0). Un mot de code sous cette ligne est perdu.
  - Panneau droit : taux d'erreur trame (FER) correspondant. RZF et WMMSE
    tombent à zéro ; les précodeurs appris se bloquent, parce que leur pire
    mot de code sature au lieu de monter avec le SNR (interférence
    résiduelle, pas bruit).

Données : results/mechanism_ber_error_floor_*.json
(experiments/mechanism_ber_error_floor.py, mots de code individuels).
Plusieurs fichiers sont fusionnés pour couvrir toute la plage de SNR.

Usage: python3 figE_se_mean_vs_tail.py [--csi perfect|pilot20]
       (depuis figures/umi_standard/, CPU)
"""
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_p = argparse.ArgumentParser()
_p.add_argument('--csi', choices=['perfect', 'pilot20'], default='perfect')
_a = _p.parse_args()

DATA_DIR = '../../results/'
CSI = _a.csi             # 'perfect' ou 'pilot20'
THRESHOLD = 1.2          # seuil de decodage QPSK r=1/2 (Shannon 1.0, LDPC 5G ~1.2)

COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LABELS = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SC — par sous-porteuse',
          'IB': 'IB — intra-bloc', 'TA-RB': 'TA-RB — agrégation par RB'}

# ── fusion des runs (un fichier peut ne couvrir qu'une partie des SNR) ──
merged, nb = {}, None
for f in sorted(glob.glob(DATA_DIR + 'mechanism_ber_error_floor_*.json')):
    d = json.load(open(f))
    if d['config'].get('csi', 'perfect') != CSI:
        continue
    nb = d['config']['num_batches']
    for m, per_snr in d['methods'].items():
        for snr, rec in per_snr.items():
            merged.setdefault(m, {})[float(snr)] = rec
if not merged:
    raise SystemExit(f"aucun run --csi {CSI} dans {DATA_DIR} ; lancer "
                      f"experiments/mechanism_ber_error_floor.py --csi {CSI}")

snrs = sorted(next(iter(merged.values())).keys())
ncw = next(iter(merged.values()))[snrs[0]]['codewords']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for m in LABELS:
    s = [x for x in snrs if x in merged[m]]
    ax1.plot(s, [merged[m][x]['rate_mean_all'] for x in s], color=COLORS[m], marker=MARKERS[m],
             linestyle='-', linewidth=2, markersize=6, label=LABELS[m])
    ax1.plot(s, [merged[m][x]['rate_min'] for x in s], color=COLORS[m], marker=MARKERS[m],
             linestyle='--', linewidth=1.6, markersize=5, alpha=0.85)
ax1.axhline(THRESHOLD, color='#cc0000', linewidth=1.2, linestyle=':')
ax1.text(snrs[0], THRESHOLD * 1.08, 'seuil de décodage QPSK r=1/2', fontsize=8, color='#cc0000')
ax1.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax1.set_ylabel('Efficacité spectrale (bps/Hz)', fontsize=12)
ax1.set_title('Trait plein : moyenne — Pointillé : pire mot de code', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8.5, loc='upper left')

floor = 1.0 / ncw
ZERO_Y = floor * 0.45          # ligne « aucune trame en échec », sous le plancher
for m in LABELS:
    s = [x for x in snrs if x in merged[m]]
    fer = np.array([merged[m][x]['fer'] for x in s], dtype=float)
    y = np.where(fer > 0, fer, ZERO_Y)
    ax2.semilogy(s, y, color=COLORS[m], linestyle='-', linewidth=2, label=LABELS[m], zorder=2)
    pos, zer = fer > 0, fer == 0
    ax2.plot(np.array(s)[pos], y[pos], color=COLORS[m], marker=MARKERS[m], linestyle='none',
             markersize=6, zorder=3)
    ax2.plot(np.array(s)[zer], y[zer], color=COLORS[m], marker=MARKERS[m], linestyle='none',
             markersize=7, markerfacecolor='white', markeredgewidth=1.6, zorder=3)
ax2.axhline(floor, color='#999999', linewidth=0.8, linestyle=':')
ax2.set_ylim(ZERO_Y * 0.6, None)
ax2.set_yticks([ZERO_Y] + [t for t in (1e-3, 1e-2) if t >= floor * 0.5])
ax2.set_yticklabels(['0'] + [f'$10^{{{int(np.log10(t))}}}$' for t in (1e-3, 1e-2) if t >= floor * 0.5])
ax2.text(snrs[-1], floor * 1.12, f'plancher de mesure : 1 trame / {ncw}',
         fontsize=8, color='#777777', ha='right')
ax2.text(snrs[-1], ZERO_Y * 1.12, 'marqueurs vides : aucune trame en échec',
         fontsize=8, color='#555555', ha='right')
ax2.set_xlabel('SNR (Eb/N0, dB)', fontsize=12)
ax2.set_ylabel('Taux d\'erreur trame (FER)', fontsize=12)
ax2.set_title(f'Conséquence : FER ({ncw} mots de code par point)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='major')
ax2.legend(fontsize=8.5, loc='upper right')

suptitle = ('Moyenne vs queue de distribution — UMi (M=8, K=4), CSI parfait' if CSI == 'perfect'
            else 'Moyenne vs queue de distribution — UMi (M=8, K=4), CSI imparfait (pilote 20 dB)')
fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
stem = 'figE_se_mean_vs_tail' + ('' if CSI == 'perfect' else '_csi_imperfect')
fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
fig.savefig(stem + '.pdf', bbox_inches='tight')
plt.close(fig)

out = {'csi': CSI, 'codewords_per_point': ncw, 'num_batches': nb, 'threshold_bps_hz': THRESHOLD,
       'snrs': snrs,
       'data': {m: {'rate_mean': [merged[m][x]['rate_mean_all'] for x in snrs if x in merged[m]],
                     'rate_min': [merged[m][x]['rate_min'] for x in snrs if x in merged[m]],
                     'fer': [merged[m][x]['fer'] for x in snrs if x in merged[m]],
                     'frame_errors': [merged[m][x]['frame_errors'] for x in snrs if x in merged[m]]}
                 for m in LABELS}}
with open(stem + '.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'✅ Figure E -> {stem}.{{png,pdf,json}}')
