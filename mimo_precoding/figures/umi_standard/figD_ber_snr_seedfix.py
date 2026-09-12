"""
figD_ber_snr_seedfix.py — Figure D (variante seedfix) : BER vs SNR, UMi
(M=8, K=4), double panneau — **gauche : CSI parfait** (9 points, TA-RB
**T=6**), **droite : CSI imparfait** (pilote 20 dB, 5 points, TA-RB **T=4**).

Différences avec figD_ber_snr.py (conservée telle quelle, elle alimente la
figure actuelle du mémoire) :
  - lit les runs à **tirages appariés** (graine Sionna remise à zéro avant
    chaque méthode) et à budget Monte-Carlo élargi, c.-à-d. les fichiers
    results/diag_ber_umi_standard_seedfix_*.json et
    results/diag_ber_csi_imperfect_seedfix_*.json (le plus récent de chaque) ;
  - le plancher de mesure est calculé depuis le budget réel
    (1 / (num_batches x 4096 bits)) au lieu d'être fixé à 1e-6 en dur, et
    annoté comme tel : un BER rapporté à 0 signifie « sous ce plancher »,
    pas « nul » ;
  - le panneau gauche est étiqueté T=6 et le droit T=4, ce qui correspond aux
    checkpoints réellement évalués (figD_ber_snr.py annonce T=4 pour les deux).

Usage: python3 figD_ber_snr_seedfix.py   (depuis figures/umi_standard/, CPU)
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '../../results/'
# Bits d'information par tirage = batch_size x K_USERS x K_BITS (lu dans le
# code : LDPC5GEncoder(k=num_data_symbols=1152, n=2304) -> rate 1/2, QPSK,
# 4 utilisateurs). Soit 1 179 648 bits par tirage de 256, donc une résolution
# de 1.7e-8 à 50 tirages -- et non 4.9e-6 comme une première estimation le
# suggérait.
K_USERS, K_BITS = 4, 1152


def newest(pattern):
    files = sorted(glob.glob(DATA_DIR + pattern))
    if not files:
        raise SystemExit(f"aucun fichier {pattern} dans {DATA_DIR} -- lancer "
                          f"experiments/eval_ber_*.py d'abord")
    return files[-1]


COLORS = {'RZF': '#2a78d6', 'WMMSE': '#eb6834', 'SC': '#1baf7a', 'IB': '#eda100', 'TA-RB': '#e87ba4'}
MARKERS = {'RZF': 'o', 'WMMSE': 's', 'SC': '^', 'IB': 'D', 'TA-RB': 'v'}
LINESTYLE = {'RZF': '--', 'WMMSE': '--', 'SC': '-', 'IB': '-', 'TA-RB': '-'}
KEY_MAP = {'RZF': 'RZF', 'WMMSE': 'WMMSE', 'SC': 'SingleSC-signed_attn',
           'IB': 'IntraRB-signed_attn', 'TA-RB': 'TA_RB_residual-signed_attn'}


def labels(tok):
    return {'RZF': 'RZF', 'WMMSE': 'WMMSE',
            'SC': 'SC — Précodeur par sous-porteuse',
            'IB': 'IB — Transformer intra-bloc',
            'TA-RB': f'TA-RB — Agrégation par RB (résiduel, T={tok})'}


fp, fi = newest('diag_ber_umi_standard_seedfix_*.json'), newest('diag_ber_csi_imperfect_seedfix_*.json')
perfect, imperfect = json.load(open(fp)), json.load(open(fi))
snr_perfect = perfect['RZF']['snr']
snr_imperfect = sorted(float(s) for s in imperfect['RZF'].keys())
floor_p = 1.0 / (perfect['metadata']['num_batches'] * perfect['metadata']['batch_size'] * K_USERS * K_BITS)
floor_i = 1.0 / (imperfect['metadata']['num_batches'] * imperfect['metadata']['batch_size'] * K_USERS * K_BITS)


def plot_panel(ax, get_ber, snr, title, floor, tok):
    lab = labels(tok)
    for k in KEY_MAP:
        ber = np.maximum(np.array(get_ber(k), dtype=float), floor)
        ax.semilogy(snr, ber, color=COLORS[k], marker=MARKERS[k], linestyle=LINESTYLE[k],
                    label=lab[k], linewidth=2, markersize=6)
    ax.axhline(floor, color='#999999', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.text(snr[0], floor * 1.15, f'plancher de mesure {floor:.1e}', fontsize=7.5, color='#777777')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_title(title, fontsize=12.5, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8.5, loc='lower left')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
plot_panel(ax1, lambda k: perfect[KEY_MAP[k]]['ber'], snr_perfect,
           f"CSI parfait — {perfect['metadata']['num_batches']} tirages/point", floor_p, 6)
ax1.set_ylabel('BER (échelle log)', fontsize=12)
plot_panel(ax2, lambda k: [imperfect[KEY_MAP[k]][f'{s:.1f}'] for s in snr_imperfect], snr_imperfect,
           f"CSI imparfait, pilote 20 dB — {imperfect['metadata']['num_batches']} tirages/point", floor_i, 4)

fig.suptitle('BER vs SNR — UMi (M=8, K=4), tirages appariés', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figD_ber_snr_seedfix.png', dpi=200, bbox_inches='tight')
fig.savefig('figD_ber_snr_seedfix.pdf', bbox_inches='tight')
plt.close(fig)

out = {'sources': {'perfect': fp.split('/')[-1], 'imperfect': fi.split('/')[-1]},
       'measurement_floor': {'perfect': floor_p, 'imperfect_pilot20dB': floor_i},
       'perfect_T6': {k: {'snr': snr_perfect, 'ber': perfect[KEY_MAP[k]]['ber']} for k in KEY_MAP},
       'imperfect_pilot20dB_T4': {k: {'snr': snr_imperfect,
                                       'ber': [imperfect[KEY_MAP[k]][f'{s:.1f}'] for s in snr_imperfect]}
                                   for k in KEY_MAP}}
with open('figD_ber_snr_seedfix.json', 'w') as f:
    json.dump(out, f, indent=2)
print('✅ Figure D (seedfix) -> figD_ber_snr_seedfix.{png,pdf,json}')
