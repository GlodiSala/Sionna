"""
experiments/complexity_energy_per_T.py — FLOPs / paramètres / énergie INT8 du
TA-RB résiduel signed_attn en fonction du nombre de tokens par RB
(T = 1, 2, 3, 4, 6, 12), UMi standard (M=8, K=4, D=128, L=4).

Alimente la colonne énergie de l'ablation T (tab:ablation_T) et la figure
Pareto débit/énergie `figures/umi_standard/figB_pareto_energy_T*.py`.

Même modèle analytique que experiments/complexity_energy_standard.py :
complexity() des classes de précodeur + compute_energy_uJ de system.py,
INT8 (Q_W = Q_A = 8), convention FLOPs réels (convention_x_ofdm=False).
Les poids sont vérifiés exacts contre trainable_variables.

Écrit results/energy_per_T_signed_attn_postfix_<date>.json. Le fichier
historique results/energy_per_T_signed_attn.json a été produit AVANT le
correctif énergie du 03/09 (accès mémoire linéaire en nombre de bits,
correctif F. Leduc-Primeau) et est conservé tel quel pour traçabilité.

Usage: python3 experiments/complexity_energy_per_T.py   (CPU suffit)
"""
import os, sys, json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precoders.signed_attention import TransformerPrecoderCleanResidualSignedAttn
from system import compute_energy_uJ
from channel_config import STANDARD_CONFIG, FFT_SIZE

NUM_OFDM, D, L, H, Q = 14, 128, 4, 4, 8
T_VALUES = [1, 2, 3, 4, 6, 12]
M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']

h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)
out = {}
for T in T_VALUES:
    m = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12,
        tokens_per_rb=T, embed_dim=D, num_heads=H, num_layers=L)
    _ = m(h_dummy, no=tf.constant(1e-3))
    real_w = int(sum(tf.size(v).numpy() for v in m.trainable_variables))
    flops, weights, acts = m.complexity(NUM_OFDM, convention_x_ofdm=False)
    assert int(weights) == real_w, f"T={T}: poids {int(weights)} vs {real_w}"
    e = compute_energy_uJ(flops, weights, acts, Q, Q)
    out[str(T)] = {'flops_M': float(flops) / 1e6, 'params': real_w, 'energy_int8_uJ': e}
    print(f'T={T:2d}  flops={float(flops)/1e6:8.1f} M  params={real_w:,}  energie={e:.4f} uJ', flush=True)

dst = 'results/energy_per_T_signed_attn_postfix_20260912.json'
with open(dst, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSauve -> {dst}')
