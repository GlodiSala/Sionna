"""
experiments/complexity_energy_massive_d384.py — recalcul énergie/
complexité pour les 3 architectures signed_attn réentraînées à
D=384,L=4 sur MASSIVE_TRUE (M=64,K=8), suite à l'investigation
"plafond 73-80% RZF" du 12/08. Même méthodologie EXACTE que
experiments/complexity_energy_standard.py (RZF/WMMSE en FP32 référence
exacte, les 3 archis neuronales en INT8) -- seul D change (128->384),
RZF/WMMSE inchangés (indépendants de D). CPU suffit.

Usage: python3 experiments/complexity_energy_massive_d384.py
"""
import os, sys, json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet

from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from system import compute_energy_uJ, compute_classical_complexity
from channel_config import MASSIVE_TRUE_CONFIG, FFT_SIZE

NUM_OFDM = 14
D, L, H = 384, 4, 4
TOKENS_PER_RB = 6
Q_NEURAL = 8
Q_CLASSICAL = 32           # FP32, RZF/WMMSE -- référence exacte

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']

# RZF/WMMSE : RECALCULÉS ici avec le même modèle d'énergie que les archis
# neuronales (compute_energy_uJ de system.py). La version précédente de ce
# script les relisait depuis results/complexity_energy_massive_true.json,
# fichier produit AVANT le correctif énergie du 03/09 (accès mémoire
# linéaire en nombre de bits, correctif F. Leduc-Primeau) -- ce qui
# mélangeait classiques pré-correctif et neuronaux post-correctif dans le
# même tableau, et faussait les rapports d'énergie. Ils restent
# indépendants de D.
h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)
out = {}
for _name, _m in (('RZF', 'RZF'), ('WMMSE', 'WMMSE')):
    _kw = {'I_wmmse': 10} if _name == 'WMMSE' else {}
    _f, _w, _a = compute_classical_complexity(_m, M, K, FFT_SIZE, NUM_OFDM, **_kw)
    out[_name] = {'flops_M': float(_f) / 1e6,
                  'energy_uj': compute_energy_uJ(_f, _w, _a, Q_CLASSICAL, Q_CLASSICAL)}
    print(f'{_name:16s} FP32 flops_M={float(_f)/1e6:.1f} energy_uJ={out[_name]["energy_uj"]:.4f}', flush=True)

models = {
    'SingleSC': SingleSCTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, embed_dim=D, num_heads=H, num_layers=L),
    'IntraRB': IntraRBTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12, embed_dim=D, num_heads=H, num_layers=L),
    'TA_RB_residual': TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=NUM_OFDM, fft_size=FFT_SIZE, rb_size=12,
        tokens_per_rb=TOKENS_PER_RB, embed_dim=D, num_heads=H, num_layers=L),
}

for name, m in models.items():
    _ = m(h_dummy, no=tf.constant(1e-3))
    real_w = int(sum(tf.size(v).numpy() for v in m.trainable_variables))
    flops, weights, acts = m.complexity(NUM_OFDM, convention_x_ofdm=False)
    assert int(weights) == real_w, f"{name}: weights mismatch {int(weights)} vs {real_w}"
    e = compute_energy_uJ(flops, weights, acts, Q_NEURAL, Q_NEURAL)
    out[name] = {'flops_M': float(flops) / 1e6, 'params': real_w, 'energy_uj_int8': e}
    print(f'{name:16s} D={D} params={real_w:,} flops_M={float(flops)/1e6:.1f} energy_uJ={e:.4f}', flush=True)

with open('results/complexity_energy_massive_capD384L4.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\n✅ Sauvé -> results/complexity_energy_massive_capD384L4.json')
