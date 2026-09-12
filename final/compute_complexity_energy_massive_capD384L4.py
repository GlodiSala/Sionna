"""
compute_complexity_energy_massive_capD384L4.py — recalcul énergie/
complexité pour les 3 architectures signed_attn réentraînées à
D=384,L=4 sur MASSIVE_TRUE (M=64,K=8), suite à l'investigation
"plafond 73-80% RZF" du 12/08. Même méthodologie EXACTE que
compute_complexity_energy_signed_attn.py (RZF/WMMSE en FP32 référence
exacte, les 3 archis neuronales en INT8) -- seul D change (128->384),
RZF/WMMSE inchangés (indépendants de D). CPU suffit.

Usage: python3 compute_complexity_energy_massive_capD384L4.py
"""
import os, sys, json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from main_finall import compute_energy_uJ
from channel_config import MASSIVE_TRUE_CONFIG, FFT_SIZE

NUM_OFDM = 14
D, L, H = 384, 4, 4
TOKENS_PER_RB = 6
Q_NEURAL = 8

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']

# RZF/WMMSE inchangés (indépendants de D) -- réutilisés depuis la référence D=128
_ref = json.load(open('results/complexity_energy_massive_true.json'))

h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)
out = {'RZF': _ref['RZF'], 'WMMSE': _ref['WMMSE']}

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
