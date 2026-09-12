"""Smoke test: confirm IntraRBTransformerPrecoder and TransformerPrecoderV4
handle the locked M64K36_ang15_los config (M=64, K=36) with a real forward
pass on a real generated channel -- not just static shape reasoning."""
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from precoder_intra_rb import IntraRBTransformerPrecoder
from precoders_w import TransformerPrecoderV4
from sionna.phy.utils import ebnodb2no

M, K = 64, 36
BATCH = 8

system = ConfigurableMIMOSystem(M, K, half_angle_deg=7.5, los=True,
                                 indoor_probability=0.0)
system.new_topology(BATCH)
no = ebnodb2no(tf.constant(10.0), system.num_bits_per_symbol, 0.5, system.rg)
h_freq = system._gen_channel(BATCH)
print("h_freq shape:", h_freq.shape, h_freq.dtype)

print("\n--- IntraRBTransformerPrecoder ---")
intra_rb = IntraRBTransformerPrecoder(
    num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12,
    embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
g_intra = intra_rb(h_freq, no=no, training=False)
print("output shape:", g_intra.shape, g_intra.dtype)
assert g_intra.shape[-2:] == (M, K), f"expected last dims ({M},{K}), got {g_intra.shape[-2:]}"
print("IntraRB OK -- num params:", intra_rb.count_params())

print("\n--- TransformerPrecoderV4 (per-carrier / RB-token baseline) ---")
v4 = TransformerPrecoderV4(
    num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12,
    tokens_per_rb=1, embed_dim=128, num_heads=4, num_layers=4, version='v4.2')
g_v4 = v4(h_freq, no=no, training=False)
print("output shape:", g_v4.shape, g_v4.dtype)
assert g_v4.shape[-2:] == (M, K), f"expected last dims ({M},{K}), got {g_v4.shape[-2:]}"
print("TransformerPrecoderV4 OK -- num params:", v4.count_params())

print("\nBoth architectures handle M=64/K=36 without shape errors.")
