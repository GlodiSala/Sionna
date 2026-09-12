"""
complexity_check.py — Comparaison FLOPs/Params/Énergie V4 vs V4.1 vs V5
+ breakdown par composant pour identifier les bottlenecks
"""

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from precoders_w import TransformerPrecoderV4, TransformerPrecoderV5

# =============================================================================
# MODÈLE D'ÉNERGIE
# =============================================================================

def energy_constants(Q):
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9
    EM   = 2.0 * EMAC
    EL   = EMAC
    return EMAC, EM, EL

def compute_energy_uJ(FLOPs, Weights, Activations, Q_W=16, Q_A=16):
    import math
    MACs = FLOPs / 2.0
    EMAC, EM,   EL   = energy_constants(Q_W)
    _,    EM_A, EL_A = energy_constants(Q_A)
    sqrt_p_W = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A = math.sqrt(64.0 * (Q_A / 16.0))
    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)
    return (EC + EW + EA) / 1e9


# =============================================================================
# BREAKDOWN ANALYTIQUE
# =============================================================================

def breakdown_v4(num_tx=8, num_rx=4, fft_size=72, num_ofdm=14,
                 rb_size=12, tokens_per_rb=1, embed_dim=128,
                 num_heads=4, num_layers=4,
                 use_learned_upsample=False):
    M  = num_tx;  K  = num_rx
    T  = (fft_size // rb_size) * tokens_per_rb
    D  = embed_dim
    fd = 5 * M + 3 * K
    N  = fft_size
    S  = rb_size // tokens_per_rb   # sc_per_token
    sc_out = M * K * 2

    def df(n,i,o): return 2.0*n*i*o
    def dp(i,o):   return i*o+o
    def mf(s,d):   return 8.0*s*d**2 + 4.0*s**2*d
    def mp(d):     return 4*(d*d+d)
    def ff(n,d):   return df(n,d,4*d)+df(n,4*d,d)
    def fp(d):     return dp(d,4*d)+dp(4*d,d)

    components = {}

    # Input projection
    components['input_proj'] = (df(T*K, fd, D) * num_ofdm, dp(fd, D))

    # Transformer blocks
    f_blk = (K*mf(T,D) + T*mf(K,D) + ff(T*K,D)) * num_ofdm
    w_blk = 2*mp(D) + fp(D)
    components[f'transformer_blocks (×{num_layers})'] = (
        f_blk * num_layers, w_blk * num_layers)

    # Joint output proj
    components['joint_output_proj'] = (
        df(T, K*D, K*M*2) * num_ofdm, dp(K*D, K*M*2))

    if use_learned_upsample:
        # V4.1 — Conv1DTranspose upsample
        f_up = 2.0 * N * S * sc_out * sc_out
        w_up = S * sc_out * sc_out + sc_out
        components['Conv1DTranspose_upsample'] = (f_up * num_ofdm, w_up)

        # V4.1 — Conv1D sc_refine (k=rb_size, input=w_up seulement)
        f_rf = 2.0 * N * rb_size * sc_out * sc_out
        w_rf = rb_size * sc_out * sc_out + sc_out
        components[f'conv1d_sc_refine (k={rb_size}, w_up_only)'] = (
            f_rf * num_ofdm, w_rf)
    else:
        # V4.0 — tf.repeat (pas de FLOPs)
        # Conv1D sc_refine (k=3 + k=1, input=w_up+h_ri → ch=tx*rx*4)
        sc_ch = M * K * 4
        f_rf  = (2.0*N*3*sc_ch*sc_ch + 2.0*N*1*sc_ch*sc_out) * num_ofdm
        w_rf  = 3*sc_ch*sc_ch + sc_ch + 1*sc_ch*sc_out + sc_out
        components['conv1d_sc_refine (k=3+1, w_up+h_ri)'] = (f_rf, w_rf)

    return components


def breakdown_v5(num_tx=8, num_rx=4, fft_size=72, num_ofdm=14,
                 embed_dim=128, num_heads=4,
                 num_intra_layers=2, num_inter_layers=2):
    M  = num_tx;  K  = num_rx
    D  = embed_dim
    fd = 2*M + 2*K + K
    N  = fft_size
    NR = fft_size // 12
    RS = 12

    def df(n,i,o): return 2.0*n*i*o
    def dp(i,o):   return i*o+o
    def mf(s,d):   return 8.0*s*d**2 + 4.0*s**2*d
    def mp(d):     return 4*(d*d+d)
    def ff(n,d):   return df(n,d,4*d)+df(n,4*d,d)
    def fp(d):     return dp(d,4*d)+dp(4*d,d)

    components = {}
    components['input_proj']  = (df(N*K, fd, D) * num_ofdm, dp(fd, D))
    components['rb_pos_bias'] = (0.0, NR * RS * D)

    f_intra = (K*mf(RS,D) + RS*mf(K,D) + ff(RS*K,D)) * NR * num_ofdm
    w_intra = 2*mp(D) + fp(D)
    components[f'intra_blocks (×{num_intra_layers}, shared_w)'] = (
        f_intra * num_intra_layers, w_intra * num_intra_layers)

    components['rb_pooling'] = (N*K*D * num_ofdm, 0.0)

    f_inter = (K*mf(NR,D) + NR*mf(K,D) + ff(NR*K,D)) * num_ofdm
    w_inter = 2*mp(D) + fp(D)
    components[f'inter_blocks (×{num_inter_layers})'] = (
        f_inter * num_inter_layers, w_inter * num_inter_layers)

    components['cross_attn_SC_RB'] = (
        NR*K*(8.0*D**2 + 4.0*RS*D) * num_ofdm, mp(D))

    components['joint_output'] = (
        df(N, K*D, K*M*2) * num_ofdm, dp(K*D, K*M*2))

    return components


# =============================================================================
# CONFIGS
# =============================================================================

CONFIGS = [
    # ── V4.0 — baseline tf.repeat ─────────────────────────────────────────
    {'name': 'V4.0 (1 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=1, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=False),
     'bkd': lambda: breakdown_v4(tokens_per_rb=1, use_learned_upsample=False)},

    {'name': 'V4.0 (2 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=2, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=False),
     'bkd': lambda: breakdown_v4(tokens_per_rb=2, use_learned_upsample=False)},

    {'name': 'V4.0 (3 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=False),
     'bkd': lambda: breakdown_v4(tokens_per_rb=3, use_learned_upsample=False)},

    {'name': 'V4.0 (4 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=4, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=False),
     'bkd': lambda: breakdown_v4(tokens_per_rb=4, use_learned_upsample=False)},

    {'name': 'V4.0 (6 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=6, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=False),
     'bkd': lambda: breakdown_v4(tokens_per_rb=6, use_learned_upsample=False)},

    # ── V4.1 — Conv1DTranspose + Conv1D(k=rb_size, w_up_only) ─────────────
    {'name': 'V4.1 (1 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=1, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=True),
     'bkd': lambda: breakdown_v4(tokens_per_rb=1, use_learned_upsample=True)},

    {'name': 'V4.1 (3 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=True),
     'bkd': lambda: breakdown_v4(tokens_per_rb=3, use_learned_upsample=True)},

    {'name': 'V4.1 (6 tok/RB)', 'arch': 'v4',
     'model': lambda: TransformerPrecoderV4(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         rb_size=12, tokens_per_rb=6, embed_dim=128, num_heads=4, num_layers=4,
         use_learned_upsample=True),
     'bkd': lambda: breakdown_v4(tokens_per_rb=6, use_learned_upsample=True)},

    # ── V5 128d ───────────────────────────────────────────────────────────
    {'name': 'V5 128d (1L+1L)', 'arch': 'v5',
     'model': lambda: TransformerPrecoderV5(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         embed_dim=128, num_heads=4,
         num_intra_layers=1, num_inter_layers=1),
     'bkd': lambda: breakdown_v5(embed_dim=128,
                                  num_intra_layers=1, num_inter_layers=1)},

    {'name': 'V5 128d (2L+1L)', 'arch': 'v5',
     'model': lambda: TransformerPrecoderV5(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         embed_dim=128, num_heads=4,
         num_intra_layers=2, num_inter_layers=1),
     'bkd': lambda: breakdown_v5(embed_dim=128,
                                  num_intra_layers=2, num_inter_layers=1)},

    {'name': 'V5 128d (2L+2L)', 'arch': 'v5',
     'model': lambda: TransformerPrecoderV5(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         embed_dim=128, num_heads=4,
         num_intra_layers=2, num_inter_layers=2),
     'bkd': lambda: breakdown_v5(embed_dim=128,
                                  num_intra_layers=2, num_inter_layers=2)},

    # ── V5 256d ───────────────────────────────────────────────────────────
    {'name': 'V5 256d (2L+1L)', 'arch': 'v5',
     'model': lambda: TransformerPrecoderV5(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         embed_dim=256, num_heads=8,
         num_intra_layers=2, num_inter_layers=1),
     'bkd': lambda: breakdown_v5(embed_dim=256, num_heads=8,
                                  num_intra_layers=2, num_inter_layers=1)},

    {'name': 'V5 256d (2L+2L)', 'arch': 'v5',
     'model': lambda: TransformerPrecoderV5(
         num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
         embed_dim=256, num_heads=8,
         num_intra_layers=2, num_inter_layers=2),
     'bkd': lambda: breakdown_v5(embed_dim=256, num_heads=8,
                                  num_intra_layers=2, num_inter_layers=2)},
]


def wmmse_complexity(M=8, K=4, N_SC=72, N_OFDM=14, I=10):
    per_iter = (
          (14.0/3.0)*K*M**3 + 12.0*K**2*M**2 + 12.0*K**2*M
        +  9.0*K*M**2 + 8.0*K*M + 5.0*K**2 + (68.0/3.0)*K)
    f    = 7.0 * I * per_iter * N_SC * N_OFDM
    acts = 2.0 * (K*M + 2.0*K**2 + M*K) * N_SC * N_OFDM
    return f, 0.0, acts


# =============================================================================
# MAIN
# =============================================================================

def main():
    dummy_h = tf.zeros([1, 4, 1, 1, 8, 14, 72], dtype=tf.complex64)

    print("\n" + "="*92)
    print("  COMPLEXITY & ENERGY ANALYSIS — V4.0 vs V4.1 vs V5  (complet)")
    print("  8×4 MIMO | 72 SC | 14 OFDM | FP16 inference")
    print("="*92)

    results = []

    for cfg in CONFIGS:
        model = cfg['model']()
        try:
            _ = model(dummy_h, training=False)
        except Exception as e:
            print(f"  ⚠️  Build error for {cfg['name']}: {e}")
            continue

        flops, weights, acts = model.complexity(num_ofdm=14)
        real_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
        energy_fp16 = compute_energy_uJ(flops, weights, acts, Q_W=16, Q_A=16)
        energy_fp32 = compute_energy_uJ(flops, weights, acts, Q_W=32, Q_A=32)
        energy_int8 = compute_energy_uJ(flops, weights, acts, Q_W=8,  Q_A=8)

        results.append({
            'name'       : cfg['name'],
            'arch'       : cfg['arch'],
            'bkd_fn'     : cfg['bkd'],
            'flops_M'    : flops   / 1e6,
            'params_K'   : weights / 1e3,
            'real_params': real_params,
            'energy_fp32': energy_fp32,
            'energy_fp16': energy_fp16,
            'energy_int8': energy_int8,
        })

    # WMMSE
    f_w, w_w, a_w = wmmse_complexity()
    e_w = compute_energy_uJ(f_w, w_w, a_w, Q_W=32, Q_A=32)
    wmmse_flops  = f_w / 1e6
    wmmse_energy = e_w

    # ── Tableau principal ──────────────────────────────────────────────────
    print(f"\n{'Method':<26} {'FLOPs(M)':>10} {'Params(K)':>10} "
          f"{'Real(K)':>8} {'FP32(µJ)':>10} {'FP16(µJ)':>10} "
          f"{'INT8(µJ)':>10} {'Match':>6}")
    print("─" * 92)
    print(f"  {'WMMSE (ref, FP32)':<24} {wmmse_flops:>10.1f} {'—':>10} "
          f"{'—':>8} {e_w:>10.4f} {'N/A':>10} {'N/A':>10} {'—':>6}")
    print("─" * 92)

    for r in results:
        match = abs(r['params_K'] - r['real_params']/1e3) < 5.0
        flag  = "✅" if match else "❌"
        print(f"  {r['name']:<24} {r['flops_M']:>10.1f} "
              f"{r['params_K']:>10.1f} {r['real_params']/1e3:>8.1f} "
              f"{r['energy_fp32']:>10.4f} {r['energy_fp16']:>10.4f} "
              f"{r['energy_int8']:>10.4f} {flag:>6}")

    # ── Pareto vs WMMSE FP32 ───────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  PARETO vs WMMSE FP32")
    print(f"  Gain énergie FP16 = WMMSE_FP32 / model_FP16")
    print(f"  Gain énergie INT8 = WMMSE_FP32 / model_INT8")
    print(f"{'='*75}")
    print(f"{'Method':<26} {'FLOPs÷':>8} {'Gain FP16':>11} "
          f"{'Gain INT8':>11} {'FP16(µJ)':>10} {'INT8(µJ)':>10}")
    print("─" * 75)

    for r in results:
        fr  = wmmse_flops  / r['flops_M']
        e16 = wmmse_energy / r['energy_fp16']
        e8  = wmmse_energy / r['energy_int8']
        print(f"  {r['name']:<24} {fr:>7.1f}× {e16:>10.1f}× "
              f"{e8:>10.1f}× {r['energy_fp16']:>10.4f} "
              f"{r['energy_int8']:>10.4f}")

    # ── Breakdown par composant ────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  BREAKDOWN PAR COMPOSANT")
    print(f"{'='*92}")

    for r in results:
        bkd     = r['bkd_fn']()
        total_f = sum(v[0] for v in bkd.values())
        total_w = sum(v[1] for v in bkd.values())

        print(f"\n  ── {r['name']}  "
              f"({r['flops_M']:.0f} MFLOPs | "
              f"{r['params_K']:.0f} K params | "
              f"FP16={r['energy_fp16']:.4f} µJ | "
              f"INT8={r['energy_int8']:.4f} µJ)")
        print(f"  {'Component':<44} {'FLOPs(M)':>10} {'%FLOPs':>8} "
              f"{'Params(K)':>10} {'%Params':>8}")
        print(f"  {'─'*82}")

        for comp, (f, w) in sorted(bkd.items(),
                                    key=lambda x: x[1][0], reverse=True):
            pf   = 100.0 * f / total_f if total_f > 0 else 0
            pw   = 100.0 * w / total_w if total_w > 0 else 0
            flag = " ⚠️" if pf > 40 else ""
            print(f"  {comp:<44} {f/1e6:>10.1f} {pf:>7.1f}% "
                  f"{w/1e3:>10.1f} {pw:>7.1f}%{flag}")

    print()


if __name__ == "__main__":
    main()