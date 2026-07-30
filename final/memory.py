# =============================================================================
# complexity_energy_analysis.py
#
# Complete FLOPs + Energy complexity analysis for your OFDM Massive MIMO system.
# Compares: ZF | RZF | WMMSE | TransformerPrecoderV3 (1,2,3 tok/RB) | Full-SC
# Across two scenarios: your current 8x4 and true massive MIMO 32x4
#
# OUTPUT: prints tables to console + saves complexity_energy_results.csv
#         in the SAME folder as this script.
#
# Lab conventions (matching ICC/PIMRC Table II):
#   FLOPs = real additions + real multiplications
#   1 complex multiply = 4 real FLOPs
#   1 complex addition = 2 real FLOPs
#   Transformer layers are real-valued → standard real FLOPs
#   Energy model: E_MAC = 0.857904*(Q/16)^1.9  (from your lab's Energy.py)
# =============================================================================

import os
import sys
import math
import numpy as np
import pandas as pd

# =============================================================================
# OUTPUT DIRECTORY — same folder as this script, always explicit
# =============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, "complexity_energy_results.csv")

# =============================================================================
# SECTION 1 — SYSTEM CONFIGURATIONS
# We study two scenarios:
#   A) Your current system  : M=8  BS antennas, K=4 users
#   B) True Massive MIMO    : M=32 BS antennas, K=4 users
# Both share the same OFDM grid and transformer hyperparameters.
# =============================================================================

CONFIGS = {
    "8x4 (Current)": dict(M=8,  K=4),
    "32x4 (Massive MIMO)": dict(M=32, K=4),
}

# OFDM / frame parameters  — same for both scenarios
N_SC    = 72      # fft_size  (total active subcarriers)
N_OFDM  = 14      # OFDM symbols per slot
N_RB    = 6       # Resource Blocks (N_SC / RB_SIZE)
RB_SIZE = 12      # subcarriers per RB

# TransformerPrecoderV3 architecture — same for both scenarios
D        = 128    # embedding dimension
N_HEADS  = 4
N_LAYERS = 4
FFN_EXP  = 4      # FFN: D -> 4D -> D

# WMMSE iterations
I_WMMSE = 10      # your code uses 10; lab paper uses 16 — we show both

# Bit-widths for energy model
Q_W = 32          # weight precision  (FP32)
Q_A = 32          # activation precision (FP32)

# =============================================================================
# SECTION 2 — LAB ENERGY MODEL  (identical to your Energy.py)
# =============================================================================

def energy_constants(Q):
    """
    Hardware energy constants from lab's Energy.py.
    Q   : bit-width (32=FP32, 16=FP16, 8=INT8)
    Returns EMAC, EM, EL  in femtojoules (fJ) when calibrated.
    Law: E_MAC = 0.857904 * (Q/16)^1.9
    """
    Q    = max(float(Q), 1e-5)
    EMAC = 0.857904 * (Q / 16) ** 1.9   # MAC energy
    EM   = 2.0 * EMAC                    # memory read/write
    EL   = EMAC                          # local register
    return EMAC, EM, EL


def compute_energy_fJ(FLOPs, Weights, Activations, Q_W=32, Q_A=32):
    """
    Compute total hardware energy in femtojoules using the lab energy model.

    Mapping from FLOPs to MACs:
        The lab's Energy.py uses MAC_operations internally.
        1 real MAC = 1 multiply + 1 accumulate = 2 real FLOPs.
        Therefore:  MACs = FLOPs / 2

    Energy breakdown (identical structure to lab's compute_energy()):
        EC  = E_MAC * (MACs + 3*Activations)          [compute]
        EW  = EM*Weights + EL*(MACs/sqrt(p_W))        [weight memory]
        EA  = 2*EM_A*Activations + EL_A*(MACs/sqrt(p_A)) [activation memory]
    """
    MACs = FLOPs / 2.0

    EMAC, EM,   EL   = energy_constants(Q_W)
    _,    EM_A, EL_A = energy_constants(Q_A)

    # Parallelism factors (from lab Energy.py)
    sqrt_p_W = math.sqrt(64.0 * (Q_W / 16.0))
    sqrt_p_A = math.sqrt(64.0 * (Q_A / 16.0))

    EC = EMAC * (MACs + 3.0 * Activations)
    EW = EM   * Weights + EL   * (MACs / sqrt_p_W)
    EA = 2.0  * EM_A * Activations + EL_A * (MACs / sqrt_p_A)

    return EC + EW + EA   # fJ


def fJ_to_uJ(e_fJ):
    return e_fJ / 1e9


# =============================================================================
# SECTION 3 — CLASSICAL PRECODER COMPLEXITY
#
# All formulas match your lab's Table II (ICC/PIMRC papers) exactly.
# The lab uses: 1 complex multiply = 4 real FLOPs (4 mults + not counting adds
# separately — the factor 7 in their ZF formula encodes both mults and adds).
#
# Lab ZF formula per channel use: 7*(2/3*K^3 + 2*K^2*M)
# where 7 = 4 real mults + 2 real adds + 1 (sign/normalisation) per cmplx op.
#
# MULTI-CARRIER ADAPTATION:
#   Classical methods run independently on each subcarrier and each OFDM symbol.
#   Total FLOPs = FLOPs_per_SC * N_SC * N_OFDM
# =============================================================================

def zf_flops_per_sc(K, M):
    """
    ZF per subcarrier, one OFDM symbol.
    Formula from lab Table II: 7 * (2/3*K^3 + 2*K^2*M)
    Operations:
      - Gram matrix  H H^†  [K×M][M×K] → [K×K]: dominant cost = 2*K^2*M complex ops
      - Cholesky solve of [K×K] system: 2/3*K^3 complex ops
      - Factor 7 converts complex op count → real FLOPs (4 mults + 2 adds + overhead)
    """
    return 7.0 * ((2.0/3.0) * K**3 + 2.0 * K**2 * M)


def rzf_flops_per_sc(K, M):
    """
    RZF = ZF + diagonal regularisation (K scalar additions, negligible).
    For correctness we add them explicitly.
    """
    return zf_flops_per_sc(K, M) + K


def wmmse_flops_per_sc(K, M, I):
    """
    WMMSE per subcarrier, one OFDM symbol, I iterations.
    Formula from lab Table II:
      I * (14/3*K*M^3 + 12*K^2*M^2 + 12*K^2*M + 9*K*M^2 + 8*K*M + 5*K^2 + 68/3*K)

    Derivation (per iteration, complex ops × 7):
      - Receive filter update U: K*(M^2 + M) complex ops → involves [M×M] inverse
      - MSE weight update W:     K ops
      - Precoder update V:       K*M^2 + K^2*M complex ops → [M×M] solve
      The lab's full expression accounts for all matrix-vector products inside
      the iterative loop. We use their exact formula verbatim.

    Note: your wmmse_precoder() uses I=10 by default; lab paper reports I=16.
    We parameterise with I so you can compare both.
    """
    per_iter = (
          (14.0/3.0) * K * M**3
        + 12.0 * K**2 * M**2
        + 12.0 * K**2 * M
        +  9.0 * K    * M**2
        +  8.0 * K    * M
        +  5.0 * K**2
        + (68.0/3.0)  * K
    )
    return 7.0 * I * per_iter


def classical_activations_per_sc(K, M):
    """
    Intermediate tensors stored per subcarrier for ZF/RZF/WMMSE.
    H: K×M complex → 2KM real values
    Gram G: K×K    → 2K^2
    Cholesky L: K×K → 2K^2
    Precoder W: M×K → 2MK
    """
    return 2.0 * (K*M + 2.0*K**2 + M*K)


def classical_total(flops_per_sc_fn, K, M, N_SC, N_OFDM, I=None):
    """Aggregate classical precoder over full OFDM slot."""
    if I is not None:
        f = flops_per_sc_fn(K, M, I)
    else:
        f = flops_per_sc_fn(K, M)
    a = classical_activations_per_sc(K, M)
    return f * N_SC * N_OFDM, 0.0, a * N_SC * N_OFDM


# =============================================================================
# SECTION 4 — TRANSFORMER PRECODER V3 COMPLEXITY
#
# Architecture (from precoders_w1.py):
#   Stage 1 — Coarse Transformer at token resolution (N_tok = N_RB * tok/RB)
#     a. Feature projection  Dense(feat_dim → D)
#     b. N_LAYERS × SeparableAttentionBlock
#     c. Coarse projection   Dense(D → M*2)
#   Stage 2 — SC-level refinement MLP (always at full N_SC resolution)
#     Dense(4 → 64, GELU) + Dense(64 → 2)
#     Applied per (SC, tx_antenna, user) position.
#
# All Transformer layers are REAL-valued.
# Real Dense(A→B) over N positions: FLOPs = 2*N*A*B  (N mults + N adds per output)
# Real Self-Attention over seq N, dim D: FLOPs = 8*N*D^2 + 4*N^2*D
#   Derivation:
#     QKV projections (×3):  3 * 2*N*D^2 = 6*N*D^2
#     Attention scores QK^T: 2*N^2*D
#     Weighted sum AV:       2*N^2*D
#     Output projection:     2*N*D^2
#     Total: 8*N*D^2 + 4*N^2*D   (softmax ~5N^2 neglected, consistent with lit.)
#
# Feature vector dimension per (token, user) — from precoders_w1.py call():
#   h_mean, h_first, h_last: each K×M complex → 2KM real → 3 features × 2KM = 6KM
#   covariance cov_matrix:   K×K complex        → 2K^2 real
#   Total feat_dim = 6*K*M + 2*K^2
# =============================================================================

def dense_flops(N, A, B):
    """FLOPs for real Dense(A→B) applied to N position vectors."""
    return 2.0 * N * A * B

def dense_weights(A, B):
    """Parameter count for Dense(A→B) including bias."""
    return A * B + B

def dense_activations(N, B):
    """Output activation storage."""
    return float(N * B)

def attn_flops(N, D):
    """
    FLOPs for one real self-attention block, sequence length N, embedding D.
    8*N*D^2 + 4*N^2*D
    """
    return 8.0 * N * D**2 + 4.0 * N**2 * D

def attn_weights(D):
    """Weight count: Q, K, V, O projections each D×D with bias."""
    return 4.0 * (D * D + D)

def attn_activations(N, D):
    """Activation footprint: Q,K,V tensors + attention matrix + output."""
    return 3.0*N*D + N*N + N*D

def separable_block_flops(N_tok, U, D):
    """
    FLOPs for ONE SeparableAttentionBlock.

    Freq-attention: U independent attention ops, each over N_tok tokens.
        → U * attn_flops(N_tok, D)

    User-attention: N_tok independent attention ops, each over U users.
        → N_tok * attn_flops(U, D)

    FFN: two dense layers D→4D→D, applied to all N_tok*U positions.
        → dense_flops(N_tok*U, D, 4D) + dense_flops(N_tok*U, 4D, D)
        = 2 * 2 * N_tok*U * D * 4D
        = 16 * N_tok * U * D^2

    Total simplifies to: 2*N_tok*U*D*(8D + N_tok + U)
    """
    freq = U     * attn_flops(N_tok, D)
    user = N_tok * attn_flops(U,     D)
    ffn  = 16.0  * N_tok * U * D**2
    return freq + user + ffn

def separable_block_weights(D):
    """Weight count for ONE SeparableAttentionBlock."""
    return (attn_weights(D)      # freq-attention
          + attn_weights(D)      # user-attention
          + dense_weights(D, 4*D) + dense_weights(4*D, D))  # FFN

def separable_block_activations(N_tok, U, D):
    """Activation footprint for ONE SeparableAttentionBlock."""
    freq = U     * attn_activations(N_tok, D)
    user = N_tok * attn_activations(U,     D)
    ffn  = N_tok * U * (4*D + D)   # hidden layer + output
    return freq + user + ffn


def transformer_v3_complexity(M, K, N_SC, N_OFDM, N_RB, RB_SIZE,
                               D, N_LAYERS, tokens_per_rb):
    """
    Full FLOPs / weights / activations for TransformerPrecoderV3
    over one complete OFDM slot (N_OFDM symbols × N_SC subcarriers).

    Stage 1 runs once per OFDM symbol → multiply by N_OFDM.
    Stage 2 runs once per OFDM symbol at full SC resolution → multiply by N_OFDM.
    Weights are shared (parameter count, not per-symbol).

    Returns: total_flops, total_weights, total_activations, breakdown_dict
    """
    N_tok = N_RB * tokens_per_rb
    U     = K
    sc_per_tok = RB_SIZE // tokens_per_rb

    # Feature dimension per (token, user) from precoders_w1.py
    feat_dim = 6 * K * M + 2 * K**2

    # ------------------------------------------------------------------
    # Stage 1a: Feature projection  Dense(feat_dim → D)
    # ------------------------------------------------------------------
    s1a_f = dense_flops(N_tok * U, feat_dim, D)
    s1a_w = dense_weights(feat_dim, D)
    s1a_a = dense_activations(N_tok * U, D)

    # ------------------------------------------------------------------
    # Stage 1b: N_LAYERS × SeparableAttentionBlock
    # ------------------------------------------------------------------
    s1b_f = N_LAYERS * separable_block_flops(N_tok, U, D)
    s1b_w = N_LAYERS * separable_block_weights(D)
    s1b_a = N_LAYERS * separable_block_activations(N_tok, U, D)

    # ------------------------------------------------------------------
    # Stage 1c: Coarse projection  Dense(D → M*2)
    # ------------------------------------------------------------------
    s1c_f = dense_flops(N_tok * U, D, M * 2)
    s1c_w = dense_weights(D, M * 2)
    s1c_a = dense_activations(N_tok * U, M * 2)

    stage1_f_per_sym = s1a_f + s1b_f + s1c_f
    stage1_w         = s1a_w + s1b_w + s1c_w
    stage1_a_per_sym = s1a_a + s1b_a + s1c_a

    # ------------------------------------------------------------------
    # Stage 2: SC-level refinement MLP
    # Dense(4→64,GELU) + Dense(64→2)
    # Applied to every (SC, tx_ant=M, user=U) position.
    # Input: [w_real, w_imag, h_real, h_imag] = 4 values per position.
    # ------------------------------------------------------------------
    n_pos_sc = N_SC * M * U

    s2_f = dense_flops(n_pos_sc, 4, 64) + dense_flops(n_pos_sc, 64, 2)
    s2_w = dense_weights(4, 64) + dense_weights(64, 2)
    s2_a = dense_activations(n_pos_sc, 64) + dense_activations(n_pos_sc, 2)

    # ------------------------------------------------------------------
    # Full slot totals
    # ------------------------------------------------------------------
    total_f = (stage1_f_per_sym + s2_f) * N_OFDM
    total_w =  stage1_w + s2_w               # weights independent of N_OFDM
    total_a = (stage1_a_per_sym + s2_a) * N_OFDM

    attn_f_slot = s1b_f * N_OFDM             # attention-only for scaling analysis

    breakdown = {
        'N_tok'         : N_tok,
        'sc_per_tok'    : sc_per_tok,
        'feat_dim'      : feat_dim,
        'stage1_f_slot' : stage1_f_per_sym * N_OFDM,
        'stage2_f_slot' : s2_f * N_OFDM,
        'attn_f_slot'   : attn_f_slot,
    }
    return total_f, total_w, total_a, breakdown


def transformer_fullsc_complexity(M, K, N_SC, N_OFDM, D, N_LAYERS):
    """
    Ablation: flat transformer where every subcarrier is its own token (N_tok = N_SC).
    No Stage 2 needed (already at full resolution).
    Simpler feature: real+imag of H only → feat_dim = 2*K*M.
    """
    N_tok    = N_SC
    U        = K
    feat_dim = 2 * K * M

    feat_f = dense_flops(N_tok * U, feat_dim, D)
    feat_w = dense_weights(feat_dim, D)
    feat_a = dense_activations(N_tok * U, D)

    attn_f = N_LAYERS * separable_block_flops(N_tok, U, D)
    attn_w = N_LAYERS * separable_block_weights(D)
    attn_a = N_LAYERS * separable_block_activations(N_tok, U, D)

    proj_f = dense_flops(N_tok * U, D, M * 2)
    proj_w = dense_weights(D, M * 2)
    proj_a = dense_activations(N_tok * U, M * 2)

    f_sym  = feat_f + attn_f + proj_f
    total_f = f_sym * N_OFDM
    total_w = feat_w + attn_w + proj_w
    total_a = (feat_a + attn_a + proj_a) * N_OFDM
    return total_f, total_w, total_a


# =============================================================================
# SECTION 5 — BUILD COMPARISON TABLES
# =============================================================================

def build_table(M, K, config_name,
                N_SC, N_OFDM, N_RB, RB_SIZE,
                D, N_LAYERS, Q_W, Q_A, I_WMMSE):
    """
    Compute all methods for a given (M, K) configuration.
    Returns a list of dicts ready for pandas.
    """
    rows = []

    def row(name, f, w, a, extra_n_tok=None, breakdown=None):
        e_fJ  = compute_energy_fJ(f, w, a, Q_W, Q_A)
        e_uJ  = fJ_to_uJ(e_fJ)
        return {
            'Config'         : config_name,
            'Method'         : name,
            'N_tokens'       : extra_n_tok if extra_n_tok is not None else N_SC,
            'FLOPs (M)'      : round(f / 1e6, 3),
            'GFLOPs'         : round(f / 1e9, 6),
            'Params (K)'     : round(w / 1e3, 1),
            'Energy (µJ)'    : round(e_uJ, 4),
            '_f'             : f,
            '_e_uJ'          : e_uJ,
            '_breakdown'     : breakdown,
        }

    # ------ Classical ------
    zf_f,    zf_w,    zf_a    = classical_total(zf_flops_per_sc,    K, M, N_SC, N_OFDM)
    rzf_f,   rzf_w,   rzf_a   = classical_total(rzf_flops_per_sc,   K, M, N_SC, N_OFDM)
    wmmse_f, wmmse_w, wmmse_a = classical_total(wmmse_flops_per_sc, K, M, N_SC, N_OFDM, I=I_WMMSE)

    rows.append(row('ZF',    zf_f,    zf_w,    zf_a))
    rows.append(row('RZF',   rzf_f,   rzf_w,   rzf_a))
    rows.append(row(f'WMMSE (I={I_WMMSE})', wmmse_f, wmmse_w, wmmse_a))

    # ------ Full-SC Transformer (ablation) ------
    fsc_f, fsc_w, fsc_a = transformer_fullsc_complexity(M, K, N_SC, N_OFDM, D, N_LAYERS)
    rows.append(row(f'Transformer Full-SC (N={N_SC})', fsc_f, fsc_w, fsc_a,
                    extra_n_tok=N_SC))

    # ------ TransformerV3 with 1, 2, 3 tok/RB ------
    for tpr in [1, 2, 3]:
        f, w, a, bd = transformer_v3_complexity(
            M, K, N_SC, N_OFDM, N_RB, RB_SIZE, D, N_LAYERS, tokens_per_rb=tpr
        )
        rows.append(row(
            f'TransformerV3  {tpr} tok/RB  (N={bd["N_tok"]})',
            f, w, a, extra_n_tok=bd['N_tok'], breakdown=bd
        ))

    # ------ Add ratio columns relative to WMMSE and ZF ------
    wmmse_f_val = wmmse_f
    wmmse_e_val = fJ_to_uJ(compute_energy_fJ(wmmse_f, wmmse_w, wmmse_a, Q_W, Q_A))
    zf_f_val    = zf_f
    zf_e_val    = fJ_to_uJ(compute_energy_fJ(zf_f, zf_w, zf_a, Q_W, Q_A))

    for r in rows:
        r['vs WMMSE FLOPs'] = f"{wmmse_f_val / r['_f']:.1f}×"
        r['vs WMMSE Energy']= f"{wmmse_e_val / r['_e_uJ']:.1f}×"
        r['vs ZF FLOPs']    = f"{r['_f'] / zf_f_val:.1f}×"

    return rows


def separator(char='=', width=140):
    print(char * width)


def print_table(df_display, title):
    separator()
    print(title)
    separator()
    # Drop internal columns
    cols = [c for c in df_display.columns if not c.startswith('_')]
    print(df_display[cols].to_string(index=False))


def print_stage_breakdown(rows_raw, config_name, zf_f_val):
    """Print Stage1/Stage2 breakdown for TransformerV3 rows."""
    separator('-', 100)
    print(f"  STAGE BREAKDOWN — {config_name}")
    print(f"  {'Method':<38} {'Stage1 (M)':>12} {'Stage2 (M)':>12} "
          f"{'Attn (M)':>12} {'S2/Total':>10} {'vs WMMSE FLOPs':>16}")
    separator('-', 100)
    for r in rows_raw:
        bd = r.get('_breakdown')
        if bd is None:
            continue
        s1  = bd['stage1_f_slot'] / 1e6
        s2  = bd['stage2_f_slot'] / 1e6
        att = bd['attn_f_slot']   / 1e6
        tot = s1 + s2
        print(f"  {r['Method']:<38} {s1:>12.2f} {s2:>12.2f} "
              f"{att:>12.2f} {s2/tot*100:>9.1f}%  {r['vs WMMSE FLOPs']:>15}")


def print_attention_scaling(K, D, N_SC, N_RB):
    """Show O(N²) attention savings from RB grouping."""
    separator('-', 100)
    print(f"  ATTENTION O(N²) SCALING  — U*(4*N²*D) + N*(4*U²*D)")
    print(f"  {'Config':<30} {'N_tokens':>10} {'N² FLOPs/block':>16} "
          f"{'vs Full-SC':>12} {'Reduction':>10}")
    separator('-', 100)
    U = K

    def n2_term(N):
        return U * (4 * N**2 * D) + N * (4 * U**2 * D)

    fsc_n2 = n2_term(N_SC)
    configs = [
        ('Full-SC', N_SC),
        ('V3  1 tok/RB', N_RB * 1),
        ('V3  2 tok/RB', N_RB * 2),
        ('V3  3 tok/RB', N_RB * 3),
    ]
    for label, n in configs:
        n2 = n2_term(n)
        ratio = fsc_n2 / n2
        print(f"  {label:<30} {n:>10} {n2:>16,.0f} {ratio:>11.1f}×  "
              f"{'(baseline)' if n == N_SC else f'{ratio:.0f}× fewer N² ops':>15}")


def print_thesis_summary(rows_raw, Q_W, Q_A):
    """Print the key ICC thesis argument."""
    separator()
    print("  THESIS SUMMARY — Why the Transformer wins despite higher FLOPs")
    separator('-', 100)

    wmmse_row = next(r for r in rows_raw if r['Method'].startswith('WMMSE'))
    zf_row    = next(r for r in rows_raw if r['Method'] == 'ZF')
    v3_rows   = [r for r in rows_raw if 'TransformerV3' in r['Method']]

    print(f"  ZF     : {zf_row['FLOPs (M)']:>10.3f} MFLOPs | "
          f"{zf_row['Energy (µJ)']:>10.4f} µJ | {N_SC*N_OFDM} sequential matrix inversions")
    print(f"  WMMSE  : {wmmse_row['FLOPs (M)']:>10.3f} MFLOPs | "
          f"{wmmse_row['Energy (µJ)']:>10.4f} µJ | {N_SC*N_OFDM} × I sequential solves")
    print()
    for r in v3_rows:
        bd = r.get('_breakdown')
        if bd is None:
            continue
        print(f"  {r['Method']:<42}: {r['FLOPs (M)']:>8.1f} MFLOPs | "
              f"{r['Energy (µJ)']:>10.4f} µJ | "
              f"{r['vs WMMSE FLOPs']:>6} cheaper than WMMSE | "
              f"1 parallel forward pass")

    # INT8 estimate for best V3
    best_v3 = v3_rows[0]
    int8_scale = (Q_W / 8.0) ** 1.9   # energy ratio FP32 / INT8
    int8_energy = best_v3['Energy (µJ)'] / int8_scale
    print()
    print(f"  INT8 estimate for {best_v3['Method']}:")
    print(f"    FP32 energy: {best_v3['Energy (µJ)']:.4f} µJ")
    print(f"    INT8 energy: {int8_energy:.4f} µJ  "
          f"(÷{int8_scale:.1f} from (32/8)^1.9 = {int8_scale:.1f})")
    print(f"    vs ZF energy: {int8_energy / zf_row['Energy (µJ)']:.1f}× more expensive")
    print()
    print(f"  Key insight: Classical methods have sequential data dependencies.")
    print(f"  The Transformer's higher FLOP count executes as ONE parallelisable")
    print(f"  forward pass → latency advantage on GPU/NPU even at higher FLOPs.")


# =============================================================================
# SECTION 6 — MAIN
# =============================================================================

def main():
    print()
    print("=" * 80)
    print("  MIMO PRECODING COMPLEXITY & ENERGY ANALYSIS")
    print(f"  Script: {__file__}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print("=" * 80)
    print(f"  Energy model: E_MAC = 0.857904*(Q/16)^1.9  |  Q_W={Q_W}, Q_A={Q_A}")
    print(f"  FLOPs convention: real adds + real mults  (1 cplx mult = 4 FLOPs)")
    print(f"  WMMSE iterations: I = {I_WMMSE}")
    print(f"  Transformer: D={D}, {N_LAYERS} layers, {N_HEADS} heads, {N_OFDM} OFDM sym/slot")
    print()

    all_rows_raw = []
    all_rows_display = []

    for config_name, cfg in CONFIGS.items():
        M_cfg = cfg['M']
        K_cfg = cfg['K']

        rows_raw = build_table(
            M=M_cfg, K=K_cfg, config_name=config_name,
            N_SC=N_SC, N_OFDM=N_OFDM, N_RB=N_RB, RB_SIZE=RB_SIZE,
            D=D, N_LAYERS=N_LAYERS, Q_W=Q_W, Q_A=Q_A, I_WMMSE=I_WMMSE
        )

        df = pd.DataFrame(rows_raw)
        all_rows_raw.extend(rows_raw)
        all_rows_display.append(df)

        # ------ Main complexity table ------
        print_table(df, f"  CONFIG: {config_name}  |  M={M_cfg} BS antennas, K={K_cfg} users")
        print()

        # ------ Stage breakdown ------
        zf_f_val = next(r['_f'] for r in rows_raw if r['Method'] == 'ZF')
        print_stage_breakdown(rows_raw, config_name, zf_f_val)
        print()

        # ------ Attention scaling ------
        print_attention_scaling(K_cfg, D, N_SC, N_RB)
        print()

        # ------ Thesis summary ------
        print_thesis_summary(rows_raw, Q_W, Q_A)
        print()

    # ------ Save CSV ------
    # Combine both configs, drop internal columns
    df_all = pd.concat(all_rows_display, ignore_index=True)
    df_save = df_all[[c for c in df_all.columns if not c.startswith('_')]]
    df_save.to_csv(OUTPUT_CSV, index=False)

    separator()
    print(f"  CSV saved to: {OUTPUT_CSV}")
    separator()


if __name__ == '__main__':
    main()