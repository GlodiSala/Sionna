# Signed-Attention Transformer Precoders for MU-MIMO OFDM Downlink

Sionna-based simulation and training code for a set of learned, permutation-
equivariant linear precoders for multi-user MIMO-OFDM downlink, evaluated
against RZF and WMMSE across four regimes: {UMi, UMa} channel x {standard
(M=8 antennas, K=4 users), massive (M=64, K=8)} scale.

## Architecture

Three precoder families, all sharing the same core idea — **SignedGateMHA**
(`precoders/signed_attention.py`): a drop-in replacement for standard softmax
multi-head attention that multiplies the softmax magnitude by a `tanh`
sign gate. Standard softmax attention can only produce a *convex combination*
of value vectors (weights ≥ 0, sum = 1); the optimal high-SNR precoder
approaches zero-forcing, which needs *signed* coefficients between users to
null interference. SignedGateMHA gives the network that degree of freedom
while keeping softmax's training stability, and stays exactly permutation-
equivariant (verified numerically: max abs. difference 0.0 under permutation
of the K users).

- **SC** (`SingleSCTransformerPrecoderSignedAttn`) — one user-attention pass
  per subcarrier, no frequency-domain structure exploited.
- **IB** (`IntraRBTransformerPrecoderSignedAttn`) — intra-resource-block:
  subcarrier-attention (unchanged, standard softmax) then user-attention
  (signed-gated), within each 12-subcarrier RB.
- **TA-RB** (`TransformerPrecoderCleanResidualSignedAttn`) — token-aggregated
  RB: each RB is pooled (mean/variance) into `T` tokens before user-attention,
  with a residual/interpolation decoder back to per-subcarrier precoding
  weights. `T` trades throughput against compute; **T=4 is the recommended
  operating point at standard scale** (best throughput/energy trade-off of
  the T-sweep — see `figures/umi_standard/figB_pareto_energy_T.png`), T=6
  was used at massive scale (no T-sweep run there).

Both classical baselines (`precoders/classical.py`) are the standard closed-
form RZF and iterative WMMSE (Shi et al. 2011-style WMMSE, `I=10` iterations).

## Repository layout

```
channel_config.py       Locked channel configs (STANDARD_CONFIG M=8/K=4,
                         MASSIVE_TRUE_CONFIG M=64/K=8), spatial user
                         clustering topology generation.
datasets.py              CachedSionnaDataset: on-disk cache of jointly-drawn
                         K-user channel samples (UMi or UMa), avoids
                         regenerating channels every training step.
system.py                MU_MIMO_System (full Sionna forward pass: LDPC
                         encode -> precode -> OFDM channel -> LMMSE equalize
                         -> LDPC decode) and SupervisedTrainer (two-phase:
                         MSE warmup against RZF, then direct sum-rate
                         finetuning).
eval_system.py            ConfigurableMIMOSystem — a lighter system used only
                         for the CSI-imperfect sweep and the classical
                         RZF/WMMSE reference (sum-rate only, no LDPC).
precoders/
  base_intra_rb.py        SingleSC / IntraRB base classes (standard softmax
                         attention) that the signed-attention variants
                         subclass.
  base_residual.py         TA-RB residual base class + shared blocks.
  signed_attention.py       SignedGateMHA and the three winning architectures
                         (see above) — the ones actually used for every
                         result in this repo.
  classical.py              RZF, WMMSE.
  rb_grouping.py            RZF with RB-level channel grouping (a reduced-
                         feedback classical reference point).
train.py                  Unified training + evaluation entrypoint (see
                         below).
classical_reference.py    RZF/WMMSE sum-rate reference for a given
                         channel x scale regime — run this first.
figures/                  Final figures, one subfolder per regime
                         (`umi_standard`, `umi_massive`, `uma_standard`,
                         `uma_massive`), each self-contained (its `fig*.py`
                         regenerates `fig*.{png,pdf,json}` from
                         `results/*.json`/`*.npy`).
results/                  Headline result files (JSON/NPY) the figures are
                         built from. NOT weights (see below).
```

## Usage

```bash
# 1. Classical reference (RZF/WMMSE), once per regime:
python3 classical_reference.py --channel umi --scale standard

# 2. Train + evaluate (CSI-perfect full SNR sweep, CSI-imperfect sweep at
#    pilot SNR=20dB) one architecture:
python3 train.py --arch ta_rb_residual --channel umi --scale standard

# Extended-budget run (used at massive scale, see note below):
python3 train.py --arch single_sc --channel umi --scale massive \
    --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget
```

Each `train.py` run writes `results/{channel}_{scale}_{arch}{tag}.json`
(throughput vs SNR, %WMMSE, CSI-imperfect retention, FLOPs/params) and a
checkpoint under `weights/` (not included in this repo — regenerate by
re-running, or ask for the pickled weights separately).

### Regenerating a figure

```bash
cd figures/umi_standard && python3 figA_sumrate_csi_perfect.py
```

Every `fig*.py` is self-contained: it reads only from `../../results/`, and
overwrites its own `.png`/`.pdf`/`.json` in place — no other figure or
regime's data is touched.

## Key findings (see the thesis for full discussion)

- **Training budget matters enormously at massive scale.** At M=64, SC/IB
  collapsed with SNR at the 83-epoch budget that was ample at M=8 (rate
  still climbing, gradient norm still increasing at the end of the budget
  — a clear convergence, not capacity, signal). Doubling to 170 epochs
  (10 warmup + 160 finetune) recovered +8 to +32 percentage points of
  %WMMSE depending on SNR. The same extended-budget test at standard scale
  gave only +0.6 to +6.0 points — 83 epochs was already sufficient there.
- **TA-RB trades peak CSI-perfect throughput for CSI-imperfect robustness,
  consistently across all four regimes tested.** It is not always the best
  architecture under perfect CSI (behind SC/IB on UMa in particular), but
  it reliably retains substantially more throughput under pilot-noise-
  degraded channel estimates (+8 to +18 percentage points of retention vs
  SC/IB at pilot SNR=20dB, every regime, no exception) — attributed to the
  per-RB mean/variance pooling in its feature extraction acting as an
  implicit denoiser.
- **Channel hardening confirmed at M=64** (RZF ≈ WMMSE, gap ≤0.01% both
  channels) — the expected massive-MIMO regime (Marzetta 2010), not an
  artifact of channel tuning: no artificial user clustering was used at
  massive scale (`MASSIVE_TRUE_CONFIG` reuses the same `CLUSTER_RADIUS_M`
  as standard scale).

## Notes on what is *not* in this repo

- Trained weights (large, regenerate via `train.py`).
- The exploratory/dead-end scripts from development (feature-engineering
  and bilinear-mixing variants that did not outperform SignedGateMHA,
  early T-sweeps run at insufficient budget, etc.) — only the final,
  validated architecture and results are kept here.
- BER curves are only produced at standard scale
  (`figures/umi_standard/figD_ber_snr.py`); at massive scale, channel
  hardening pushes the bit error rate low enough that the Monte-Carlo
  budget used elsewhere in this repo (a few hundred codewords per point)
  cannot resolve it — a genuinely larger budget would be needed, not
  attempted here.
