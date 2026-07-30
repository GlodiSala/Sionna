"""
Locked channel configurations for training and final evaluation (Stage 3/4).
=============================================================================

TWO configs are locked in, same mechanism, different scale:

  STANDARD_CONFIG : M=8,  K=4  (50% loading) -- standard MIMO baseline
  MASSIVE_CONFIG  : M=64, K=32 (50% loading) -- genuine massive MIMO

Both use: UMi scenario, FFT_SIZE=96, 15 deg azimuth drop window, forced
LOS, indoor_probability=0. SNR: characterize/train across a range that
includes 5 and 10 dB (see below -- the gap is larger at 5dB, smaller but
still real at 10dB, for both configs).

WHY THIS CONFIG (validation trail -- re-run scripts still exist under
final/wmmse_convergence_check.py, final/step2_antenna_sweep.py,
final/step2b_m64k32.py, final/step2c_snr10_crossscale.py,
final/step3_massive_mimo_test.py, final/step3_followup.py if this needs
re-deriving):

## Part 1 -- establishing the mechanism (originally at M=64/K=36, FFT=72)

1. Non-orthogonality levers that Sionna actually exposes (investigated by
   reading tr38901 source): per-link angular/delay spread (ASA/ASD/DS) are
   drawn stochastically from 3GPP log-normal tables keyed only by LOS/NLOS
   state -- NOT directly settable. What IS controllable: (a) the physical
   azimuth window UTs are dropped into (custom, see gen_topology_custom in
   wmmse_convergence_check.py -- Sionna's own gen_single_sector_topology
   only offers a fixed 120 deg sector), (b) forced LOS/NLOS via
   set_topology(los=...), (c) indoor_probability, (d) K/M loading ratio,
   (e) BS antenna element spacing (untested, separate mutual-coupling
   mechanism). InH scenario does not exist in this Sionna version (only
   UMi/UMa/RMa) -- not an option. UMi vs UMa are structurally equivalent
   for our purposes; UMi kept for continuity. Counter-intuitively, forcing
   LOS matters because LOS uses fewer/tighter clusters (UMi: 12 clusters,
   3 deg spread) than NLOS (19 clusters, 10 deg spread) -- LOS is closer to
   a single dominant steering vector, so users need real physical angular
   separation to be resolved; NLOS's richer scattering actually
   decorrelates users even when physically close.

2. WMMSE convergence ruled out as an explanation for any RZF/WMMSE gap
   (paired same-channel comparison, cross-validated on GPU and CPU): the
   gap GROWS monotonically with more WMMSE iterations (10->30->50), the
   opposite of what under-convergence would predict. WMMSE stays at its
   documented 10-iteration default (precoders_w.py:wmmse_precoder) --
   running more iterations does not help under Sionna's reporting metric.

3. Lever isolation at M=64/K=36 (FFT=72 at the time), SNR=10dB, WMMSE@10
   iters, paired + reference-adjusted vs a M64K8_wide_nlos baseline:
     REF  K=8,  wide(120deg), NLOS  : +0.00%
     A    K=8,  narrow(15deg), LOS  : +0.01%  (angular/LOS lever ALONE: ~nothing)
     B    K=36, wide(120deg), LOS   : +0.33%  (loading lever ALONE: small)
     C    K=36, narrow(15deg), LOS  : +2.53% +/- 0.55  (BOTH combined: real,
                                                         ~8x superadditive)
   Narrowing further (8deg, 5deg) does NOT reliably grow the gap beyond
   ~15deg -- the angular lever saturates around 15deg. SNR=5 re-verified
   with a freshly matched reference (not assumed from SNR=10): REF stayed
   at +0.00%, C rose to +4.65% +/- 0.65 (~2x the SNR=10 value). Confirmed:
   BOTH high loading AND narrow angle+LOS are needed together; neither
   alone is sufficient, and the SNR=5 point shows a bigger gap than SNR=10.

## Part 2 -- Step 0: FFT_SIZE 72->96

Changed to a still-valid 5G NR grid (96 = 8 RBs x 12 SC, vs 72 = 6 RBs x
12 SC) as part of a follow-up investigation into less hand-tuned
non-orthogonality levers plus an M-sweep at realistic K=4/K=8. Fixed
hardcoded 72s in the active pipeline (wmmse_convergence_check.py,
compare_rb_grouping.py, main_finall.py, datasets.py/precoders_w.py/
precoder_intra_rb.py default args); left legacy/superseded scripts
(main.py, main_final.py, main_ee.py, eval_only.py, compare_models.py,
precoders_w1.py) and historical-record analysis scripts (plot_sumrate_
final.py, memory.py, validation.py, timing.py, complexity.py, run_
diagnostics.py, correlation.py, rb_correlation_analysis.py, test_intra_
rb.py) untouched. Verified via real forward pass at FFT=96 for K=4/K=8:
kronecker pilot pattern, pilot_ofdm_symbol_indices=[2,11], num_ofdm=14 all
work cleanly; both IntraRB and TransformerPrecoderV4 produce correct
shape, no NaN, exact per-user power normalization.

## Part 3 -- Step 1: spatial consistency and CDL investigated, both ruled out

Spatial consistency (TR38.901 sec 7.6.3, correlated LSP generation across
nearby users): confirmed REAL and unconditionally ACTIVE in Sionna's UMi/
UMa/RMa (lsp.py, using the corrDist* JSON table values, exp(-d/D) model,
triggered automatically by set_topology with real coordinates -- no
toggle exists, always on). BUT it only correlates the 7 large-scale
parameters (DS, ASD/ASA/ZSA/ZSD, SF, K-factor) -- scalar summary stats.
Cluster/ray-level geometry (the actual per-path AoA/AoD values that
determine steering-vector similarity, i.e. what actually matters for
precoding difficulty) is confirmed NOT spatially correlated (rays.py, no
corrDist/spatial reference anywhere). LoS/NLoS state itself is also drawn
i.i.d. per user, not correlated. Conclusion: this has been active in every
UMi run we've ever done and clearly wasn't sufficient on its own (Part 1's
lever isolation already proves narrow-angle-window is still necessary) --
not a usable substitute.

CDL (Clustered Delay Line): confirmed single-link only (num_rx/num_tx
hardcoded to 1 per call, cdl.py), no set_topology/multi-user concept.
Combining K CDL draws would require manually instantiating K separate CDL
objects with different fixed ut_orientation, but the underlying cluster/
ray geometry is independently re-randomized per instance regardless --
same fundamental ceiling as spatial consistency, but far more integration
cost (manual K-way instantiation, manual H-matrix assembly, incompatible
with the existing StreamManagement/ResourceGrid pipeline). Not worth it.

Correction for the record: CDL-D and CDL-E are the LOS-dominant CDL
profiles (not CDL-C/D as originally guessed when scoping this
investigation) -- moot since CDL wasn't used, noted only so it doesn't
get misremembered later.

Decision: proceed with the existing forced-LOS + narrow-angle-window
mechanism (Part 1) for the M-sweep, since neither alternative beats it.

## Part 4 -- Step 2: M-sweep at fixed K=4 and K=8, FFT=96, SNR=5dB

M in {8,16,32,64,128}, paired reference (wide/NLOS) vs hard (15deg/LOS) at
each M, same K. Ref-adjusted gap:
  K=4:  M=8: +3.99%   M=16: +1.15%  M=32: +0.16%  M=64: +0.02%  M=128: +0.00%
  K=8:  M=8: +6.41%*  M=16: +4.63%  M=32: +0.87%  M=64: +0.05%  M=128: +0.00%
  (*M=8/K=8 is K=M, 100% loading -- even the REFERENCE arm showed a large
  +12.79% gap here, meaning the reference itself picked up a genuine
  loading effect at this critical fully-loaded boundary, not just the
  known metric artifact it's supposed to control for. Treated as a noted
  anomaly, not chased further -- not needed once the loading-ratio pattern
  below was established from cleaner points.)

Finding: the gap vanishes by M=64-128 for fixed small K, exactly as
massive-MIMO favorable-propagation theory predicts. But the REAL pattern
is that ref-adjusted gap tracks the LOADING RATIO K/M, not M or K
individually: ~50% loading gives 3.99% (M8K4) and 4.63% (M16K8); ~25%
gives 0.16-0.87%; ~12.5% gives <0.2%; ~6% is ~0. This is consistent with
Part 1's M64/K36 finding (56% loading, 2.53-4.65% gap) -- loading ratio,
not absolute scale, is the driver.

## Part 5 -- confirming M=64/K=32 (50% loading, genuine massive-MIMO scale)

Tested M=64/K=32 (K=32 divides FFT=96 cleanly, 96/32=3) at SNR=5 and
SNR=10, same paired methodology:
  SNR=5  : REF +0.02%, HARD +4.08% +/- 0.28  -> ref-adjusted +4.06%
  SNR=10 : REF +0.00%, HARD +0.84% +/- 0.24  -> ref-adjusted +0.83%

At SNR=5, this lands right alongside M8K4 (3.99%) and M16K8 (4.63%) --
clean confirmation that loading ratio (not scale) governs the gap, holding
from standard MIMO up to a genuine 64-antenna array.

Cross-checked SNR=10 numbers for M8K4 (+2.93%) and M16K8 (+3.79%) too, to
see if the SNR-dependence itself is scale-invariant. IT IS NOT: at SNR=10,
M64K32's gap (0.83%) is noticeably smaller, proportionally, than M8K4's
(2.93%) or M16K8's (3.79%) -- i.e. the massive-MIMO case's gap decays
faster with increasing SNR than the smaller-scale cases do. Full table:

  Config        Loading   Gap @ SNR=5    Gap @ SNR=10
  M=8,  K=4     50%       +3.99%         +2.93%
  M=16, K=8     50%       +4.63%         +3.79%
  M=64, K=32    50%       +4.06%         +0.83%

This is an honest, reportable finding, not swept under the rug: matched
loading ratio gives a consistent gap at SNR=5 across 8x scale, but the
SNR-decay rate of that gap is itself scale-dependent (faster decay at
larger M) -- i.e. the two configs below are "same mechanism, same gap
size at low SNR" but NOT "identical behavior at every SNR." Training/eval
should therefore weight the 5-10dB region rather than treat a single SNR
point as representative of both scales.

## FINAL DECISION

STANDARD_CONFIG (M=8, K=4) and MASSIVE_CONFIG (M=64, K=32) locked in as
the Stage 3/4 training/eval channel configs. Same mechanism (forced LOS,
15deg window, indoor_probability=0 for hygiene), same 50% loading ratio,
different absolute scale -- the "standard MIMO vs massive MIMO, same
channel-hardening story" comparison. Further dials (SNR=0, BS antenna
sub-half-wavelength spacing, K=M full loading, spatial consistency, CDL)
were investigated where relevant and explicitly NOT pursued further --
diminishing returns / already ruled out.
"""

import numpy as np

# =============================================================================
# LOCKED CONFIGS
# =============================================================================

STANDARD_CONFIG = dict(
    NUM_TX=8, NUM_RX=4,            # M, K -- 50% loading, standard MIMO
    HALF_ANGLE_DEG=7.5, FORCE_LOS=True, INDOOR_PROBABILITY=0.0,
)

MASSIVE_CONFIG = dict(
    NUM_TX=64, NUM_RX=32,          # M, K -- 50% loading, genuine massive MIMO
    HALF_ANGLE_DEG=7.5, FORCE_LOS=True, INDOOR_PROBABILITY=0.0,
)

SCENARIO = 'umi'
FFT_SIZE = 96

# SNR range for training/eval -- must include the 5-10dB region where the
# validated gap is largest (already satisfied by main_finall.py's
# SNR_MIN_TRAIN=5/SNR_MAX_TRAIN=25 and EVALUATION_SNR_RANGE). Note the gap's
# SNR-decay rate is scale-dependent (Part 5) -- don't rely on a single SNR
# point to characterize both configs.
SNR_RANGE_DB = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], dtype=np.float32)


def gen_topology_locked(batch_size, num_ut, half_angle_deg=7.5,
                         indoor_probability=0.0, scenario=SCENARIO):
    """Reusable topology draw for the locked mechanism (forced LOS + narrow
    azimuth window). Thin wrapper around
    wmmse_convergence_check.gen_topology_custom (the validated
    implementation used throughout Steps 1-3) so training/dataset code
    shares one source of truth with the experiments that justified this
    config. Pass num_ut / half_angle_deg explicitly for whichever of
    STANDARD_CONFIG / MASSIVE_CONFIG (or a future config) is in use."""
    from wmmse_convergence_check import gen_topology_custom
    return gen_topology_custom(batch_size, num_ut, scenario,
                                half_angle_deg=half_angle_deg,
                                indoor_probability=indoor_probability)


def set_locked_topology(channel_model, batch_size, num_ut,
                         half_angle_deg=7.5, force_los=True,
                         indoor_probability=0.0, scenario=SCENARIO):
    """Draw a topology under the locked mechanism and apply it to
    channel_model (forces LOS via set_topology, since gen_topology_locked's
    output tuple carries no LOS info -- LOS is forced at set_topology
    time)."""
    topology = gen_topology_locked(batch_size, num_ut, half_angle_deg,
                                    indoor_probability, scenario)
    channel_model.set_topology(*topology, los=force_los)
