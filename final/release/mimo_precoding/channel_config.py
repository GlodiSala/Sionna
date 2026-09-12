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

## FINAL DECISION (superseded -- see "REVISION 2026-08-07" below)

STANDARD_CONFIG (M=8, K=4) and MASSIVE_CONFIG (M=64, K=32) locked in as
the Stage 3/4 training/eval channel configs. Same mechanism (forced LOS,
15deg window, indoor_probability=0 for hygiene), same 50% loading ratio,
different absolute scale -- the "standard MIMO vs massive MIMO, same
channel-hardening story" comparison. Further dials (SNR=0, BS antenna
sub-half-wavelength spacing, K=M full loading, spatial consistency, CDL)
were investigated where relevant and explicitly NOT pursued further --
diminishing returns / already ruled out.

## REVISION 2026-08-07 -- forced-LOS replaced with forced-NLOS

The forced-LOS + 7.5deg window channel above was found (empirically, not
assumed) to be quasi frequency-flat: |rho(lag)| stayed above 0.80 across
the ENTIRE 96-SC band (delay-domain check: >99% of power in a single tap).
This defeats the premise of every frequency-domain architecture in this
thesis (RB grouping, T-token compression, intra-RB windows) -- there is no
frequency structure left to exploit, so window-size / token-count sweeps
on that channel are non-discriminant by construction (verified: an RB-size
sweep 4/6/12/24/48 SC showed no consistent trend on the forced-LOS channel).

Switched FORCE_LOS -> forced NLOS (los=False, was True), HALF_ANGLE_DEG
and INDOOR_PROBABILITY unchanged. Measured on both scales (paired RZF/WMMSE,
same methodology as classical_comparison.py, WMMSE already bisection-fixed):

  Config          rho@48sc  rho@95sc  RZF-WMMSE gap (max, always at 0dB)
  M=8,  K=4       0.606     0.458     +0.62%  (1/5 SNR points >0.3%)
  M=64, K=32      0.571     0.413     +0.87%  (1/5 SNR points >0.3%)

Frequency selectivity is now real and consistent across both scales
(rho@95sc ~0.41-0.46, vs >0.80 before). The RZF-WMMSE gap shrank a lot
(was +9.08% at M8K4 under forced LOS) and now only shows up at the 0dB
point -- tested wider angles (15/30/60deg) and natural/stochastic LOS as
alternatives, all gave an equal or smaller gap with equal or worse
selectivity, so narrow-7.5deg+NLOS is the best point on this tradeoff
found so far. A K=M (100% loading) variant gave a much bigger, wider-SNR
gap (+35.8% at M8K4, 5/5 points) but was rejected: K=M contradicts the
M>>K massive-MIMO definition this thesis uses, independent of the
resulting gap size. A modest gap concentrated at 0dB is accepted as
sufficient -- the priority is a real, measured frequency-selective
channel (needed to justify the frequency-domain architectures existing
at all), not gap magnitude.

## REVISION 2026-08-07 (b) -- narrow-angle-window replaced with spatial
## user clustering; MASSIVE_CONFIG rescaled from M64K32 to M32K8

The REVISION (a) NLOS channel above (narrow7.5deg+NLOS) was re-measured
with the same rigor as classical_comparison.py (10 batches instead of
6-8) and the "modest gap at 0dB" turned out to be mostly noise: M8K4 gap
dropped to +0.34% at 0dB and ~0 or negative everywhere else; M64K32 gap
dropped to +0.03% at 0dB, ~0 elsewhere. Root cause identified by
measuring actual UT geometry: even the narrow 7.5deg *global* sector
window still lets users land far apart from EACH OTHER (mean pairwise
angular separation 39.6deg under the plain 3GPP topology, confirmed by
direct measurement) -- narrowing the sector's angle from the BS does not
by itself force users close to each other, and under NLOS each user's
channel is dominated by its own independent multipath draw (spatial
consistency only correlates 7 scalar LSPs, not cluster/ray geometry --
see Part 3 above), so global angular proximity from the BS's viewpoint
stopped translating into channel-vector correlation the way it did
under the old forced-LOS mechanism (where channel ~= a single steering
vector, so angle-from-BS proximity WAS user-to-user proximity).

Fix: stopped narrowing the sector window (now the plain 3GPP standard
120deg sector, gen_single_sector_topology, los=False, no custom angle)
and instead force users to be physically close TO EACH OTHER: draw one
cluster-center location uniformly in the standard sector (same
distribution as a normal drop), then draw each of the K users uniformly
inside a disk of radius R around that center. This is a literature-
standard way to model spatially-correlated co-scheduled users (users
close together really do see correlated multipath) and is NOT the same
lever as narrowing the sector angle (confirmed by direct comparison --
sector narrowing alone, even down to 5deg, without user-to-user
clustering, was tested during REVISION (a) and gave ~0 gap under NLOS
once measured with 10-batch rigor).

Swept radius R on M8K4 (STANDARD): R=5/15/30/60m all gave only 1-2/5 SNR
points with gap>0.3%; R=20m (interpolated) was the actual optimum: 3/5
points (0dB +2.79%, 5dB +0.97%, 10dB +0.35%), and BETTER selectivity
than R=30/60m (rho@95sc=0.314, vs 0.358-0.394 for other radii tested).
R=20m locked for STANDARD_CONFIG.

MASSIVE at the original M=64 scale: swept K in {4,8,16} x R in
{5,10,20}m (9 combinations) -- NONE opened a gap above 0/5 points
(max +0.27% at K=16,R=20m). This is genuine massive-MIMO channel
hardening / favorable propagation (Marzetta 2010; the effect that makes
simple linear precoding asymptotically optimal as M grows), confirmed
by exhaustive search rather than assumed -- not a failure to find the
right lever. Per user direction: the RZF-WMMSE gap is a secondary
indicator of "is there room for a learned precoder to add value", not
the goal itself -- a MASSIVE config with no gap would still be a valid,
literature-consistent result to report (the learned precoder would
just have less headroom to beat the baselines there, which is itself
worth stating plainly).

Before accepting a gap-less MASSIVE, tried one rescale: M=32 instead of
M=64 (4x STANDARD, less extreme than 8x, already referenced as a
plausible massive-MIMO scale in Chapter 3). Swept K in {4,8} x R in
{5,10,20}m: K=8,R=5m was the only combination to open a real gap: 2/5
SNR points (0dB +0.97%, 5dB +0.39%, 10dB +0.25% just under the 0.3%
bar), rho@95sc=0.370 (real, consistent selectivity), M/K=4 (not
degenerate -- nowhere near K=M). Locked as MASSIVE_CONFIG.

Both radii differ (20m vs 5m) because they compensate for different
antenna array sizes: an M=32 array has much finer angular resolution
than M=8, so it can resolve users that would appear "merged" to a
smaller array -- users need to be physically closer together for their
channels to appear correlated to a larger array. This is the same
physical mechanism (user-to-user spatial correlation) at both scales,
just with a scale-appropriate radius, exactly like REVISION (a)'s
single half_angle_deg=7.5 was reused unchanged at both M=8 and M=64
(there, the same angle worked at both scales because the OLD mechanism
was BS-relative, not encoding array resolution -- the NEW mechanism is
inherently array-aware, which is arguably more physically principled).

MASSIVE_CONFIG rescaled from M=64,K=32 to M=32,K=8 as a direct
consequence -- every M64K32 result computed before this revision
(complexity/energy numbers, classical_comparison, any smoke test) is
now for a superseded scale and needs regenerating at M32K8 wherever
still relevant.
"""

import numpy as np
import tensorflow as tf

# =============================================================================
# LOCKED CONFIGS  (REVISION 2026-08-07 (b) -- spatial user clustering)
# =============================================================================

STANDARD_CONFIG = dict(
    NUM_TX=8, NUM_RX=4,            # M, K -- not a 50%-loading pair anymore by
                                    # design choice, just the pre-existing
                                    # standard-MIMO scale, M/K=2
    CLUSTER_RADIUS_M=20.0,         # users drawn in a 20m disk around a random
                                    # cluster center in the standard 120deg
                                    # sector -- see REVISION (b) note above
    INDOOR_PROBABILITY=0.0,
    # legacy fields kept for any code still reading them; no longer used by
    # gen_topology_clustered (sector angle is now the plain 3GPP 120deg
    # standard, not a custom narrow window; LOS is forced False inside
    # set_locked_topology regardless of this value)
    HALF_ANGLE_DEG=60.0, FORCE_LOS=False,
)

MASSIVE_CONFIG = dict(
    NUM_TX=32, NUM_RX=8,           # M, K -- RESCALED 2026-08-07 (b) from
                                    # M=64,K=32: M=64 showed zero RZF-WMMSE
                                    # gap under every (K, cluster radius)
                                    # combination tried (channel hardening,
                                    # see REVISION (b)) ; M=32 (4x STANDARD)
                                    # does show a real gap at K=8. M/K=4.
    CLUSTER_RADIUS_M=5.0,          # tighter than STANDARD's 20m -- a larger
                                    # antenna array resolves users more
                                    # finely, so users must be physically
                                    # closer to appear correlated to it.
    INDOOR_PROBABILITY=0.0,
    HALF_ANGLE_DEG=60.0, FORCE_LOS=False,   # legacy fields, see STANDARD_CONFIG
)

MASSIVE_TRUE_CONFIG = dict(
    NUM_TX=64, NUM_RX=8,            # M, K -- Priorité 3 (demande utilisateur,
                                     # 8/8) : redéfinition volontaire à M=64
                                     # (définition usuelle du MIMO massif,
                                     # cohérente avec le titre du mémoire) --
                                     # PAS le MASSIVE_CONFIG ci-dessus
                                     # (M=32, retenu le 7/8 uniquement parce
                                     # que M=64 donnait un gap RZF-WMMSE nul).
                                     # Ici, AUCUN objectif de gap forcé :
                                     # si RZF est quasi-optimal à cette
                                     # échelle, c'est un résultat honnête à
                                     # documenter (channel hardening,
                                     # Marzetta 2010), pas à contourner.
    CLUSTER_RADIUS_M=20.0,          # canal STANDARD tel quel (même R que
                                     # STANDARD_CONFIG) -- PAS de clustering
                                     # resserré artificiellement pour forcer
                                     # un gap.
    INDOOR_PROBABILITY=0.0,
    HALF_ANGLE_DEG=60.0, FORCE_LOS=False,
)

SCENARIO = 'umi'
FFT_SIZE = 96

# SNR range for training/eval -- must include the 0-5dB region where the
# validated gap lives for BOTH configs (see REVISION (b): gap is at 0/5/10dB
# for STANDARD, 0/5dB for MASSIVE, ~0 above that for both).
SNR_RANGE_DB = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20], dtype=np.float32)

PI = np.pi


def gen_topology_clustered(batch_size, num_ut, scenario, cluster_radius_m,
                            indoor_probability=0.0):
    """K users drawn inside a disk of radius cluster_radius_m around a
    cluster-center location that is itself drawn from the SAME distribution
    as Sionna's plain gen_single_sector_topology (standard 120deg sector,
    uniform-area placement) -- only the dispersion of users AROUND that
    center is changed. This is what actually opens a real RZF-WMMSE gap
    under NLOS (see REVISION (b) in this module's docstring) -- narrowing
    the sector's angle from the BS, tried first, does not, because it
    doesn't force users close to EACH OTHER, only close to the BS's
    boresight, which stopped mattering once LOS was dropped for NLOS
    (each user's NLOS channel is dominated by its own independent
    multipath draw, not by its angle from the BS).

    Physical justification: co-located/clustered users sharing local
    scatterers is a standard way to model spatially-correlated MU-MIMO
    scheduling groups in the literature; this is a different, independent
    lever from sector-angle narrowing (confirmed empirically, not assumed).
    """
    from sionna.phy.channel.utils import (relocate_uts, random_ut_properties,
                                           set_3gpp_scenario_parameters)
    from sionna.phy.config import config

    (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
     indoor_probability, min_ut_velocity, max_ut_velocity) = \
        set_3gpp_scenario_parameters(scenario, indoor_probability=indoor_probability)

    rdtype = config.tf_rdtype
    bs_loc = tf.stack([tf.zeros([batch_size, 1], rdtype),
                        tf.zeros([batch_size, 1], rdtype),
                        tf.fill([batch_size, 1], bs_height)], axis=-1)
    sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
    bs_downtilt = 0.5 * PI - tf.math.atan(sector_center / bs_height)
    bs_yaw = tf.constant(PI / 3.0, rdtype)
    bs_orientation = tf.stack([tf.fill([batch_size, 1], bs_yaw),
                                tf.fill([batch_size, 1], bs_downtilt),
                                tf.zeros([batch_size, 1], rdtype)], axis=-1)

    half_angle = tf.constant(60.0 * PI / 180.0, rdtype)   # standard 120deg sector
    d_min = tf.cast(min_bs_ut_dist, rdtype)
    r_max = tf.cast(isd * 0.5, rdtype)

    # cluster center: 1 per batch element, standard uniform-area distribution
    alpha_c = config.tf_rng.uniform([batch_size, 1], minval=-half_angle, maxval=half_angle, dtype=rdtype)
    dist2_c = config.tf_rng.uniform([batch_size, 1], minval=d_min**2, maxval=r_max**2, dtype=rdtype)
    dist_c = tf.sqrt(dist2_c)
    center_xy = tf.stack([dist_c * tf.math.cos(alpha_c), dist_c * tf.math.sin(alpha_c)], axis=-1)[:, 0, :]

    # user offsets: uniform disk of radius cluster_radius_m around the center
    R = tf.constant(cluster_radius_m, rdtype)
    ang_off = config.tf_rng.uniform([batch_size, num_ut], minval=0.0, maxval=2 * PI, dtype=rdtype)
    rad_off = R * tf.sqrt(config.tf_rng.uniform([batch_size, num_ut], minval=0.0, maxval=1.0, dtype=rdtype))
    ut_x = center_xy[:, 0:1] + rad_off * tf.math.cos(ang_off)
    ut_y = center_xy[:, 1:2] + rad_off * tf.math.sin(ang_off)

    # safety: keep users at/beyond min_bs_ut_dist (rare, cluster near BS)
    dist_bs = tf.sqrt(ut_x**2 + ut_y**2)
    scale = tf.maximum(1.0, d_min / tf.maximum(dist_bs, 1e-3))
    ut_x, ut_y = ut_x * scale, ut_y * scale

    ut_loc_xy = tf.stack([ut_x, ut_y], axis=-1)
    ut_loc_z = config.tf_rng.uniform([batch_size, num_ut, 1], minval=min_ut_height,
                                      maxval=max_ut_height, dtype=rdtype)
    ut_loc = tf.concat([ut_loc_xy, ut_loc_z], axis=-1)

    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size, num_ut, indoor_probability, min_ut_velocity, max_ut_velocity)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


def gen_topology_locked(batch_size, num_ut, cluster_radius_m=20.0,
                         indoor_probability=0.0, scenario=SCENARIO):
    """Reusable topology draw for the locked mechanism (spatial user
    clustering, REVISION (b)). Pass num_ut / cluster_radius_m explicitly for
    whichever of STANDARD_CONFIG / MASSIVE_CONFIG (or a future config) is
    in use -- radius should scale down as num_tx (array size) grows, see
    module docstring."""
    return gen_topology_clustered(batch_size, num_ut, scenario, cluster_radius_m,
                                   indoor_probability=indoor_probability)


def set_locked_topology(channel_model, batch_size, num_ut,
                         cluster_radius_m=20.0, force_los=False,
                         indoor_probability=0.0, scenario=SCENARIO,
                         half_angle_deg=None):
    """Draw a topology under the locked mechanism and apply it to
    channel_model. force_los is kept as a parameter for interface stability
    but REVISION (b)'s mechanism is validated under NLOS only (los=False) --
    don't pass True without re-validating the gap first.

    half_angle_deg is accepted and IGNORED (kept only so old call sites
    passing CHOSEN_CONFIG['HALF_ANGLE_DEG'] positionally/by-keyword don't
    crash) -- the sector angle is now always the plain 3GPP 120deg standard;
    call sites should migrate to passing CLUSTER_RADIUS_M instead."""
    topology = gen_topology_locked(batch_size, num_ut, cluster_radius_m,
                                    indoor_probability, scenario)
    channel_model.set_topology(*topology, los=force_los)
