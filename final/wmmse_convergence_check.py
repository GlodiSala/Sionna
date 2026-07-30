"""
Step 1 — WMMSE convergence check.

Question: is the WMMSE-vs-RZF sum-rate gap seen at higher user counts (K)
a real precoding-difficulty effect, or is it just WMMSE's fixed 10-iteration
budget under-converging as the problem (K users, M antennas) gets bigger?

This script reuses the exact `rzf_precoder` / `wmmse_precoder` implementations
from precoders_w.py and the channel/OFDM pipeline pattern from
compare_rb_grouping.py, but makes M (num_tx), K (num_rx), the user angular
window, LOS state and indoor probability all configurable, and sweeps
WMMSE's num_iterations to see whether more iterations close any gap.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from precoders_w import rzf_precoder, wmmse_precoder

from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, RemoveNulledSubcarriers,
                             ResourceGridMapper, LMMSEEqualizer,
                             LMMSEPostEqualizationSINR, RZFPrecodedChannel)
from sionna.phy.channel import (ApplyOFDMChannel, subcarrier_frequencies,
                                 cir_to_ofdm_channel)
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.channel.utils import (relocate_uts, random_ut_properties,
                                       set_3gpp_scenario_parameters)
from sionna.phy.config import config
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

FFT_SIZE = 96   # 8 RBs x 12 SC (was 72 = 6 RBs x 12 SC)
NUM_OFDM = 14
PI = np.pi


# =============================================================================
# Custom topology generator: configurable azimuth half-angle window
# =============================================================================

def gen_topology_custom(batch_size, num_ut, scenario, half_angle_deg=60.0,
                         indoor_probability=None):
    """Same statistics as sionna's gen_single_sector_topology, except UTs are
    dropped within an azimuth window of +/- half_angle_deg (default 60 deg =
    full 120 deg 3GPP sector, i.e. reproduces the original behaviour)
    instead of the fixed two-halves-of-120-deg sector.
    """
    (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
     indoor_probability, min_ut_velocity, max_ut_velocity) = \
        set_3gpp_scenario_parameters(scenario,
                                      indoor_probability=indoor_probability)

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

    half_angle = tf.constant(half_angle_deg * PI / 180.0, rdtype)
    d_min = tf.cast(min_bs_ut_dist, rdtype)
    r = tf.cast(isd * 0.5, rdtype)

    alpha = config.tf_rng.uniform([batch_size, num_ut], minval=-half_angle,
                                   maxval=half_angle, dtype=rdtype)
    # uniform-area sampling within the window (same sqrt-density trick as
    # sionna's drop_uts_in_sector, just without the two-halves split)
    distance2 = config.tf_rng.uniform([batch_size, num_ut],
                                       minval=d_min ** 2, maxval=r ** 2,
                                       dtype=rdtype)
    distance = tf.sqrt(distance2)
    ut_loc_xy = tf.stack([distance * tf.math.cos(alpha),
                          distance * tf.math.sin(alpha)], axis=-1)
    ut_loc_xy = relocate_uts(ut_loc_xy, tf.constant(0, tf.int32),
                              tf.zeros([2], rdtype))

    ut_loc_z = config.tf_rng.uniform([batch_size, num_ut, 1],
                                      minval=min_ut_height,
                                      maxval=max_ut_height, dtype=rdtype)
    ut_loc = tf.concat([ut_loc_xy, ut_loc_z], axis=-1)

    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size, num_ut, indoor_probability, min_ut_velocity,
        max_ut_velocity)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, \
        in_state


# =============================================================================
# Configurable MU-MIMO system
# =============================================================================

class ConfigurableMIMOSystem(tf.keras.Model):

    def __init__(self, num_tx, num_rx, half_angle_deg=60.0, los=None,
                 indoor_probability=None):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.half_angle_deg = half_angle_deg
        self.los = los
        self.indoor_probability = indoor_probability
        self.num_bits_per_symbol = 2

        rx_tx_association = np.ones([num_rx, 1])
        self.sm = StreamManagement(rx_tx_association, num_streams_per_tx=num_rx)

        self.rg = ResourceGrid(
            num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
            subcarrier_spacing=30e3, num_tx=1,
            num_streams_per_tx=num_rx, cyclic_prefix_length=6,
            pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1, polarization="single",
            polarization_type="V", antenna_pattern="omni",
            carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(
            num_rows=1, num_cols=int(num_tx / 2), polarization="dual",
            polarization_type="cross", antenna_pattern="38.901",
            carrier_frequency=2.6e9)

        self.channel_model = UMi(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(
            int(self.rg.num_data_symbols),
            int(self.rg.num_data_symbols * 2))
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.channel_freq = ApplyOFDMChannel(add_awgn=True)
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(self.rg, self.sm)
        self.remove_nulled = RemoveNulledSubcarriers(self.rg)
        self.precoded_channel_helper = RZFPrecodedChannel(self.rg, self.sm)

    def new_topology(self, batch_size):
        topology = gen_topology_custom(
            batch_size, self.num_rx, "umi",
            half_angle_deg=self.half_angle_deg,
            indoor_probability=self.indoor_probability)
        self.channel_model.set_topology(*topology, los=self.los)

    def _gen_channel(self, batch_size):
        cir = self.channel_model(batch_size, self.rg.num_ofdm_symbols,
                                  1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        return self.remove_nulled(h_freq)

    def _sum_rate(self, h_freq, g, no):
        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        sum_rate = tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0),
                           axis=[1, 2, 4]),
            axis=0))
        return sum_rate

    @tf.function(reduce_retracing=True)
    def channel_and_no(self, batch_size, snr_db):
        """Generate one channel realization + noise level, to be reused by
        every precoder so RZF/WMMSE(iters) are compared on IDENTICAL
        channels (paired comparison, no topology-sampling noise)."""
        no = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)
        h_freq = self._gen_channel(batch_size)
        return h_freq, no

    @tf.function(reduce_retracing=True)
    def eval_rzf_from_h(self, h_freq, no):
        g = rzf_precoder(h_freq, stream_management=self.sm, no=no)
        return self._sum_rate(h_freq, g, no)

    @tf.function(reduce_retracing=True)
    def eval_wmmse_from_h(self, h_freq, no, num_iterations):
        g = wmmse_precoder(h_freq, no=no, stream_management=self.sm,
                           num_iterations=num_iterations)
        return self._sum_rate(h_freq, g, no)


def run_config(name, num_tx, num_rx, snr_db, half_angle_deg, los,
               indoor_probability, iter_list, batch_size, num_batches):
    print(f"\n{'='*70}\n{name}  (M={num_tx}, K={num_rx}, SNR={snr_db} dB)\n{'='*70}")
    system = ConfigurableMIMOSystem(num_tx, num_rx,
                                     half_angle_deg=half_angle_deg,
                                     los=los,
                                     indoor_probability=indoor_probability)
    bs = tf.constant(batch_size, dtype=tf.int32)
    snr_t = tf.constant(float(snr_db), dtype=tf.float32)

    # Paired comparison: same channel realizations reused across RZF and
    # every WMMSE iteration count, so differences reflect the precoder/
    # iteration count only, not topology-sampling noise.
    rzf_rates = []
    wmmse_rates = {it: [] for it in iter_list}
    gap_pct_per_batch = {it: [] for it in iter_list}
    for _ in range(num_batches):
        system.new_topology(batch_size)
        h_freq, no = system.channel_and_no(bs, snr_t)
        r_rzf = float(system.eval_rzf_from_h(h_freq, no))
        rzf_rates.append(r_rzf)
        for it in iter_list:
            r_w = float(system.eval_wmmse_from_h(h_freq, no, it))
            wmmse_rates[it].append(r_w)
            gap_pct_per_batch[it].append(100.0 * (r_rzf - r_w) / r_rzf)

    rzf_rate = float(np.mean(rzf_rates))
    print(f"  RZF                          : {rzf_rate:7.3f} bps/Hz  "
          f"(std over batches={np.std(rzf_rates):.3f})")

    results = {'config': name, 'rzf': rzf_rate, 'wmmse': {}}
    for it in iter_list:
        rate = float(np.mean(wmmse_rates[it]))
        gap_pct_mean = float(np.mean(gap_pct_per_batch[it]))
        gap_pct_std = float(np.std(gap_pct_per_batch[it]))
        results['wmmse'][it] = rate
        print(f"  WMMSE (iters={it:3d})            : {rate:7.3f} bps/Hz  "
              f"(paired gap vs RZF: {gap_pct_mean:+.2f}% +/- {gap_pct_std:.2f}, "
              f"std over batches={np.std(wmmse_rates[it]):.3f})")

    return results


if __name__ == '__main__':
    BATCH_SIZE = 48
    NUM_BATCHES = 6
    ITER_LIST = [10, 30, 50]

    all_results = []

    # Approximates "M64K36_ang15_los_indoor0" @ SNR=5
    all_results.append(run_config(
        "M64K36_ang15_los_indoor0", num_tx=64, num_rx=36, snr_db=5.0,
        half_angle_deg=7.5, los=True, indoor_probability=0.0,
        iter_list=ITER_LIST, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES))

    # Approximates "load_8x8_nlos_default" @ SNR=15
    all_results.append(run_config(
        "load_8x8_nlos_default", num_tx=8, num_rx=8, snr_db=15.0,
        half_angle_deg=60.0, los=False, indoor_probability=None,
        iter_list=ITER_LIST, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES))

    os.makedirs('./results', exist_ok=True)
    np.save('./results/wmmse_convergence_check.npy', all_results)
    print("\nSaved to ./results/wmmse_convergence_check.npy")
