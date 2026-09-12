"""
diag_uma_selectivity.py — demande utilisateur (17h15) : le canal actuel
(UMi, clustering R=20m) donne rho(12sc)=0.797 -- sélectivité réelle mais
peut-être trop modeste pour qu'un mécanisme cross-SC ait un avantage net
sous CSI parfait. Teste UMa (delay spread 3GPP plus large que UMi ->
décorrélation fréquentielle plus rapide attendue) avec le MÊME mécanisme
de clustering spatial déjà validé (K users proches, R=20m, NLOS).

Mesure (1) corrélation à lags courts 1-12sc + référence longue distance,
(2) gap RZF-WMMSE (même rigueur que classical_comparison.py, 10 batches).

Ne modifie/n'invalide RIEN -- mesure seulement, propose, n'applique pas
sans confirmation explicite.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_uma_selectivity.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import AntennaArray, UMa
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.utils import ebnodb2no
from precoders_w import rzf_precoder, wmmse_precoder
from channel_config import STANDARD_CONFIG, gen_topology_clustered

M, K, R = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX'], STANDARD_CONFIG['CLUSTER_RADIUS_M']
FFT_SIZE, NUM_OFDM = 96, 14
SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 10
BATCH = 16
LAGS = list(range(1, 13)) + [24, 48, 95]


class UMaSystem:
    """Miroir minimal de ConfigurableMIMOSystem/ClusteredSystem mais avec
    UMa au lieu de UMi -- même clustering spatial (gen_topology_clustered,
    déjà validé §0/§1), même RG (pas de guard/dc_null, comme wmmse_
    convergence_check.py, pour comparabilité directe avec classical_
    comparison.py)."""
    def __init__(self, num_tx, num_rx, cluster_radius_m):
        self.num_tx, self.num_rx = num_tx, num_rx
        self.cluster_radius_m = cluster_radius_m
        self.num_bits_per_symbol = 2

        self.sm = StreamManagement(np.ones([num_rx, 1]), num_streams_per_tx=num_rx)
        self.rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
                                subcarrier_spacing=30e3, num_tx=1,
                                num_streams_per_tx=num_rx, cyclic_prefix_length=6,
                                pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])
        self.ut_array = AntennaArray(num_rows=1, num_cols=1, polarization="single",
                                      polarization_type="V", antenna_pattern="omni",
                                      carrier_frequency=2.6e9)
        self.bs_array = AntennaArray(num_rows=1, num_cols=int(num_tx / 2), polarization="dual",
                                      polarization_type="cross", antenna_pattern="38.901",
                                      carrier_frequency=2.6e9)
        self.channel_model = UMa(carrier_frequency=2.6e9, o2i_model="low",
                                  ut_array=self.ut_array, bs_array=self.bs_array,
                                  direction='downlink', enable_pathloss=False,
                                  enable_shadow_fading=False)
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self.precoded_channel_helper = RZFPrecodedChannel(self.rg, self.sm)
        self.lmmse_sinr = LMMSEPostEqualizationSINR(self.rg, self.sm)

    def new_topology(self, batch_size):
        topology = gen_topology_clustered(batch_size, self.num_rx, 'uma',
                                           self.cluster_radius_m, indoor_probability=0.0)
        self.channel_model.set_topology(*topology, los=False)

    def channel_and_no(self, batch_size, snr_db):
        no = ebnodb2no(snr_db, self.num_bits_per_symbol, 0.5, self.rg)
        cir = self.channel_model(batch_size, self.rg.num_ofdm_symbols, 1.0 / self.rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(self.frequencies, *cir, normalize=True)
        return h_freq, no

    def _sum_rate(self, h_freq, g, no):
        h_eff = self.precoded_channel_helper.compute_effective_channel(h_freq, g)
        sinr = self.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        return tf.reduce_sum(tf.reduce_mean(
            tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0), axis=[1, 2, 4]), axis=0))


def freq_correlation(h_freq_np, lags):
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    return {lag: float(np.abs(np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))) / den) for lag in lags}


if __name__ == '__main__':
    print(f'\n{"="*70}\nUMa clustering M={M} K={K} R={R}m NLOS\n{"="*70}', flush=True)
    system = UMaSystem(M, K, R)

    # -- corrélation lags courts --
    all_corrs = {lag: [] for lag in LAGS}
    for _ in range(20):
        system.new_topology(BATCH)
        h_freq, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(15.0, tf.float32))
        c = freq_correlation(h_freq.numpy(), LAGS)
        for lag in LAGS:
            all_corrs[lag].append(c[lag])
    corr_mean = {lag: float(np.mean(all_corrs[lag])) for lag in LAGS}
    print(f'{"lag(SC)":>8} {"rho UMa":>9} {"rho UMi(réf)":>13}')
    umi_ref = {1: 0.996, 2: 0.985, 3: 0.970, 4: 0.951, 5: 0.932, 6: 0.912, 7: 0.892,
               8: 0.873, 9: 0.854, 10: 0.834, 11: 0.815, 12: 0.797, 24: 0.646, 48: 0.494, 95: 0.363}
    for lag in LAGS:
        print(f'{lag:>8} {corr_mean[lag]:>9.4f} {umi_ref.get(lag, float("nan")):>13.4f}')
    decay = corr_mean[12] / corr_mean[1]
    print(f'\nDécroissance intra-RB UMa : rho(12)/rho(1) = {decay:.3f} (UMi: 0.800)')

    # -- gap RZF-WMMSE --
    print(f'\n{"="*70}\nGap RZF-WMMSE UMa (10 batches, comme classical_comparison.py)\n{"="*70}')
    gaps = {}
    for snr in SNR_POINTS:
        snr_t = tf.constant(snr, tf.float32)
        r_rzf, r_wmmse = [], []
        for _ in range(NUM_BATCHES):
            system.new_topology(BATCH)
            h_freq, no = system.channel_and_no(tf.constant(BATCH, tf.int32), snr_t)
            g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            r_rzf.append(float(system._sum_rate(h_freq, g_rzf, no)))
            g_wmmse = wmmse_precoder(h_freq, no=no, stream_management=system.sm, num_iterations=10)
            r_wmmse.append(float(system._sum_rate(h_freq, g_wmmse, no)))
        rzf_m, wmmse_m = np.mean(r_rzf), np.mean(r_wmmse)
        gap_pct = 100.0 * (wmmse_m - rzf_m) / max(rzf_m, 1e-6)
        gaps[snr] = (rzf_m, wmmse_m, gap_pct)
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        print(f'  SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}')

    result = {'corr_mean': corr_mean, 'decay_12_over_1': decay,
              'gaps': {str(k): v for k, v in gaps.items()}}
    os.makedirs('results', exist_ok=True)
    with open('results/diag_uma_selectivity.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\nSauvé -> results/diag_uma_selectivity.json')
    print('\n⚠️  Mesure seulement -- rien invalidé/régénéré. Attente confirmation utilisateur.')
