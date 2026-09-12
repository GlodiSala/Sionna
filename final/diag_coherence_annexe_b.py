"""
diag_coherence_annexe_b.py — Annexe B (demande utilisateur) : ρ(Δn)
pour les 4 combinaisons canal x état LOS (UMi-LOS, UMi-NLOS, UMa-LOS,
UMa-NLOS), config STANDARD_CONFIG verrouillée actuelle (M=8,K=4,
R=20m), même méthodologie que diag_uma_selectivity.py (20 tirages x
batch 16, lags 1-12 + 24/48/95). Le pipeline final n'utilise QUE
NLOS -- les points LOS sont une mesure de caractérisation du canal
pour l'annexe, pas une config jamais entraînée dessus.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_coherence_annexe_b.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid
from channel_config import STANDARD_CONFIG, set_locked_topology

M, K, R = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX'], STANDARD_CONFIG['CLUSTER_RADIUS_M']
FFT_SIZE, NUM_OFDM = 96, 14
LAGS = list(range(1, 13)) + [24, 48, 95]
NUM_DRAWS = 20
BATCH = 16
DATA_SNR = 15.0  # non-influent sur rho (mesure sur H, pas de bruit ajouté) -- gardé pour cohérence de style


def build_arrays():
    ut_array = AntennaArray(num_rows=1, num_cols=1, polarization='single',
                             polarization_type='V', antenna_pattern='omni', carrier_frequency=2.6e9)
    bs_array = AntennaArray(num_rows=1, num_cols=M // 2, polarization='dual',
                             polarization_type='cross', antenna_pattern='38.901', carrier_frequency=2.6e9)
    return ut_array, bs_array


def freq_correlation(h_freq_np, lags):
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    return {lag: float(np.abs(np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))) / den) for lag in lags}


def measure(channel_model, scenario, force_los, frequencies, ofdm_symbol_duration):
    all_corrs = {lag: [] for lag in LAGS}
    for _ in range(NUM_DRAWS):
        set_locked_topology(channel_model, BATCH, num_ut=K, cluster_radius_m=R,
                             force_los=force_los, indoor_probability=STANDARD_CONFIG['INDOOR_PROBABILITY'],
                             scenario=scenario)
        cir = channel_model(BATCH, NUM_OFDM, 1.0 / ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
        c = freq_correlation(h_freq.numpy(), LAGS)
        for lag in LAGS:
            all_corrs[lag].append(c[lag])
    corr_mean = {str(lag): float(np.mean(all_corrs[lag])) for lag in LAGS}
    return corr_mean


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    ut_array, bs_array = build_arrays()
    frequencies = subcarrier_frequencies(FFT_SIZE, 30e3)
    rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE, subcarrier_spacing=30e3,
                       num_tx=1, num_streams_per_tx=K, cyclic_prefix_length=6,
                       pilot_pattern='kronecker', pilot_ofdm_symbol_indices=[2, 11])
    ofdm_symbol_duration = rg.ofdm_symbol_duration

    scenarios = {
        'UMi-LOS':  ('umi', UMi, True),
        'UMi-NLOS': ('umi', UMi, False),
        'UMa-LOS':  ('uma', UMa, True),
        'UMa-NLOS': ('uma', UMa, False),
    }

    results = {}
    for name, (scenario, cls, force_los) in scenarios.items():
        print(f"\n{'='*70}\n{name}  (M={M},K={K},R={R}m, {'LOS force' if force_los else 'NLOS'})\n{'='*70}", flush=True)
        channel_model = cls(carrier_frequency=2.6e9, o2i_model='low', ut_array=ut_array, bs_array=bs_array,
                             direction='downlink', enable_pathloss=False, enable_shadow_fading=False)
        corr_mean = measure(channel_model, scenario, force_los, frequencies, ofdm_symbol_duration)
        decay = corr_mean['12'] / corr_mean['1']
        results[name] = {'corr_mean': corr_mean, 'decay_12_over_1': decay}
        print(f"  rho(1)={corr_mean['1']:.4f}  rho(12)={corr_mean['12']:.4f}  "
              f"rho(24)={corr_mean['24']:.4f}  rho(48)={corr_mean['48']:.4f}  rho(95)={corr_mean['95']:.4f}  "
              f"decay(12/1)={decay:.3f}", flush=True)
        with open('results/diag_coherence_annexe_b.json', 'w') as f:
            json.dump(results, f, indent=2)   # sauvé après chaque scénario -- reprenable

    print("\n✅ Sauvé -> results/diag_coherence_annexe_b.json")
