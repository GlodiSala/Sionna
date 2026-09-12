"""
experiments/channel_coherence_bandwidth.py — Investigation demandée (François,
round 2, tab:coherence) : bande de cohérence EMPIRIQUE par scénario
(UMi/UMa x LOS/NLOS), extraite par seuillage du coefficient de
corrélation fréquentielle rho(Delta n), pour comparaison directe avec
la valeur analytique Bc ~ 1/(5*sigma_tau) déjà présente dans le texte
(7-Theme1.tex, tab:coherence).

NOUVEAU FICHIER -- ne touche à rien d'existant. Réutilise EXACTEMENT
le mécanisme de construction de canal du mémoire (channel_config.py,
STANDARD_CONFIG M=8/K=4/R=20m, set_locked_topology, Section
sec:channel_construction de 7-Theme1.tex) -- même config que celle
qui a produit les rho(12) déjà cités en prose dans le texte actuel
(UMi~0,89 / UMa~0,65, cf. experiments/channel_coherence.py, résultats
results/diag_coherence_annexe_b.json). Ce script ÉTEND cette mesure à
une grille de lags DENSE (chaque sous-porteuse, Delta n = 1..95, pas
un sous-ensemble épars) pour pouvoir interpoler précisément le lag où
rho(Delta n) franchit un seuil donné -> bande de cohérence empirique.

Seuils de corrélation utilisés (convention standard de la littérature,
Rappaport, "Wireless Communications: Principles and Practice", 2e éd.,
section 5.6.3 -- PAS inventés) :
  - rho > 0.9  : Bc "stricte"  (formule usuelle associée : Bc ~ 1/(50*sigma_tau))
  - rho > 0.5  : Bc "relâchée" (formule usuelle associée : Bc ~ 1/(5*sigma_tau))
Les DEUX sont calculés empiriquement ici, sans présupposer laquelle des
deux formules analytiques du texte actuel leur correspond -- la
comparaison brute (section 4 du rapport) permet de trancher.

Méthodologie de mesure identique à experiments/channel_coherence.py :
  rho(lag) = |mean(H[...,:-lag] * conj(H[...,lag:]))| / mean(|H|^2)
moyennée sur tirages (NUM_DRAWS), batch, utilisateurs, antennes.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/channel_coherence_bandwidth.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import AntennaArray, UMi, UMa
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.config import config as sionna_config
from channel_config import STANDARD_CONFIG, set_locked_topology

SEED = 2026                        # reproductibilité -- experiments/channel_coherence.py n'en fixait pas
                                    # (d'où la variation run-to-run observée, ~+/-0.02-0.03 sur rho(12)
                                    # UMa-NLOS notamment -- voir rapport), corrigé ici.
sionna_config.seed = SEED

M, K, R = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX'], STANDARD_CONFIG['CLUSTER_RADIUS_M']
FFT_SIZE, NUM_OFDM = 96, 14
SUBCARRIER_SPACING = 30e3          # Hz -- tab:sys_params, Delta f = 30 kHz
CARRIER_FREQ = 2.6e9               # tab:sys_params, "Fréquence porteuse 2,6 GHz"
LAGS = list(range(1, 96))          # DENSE : chaque sous-porteuse (vs sous-ensemble épars de experiments/channel_coherence.py)
NUM_DRAWS = 100                    # >> experiments/channel_coherence.py (20) -- réduit le bruit Monte-Carlo observé
                                    # sur UMa-NLOS (canal le plus sensible, delay spread le plus grand)
BATCH = 16
THRESHOLDS = [0.9, 0.5]            # Rappaport : 0.9 (stricte) et 0.5 (relâchée) -- voir docstring


def build_arrays():
    ut_array = AntennaArray(num_rows=1, num_cols=1, polarization='single',
                             polarization_type='V', antenna_pattern='omni', carrier_frequency=CARRIER_FREQ)
    bs_array = AntennaArray(num_rows=1, num_cols=M // 2, polarization='dual',
                             polarization_type='cross', antenna_pattern='38.901', carrier_frequency=CARRIER_FREQ)
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
    return {lag: float(np.mean(all_corrs[lag])) for lag in LAGS}


def empirical_bc(corr_mean, threshold):
    """Premier lag (en nb de sous-porteuses) où rho(Delta n) descend sous
    threshold, interpolation linéaire entre les deux lags encadrants pour
    une estimation continue. Retourne (lag_frac, Bc_Hz). None si rho ne
    descend jamais sous threshold sur la plage mesurée (canal quasi-plat
    sur toute la bande -- rapporté explicitement, pas extrapolé)."""
    lags_sorted = sorted(corr_mean.keys())
    prev_lag, prev_rho = 0, 1.0   # rho(0) = 1 par définition
    for lag in lags_sorted:
        rho = corr_mean[lag]
        if rho < threshold:
            if prev_rho == rho:
                frac = lag
            else:
                frac = prev_lag + (prev_rho - threshold) / (prev_rho - rho) * (lag - prev_lag)
            return float(frac), float(frac * SUBCARRIER_SPACING)
        prev_lag, prev_rho = lag, rho
    return None, None   # jamais descendu sous le seuil sur Delta n in [1,95]


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    ut_array, bs_array = build_arrays()
    frequencies = subcarrier_frequencies(FFT_SIZE, SUBCARRIER_SPACING)
    rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE, subcarrier_spacing=SUBCARRIER_SPACING,
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
        channel_model = cls(carrier_frequency=CARRIER_FREQ, o2i_model='low', ut_array=ut_array, bs_array=bs_array,
                             direction='downlink', enable_pathloss=False, enable_shadow_fading=False)
        corr_mean = measure(channel_model, scenario, force_los, frequencies, ofdm_symbol_duration)

        row = {'corr_mean': corr_mean}
        for th in THRESHOLDS:
            lag_frac, bc_hz = empirical_bc(corr_mean, th)
            row[f'Bc_empirical_threshold_{th}'] = {
                'lag_subcarriers': lag_frac,
                'Bc_Hz': bc_hz,
                'Bc_MHz': (bc_hz / 1e6) if bc_hz is not None else None,
                'Bc_RB': (lag_frac / 12.0) if lag_frac is not None else None,   # 1 RB = 12 SC
            }
            if lag_frac is not None:
                print(f"  seuil rho>{th} : franchi à Delta n={lag_frac:.2f} SC -> "
                      f"Bc_empirique={bc_hz/1e6:.3f} MHz ({lag_frac/12.0:.3f} RB)", flush=True)
            else:
                print(f"  seuil rho>{th} : JAMAIS franchi sur Delta n in [1,95] "
                      f"(rho(95)={corr_mean[95]:.4f} >= {th}) -- canal quasi-plat sur toute la bande mesurée", flush=True)

        results[name] = row
        with open('results/diag_coherence_bandwidth_threshold.json', 'w') as f:
            json.dump(results, f, indent=2)   # sauvé après chaque scénario -- reprenable

    print("\n✅ Sauvé -> results/diag_coherence_bandwidth_threshold.json")
