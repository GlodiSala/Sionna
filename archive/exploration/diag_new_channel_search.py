"""
diag_new_channel_search.py — recherche d'un canal qui restaure une vraie
sélectivité fréquentielle TOUT en gardant un gap RZF-WMMSE mesurable.

Le mécanisme verrouillé actuel (LOS forcé + fenêtre azimutale étroite
7.5°) donne un canal quasi mono-trajet (>99% de la puissance dans un
seul tap retard, |rho|>0.80 même à 96 SC de lag) -- aucune sélectivité
fréquentielle à exploiter par une architecture multi-porteuses.

Hypothèse testée : angle et LOS/NLOS sont deux leviers en partie
indépendants dans Sionna (angle = fenêtre de dépôt spatiale, LOS/NLOS =
état tiré séparément qui pilote les tables 3GPP de delay/angular spread).
On peut donc garder la fenêtre étroite (qui créait le gap RZF-WMMSE
avec le loading ratio, cf. channel_config.py Part 1) tout en forçant
NLOS (qui devrait réintroduire un vrai delay spread => sélectivité
fréquentielle), une combinaison jamais testée dans l'historique
(channel_config.py ne teste que narrow+LOS vs wide+NLOS).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_new_channel_search.py
"""
import os, sys, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

M, K = 8, 4   # STANDARD_CONFIG scale, recherche rapide d'abord
SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 8
BATCH_SIZE = 32

CANDIDATES = {
    'actuel (narrow7.5+LOS)':      dict(half_angle_deg=7.5,  los=True,  indoor_probability=0.0),
    'narrow7.5+NLOS':              dict(half_angle_deg=7.5,  los=False, indoor_probability=0.0),
    'narrow15+NLOS':               dict(half_angle_deg=15.0, los=False, indoor_probability=0.0),
    'wide30+NLOS':                 dict(half_angle_deg=30.0, los=False, indoor_probability=0.0),
    'narrow7.5+LOS_naturel(None)': dict(half_angle_deg=7.5,  los=None,  indoor_probability=0.0),
    'wide60+LOS(REF historique)':  dict(half_angle_deg=60.0, los=True,  indoor_probability=0.0),
}


def freq_correlation(h_freq_np, lags=(4, 8, 12, 24, 48, 95)):
    """h_freq_np : [B, K, 1, 1, M, ofdm, fft] complexe -> corrélation sur fft, symbole 0."""
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]  # [B,K,M,fft]
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    out = {}
    for lag in lags:
        num = np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))
        out[lag] = float(np.abs(num) / den)
    return out


def rzf_wmmse_gap(system, snr_points, num_batches, batch_size):
    from precoders_w import rzf_precoder, wmmse_precoder
    gaps = {}
    for snr in snr_points:
        snr_t = tf.constant(snr, tf.float32)
        r_rzf, r_wmmse = [], []
        for _ in range(num_batches):
            system.new_topology(batch_size)
            h_freq, no = system.channel_and_no(tf.constant(batch_size, tf.int32), snr_t)
            g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            r_rzf.append(float(system._sum_rate(h_freq, g_rzf, no)))
            r_wmmse.append(float(system.eval_wmmse_from_h(h_freq, no, 10)))
        rzf_m, wmmse_m = np.mean(r_rzf), np.mean(r_wmmse)
        gap_pct = 100.0 * (wmmse_m - rzf_m) / max(rzf_m, 1e-6)
        gaps[snr] = (rzf_m, wmmse_m, gap_pct)
    return gaps


results = {}
for name, cfg in CANDIDATES.items():
    print(f'\n{"="*75}\n{name}  {cfg}\n{"="*75}')
    t0 = time.time()
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['half_angle_deg'],
                                     los=cfg['los'], indoor_probability=cfg['indoor_probability'])
    system.new_topology(200)
    h_freq, _ = system.channel_and_no(tf.constant(200, tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation fréquentielle |rho(lag)| :',
          ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()))

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, BATCH_SIZE)
    print('  Gap RZF-WMMSE par SNR :')
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%')

    max_gap = max(g[2] for g in gaps.values())
    results[name] = {'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()}, 'max_gap_pct': max_gap}
    print(f'  -> max gap sur la plage SNR = {max_gap:+.2f}%  ({time.time()-t0:.0f}s)')
    del system
    tf.keras.backend.clear_session()

print(f'\n\n{"="*75}\nRÉSUMÉ RECHERCHE CANAL (M={M},K={K})\n{"="*75}')
print(f'{"config":<32} {"rho@12sc":>9} {"rho@48sc":>9} {"rho@95sc":>9} {"max_gap%":>9}')
for name, r in results.items():
    c = r['corr']
    print(f'{name:<32} {c[12]:>9.3f} {c[48]:>9.3f} {c[95]:>9.3f} {r["max_gap_pct"]:>9.2f}')

import json
with open('results/diag_new_channel_search.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_new_channel_search.json')
