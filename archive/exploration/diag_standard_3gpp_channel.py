"""
diag_standard_3gpp_channel.py — Étape 2/3 : canal 3GPP STANDARD, pas de
réglage custom. Topologie = gen_single_sector_topology (fonction stock
Sionna, secteur 120° standard, paramètres 3GPP par défaut), LOS state
forcé à NLOS (scénario reconnu/documenté dans la littérature -- pas un
angle bricolé par recherche).

Teste :
  - STANDARD : M=8, K=4
  - MASSIVE  : M=64, K=4
  - MASSIVE  : M=64, K=8

Pour chacun : corrélation fréquentielle (lag 4/8/12/24/48/95 SC) + gap
RZF-WMMSE apparié sur 5 points SNR [0,5,10,15,20]dB, 10 batches
(même rigueur que classical_comparison.py).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_standard_3gpp_channel.py
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from sionna.phy.channel import gen_single_sector_topology

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 10   # aligné sur classical_comparison.py -- rigueur nécessaire
                    # (6-8 batches se sont révélés trop bruités hier)


class StandardNLOSSystem(ConfigurableMIMOSystem):
    """Sous-classe : topologie 3GPP standard stock (secteur 120°, pas de
    fenêtre d'angle custom), LOS state forcé à NLOS -- scénario standard
    documenté, pas un réglage recherché."""
    def new_topology(self, batch_size):
        topology = gen_single_sector_topology(
            batch_size, self.num_rx, self.scenario if hasattr(self, 'scenario') else 'umi',
            indoor_probability=0.0)
        self.channel_model.set_topology(*topology, los=False)


def freq_correlation(h_freq_np, lags=(4, 8, 12, 24, 48, 95)):
    h = np.squeeze(h_freq_np, axis=(2, 3))[:, :, :, 0, :]
    N = h.shape[-1]
    den = np.mean(np.abs(h) ** 2)
    return {lag: float(np.abs(np.mean(h[..., :N - lag] * np.conj(h[..., lag:]))) / den) for lag in lags}


def rzf_wmmse_gap(system, snr_points, num_batches, batch_size):
    from precoders_w import rzf_precoder
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


CONFIGS = {
    'STANDARD (M=8,K=4)':  dict(M=8,  K=4, batch=16),
    'MASSIVE (M=64,K=4)':  dict(M=64, K=4, batch=8),
    'MASSIVE (M=64,K=8)':  dict(M=64, K=8, batch=8),
}

results = {}
for name, cfg in CONFIGS.items():
    M, K, BATCH_SIZE = cfg['M'], cfg['K'], cfg['batch']
    print(f'\n{"="*75}\n{name}  M/K={M/K:.1f}\n{"="*75}', flush=True)
    t0 = time.time()
    system = StandardNLOSSystem(M, K)  # half_angle_deg/los/indoor_probability des kwargs
                                        # ConfigurableMIMOSystem sont ignorés -- new_topology
                                        # est complètement remplacée ci-dessus
    system.scenario = 'umi'
    system.new_topology(min(32, BATCH_SIZE * 2))
    h_freq, _ = system.channel_and_no(tf.constant(min(32, BATCH_SIZE * 2), tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()), flush=True)

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, BATCH_SIZE)
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    max_gap = max(g[2] for g in gaps.values())
    results[name] = {'M': M, 'K': K, 'M_over_K': M / K, 'corr': corr,
                      'gaps': {str(k): v for k, v in gaps.items()},
                      'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ CANAL 3GPP STANDARD (UMi NLOS, secteur 120° stock, pas de réglage custom) ===')
print(f'{"config":<22} {"M/K":>6} {"rho@48sc":>9} {"rho@95sc":>9} {"max_gap%":>9} {"pts>0.3%":>9}')
for name, r in results.items():
    c = r['corr']
    print(f'{name:<22} {r["M_over_K"]:>6.1f} {c[48]:>9.3f} {c[95]:>9.3f} {r["max_gap_pct"]:>9.2f} {r["n_points_with_gap_gt_0.3pct"]:>9d}')

with open('results/diag_standard_3gpp_channel.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_standard_3gpp_channel.json')
