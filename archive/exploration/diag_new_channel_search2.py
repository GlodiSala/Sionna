"""
diag_new_channel_search2.py — Suite de diag_new_channel_search.py :
teste le loading ratio (K/M) combiné à NLOS/angle pour chercher un
gap RZF-WMMSE qui survit à plus d'un point SNR, tout en gardant la
sélectivité fréquentielle restaurée par NLOS.

M=8 fixe (échelle STANDARD), K variable (6, 8) x angle/LOS variable.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_new_channel_search2.py
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

M = 8
SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 8
BATCH_SIZE = 32

CANDIDATES = {
    'K6(75%)_narrow7.5+NLOS':  dict(K=6, half_angle_deg=7.5,  los=False, indoor_probability=0.0),
    'K8(100%)_narrow7.5+NLOS': dict(K=8, half_angle_deg=7.5,  los=False, indoor_probability=0.0),
    'K6(75%)_narrow15+NLOS':   dict(K=6, half_angle_deg=15.0, los=False, indoor_probability=0.0),
    'K6(75%)_wide30+NLOS':     dict(K=6, half_angle_deg=30.0, los=False, indoor_probability=0.0),
    'K6(75%)_narrow7.5+LOSnat':dict(K=6, half_angle_deg=7.5,  los=None,  indoor_probability=0.0),
}


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


results = {}
for name, cfg in CANDIDATES.items():
    K = cfg['K']
    print(f'\n{"="*75}\n{name}  M={M} K={K}  {cfg}\n{"="*75}')
    t0 = time.time()
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['half_angle_deg'],
                                     los=cfg['los'], indoor_probability=cfg['indoor_probability'])
    system.new_topology(200)
    h_freq, _ = system.channel_and_no(tf.constant(200, tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()))

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, BATCH_SIZE)
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}')

    max_gap = max(g[2] for g in gaps.values())
    results[name] = {'K': K, 'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()},
                      'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)')
    del system
    tf.keras.backend.clear_session()

print(f'\n\n{"="*75}\nRÉSUMÉ (loading ratio x NLOS/angle)\n{"="*75}')
print(f'{"config":<28} {"rho@48sc":>9} {"rho@95sc":>9} {"max_gap%":>9} {"pts>0.3%":>9}')
for name, r in results.items():
    c = r['corr']
    print(f'{name:<28} {c[48]:>9.3f} {c[95]:>9.3f} {r["max_gap_pct"]:>9.2f} {r["n_points_with_gap_gt_0.3pct"]:>9d}')

with open('results/diag_new_channel_search2.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_new_channel_search2.json')
