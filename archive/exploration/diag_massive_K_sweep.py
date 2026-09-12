"""
diag_massive_K_sweep.py — Volet canal : sweep du ratio M/K à M=64 fixe
(K=4,8,16 ; K=32 déjà couvert par classical_comparison_newchannel),
canal narrow7.5+NLOS déjà choisi (sélectivité fréquentielle stable,
pas remesurée ici -- déjà confirmée indépendante du K testé).

Objectif : trouver le meilleur compromis M>>K (massif au sens propre)
vs gap RZF-WMMSE observable sur plusieurs points SNR.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_massive_K_sweep.py
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

M = 64
SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 10  # aligné sur la rigueur de classical_comparison.py (6-8 batches
                   # se sont révélés trop bruités pour distinguer un vrai gap du bruit)
BATCH_SIZE = 16
HALF_ANGLE_DEG, LOS, INDOOR = 7.5, False, 0.0   # canal déjà verrouillé

K_VALUES = [4, 8, 16]


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
for K in K_VALUES:
    print(f'\n{"="*75}\nM={M} K={K}  (M/K={M/K:.1f})\n{"="*75}', flush=True)
    t0 = time.time()
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=HALF_ANGLE_DEG,
                                     los=LOS, indoor_probability=INDOOR)
    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, BATCH_SIZE)
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    max_gap = max(g[2] for g in gaps.values())
    results[str(K)] = {'M': M, 'K': K, 'M_over_K': M / K,
                        'gaps': {str(k): v for k, v in gaps.items()},
                        'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> M/K={M/K:.1f} | max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ SWEEP M/K (M=64, canal narrow7.5+NLOS) ===')
print(f'{"K":>4} {"M/K":>6} {"max_gap%":>9} {"pts>0.3%":>9} | gap par SNR')
for K, r in results.items():
    g = r['gaps']
    gaps_str = ' '.join(f'{g[str(s)][2]:+.2f}' for s in SNR_POINTS)
    print(f'{K:>4} {r["M_over_K"]:>6.1f} {r["max_gap_pct"]:>9.2f} {r["n_points_with_gap_gt_0.3pct"]:>9d} | {gaps_str}')

with open('results/diag_massive_K_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_massive_K_sweep.json')
