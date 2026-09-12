"""
diag_m32_final_round.py — Dernier tour : M=32 (massif au sens M≫K,
4x STANDARD, mentionné Chapitre 3), K=4 et K=8, rayons R=20/10/5m
(liberté totale sur R cette fois, tant que la sélectivité reste réelle).
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from diag_spatial_cluster_channel import ClusteredSystem, freq_correlation, rzf_wmmse_gap, SNR_POINTS, NUM_BATCHES

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

M = 32
TESTS = [(4, 20.0), (8, 20.0), (4, 10.0), (8, 10.0), (4, 5.0), (8, 5.0)]

results = {}
for K, R in TESTS:
    batch = 16
    name = f'K={K},R={R}m'
    print(f'\n{"="*75}\nM=32,{name}  M/K={M/K:.1f}\n{"="*75}', flush=True)
    t0 = time.time()
    system = ClusteredSystem(M, K)
    system.cluster_radius_m = R
    system.new_topology(32)
    h_freq, _ = system.channel_and_no(tf.constant(32, tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()), flush=True)

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, batch)
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    max_gap = max(g[2] for g in gaps.values())
    results[name] = {'K': K, 'R': R, 'M_over_K': M / K, 'corr': corr,
                      'gaps': {str(k): v for k, v in gaps.items()},
                      'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ M=32, dernier tour ===')
print(f'{"K,R":<14} {"M/K":>6} {"rho@95sc":>9} {"max_gap%":>9} {"pts>0.3%":>9}')
for name, r in results.items():
    print(f'{name:<14} {r["M_over_K"]:>6.1f} {r["corr"][95]:>9.3f} {r["max_gap_pct"]:>9.2f} {r["n_points_with_gap_gt_0.3pct"]:>9d}')

with open('results/diag_m32_final_round.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_m32_final_round.json')
