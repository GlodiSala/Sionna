"""
diag_spatial_cluster_r20_k8.py — deux tests rapides demandés :
1. R=20m sur STANDARD (M8,K4) -- interpole entre 15 et 30
2. R=30m combiné à K=8 sur STANDARD (M=8,K=8, 100% loading) -- un seul point

Réutilise gen_topology_clustered de diag_spatial_cluster_channel.py.
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from diag_spatial_cluster_channel import (ClusteredSystem, freq_correlation, rzf_wmmse_gap, SNR_POINTS, NUM_BATCHES)

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

TESTS = [
    ('R=20m, K=4', dict(M=8, K=4, R=20.0, batch=16)),
    ('R=30m, K=8', dict(M=8, K=8, R=30.0, batch=16)),
]

results = {}
for name, cfg in TESTS:
    print(f'\n{"="*75}\n{name}\n{"="*75}', flush=True)
    t0 = time.time()
    system = ClusteredSystem(cfg['M'], cfg['K'])
    system.cluster_radius_m = cfg['R']
    system.new_topology(32)
    h_freq, _ = system.channel_and_no(tf.constant(32, tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()), flush=True)

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, cfg['batch'])
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    max_gap = max(g[2] for g in gaps.values())
    results[name] = {'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()},
                      'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ R=20 vs R=30+K=8 ===')
for name, r in results.items():
    print(f'{name}: rho@95sc={r["corr"][95]:.3f} max_gap={r["max_gap_pct"]:+.2f}% pts={r["n_points_with_gap_gt_0.3pct"]}/5')

with open('results/diag_spatial_cluster_r20_k8.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_spatial_cluster_r20_k8.json')
