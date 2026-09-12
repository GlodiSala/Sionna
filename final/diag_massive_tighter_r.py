"""
diag_massive_tighter_r.py — dernier test ciblé : M=64,K=16 (le meilleur
K du round précédent) avec un rayon de cluster plus serré (R=5m, 10m),
justifié par la résolution angulaire plus fine d'un réseau à 64 antennes
(nécessite une proximité physique plus stricte pour produire une
corrélation inter-utilisateurs comparable à celle observée à M=8).
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

M, K, BATCH = 64, 16, 8
R_VALUES = [5.0, 10.0]

results = {}
for R in R_VALUES:
    print(f'\n{"="*75}\nMASSIVE M=64,K=16  R={R}m\n{"="*75}', flush=True)
    t0 = time.time()
    system = ClusteredSystem(M, K)
    system.cluster_radius_m = R
    system.new_topology(16)
    h_freq, _ = system.channel_and_no(tf.constant(16, tf.int32), tf.constant(10.0, tf.float32))
    corr = freq_correlation(h_freq.numpy())
    print('  Corrélation :', ' '.join(f'{l}sc={v:.3f}' for l, v in corr.items()), flush=True)

    gaps = rzf_wmmse_gap(system, SNR_POINTS, NUM_BATCHES, BATCH)
    n_points_with_gap = 0
    for snr, (rzf_m, wmmse_m, gap_pct) in gaps.items():
        flag = '  <-- gap' if gap_pct > 0.3 else ''
        if gap_pct > 0.3:
            n_points_with_gap += 1
        print(f'    SNR={snr:5.1f}dB | RZF={rzf_m:7.2f} | WMMSE={wmmse_m:7.2f} | gap={gap_pct:+.2f}%{flag}', flush=True)

    max_gap = max(g[2] for g in gaps.values())
    results[str(R)] = {'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()},
                        'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ M=64,K=16, rayons serrés ===')
for R, r in results.items():
    print(f'R={R}m: rho@95sc={r["corr"][95]:.3f} max_gap={r["max_gap_pct"]:+.2f}% pts={r["n_points_with_gap_gt_0.3pct"]}/5')

with open('results/diag_massive_tighter_r.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_massive_tighter_r.json')
