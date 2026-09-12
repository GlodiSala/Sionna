"""
diag_new_channel_massive.py — Vérifie le candidat gagnant (narrow7.5 +
NLOS + loading 100%, K=M) à l'échelle MASSIVE (M=64), pour confirmer
qu'il tient aux deux échelles avant verrouillage.

Teste aussi M=64,K=32 (50% loading, le ratio "déjà en place") en
comparaison, pour voir si le loading 100% est nécessaire à cette
échelle aussi (attendu, vu le channel-hardening documenté dans
channel_config.py à grand M).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_new_channel_massive.py
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
NUM_BATCHES = 6
BATCH_SIZE = 8    # M=64 -> matrices plus grosses, batch réduit pour rester léger
                  # (encore réduit de 16->8 : GPU0 partagé avec le job LR schedules, ~24GB déjà pris)

CANDIDATES = {
    'M64K32(50%)_narrow7.5+NLOS':  dict(K=32, half_angle_deg=7.5, los=False, indoor_probability=0.0),
    'M64K64(100%)_narrow7.5+NLOS': dict(K=64, half_angle_deg=7.5, los=False, indoor_probability=0.0),
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
    print(f'\n{"="*75}\n{name}  M={M} K={K}\n{"="*75}', flush=True)
    t0 = time.time()
    system = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['half_angle_deg'],
                                     los=cfg['los'], indoor_probability=cfg['indoor_probability'])
    system.new_topology(100)
    h_freq, _ = system.channel_and_no(tf.constant(100, tf.int32), tf.constant(10.0, tf.float32))
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
    results[name] = {'K': K, 'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()},
                      'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
    print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
    del system
    tf.keras.backend.clear_session()

print(f'\n\n=== RÉSUMÉ MASSIVE (M=64) ===')
for name, r in results.items():
    print(f'{name}: rho@95sc={r["corr"][95]:.3f} max_gap={r["max_gap_pct"]:+.2f}% pts={r["n_points_with_gap_gt_0.3pct"]}/5')

with open('results/diag_new_channel_massive.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSauvé -> results/diag_new_channel_massive.json')
