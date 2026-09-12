"""
diag_wmmse_nlos_check.py — Étape 1 (priorité) : revalide le fix WMMSE
(bisection exacte sur mu, Shi et al. 2011 Algorithm 1) SOUS NLOS.
Le fix du 29/07 n'a été instrumenté/validé que sous LOS verrouillé.

Deux vérifications :
  A. Non-décroissance du WSR interne à CHAQUE itération (reproduit le
     diagnostic qui avait détecté le bug de renormalisation a posteriori
     -- 14/14 violations avant fix, 0/14 après, sous LOS). Sous NLOS ici.
  B. Gap RZF-WMMSE à 10 vs 50 itérations : doit rester quasi stable
     (WMMSE >= RZF de plus en plus, pas l'inverse) si l'algorithme
     converge correctement.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_wmmse_nlos_check.py
"""
import os, sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from precoders_w import _get_desired_channels, rzf_precoder
from sionna.phy.mimo import rzf_precoding_matrix

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

M, K = 8, 4
BATCH_SIZE = 8
SNR_TEST_DB = 15.0   # zone où le mal-conditionnement est modéré -- reproductible
NUM_BATCHES_INSTRUMENT = 3


def wmmse_instrumented(h_freq, no, stream_management, num_iterations,
                        bisection_iters=30, bisection_doublings=30):
    """Copie de precoders_w.wmmse_precoder, +trace du WSR interne à chaque itération."""
    h_pc = _get_desired_channels(h_freq, stream_management)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    s = tf.shape(H)
    Kd = s[3]
    K_float = tf.cast(Kd, tf.float64)
    no_scalar = tf.cast(tf.reshape(no, []), tf.float64)
    no_val = tf.reshape(no_scalar, [1, 1, 1, 1])

    V = tf.cast(tf.squeeze(rzf_precoding_matrix(h_pc, alpha=tf.cast(no_scalar, tf.float32)), axis=1), tf.complex128)

    wsr_trace = []
    for it in range(num_iterations):
        HV = tf.matmul(H, V)
        signal = tf.linalg.diag_part(HV)
        tot_pwr = tf.reduce_sum(tf.abs(HV) ** 2, axis=-1)
        denom = tf.cast(tot_pwr + no_val, H.dtype)
        U = tf.math.conj(signal) / denom
        mse = 1.0 - tf.math.real(U * signal)
        mse = tf.maximum(mse, 1e-12)
        wsr = -tf.reduce_sum(tf.math.log(mse) / tf.math.log(tf.constant(2.0, tf.float64)), axis=-1)  # bps/Hz-like, somme sur users
        wsr_trace.append(float(tf.reduce_mean(wsr)))

        W = 1.0 / tf.maximum(mse, 1e-7)
        weights = tf.cast(W * tf.abs(U) ** 2, H.dtype)
        H_H = tf.linalg.adjoint(H)
        weighted_HH = H_H * weights[:, :, :, None, :]
        A = tf.matmul(weighted_HH, H)
        target = tf.cast(W, H.dtype) * tf.math.conj(U)
        B_rhs = H_H * target[:, :, :, None, :]

        with tf.device('/CPU:0'):
            sigma, Q = tf.linalg.eigh(A)
        sigma = tf.maximum(tf.math.real(sigma), 0.0)
        C = tf.matmul(Q, B_rhs, adjoint_a=True)
        p = tf.reduce_sum(tf.abs(C) ** 2, axis=-1)

        def power_at(mu):
            denom_m = tf.maximum(sigma + mu, 1e-30)
            return tf.reduce_sum(p / (denom_m ** 2), axis=-1, keepdims=True)

        lo = tf.zeros_like(sigma[..., :1])
        hi = tf.fill(tf.shape(sigma[..., :1]), tf.constant(1.0, tf.float64))
        for _ in range(bisection_doublings):
            hi = tf.where(power_at(hi) > K_float, hi * 2.0, hi)
        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            too_much = power_at(mid) > K_float
            lo = tf.where(too_much, mid, lo)
            hi = tf.where(too_much, hi, mid)
        mu_star = 0.5 * (lo + hi)
        inv_diag = 1.0 / tf.maximum(sigma + mu_star, 1e-30)
        V = tf.matmul(Q, tf.cast(inv_diag, H.dtype)[..., None] * C)

    return V, wsr_trace


print('=== A. Non-décroissance du WSR interne, sous NLOS (narrow7.5+NLOS, M8K4) ===')
system = ConfigurableMIMOSystem(M, K, half_angle_deg=7.5, los=False, indoor_probability=0.0)
total_violations, total_checks = 0, 0
for b in range(NUM_BATCHES_INSTRUMENT):
    system.new_topology(BATCH_SIZE)
    h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), tf.constant(SNR_TEST_DB, tf.float32))
    _, wsr_trace = wmmse_instrumented(h_freq, no, system.sm, num_iterations=15)
    diffs = np.diff(wsr_trace)
    n_viol = int(np.sum(diffs < -1e-6))
    total_violations += n_viol
    total_checks += len(diffs)
    print(f'  batch {b}: WSR trace = ' + ' '.join(f'{v:.3f}' for v in wsr_trace))
    print(f'    violations (WSR décroît) : {n_viol}/{len(diffs)}')

print(f'\n  TOTAL : {total_violations}/{total_checks} violations de non-décroissance sous NLOS')
verdict_A = 'OK — WSR non-décroissant' if total_violations == 0 else 'PROBLÈME — WSR décroît par endroits'
print(f'  VERDICT A : {verdict_A}')

print('\n=== B. Gap RZF-WMMSE à 10 vs 50 itérations, sous NLOS ===')
from precoders_w import wmmse_precoder
for n_iter in [10, 50]:
    r_rzf, r_wmmse = [], []
    for b in range(NUM_BATCHES_INSTRUMENT):
        system.new_topology(BATCH_SIZE)
        h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), tf.constant(SNR_TEST_DB, tf.float32))
        g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
        g_wmmse = wmmse_precoder(h_freq, no, system.sm, num_iterations=n_iter)
        r_rzf.append(float(system._sum_rate(h_freq, g_rzf, no)))
        r_wmmse.append(float(system._sum_rate(h_freq, g_wmmse, no)))
    rzf_m, wmmse_m = np.mean(r_rzf), np.mean(r_wmmse)
    gap_pct = 100.0 * (wmmse_m - rzf_m) / max(rzf_m, 1e-6)
    print(f'  {n_iter:3d} iters : RZF={rzf_m:.3f}  WMMSE={wmmse_m:.3f}  gap={gap_pct:+.3f}%')

print('\n=== VERDICT FINAL ÉTAPE 1 ===')
print(f'  A (non-décroissance WSR) : {verdict_A}')
print('  B (gap 10 vs 50 iters)   : voir ci-dessus -- stable/petit == OK, grandit fort == problème')
