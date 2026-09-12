"""
diag_wmmse_iters_check.py — Diagnostic direct pour la question du directeur
sur Tableau 3.9 (M=8, K=4, canal UMi verrouille, WMMSE_ITERS=10 vs 100).

Ne modifie AUCUN fichier existant. Reutilise EXACTEMENT le meme mecanisme
que classical_comparison.py (LockedClusterSystem, STANDARD_CONFIG, meme
BATCH_SIZE/NUM_BATCHES, meme boucle sur les 9 points de SNR_RANGE_DB dans
le meme ordre) pour que les tirages de canal soient IDENTIQUES a ceux qui
ont produit results/classical_comparison_M8K4.npy -- le seed=42 est pose
au moment de l'import de wmmse_convergence_check (effet de bord), et la
sequence d'appels config.tf_rng.uniform(...) ne depend que de l'ordre des
appels new_topology()/channel_and_no(), pas des precodeurs eux-memes (qui
ne consomment aucun alea). On boucle donc sur les 9 SNR (pas seulement les
5 demandes) pour ne pas desynchroniser l'etat du RNG.

Rapporte, sans interpretation :
  1. Sum-rate RZF-Full / WMMSE@10 / WMMSE@100 pour les 5 SNR 0/5/10/15/20 dB
  2. Trace du WSR interne a chaque iteration (jusqu'a 100) pour SNR=0dB et
     SNR=20dB, 1er batch (meme methode que diag_wmmse_nlos_check.py)
  3. Allocation de puissance moyenne par utilisateur (RZF / WMMSE@10 /
     WMMSE@100), moyennee sur tous les batches, pour les 5 SNR demandes

Usage: LD_LIBRARY_PATH=<conda_env>/lib python3 diag_wmmse_iters_check.py
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

# Import chain identique a classical_comparison.py -- pose le seed=42 via
# l'effet de bord du module wmmse_convergence_check (tf.random.set_seed(42),
# np.random.seed(42)) AVANT tout tirage de canal.
from classical_comparison import LockedClusterSystem, BATCH_SIZE, NUM_BATCHES, RB_SIZE
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from channel_config import STANDARD_CONFIG, SNR_RANGE_DB
from sionna.phy.mimo import rzf_precoding_matrix

REPORT_SNRS  = [0.0, 5.0, 10.0, 15.0, 20.0]
TRACE_SNRS   = [0.0, 20.0]
WMMSE_ITERS_A = 10
WMMSE_ITERS_B = 100
TRACE_ITERS   = 100

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']


def user_power(g):
    """g: [B,1,ofdm,fft,M,K] -> moyenne (sur B,ofdm,fft) de ||v_k||^2 par
    utilisateur k, forme [K]."""
    p = tf.reduce_sum(tf.abs(g) ** 2, axis=-2)         # [B,1,ofdm,fft,K]
    return tf.reduce_mean(p, axis=[0, 1, 2, 3]).numpy()  # [K]


def wmmse_instrumented(h_freq, no, stream_management, num_iterations):
    """Copie de precoders_w.wmmse_precoder (meme algo -- bisection exacte
    sur mu, Shi et al. 2011 Algorithm 1) + trace du WSR interne et de la
    puissance par utilisateur a CHAQUE iteration. Meme formule WSR que
    diag_wmmse_nlos_check.py : wsr = -sum_k log2(mse_k) (proxy interne a
    l'algo, pas la metrique LMMSEPostEqualizationSINR de _sum_rate)."""
    h_pc = _get_desired_channels(h_freq, stream_management)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    s = tf.shape(H)
    Kd = s[3]
    K_float = tf.cast(Kd, tf.float64)
    no_scalar = tf.cast(tf.reshape(no, []), tf.float64)
    no_val = tf.reshape(no_scalar, [1, 1, 1, 1])

    V = tf.cast(tf.squeeze(
        rzf_precoding_matrix(h_pc, alpha=tf.cast(no_scalar, tf.float32)), axis=1),
        tf.complex128)

    wsr_trace = []
    power_trace = []  # liste de [K] (puissance moyenne par user a cette iter)
    for it in range(num_iterations):
        HV = tf.matmul(H, V)
        signal = tf.linalg.diag_part(HV)
        tot_pwr = tf.reduce_sum(tf.abs(HV) ** 2, axis=-1)
        denom = tf.cast(tot_pwr + no_val, H.dtype)
        U = tf.math.conj(signal) / denom
        mse = 1.0 - tf.math.real(U * signal)
        mse = tf.maximum(mse, 1e-12)
        wsr = -tf.reduce_sum(tf.math.log(mse) / tf.math.log(tf.constant(2.0, tf.float64)), axis=-1)
        wsr_trace.append(float(tf.reduce_mean(wsr)))

        pwr_k = tf.reduce_sum(tf.abs(V) ** 2, axis=-2)  # [...,K] puissance par user, cette iter
        power_trace.append(tf.reduce_mean(pwr_k, axis=[0, 1, 2]).numpy().astype(np.float64))

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
        for _ in range(30):
            hi = tf.where(power_at(hi) > K_float, hi * 2.0, hi)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            too_much = power_at(mid) > K_float
            lo = tf.where(too_much, mid, lo)
            hi = tf.where(too_much, hi, mid)
        mu_star = 0.5 * (lo + hi)
        inv_diag = 1.0 / tf.maximum(sigma + mu_star, 1e-30)
        V = tf.matmul(Q, tf.cast(inv_diag, H.dtype)[..., None] * C)

    return V, wsr_trace, power_trace


def main():
    t0 = time.time()
    system = LockedClusterSystem(M, K)
    system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

    rates = {snr: {'rzf': [], 'wmmse10': [], 'wmmse100': []} for snr in REPORT_SNRS}
    powers = {snr: {'rzf': [], 'wmmse10': [], 'wmmse100': []} for snr in REPORT_SNRS}
    traces = {}

    for snr in SNR_RANGE_DB:
        snr_f = float(snr)
        is_report = snr_f in REPORT_SNRS
        snr_t = tf.constant(snr_f, dtype=tf.float32)

        for b in range(NUM_BATCHES):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)

            if not is_report:
                continue  # on tire quand meme le canal (RNG) mais on ne calcule rien

            g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            r_rzf = float(system._sum_rate(h_freq, g_rzf, no))

            g_w10 = wmmse_precoder(h_freq, no, system.sm, num_iterations=WMMSE_ITERS_A)
            r_w10 = float(system._sum_rate(h_freq, g_w10, no))

            g_w100 = wmmse_precoder(h_freq, no, system.sm, num_iterations=WMMSE_ITERS_B)
            r_w100 = float(system._sum_rate(h_freq, g_w100, no))

            rates[snr_f]['rzf'].append(r_rzf)
            rates[snr_f]['wmmse10'].append(r_w10)
            rates[snr_f]['wmmse100'].append(r_w100)

            powers[snr_f]['rzf'].append(user_power(g_rzf))
            powers[snr_f]['wmmse10'].append(user_power(g_w10))
            powers[snr_f]['wmmse100'].append(user_power(g_w100))

            if b == 0 and snr_f in TRACE_SNRS:
                _, wsr_trace, power_trace = wmmse_instrumented(
                    h_freq, no, system.sm, num_iterations=TRACE_ITERS)
                traces[snr_f] = {'wsr': wsr_trace, 'power': [p.tolist() for p in power_trace]}

            print(f"  SNR={snr_f:5.1f} batch={b} | RZF={r_rzf:.4f} "
                  f"WMMSE10={r_w10:.4f} WMMSE100={r_w100:.4f}", flush=True)

        if is_report:
            print(f"SNR={snr_f} dB done ({time.time()-t0:.0f}s elapsed)\n", flush=True)

    out = {
        'rates': {snr: {k: v for k, v in d.items()} for snr, d in rates.items()},
        'powers': {snr: {k: [p.tolist() for p in v] for k, v in d.items()} for snr, d in powers.items()},
        'traces': traces,
    }
    with open('diag_wmmse_iters_check_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print("\n=== RESUME SUM-RATE (moyenne sur batches) ===")
    print(f"{'SNR':>6} {'RZF':>10} {'WMMSE@10':>10} {'gap10%':>8} {'WMMSE@100':>10} {'gap100%':>8}")
    for snr in REPORT_SNRS:
        rzf_m = np.mean(rates[snr]['rzf'])
        w10_m = np.mean(rates[snr]['wmmse10'])
        w100_m = np.mean(rates[snr]['wmmse100'])
        g10 = 100 * (w10_m - rzf_m) / rzf_m
        g100 = 100 * (w100_m - rzf_m) / rzf_m
        print(f"{snr:6.1f} {rzf_m:10.4f} {w10_m:10.4f} {g10:+8.3f} {w100_m:10.4f} {g100:+8.3f}")

    print("\n=== ALLOCATION DE PUISSANCE PAR UTILISATEUR (moyenne sur batches) ===")
    for snr in REPORT_SNRS:
        print(f"\n  SNR={snr} dB")
        for meth in ('rzf', 'wmmse10', 'wmmse100'):
            arr = np.mean(np.stack(powers[snr][meth], axis=0), axis=0)
            print(f"    {meth:9s} : " + " ".join(f"user{k}={arr[k]:.4f}" for k in range(K))
                  + f"  (sum={arr.sum():.4f}, std={arr.std():.4f})")

    print("\n=== TRACE WSR INTERNE (100 iterations, 1er batch) ===")
    for snr in TRACE_SNRS:
        tr = traces[snr]['wsr']
        diffs = np.diff(tr)
        n_viol = int(np.sum(diffs < -1e-6))
        print(f"\n  SNR={snr} dB -- WSR[1..10] = " + " ".join(f"{v:.4f}" for v in tr[:10]))
        print(f"  SNR={snr} dB -- WSR[91..100] = " + " ".join(f"{v:.4f}" for v in tr[90:100]))
        print(f"  delta(WSR[99]-WSR[98]) = {tr[99]-tr[98]:.6f}   delta(WSR[9]-WSR[8]) = {tr[9]-tr[8]:.6f}")
        print(f"  violations non-decroissance sur 100 iters : {n_viol}/{len(diffs)}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("Resultats complets ecrits dans diag_wmmse_iters_check_results.json")


if __name__ == '__main__':
    main()
