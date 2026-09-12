"""
diag_spatial_cluster_channel.py — Étape suivante (diagnostic proximité
spatiale) : au lieu de restreindre l'angle du secteur global (déjà
testé, insuffisant sous NLOS), on rapproche les UTILISATEURS ENTRE EUX
en les tirant dans un petit disque autour d'un centre de cluster tiré
aléatoirement dans le secteur -- augmente leur corrélation spatiale
mutuelle sans changer la distribution globale (BS-secteur) des clusters.

Raisonnement physique : sous NLOS, le canal de chaque UT est dominé par
son propre tirage de clusters/rayons multipath (indépendant), donc la
seule proximité angulaire GLOBALE (vue du BS) ne suffit pas à corréler
les vecteurs canal -- il faut que les UTILISATEURS SOIENT PHYSIQUEMENT
PROCHES les uns des autres pour que leurs multipaths se ressemblent
(diffuseurs locaux partagés, spatial consistency 3GPP).

Teste plusieurs rayons de cluster R (5/15/30/60m) sur STANDARD (M8K4)
d'abord, scénario NLOS (los=False), sector 120° standard sinon.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_spatial_cluster_channel.py
"""
import os, sys, time, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from sionna.phy.channel.utils import relocate_uts, random_ut_properties, set_3gpp_scenario_parameters
from sionna.phy.config import config

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

PI = np.pi
SNR_POINTS = [0.0, 5.0, 10.0, 15.0, 20.0]
NUM_BATCHES = 10
CLUSTER_RADII = [5.0, 15.0, 30.0, 60.0]   # metres -- 60m ~= comportement large (contrôle)


def gen_topology_clustered(batch_size, num_ut, scenario, cluster_radius_m,
                            indoor_probability=0.0):
    """K utilisateurs tirés dans un disque de rayon cluster_radius_m autour
    d'un centre de cluster tiré uniformément dans le secteur 120° standard
    -- le centre suit la même distribution que gen_single_sector_topology,
    seule la dispersion AUTOUR du centre change."""
    (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
     indoor_probability, min_ut_velocity, max_ut_velocity) = \
        set_3gpp_scenario_parameters(scenario, indoor_probability=indoor_probability)

    rdtype = config.tf_rdtype
    bs_loc = tf.stack([tf.zeros([batch_size, 1], rdtype),
                        tf.zeros([batch_size, 1], rdtype),
                        tf.fill([batch_size, 1], bs_height)], axis=-1)
    sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
    bs_downtilt = 0.5 * PI - tf.math.atan(sector_center / bs_height)
    bs_yaw = tf.constant(PI / 3.0, rdtype)
    bs_orientation = tf.stack([tf.fill([batch_size, 1], bs_yaw),
                                tf.fill([batch_size, 1], bs_downtilt),
                                tf.zeros([batch_size, 1], rdtype)], axis=-1)

    half_angle = tf.constant(60.0 * PI / 180.0, rdtype)   # secteur 120° standard
    d_min = tf.cast(min_bs_ut_dist, rdtype)
    r_max = tf.cast(isd * 0.5, rdtype)

    # centre de cluster : 1 par batch, distribution standard (uniform-area)
    alpha_c = config.tf_rng.uniform([batch_size, 1], minval=-half_angle, maxval=half_angle, dtype=rdtype)
    dist2_c = config.tf_rng.uniform([batch_size, 1], minval=d_min**2, maxval=r_max**2, dtype=rdtype)
    dist_c = tf.sqrt(dist2_c)
    center_xy = tf.stack([dist_c * tf.math.cos(alpha_c), dist_c * tf.math.sin(alpha_c)], axis=-1)[:, 0, :]  # [B,2]

    # offsets utilisateurs : disque uniforme de rayon cluster_radius_m autour du centre
    R = tf.constant(cluster_radius_m, rdtype)
    ang_off = config.tf_rng.uniform([batch_size, num_ut], minval=0.0, maxval=2 * PI, dtype=rdtype)
    rad_off = R * tf.sqrt(config.tf_rng.uniform([batch_size, num_ut], minval=0.0, maxval=1.0, dtype=rdtype))
    off_x = rad_off * tf.math.cos(ang_off)
    off_y = rad_off * tf.math.sin(ang_off)
    ut_x = center_xy[:, 0:1] + off_x
    ut_y = center_xy[:, 1:2] + off_y

    # garde-fou : distance minimale au BS (rare, cluster proche du BS)
    dist_bs = tf.sqrt(ut_x**2 + ut_y**2)
    scale = tf.maximum(1.0, d_min / tf.maximum(dist_bs, 1e-3))
    ut_x, ut_y = ut_x * scale, ut_y * scale

    ut_loc_xy = tf.stack([ut_x, ut_y], axis=-1)
    ut_loc_z = config.tf_rng.uniform([batch_size, num_ut, 1], minval=min_ut_height, maxval=max_ut_height, dtype=rdtype)
    ut_loc = tf.concat([ut_loc_xy, ut_loc_z], axis=-1)

    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size, num_ut, indoor_probability, min_ut_velocity, max_ut_velocity)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


class ClusteredSystem(ConfigurableMIMOSystem):
    cluster_radius_m = 15.0
    def new_topology(self, batch_size):
        topology = gen_topology_clustered(batch_size, self.num_rx, 'umi',
                                           self.cluster_radius_m, indoor_probability=0.0)
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


def _run_sweep():
    M, K, BATCH_SIZE = 8, 4, 16
    results = {}
    for R in CLUSTER_RADII:
        print(f'\n{"="*75}\nSTANDARD M=8,K=4 -- rayon de cluster R={R}m\n{"="*75}', flush=True)
        t0 = time.time()
        system = ClusteredSystem(M, K)
        system.cluster_radius_m = R
        system.new_topology(32)
        h_freq, _ = system.channel_and_no(tf.constant(32, tf.int32), tf.constant(10.0, tf.float32))
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
        results[str(R)] = {'corr': corr, 'gaps': {str(k): v for k, v in gaps.items()},
                            'max_gap_pct': max_gap, 'n_points_with_gap_gt_0.3pct': n_points_with_gap}
        print(f'  -> max_gap={max_gap:+.2f}% | points avec gap>0.3%: {n_points_with_gap}/5 | ({time.time()-t0:.0f}s)', flush=True)
        del system
        tf.keras.backend.clear_session()

    print(f'\n\n=== RÉSUMÉ PROXIMITÉ SPATIALE (STANDARD M8K4, NLOS) ===')
    print(f'{"R(m)":>6} {"rho@48sc":>9} {"rho@95sc":>9} {"max_gap%":>9} {"pts>0.3%":>9}')
    for R, r in results.items():
        c = r['corr']
        print(f'{R:>6} {c[48]:>9.3f} {c[95]:>9.3f} {r["max_gap_pct"]:>9.2f} {r["n_points_with_gap_gt_0.3pct"]:>9d}')

    with open('results/diag_spatial_cluster_channel.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSauvé -> results/diag_spatial_cluster_channel.json')


if __name__ == '__main__':
    _run_sweep()
