"""
diag_csi_imperfect_massive_2regimes_seedfix.py -- CIBLE 3 (regimes 2 et 3) :
version corrigee de diag_csi_imperfect_massive_uma_capD384L4.py (UMa-massif
D=384) et diag_csi_imperfect_massive_extbudget_sweep.py (UMi-massif D=128,
CSI imparfait), memes corrections que classical_comparison_seedfix_v2.py :
1) sionna_config.seed = SEED (config.tf_rng n'est pas seede par
   tf.random.set_seed).
2) double metrique sur les MEMES tirages : Sionna (_sum_rate/
   LMMSEPostEqualizationSINR) ET SINR manuel direct (sans lmmse_matrix).
Canal deja partage dans les scripts originaux (h_true tire une fois par
draw, reutilise pour RZF/WMMSE/3 reseaux) -- rien a changer sur ce point.
Checkpoints deja verifies sains (epoch 170 pour D384 UMa, extbudget epoch
160 pour D128 UMi) -- pas de bug 3 ici.

Ne modifie aucun fichier existant.
"""
import os, sys, json, pickle, time, hashlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from precoders_w import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_DRAWS = 6      # allege vs 12 original -- compromis temps/disque pour cette worktree
BATCH = 16
RNG_SEED = 2026
METHODS = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']

REGIMES = {
    'uma_massive_D384': {
        'embed_dim': 384, 'tokens_per_rb': 6,
        'scenario': 'uma',
        'ckpt_jsons': {
            'SingleSC': 'results/diag_massive_gap_fullbudget_uma_single_sc_cap_d384l4.json',
            'IntraRB':  'results/diag_massive_gap_fullbudget_uma_intra_rb_cap_d384l4.json',
            'TA_RB_residual': 'results/diag_massive_gap_fullbudget_uma_ta_rb_residual_cap_d384l4.json',
        },
    },
    'umi_massive_D128': {
        'embed_dim': 128, 'tokens_per_rb': 6,
        'scenario': 'umi',
        'ckpt_jsons': {
            'SingleSC': 'results/diag_massive_true_single_sc_extbudget.json',
            'IntraRB':  'results/diag_massive_true_intra_rb_extbudget.json',
            'TA_RB_residual': 'results/diag_massive_true_ta_rb_residual_extbudget.json',
        },
    },
}


def resolve_ckpt(json_path):
    with open(json_path) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable: {ckpt}"
    return ckpt


def build_neural(kind, embed_dim, tokens_per_rb):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=embed_dim, num_heads=4, num_layers=4)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=tokens_per_rb, **kwargs)
    raise ValueError(kind)


def make_system(scenario):
    if scenario == 'uma':
        from sionna.phy.channel.tr38901 import UMa

        class LockedSystemUMa(ConfigurableMIMOSystem):
            cluster_radius_m = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
            indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

            def __init__(self, num_tx, num_rx):
                super().__init__(num_tx, num_rx)
                self.channel_model = UMa(
                    carrier_frequency=2.6e9, o2i_model='low',
                    ut_array=self.ut_array, bs_array=self.bs_array,
                    direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

            def new_topology(self, batch_size):
                set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                                     cluster_radius_m=self.cluster_radius_m,
                                     force_los=MASSIVE_TRUE_CONFIG['FORCE_LOS'],
                                     indoor_probability=self.indoor_probability)
        return LockedSystemUMa(M, K)

    class LockedSystem(ConfigurableMIMOSystem):
        cluster_radius_m = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
        indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

        def new_topology(self, batch_size):
            set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                                 cluster_radius_m=self.cluster_radius_m,
                                 force_los=MASSIVE_TRUE_CONFIG['FORCE_LOS'],
                                 indoor_probability=self.indoor_probability)
    return LockedSystem(M, K)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


def manual_sinr_rate(system, h_true, g, no):
    h_pc = _get_desired_channels(h_true, system.sm)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal = tf.linalg.diag_part(HW)
    signal_pwr = tf.abs(signal) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    no_val = tf.cast(tf.reshape(no, []), tf.float64)
    sinr = signal_pwr / (interference_pwr + no_val)
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    Y = tf.reduce_mean(rate, axis=[1, 2])
    return float(tf.reduce_sum(tf.reduce_mean(Y, axis=0)))


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_regime(regime_name, cfg):
    t0 = time.time()
    print(f"\n{'='*70}\nREGIME {regime_name}\n{'='*70}", flush=True)
    system = make_system(cfg['scenario'])
    D = cfg['embed_dim']

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    neural_models, ckpts = {}, {}
    for name, jpath in cfg['ckpt_jsons'].items():
        ckpt = resolve_ckpt(jpath)
        ckpts[name] = ckpt
        no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        m = build_neural(name, D, cfg['tokens_per_rb'])
        load_weights(m, ckpt, dummy_h, no0)
        neural_models[name] = m
        print(f'OK {name} (D={D}) charge depuis {ckpt}', flush=True)

    rng = np.random.RandomState(RNG_SEED)
    sionna_out = {m: {} for m in METHODS}
    manual_out = {m: {} for m in METHODS}

    for data_snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(data_snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_sionna = {m: [] for m in METHODS}
        r_manual = {m: [] for m in METHODS}
        for draw in range(NUM_DRAWS):
            system.new_topology(BATCH)
            h_true, _ = system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(data_snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_rzf = rzf_precoder(h_est, stream_management=system.sm, no=no)
            r_sionna['RZF'].append(float(system._sum_rate(h_true, g_rzf, no)))
            r_manual['RZF'].append(manual_sinr_rate(system, h_true, g_rzf, no))

            g_wmmse = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            r_sionna['WMMSE'].append(float(system._sum_rate(h_true, g_wmmse, no)))
            r_manual['WMMSE'].append(manual_sinr_rate(system, h_true, g_wmmse, no))

            for name, model in neural_models.items():
                g_n = model(h_est, no=no, training=False)
                r_sionna[name].append(float(system._sum_rate(h_true, g_n, no)))
                r_manual[name].append(manual_sinr_rate(system, h_true, g_n, no))

        for m in METHODS:
            sionna_out[m][data_snr] = float(np.mean(r_sionna[m]))
            manual_out[m][data_snr] = float(np.mean(r_manual[m]))
        print(f"SNR={data_snr:5.1f} | " + " ".join(
            f"{m}:{sionna_out[m][data_snr]:.2f}/{manual_out[m][data_snr]:.2f}" for m in METHODS), flush=True)

    meta = {
        'seed': SEED, 'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True, 'regime': regime_name, 'scenario': cfg['scenario'],
        'embed_dim': D, 'tokens_per_rb': cfg['tokens_per_rb'],
        'pilot_snr_db': PILOT_SNR_DB, 'data_snrs': DATA_SNRS_DB,
        'num_draws': NUM_DRAWS, 'batch': BATCH,
        'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of('precoders_w.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'checkpoints': ckpts, 'M': M, 'K': K,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }
    date_tag = time.strftime('%Y%m%d_%H%M%S')
    out_sionna = {'metric': 'sionna_LMMSEPostEqualizationSINR (_sum_rate)', 'table': sionna_out, 'metadata': meta}
    out_manual = {'metric': 'manual_sinr', 'table': manual_out, 'metadata': meta}
    p_sionna = f'results/diag_csi_imperfect_{regime_name}_seedfix_sionna_{date_tag}.json'
    p_manual = f'results/diag_csi_imperfect_{regime_name}_seedfix_manual_sinr_{date_tag}.json'
    with open(p_sionna, 'w') as f: json.dump(out_sionna, f, indent=2)
    with open(p_manual, 'w') as f: json.dump(out_manual, f, indent=2)
    print(f"{regime_name}: sauvegarde -> {p_sionna}\n              -> {p_manual}")
    print(f"{regime_name}: time {time.time()-t0:.0f}s")
    return p_sionna, p_manual, sionna_out, manual_out


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--regimes', nargs='+', default=list(REGIMES.keys()))
    args = p.parse_args()
    for r in args.regimes:
        run_regime(r, REGIMES[r])
