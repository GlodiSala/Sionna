"""
experiments/eval_csi_conserve_4regimes.py -- calcule le "debit conserve"
(tab:results_csi, tab:results_csi_4regimes) pour les 4 regimes du memoire
(UMi validation, UMa validation, UMi massif D384, UMa massif D384) avec la
propriete qui manquait a TOUS les scripts CSI-imparfait existants
(experiments/eval_csi_imperfect_*.py) : ces scripts calculent le precodeur
sur h_est et evaluent sur h_true, mais ne calculent PAS aussi le precodeur
sur h_true (condition CSI parfait) DANS LE MEME TIRAGE -- le "debit
conserve" publie dans le memoire vient donc de deux scripts separes, avec
des graines et/ou des points de fonctionnement TA-RB differents (verifie :
validation UMi = seed42/T4 cote parfait vs seed2026/T6 cote imparfait ;
validation UMa = memes checkpoints mais seed42 vs seed2026 ; massif UMi
D384 = seed42 des deux cotes mais num_batches=5 vs num_draws=4, scripts
differents ; massif UMa D384 = pas de fichier CSI-parfait seedfix du tout).

Ce script fait les deux conditions dans LA MEME boucle, sur LE MEME tirage
h_true, une seule graine par regime :
    g_parfait   = precodeur(h_true, ...)
    g_imparfait = precodeur(h_est,  ...)          h_est = noisy_channel(h_true, pilote 20dB)
    debit_parfait[m,snr]   += manual_sinr_rate(h_true, g_parfait,   no)
    debit_imparfait[m,snr] += manual_sinr_rate(h_true, g_imparfait, no)
debit_conserve[m,snr] = 100 * mean(debit_imparfait) / mean(debit_parfait)

Metrique : manual_sinr uniquement (SINR_k = |h_k^H w_k|^2 /
(sum_{j!=k}|h_k^H w_j|^2 + no), eq. 2.2 du memoire, precoders/classical.py),
PAS la metrique Sionna _sum_rate -- coherent avec les autres tableaux du
chapitre 3.

Points de controle (repris tels quels des runs perfect-CSI existants
lorsqu'ils sont coherents avec le memoire ; corrige pour UMi validation,
voir REGIMES['umi_standard']) :
  - validation (UMi, UMa) : TA-RB a T=4
  - massif (UMi, UMa)     : TA-RB a T=6, D=384

N'ecrase AUCUN fichier existant : ecrit results/eval_csi_conserve_<regime>_<timestamp>.json

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/eval_csi_conserve_4regimes.py [--regimes r1 r2 ...]
"""
import os, sys, json, pickle, time, hashlib, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 2026
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

from sionna.phy.channel.tr38901 import UMa
from sionna.phy.utils import ebnodb2no
from eval_system import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, MASSIVE_TRUE_CONFIG, set_locked_topology, gen_topology_clustered
from precoders.classical import rzf_precoder, wmmse_precoder, _get_desired_channels
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
METHODS = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB', 'TA_RB_residual']
NAME_MAP = {'SingleSC': 'SC', 'IntraRB': 'IB', 'TA_RB_residual': 'TA-RB'}


class LockedSystemUMi(ConfigurableMIMOSystem):
    def __init__(self, num_tx, num_rx, cfg):
        super().__init__(num_tx, num_rx)
        self.cluster_radius_m = cfg['CLUSTER_RADIUS_M']
        self.indoor_probability = cfg['INDOOR_PROBABILITY']
        self.force_los = cfg.get('FORCE_LOS', False)

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=self.force_los,
                             indoor_probability=self.indoor_probability)


class LockedSystemUMa(ConfigurableMIMOSystem):
    def __init__(self, num_tx, num_rx, cfg):
        super().__init__(num_tx, num_rx)
        self.cluster_radius_m = cfg['CLUSTER_RADIUS_M']
        self.indoor_probability = cfg['INDOOR_PROBABILITY']
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model='low',
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        topology = gen_topology_clustered(
            batch_size, self.num_rx, 'uma', self.cluster_radius_m,
            indoor_probability=self.indoor_probability)
        self.channel_model.set_topology(*topology, los=False)


REGIMES = {
    'umi_standard': {
        'label': 'UMi validation (M=8,K=4)',
        'system_cls': LockedSystemUMi, 'cfg': STANDARD_CONFIG,
        'embed_dim': 128, 'tokens_per_rb': 4, 'num_draws': 20, 'batch': 32,
        'checkpoints': {
            'SingleSC': 'weights/front_a_signed_attn/best_20260808_103631',
            'IntraRB': 'weights/IntraRB_signed_attn_4L_128d/best_20260808_113308',
            'TA_RB_residual': 'weights/TA_RB_residual_signed_attn_T4_4L_128d/best_20260808_181436',
        },
        'checkpoint_note': ('CORRIGE : experiments/eval_csi_imperfect_umi_standard.py utilise le '
                             'checkpoint TA-RB a T=6 (tokens_per_rb=6, '
                             'diag_front_a_signed_attn_ta_rb_residual.json) pour ce regime, alors que '
                             'le point de controle memoire pour la validation est T=4 -- meme '
                             'checkpoint que classical_comparison_M8K4_seedfix (T4_4L_128d/'
                             'best_20260808_181436). Utilise T=4 ici.'),
    },
    'uma_standard': {
        'label': 'UMa validation (M=8,K=4)',
        'system_cls': LockedSystemUMa, 'cfg': STANDARD_CONFIG,
        'embed_dim': 128, 'tokens_per_rb': 4, 'num_draws': 20, 'batch': 32,
        'checkpoints': {
            'SingleSC': 'weights/UMa_SingleSC_signed_attn_4L_128d/best_20260808_231408',
            'IntraRB': 'weights/UMa_IntraRB_signed_attn_4L_128d/best_20260809_005458',
            'TA_RB_residual': 'weights/UMa_TA_RB_residual_signed_attn_4tok_4L_128d/best_20260809_001031',
        },
        'checkpoint_note': None,
    },
    'umi_massive_D384': {
        'label': 'UMi massif (M=64,K=8,D=384)',
        'system_cls': LockedSystemUMi, 'cfg': MASSIVE_TRUE_CONFIG,
        'embed_dim': 384, 'tokens_per_rb': 6, 'num_draws': 12, 'batch': 16,
        'checkpoints': {
            'SingleSC': 'weights/MASSIVE_TRUE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_014548',
            'IntraRB': 'weights/MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_021303',
            'TA_RB_residual': 'weights/MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_022738',
        },
        'checkpoint_note': None,
    },
    'uma_massive_D384': {
        'label': 'UMa massif (M=64,K=8,D=384)',
        'system_cls': LockedSystemUMa, 'cfg': MASSIVE_TRUE_CONFIG,
        'embed_dim': 384, 'tokens_per_rb': 6, 'num_draws': 12, 'batch': 16,
        'checkpoints': {
            'SingleSC': 'weights/UMa_MASSIVE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_231346',
            'IntraRB': 'weights/UMa_MASSIVE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_233842',
            'TA_RB_residual': 'weights/UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_235531',
        },
        'checkpoint_note': ("Aucun fichier resultat CSI-parfait seedfix n'existait pour ce regime "
                             "avant ce script -- valeurs 'parfait' produites ici pour la premiere fois."),
    },
}


def build_neural(kind, M, K, D, L, tokens_per_rb):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=D, num_heads=4, num_layers=L)
    if kind == 'SingleSC':
        return SingleSCTransformerPrecoderSignedAttn(**kwargs)
    if kind == 'IntraRB':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kwargs)
    if kind == 'TA_RB_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=tokens_per_rb, **kwargs)
    raise ValueError(kind)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def noisy_channel(h_freq, pilot_snr_db, rng):
    sigma2 = 1.0 / 10.0 ** (pilot_snr_db / 10.0)
    nre = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    nim = rng.normal(0, np.sqrt(sigma2 / 2), size=h_freq.shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(nre, nim), h_freq.dtype)


def manual_sinr_rate(sm, h_true, g, no):
    """SINR_k = |h_k^H w_k|^2 / (sum_{j!=k}|h_k^H w_j|^2 + no), eq. (2.2),
    evaluee sur h_true quel que soit le canal utilise pour calculer g."""
    if len(g.shape) == 5:
        g = tf.expand_dims(g, axis=1)
    h_pc = _get_desired_channels(h_true, sm)
    H = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)
    W = tf.cast(tf.squeeze(g, axis=1), tf.complex128)
    HW = tf.matmul(H, W)
    signal_pwr = tf.abs(tf.linalg.diag_part(HW)) ** 2
    tot_pwr = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interference_pwr = tf.maximum(tot_pwr - signal_pwr, tf.constant(0.0, tf.float64))
    sinr = signal_pwr / (interference_pwr + tf.cast(tf.reshape(no, []), tf.float64))
    rate = tf.math.log(1.0 + sinr) / tf.math.log(tf.constant(2.0, tf.float64))
    return float(tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(rate, axis=[1, 2]), axis=0)))


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_regime(name, cfg):
    t0 = time.time()
    print(f"\n{'='*78}\nREGIME {name} -- {cfg['label']}\n{'='*78}", flush=True)
    M, K = cfg['cfg']['NUM_TX'], cfg['cfg']['NUM_RX']
    D, L, T = cfg['embed_dim'], 4, cfg['tokens_per_rb']
    system = cfg['system_cls'](M, K, cfg['cfg'])

    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    no0 = ebnodb2no(tf.constant(15.0, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    neural_models = {}
    for nm, ckpt in cfg['checkpoints'].items():
        assert os.path.isfile(os.path.join(ckpt, 'weights.pkl')), f"checkpoint introuvable: {ckpt}"
        model = build_neural(nm, M, K, D, L, T)
        load_weights(model, ckpt, dummy_h, no0)
        neural_models[nm] = model
        print(f'  OK {nm} <- {ckpt}', flush=True)

    rng = np.random.RandomState(SEED)
    perfect = {m: {} for m in METHODS}
    imperfect = {m: {} for m in METHODS}
    raw_perfect = {m: {} for m in METHODS}
    raw_imperfect = {m: {} for m in METHODS}

    for snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        rp = {m: [] for m in METHODS}
        ri = {m: [] for m in METHODS}
        for _ in range(cfg['num_draws']):
            system.new_topology(cfg['batch'])
            h_true, _ = system.channel_and_no(tf.constant(cfg['batch'], tf.int32), tf.constant(snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            g_p = rzf_precoder(h_true, stream_management=system.sm, no=no)
            g_i = rzf_precoder(h_est, stream_management=system.sm, no=no)
            rp['RZF'].append(manual_sinr_rate(system.sm, h_true, g_p, no))
            ri['RZF'].append(manual_sinr_rate(system.sm, h_true, g_i, no))

            g_p = wmmse_precoder(h_true, no=no, stream_management=system.sm, num_iterations=10)
            g_i = wmmse_precoder(h_est, no=no, stream_management=system.sm, num_iterations=10)
            rp['WMMSE'].append(manual_sinr_rate(system.sm, h_true, g_p, no))
            ri['WMMSE'].append(manual_sinr_rate(system.sm, h_true, g_i, no))

            for nm, model in neural_models.items():
                g_p = model(h_true, no=no, training=False)
                g_i = model(h_est, no=no, training=False)
                rp[nm].append(manual_sinr_rate(system.sm, h_true, g_p, no))
                ri[nm].append(manual_sinr_rate(system.sm, h_true, g_i, no))

        for m in METHODS:
            perfect[m][snr] = float(np.mean(rp[m]))
            imperfect[m][snr] = float(np.mean(ri[m]))
            raw_perfect[m][snr] = rp[m]
            raw_imperfect[m][snr] = ri[m]
        print(f"  SNR={snr:5.1f} | " + " ".join(
            f"{NAME_MAP.get(m,m)}: parfait={perfect[m][snr]:.3f} imparfait={imperfect[m][snr]:.3f} "
            f"conserve={100*imperfect[m][snr]/perfect[m][snr]:.1f}%" for m in METHODS), flush=True)

    conserved = {m: {snr: 100.0 * imperfect[m][snr] / perfect[m][snr] for snr in DATA_SNRS_DB} for m in METHODS}

    meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'shared_channel': True,
        'shared_channel_note': ('h_true tire UNE fois par (snr, draw), reutilise pour calculer g '
                                 'sous les DEUX conditions CSI (h_true et h_est=noisy_channel(h_true)) '
                                 'et pour les 5 methodes -- perfect et imperfect viennent donc du meme '
                                 'tirage de canal, contrairement aux fichiers results/ pre-existants.'),
        'metric': 'manual_sinr (SINR_k=|h_k^Hw_k|^2/(sum_j!=k|h_k^Hw_j|^2+no), eq 2.2, precoders/classical.py)',
        'regime': name, 'label': cfg['label'],
        'embed_dim': D, 'num_layers': L, 'tokens_per_rb': T,
        'checkpoints': cfg['checkpoints'], 'checkpoint_note': cfg['checkpoint_note'],
        'batch_size': cfg['batch'], 'num_draws': cfg['num_draws'],
        'pilot_snr_db': PILOT_SNR_DB, 'data_snrs_db': DATA_SNRS_DB,
        'config': cfg['cfg'],
        'script': os.path.abspath(__file__), 'script_sha256_16': sha256_of(__file__),
        'precoders_classical_sha256_16': sha256_of('precoders/classical.py'),
        'channel_config_sha256_16': sha256_of('channel_config.py'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }

    tag = time.strftime('%Y%m%d_%H%M%S')
    out = {
        'metadata': meta,
        'table_debit_conserve_pct': {NAME_MAP.get(m, m): [conserved[m][s] for s in DATA_SNRS_DB] for m in METHODS},
        'table_debit_parfait_bps_hz': {NAME_MAP.get(m, m): [perfect[m][s] for s in DATA_SNRS_DB] for m in METHODS},
        'table_debit_imparfait_bps_hz': {NAME_MAP.get(m, m): [imperfect[m][s] for s in DATA_SNRS_DB] for m in METHODS},
        'snr_db': DATA_SNRS_DB,
        'raw_per_draw': {'perfect': raw_perfect, 'imperfect': raw_imperfect},
    }
    path = f'results/eval_csi_conserve_{name}_seedfix_{tag}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Sauvegarde -> {path}")
    print(f"  Temps regime: {time.time()-t0:.0f}s")
    return path


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('--regimes', nargs='+', default=list(REGIMES.keys()))
    args = p.parse_args()
    paths = []
    for r in args.regimes:
        paths.append(run_regime(r, REGIMES[r]))
    print("\n" + "="*78)
    print("Fichiers produits:")
    for p_ in paths:
        print(" ", p_)
