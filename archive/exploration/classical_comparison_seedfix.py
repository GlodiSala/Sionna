"""
classical_comparison_seedfix.py -- Version corrigee de classical_comparison.py
avec le vrai fix de graine.

BUG (confirme par investigation) : sionna/phy/config.py cree config.tf_rng
comme tf.random.Generator.from_non_deterministic_state() par defaut. Tous
les tirages de topologie (channel_config.py: gen_topology_clustered,
gen_topology_locked) utilisent config.tf_rng.uniform(...), PAS le RNG
global TensorFlow. tf.random.set_seed(42) (pose par wmmse_convergence_
check.py, herite par classical_comparison.py via import) ne seed donc PAS
les tirages de canal -- chaque run tire une topologie reellement aleatoire
(entropie OS), malgre l'apparence de reproductibilite.

FIX : sionna_config.seed = SEED (property setter de sionna.phy.config.Config,
qui appelle self.tf_rng.reset_from_seed(seed) -- seul moyen de fixer
reellement config.tf_rng).

Ne modifie PAS classical_comparison.py (toujours utilise pour produire
d'autres tableaux/figures du memoire) -- fichier separe, comme demande.

Usage:
    python3 classical_comparison_seedfix.py --verify     # RZF seul, rapide,
                                                            # pour verifier
                                                            # la reproductibilite
    python3 classical_comparison_seedfix.py               # table complete
                                                            # (6 methodes),
                                                            # sauvegardee en JSON
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # robuste au cwd du lanceur (relatif utilise partout ci-dessous)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED   # <-- LE FIX : seed reellement config.tf_rng

from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import cir_to_ofdm_channel

from classical_comparison import LockedClusterSystem, BATCH_SIZE, RB_SIZE, WMMSE_ITERS
from compare_rb_grouping import rzf_precoder_with_rb_grouping
from precoders_w import rzf_precoder
from channel_config import STANDARD_CONFIG
from main_finall import MU_MIMO_System
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']

CKPTS = {
    'SC':    ('single_sc',       'results/diag_front_a_signed_attn.json'),
    'IB':    ('intra_rb',        'results/diag_front_a_signed_attn_intra_rb.json'),
    'TA-RB': ('ta_rb_residual',  'results/diag_tarb_residual_signed_attn_T4_train.json'),
}


def build_neural_system(name):
    arch, json_path = CKPTS[name]
    with open(json_path) as f:
        ckpt = json.load(f)['best_ckpt']
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch, embed_dim=128,
                             num_heads=4, num_layers=4, tokens_per_rb=4)
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols,
                  fft_size=system.rg.fft_size, embed_dim=128, num_heads=4,
                  num_layers=4, snr_aware=True)
    if name == 'SC':
        system.precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    elif name == 'IB':
        system.precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    else:
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=4, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
    ok = system.load_weights_from(ckpt)
    assert ok, f"chargement echoue pour {name}"
    return system, ckpt


def gen_h_no(system, batch_size, snr_db):
    """Meme mecanisme que MU_MIMO_System.call() (main_finall.py:420-432)."""
    no = ebnodb2no(snr_db, system.num_bits_per_symbol, 0.5, system.rg)
    cir = system.channel_model(batch_size, system.rg.num_ofdm_symbols,
                                1.0 / system.rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
    h_freq = system.remove_nulled(h_freq)
    return h_freq, no


def sum_rate_from_g(system, h_freq, g, no):
    """Meme formule que ConfigurableMIMOSystem._sum_rate (wmmse_convergence_
    check.py:174-181), reutilisee via system.ch_helper/system.lmmse_sinr
    (MU_MIMO_System a les deux -- main_finall.py:308-313)."""
    h_eff = system.ch_helper.compute_effective_channel(h_freq, g)
    sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
    X = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
    return float(tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(X, axis=[1, 2, 4]), axis=0)))


def run_classical(num_batches):
    """RZF (par SC), RZF (par RB12), WMMSE -- meme h_freq pour les trois,
    comme classical_comparison.py:72-83 (paired comparison)."""
    system = LockedClusterSystem(M, K)
    system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']

    out = {'rzf_sc': {}, 'rzf_rb12': {}, 'wmmse': {}}
    for snr in REPORT_SNRS:
        snr_t = tf.constant(snr, tf.float32)
        rzf_l, rb_l, wmmse_l = [], [], []
        for b in range(num_batches):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
            g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
            rzf_l.append(float(system._sum_rate(h_freq, g_rzf, no)))
            g_rb = rzf_precoder_with_rb_grouping(h_freq, stream_management=system.sm, rb_size=RB_SIZE)
            rb_l.append(float(system._sum_rate(h_freq, g_rb, no)))
            wmmse_l.append(float(system.eval_wmmse_from_h(h_freq, no, WMMSE_ITERS)))
        out['rzf_sc'][snr] = rzf_l
        out['rzf_rb12'][snr] = rb_l
        out['wmmse'][snr] = wmmse_l
        print(f"[classical] SNR={snr:5.1f} | RZF-SC={np.mean(rzf_l):.4f} "
              f"RZF-RB12={np.mean(rb_l):.4f} WMMSE={np.mean(wmmse_l):.4f}", flush=True)
    return out


def run_neural(name, num_batches):
    system, ckpt = build_neural_system(name)
    out = {}
    for snr in REPORT_SNRS:
        snr_t = tf.constant(snr, tf.float32)
        rates = []
        for b in range(num_batches):
            system.new_topology(BATCH_SIZE)
            h_freq, no = gen_h_no(system, BATCH_SIZE, snr_t)
            g = system._call_precoder(h_freq, no, training=False)
            rates.append(sum_rate_from_g(system, h_freq, g, no))
        out[snr] = rates
        print(f"[{name}] SNR={snr:5.1f} | rate={np.mean(rates):.4f}", flush=True)
    return out, ckpt


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true',
                    help='RZF-SC seul, num_batches reduit -- pour test de reproductibilite (lancer 2x)')
    p.add_argument('--num_batches', type=int, default=None)
    args = p.parse_args()

    t0 = time.time()
    if args.verify:
        nb = args.num_batches or 5
        system = LockedClusterSystem(M, K)
        system.cluster_radius_m   = STANDARD_CONFIG['CLUSTER_RADIUS_M']
        system.indoor_probability = STANDARD_CONFIG['INDOOR_PROBABILITY']
        rzf_out = {}
        for snr in REPORT_SNRS:
            snr_t = tf.constant(snr, tf.float32)
            vals = []
            for b in range(nb):
                system.new_topology(BATCH_SIZE)
                h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
                g_rzf = rzf_precoder(h_freq, stream_management=system.sm, no=no)
                vals.append(float(system._sum_rate(h_freq, g_rzf, no)))
            rzf_out[snr] = vals
            print(f"SNR={snr:5.1f} RZF per-batch = {[round(v,6) for v in vals]}", flush=True)
        print(f"\nRZF_VERIFY_JSON={json.dumps(rzf_out)}")
        print(f"Total time: {time.time()-t0:.0f}s")
        return

    nb = args.num_batches or 10
    classical = run_classical(nb)
    neural = {}
    ckpts = {}
    for name in ['SC', 'IB', 'TA-RB']:
        neural[name], ckpts[name] = run_neural(name, nb)

    table = {'snr': REPORT_SNRS}
    for key in ['rzf_sc', 'rzf_rb12', 'wmmse']:
        table[key] = [float(np.mean(classical[key][s])) for s in REPORT_SNRS]
    for name in ['SC', 'IB', 'TA-RB']:
        table[name] = [float(np.mean(neural[name][s])) for s in REPORT_SNRS]

    meta = {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED (config.tf_rng.reset_from_seed)',
        'script': os.path.abspath(__file__),
        'script_sha256_16': sha256_of(__file__),
        'precoders_w_sha256_16': sha256_of(os.path.join(os.path.dirname(__file__), 'precoders_w.py')),
        'channel_config_sha256_16': sha256_of(os.path.join(os.path.dirname(__file__), 'channel_config.py')),
        'checkpoints': ckpts,
        'batch_size': BATCH_SIZE,
        'num_batches': nb,
        'rb_size': RB_SIZE,
        'wmmse_iters': WMMSE_ITERS,
        'config': STANDARD_CONFIG,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'raw_per_batch': {
            'rzf_sc': classical['rzf_sc'], 'rzf_rb12': classical['rzf_rb12'], 'wmmse': classical['wmmse'],
            'SC': neural['SC'], 'IB': neural['IB'], 'TA-RB': neural['TA-RB'],
        },
    }

    out = {'table': table, 'metadata': meta}
    date_tag = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'classical_comparison_M8K4_seedfix_{date_tag}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n=== TABLE 3.9 REGENEREE (seed fixe, M8K4, UMi, CSI parfait) ===")
    print(f"{'SNR':>6} {'RZF-SC':>9} {'RZF-RB12':>9} {'WMMSE':>9} {'SC':>9} {'IB':>9} {'TA-RB':>9}")
    for i, s in enumerate(REPORT_SNRS):
        print(f"{s:6.1f} {table['rzf_sc'][i]:9.3f} {table['rzf_rb12'][i]:9.3f} "
              f"{table['wmmse'][i]:9.3f} {table['SC'][i]:9.3f} {table['IB'][i]:9.3f} {table['TA-RB'][i]:9.3f}")

    print(f"\nSauvegarde -> {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
