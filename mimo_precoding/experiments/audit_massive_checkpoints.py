"""
experiments/audit_massive_checkpoints.py -- verifie si le checkpoint "extbudget" (D=128,
massif M64K8) reproduit les valeurs deja publiees dans le memoire pour SC/IB
(72.2/80.9/86.0 bps/Hz a 10/15/20dB pour SC), au lieu du checkpoint "baseline"
(70 epoques, sous-entraine) utilise par erreur dans
experiments/eval_massive_d128_d384.py.

Ne modifie aucun fichier existant. N'evalue QUE SC et IB (pas RZF/WMMSE/TA-RB,
pas la table complete) -- verification ciblee demandee.
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
from sionna.phy.config import config as sionna_config
sionna_config.seed = SEED

import pickle
from eval_system import ConfigurableMIMOSystem
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from system import MU_MIMO_System
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn)

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
REPORT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
BATCH_SIZE = 16
EMBED_DIM = 128

# Checkpoint utilise PAR ERREUR par experiments/eval_massive_d128_d384.py
CKPT_WRONG = {
    'SC': 'results/diag_massive_true_single_sc.json',
    'IB': 'results/diag_massive_true_intra_rb.json',
}
# Checkpoint candidat (entrainement etendu, epoch 161 vs 70)
CKPT_EXTBUDGET = {
    'SC': 'results/diag_massive_true_single_sc_extbudget.json',
    'IB': 'results/diag_massive_true_intra_rb_extbudget.json',
}

PUBLISHED_SC = {10.0: 72.2, 15.0: 80.9, 20.0: 86.0}  # valeurs du memoire (tab:results_massive, D=128)


class LockedClusterSystemMassive(ConfigurableMIMOSystem):
    cluster_radius_m   = MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY']

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def build_neural_precoder(name, ckpt_json):
    with open(ckpt_json) as f:
        ckpt = json.load(f)['best_ckpt']
    dummy_system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='single_sc',
                                   embed_dim=EMBED_DIM, num_heads=4, num_layers=4, tokens_per_rb=6)
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=dummy_system.rg.num_ofdm_symbols,
                  fft_size=dummy_system.rg.fft_size, embed_dim=EMBED_DIM, num_heads=4,
                  num_layers=4, snr_aware=True)
    if name == 'SC':
        precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    else:
        precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    dummy_h = tf.zeros([1, K, 1, 1, M, dummy_system.rg.num_ofdm_symbols, dummy_system.rg.fft_size],
                        dtype=tf.complex64)
    _ = precoder(dummy_h, no=tf.constant(0.01), training=False)
    weights_path = os.path.join(ckpt, 'weights.pkl') if os.path.isdir(ckpt) else ckpt
    with open(weights_path, 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(precoder.trainable_variables, ws):
        v.assign(w)
    print(f"OK poids charges pour {name} depuis {ckpt}", flush=True)
    return precoder


def sum_rate(system, h_freq, g, no):
    return float(system._sum_rate(h_freq, g, no))


def eval_ckpt_set(label, ckpt_map, num_batches=5):
    system = LockedClusterSystemMassive(M, K)
    precoders = {name: build_neural_precoder(name, ckpt_map[name]) for name in ['SC', 'IB']}
    out = {name: {s: [] for s in REPORT_SNRS} for name in ['SC', 'IB']}
    for snr in REPORT_SNRS:
        snr_t = tf.constant(snr, tf.float32)
        for b in range(num_batches):
            system.new_topology(BATCH_SIZE)
            h_freq, no = system.channel_and_no(tf.constant(BATCH_SIZE, tf.int32), snr_t)
            for name in ['SC', 'IB']:
                g = precoders[name](h_freq, no=no, training=False)
                out[name][snr].append(sum_rate(system, h_freq, g, no))
        print(f"[{label}] SNR={snr:5.1f} | SC={np.mean(out['SC'][snr]):.3f} IB={np.mean(out['IB'][snr]):.3f}", flush=True)
    return {name: {s: float(np.mean(v)) for s, v in d.items()} for name, d in out.items()}


t0 = time.time()
print("=== CHECKPOINT UTILISE PAR experiments/eval_massive_d128_d384.py (probable cause du bug) ===")
res_wrong = eval_ckpt_set('wrong(70ep)', CKPT_WRONG, num_batches=5)

print("\n=== CHECKPOINT CANDIDAT 'extbudget' (161 epoques) ===")
res_ext = eval_ckpt_set('extbudget(161ep)', CKPT_EXTBUDGET, num_batches=5)

print("\n=== COMPARAISON FINALE ===")
print(f"{'SNR':>6} {'SC_wrong':>10} {'SC_ext':>10} {'SC_publie':>10} {'IB_wrong':>10} {'IB_ext':>10}")
for s in REPORT_SNRS:
    pub = PUBLISHED_SC.get(s, float('nan'))
    print(f"{s:6.1f} {res_wrong['SC'][s]:10.3f} {res_ext['SC'][s]:10.3f} {pub:10.3f} "
          f"{res_wrong['IB'][s]:10.3f} {res_ext['IB'][s]:10.3f}")

out = {
    'checkpoint_wrong': CKPT_WRONG, 'checkpoint_extbudget': CKPT_EXTBUDGET,
    'published_SC_values': PUBLISHED_SC,
    'result_wrong_checkpoint': res_wrong,
    'result_extbudget_checkpoint': res_ext,
    'seed': SEED, 'num_batches': 5, 'batch_size': BATCH_SIZE, 'embed_dim': EMBED_DIM,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
}
with open('results/check_d128_extbudget_ckpt_result.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSauvegarde -> results/check_d128_extbudget_ckpt_result.json")
print(f"Total time: {time.time()-t0:.0f}s")
