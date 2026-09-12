"""
diag_tarb_residual_uma_train.py — Front B (demande utilisateur) : complète
le tableau UMa (seuls SingleSC/IntraRB y ont un run complet, cf. SESSION_LOG_
20260808.md) avec TA-RB résiduel (le standard TA-RB actuel, adopté sur UMi
le 7/8 -- cf. diag_tarb_residual_test.py), même protocole baseline (83ep :
3 warmup + 80 finetune, cosine_long, steps_per_epoch=150), même cache UMa
déjà généré (8000 échantillons joints M8K4, réutilisé tel quel pour
comparabilité directe avec les runs single_sc/intra_rb UMa déjà faits).

Puis éval CSI imparfait sur le checkpoint frais (même méthodologie que
diag_tarb_residual_test.py sur UMi) pour voir si l'avantage de débruitage
se retrouve sur UMa.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_tarb_residual_uma_train.py
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import UMa
from main_finall import MU_MIMO_System, SupervisedTrainer
from datasets import CachedSionnaDataset
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, gen_topology_clustered
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--finetune_epochs', type=int, default=80)
_p.add_argument('--warmup_epochs', type=int, default=3)
_args = _p.parse_args()

M, K, R        = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX'], STANDARD_CONFIG['CLUSTER_RADIUS_M']
STEPS_PER_EPOCH = 150
BATCH_SIZE      = 256
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
DATASET_SIZE    = 8000
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_uma_{DATASET_SIZE//1000}k_{M}x{K}.npz'

_ref = np.load('results/classical_comparison_M8K4_uma.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}


class UMaLockedSystem(ConfigurableMIMOSystem):
    """Même pattern que diag_classical_comparison_uma.py /
    diag_intrarb_gradient_layers_uma.py -- pour l'éval CSI imparfait
    uniquement (tirage dynamique de nouvelles topologies UMa)."""
    cluster_radius_m   = R
    indoor_probability = 0.0

    def __init__(self, num_tx, num_rx):
        super().__init__(num_tx, num_rx)
        self.channel_model = UMa(
            carrier_frequency=2.6e9, o2i_model="low",
            ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink',
            enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        topology = gen_topology_clustered(
            batch_size, self.num_rx, 'uma', self.cluster_radius_m,
            indoor_probability=self.indoor_probability)
        self.channel_model.set_topology(*topology, los=False)


def eval_perfect(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=128):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    rates = []
    for _ in range(num_batches):
        h = dataset.get_batch(batch_size)
        g = system._call_precoder(h, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr  = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate  = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))
        rates.append(float(rate))
    return float(np.mean(rates))


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2  = 1.0 / snr_lin
    shape   = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


def eval_csi_imperfect(fresh_system, precoder_fn, snr_db, pilot_snr_db, rng, num_draws=20, batch=32):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), 2, 0.5, fresh_system.rg)
    rates = []
    for _ in range(num_draws):
        fresh_system.new_topology(batch)
        h_true, _ = fresh_system.channel_and_no(tf.constant(batch, tf.int32), tf.constant(snr_db, tf.float32))
        h_est = noisy_channel(h_true, pilot_snr_db, rng)
        g = precoder_fn(h_est, no)
        rates.append(float(fresh_system._sum_rate(h_true, g, no)))
    return float(np.mean(rates))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=CACHE_FILE,
        cluster_radius_m=R, indoor_probability=0.0, scenario='uma', seed=42)

    # ── Entraînement TA-RB décodeur résiduel sur UMa ──────────────────────
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='ta_rb_residual',
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6)
    trainer = SupervisedTrainer(
        system, dataset, run_name='TA_RB_residual_uma_6tok_4L_128d',
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0
    print(f'\n✅ Entraînement fini en {train_time/60:.1f}min, meilleur checkpoint: {trainer.ckpt.best_ckpt_path}')

    # ── Éval CSI parfait ────────────────────────────────────────────────
    evals_perfect = {snr: eval_perfect(system, dataset, snr) for snr in EVAL_SNRS}
    pct_perfect = {snr: 100.0 * evals_perfect[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}

    print(f'\n{"="*70}\nCSI PARFAIT (UMa) -- TA-RB résiduel\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | résiduel={evals_perfect[snr]:7.2f} ({pct_perfect[snr]:5.1f}% WMMSE)')

    # ── Éval CSI imparfait (comparaison directe, poids fraîchement entraînés) ──
    fresh_system = UMaLockedSystem(M, K)
    rng = np.random.RandomState(2026)

    def precoder_fn(h_est, no):
        return system._call_precoder(h_est, no, training=False)

    print(f'\n{"="*70}\nCSI IMPARFAIT (UMa) -- TA-RB résiduel, data SNR=15dB, pilote {{parfait,20dB,10dB}}\n{"="*70}')
    csi_results = {}
    for pilot in [None, 20.0, 10.0]:
        cond = 'perfect' if pilot is None else f'pilot{int(pilot)}dB'
        if pilot is None:
            r = eval_csi_imperfect(fresh_system, precoder_fn, 15.0, 1000.0, rng)
        else:
            r = eval_csi_imperfect(fresh_system, precoder_fn, 15.0, pilot, rng)
        csi_results[cond] = r
        print(f'  {cond:>12s} : {r:.2f}')

    pct20 = 100.0 * csi_results['pilot20dB'] / csi_results['perfect']
    pct10 = 100.0 * csi_results['pilot10dB'] / csi_results['perfect']
    print(f'\n  %retenu pilot20dB={pct20:.1f}% | pilot10dB={pct10:.1f}%  '
          f'(réf UMi résiduel : 87.7%/54.6%)')

    out = {'perfect': evals_perfect, 'pct_of_wmmse_perfect': pct_perfect,
           'csi_imperfect': csi_results, 'pct_retained_20dB': pct20, 'pct_retained_10dB': pct10,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path}
    with open('results/diag_tarb_residual_uma_test.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_tarb_residual_uma_test.json')
