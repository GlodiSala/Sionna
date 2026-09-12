"""
diag_tarb_residual_test.py — Partie 2 (demande utilisateur) : entraîne
TransformerPrecoderCleanResidual (décodeur résiduel : interpolation
linéaire FIXE entre tokens + correction apprise, au lieu du Conv1D
Transpose appris) et compare à TA-RB actuel (T=6, décodeur appris complet)
sur DEUX régimes -- CSI parfait (l'écart à haut SNR se réduit-il ?) ET
CSI imparfait (le bénéfice de débruitage est-il préservé ?).

Protocole d'entraînement : identique au run STANDARD validé (cosine_long,
warmup=3ep, finetune=80ep, steps_per_epoch=150) -- SAUF si le test
warmup+rampe (en cours en parallèle) montre un gain net, auquel cas à
reconsidérer pour un run définitif (ce script utilise le protocole
ACTUELLEMENT validé, pas encore le résultat du test rampe).

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_tarb_residual_test.py [--finetune_epochs 80]
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer, CHOSEN_CONFIG, DATASET_SIZE, NUM_TX, NUM_RX
from datasets import CachedSionnaDataset
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, set_locked_topology
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--finetune_epochs', type=int, default=80)
_p.add_argument('--warmup_epochs', type=int, default=3)
_args = _p.parse_args()

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
STEPS_PER_EPOCH = 150
BATCH_SIZE      = 256
LR              = 1e-3
EVAL_SNRS       = [15.0, 17.5, 20.0]
EVAL_BATCHES    = 20

_ref = np.load('results/classical_comparison_M8K4.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


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
    dummy = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=42)

    # ── Entraînement TA-RB décodeur résiduel ──────────────────────────────
    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='ta_rb_residual',
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6)
    trainer = SupervisedTrainer(
        system, dataset, run_name='TA_RB_residual_6tok_4L_128d',
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

    print(f'\n{"="*70}\nCSI PARFAIT -- TA-RB résiduel vs référence TA-RB T=6 original\n{"="*70}')
    tarb_orig = {15.0: 28.20, 17.5: 29.67, 20.0: 30.73}   # run complet ÉTAPE 4, déjà loggé
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | résiduel={evals_perfect[snr]:7.2f} ({pct_perfect[snr]:5.1f}% WMMSE) | '
              f'original T=6={tarb_orig[snr]:7.2f} | delta={evals_perfect[snr]-tarb_orig[snr]:+.2f}')

    # ── Éval CSI imparfait (comparaison directe, poids fraîchement entraînés) ──
    fresh_system = LockedSystem(M, K)
    rng = np.random.RandomState(2026)

    def precoder_fn(h_est, no):
        return system._call_precoder(h_est, no, training=False)

    print(f'\n{"="*70}\nCSI IMPARFAIT -- TA-RB résiduel, data SNR=15dB, pilote {{parfait,20dB,10dB}}\n{"="*70}')
    csi_results = {}
    for pilot in [None, 20.0, 10.0]:
        cond = 'perfect' if pilot is None else f'pilot{int(pilot)}dB'
        if pilot is None:
            r = eval_csi_imperfect(fresh_system, precoder_fn, 15.0, 1000.0, rng)  # ~no noise proxy
        else:
            r = eval_csi_imperfect(fresh_system, precoder_fn, 15.0, pilot, rng)
        csi_results[cond] = r
        print(f'  {cond:>12s} : {r:.2f}')

    pct20 = 100.0 * csi_results['pilot20dB'] / csi_results['perfect']
    pct10 = 100.0 * csi_results['pilot10dB'] / csi_results['perfect']
    print(f'\n  %retenu pilot20dB={pct20:.1f}% (TA-RB T=6 original: 85.7%) | '
          f'pilot10dB={pct10:.1f}% (original: 51.0%)')

    out = {'perfect': evals_perfect, 'pct_of_wmmse_perfect': pct_perfect,
           'csi_imperfect': csi_results, 'pct_retained_20dB': pct20, 'pct_retained_10dB': pct10,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path}
    with open('results/diag_tarb_residual_test.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_tarb_residual_test.json')
