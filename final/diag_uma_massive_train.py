"""
diag_uma_massive_train.py — Priorité 4 (nuit, demande utilisateur) :
UMa MASSIVE (M=64, K=8), budget étendu D'EMBLÉE (10 warmup + 160
finetune -- pas de run 83ep intermédiaire, cf. diagnostic budget
MASSIVE déjà tranché : le budget court est insuffisant à cette échelle
pour SC/IB, donc inutile de le re-tester ici). Fusion de deux patterns
déjà validés cette nuit :
- `diag_massive_true_train.py` (dims M=64/K=8, batch réduit training
  16/16/32, DATASET_SIZE réduit -- contrainte mémoire M=64)
- `diag_uma_signed_attn_train.py` (canal UMa via scenario='uma',
  UMaLockedSystem pour l'éval CSI imparfait)

Dataset généré à la volée (pas encore en cache pour cette combinaison
inédite) -- batch CIR-gen réduit à 32 (cf. leçon MASSIVE_TRUE : 128
OOM à M=64, cf. diag_massive_true_smoke.py).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_uma_massive_train.py --arch single_sc
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
from channel_config import MASSIVE_TRUE_CONFIG, set_locked_topology
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.utils import ebnodb2no

_p = argparse.ArgumentParser()
_p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb', 'ta_rb_residual'])
_p.add_argument('--finetune_epochs', type=int, default=160)
_p.add_argument('--warmup_epochs', type=int, default=10)
_p.add_argument('--tag', type=str, default='')
_args = _p.parse_args()

M, K, R = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX'], MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M']
STEPS_PER_EPOCH = 150
LR              = 1e-3
DATASET_SIZE    = 4000
CACHE_FILE      = f'/export/tmp/sala/sionna_joint_uma_massive_{DATASET_SIZE//1000}k_{M}x{K}.npz'
EVAL_SNRS       = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES    = 20
PILOT_SNR_DB    = 20.0
NUM_DRAWS       = 12
CSI_BATCH       = 16   # même réduction que diag_csi_imperfect_massive_true_sweep.py
SEED            = 2026
BATCH_SIZE      = {'single_sc': 16, 'intra_rb': 16, 'ta_rb_residual': 32}[_args.arch]

_ref = np.load('results/classical_comparison_M64K8_true_uma.npy', allow_pickle=True).item()
WMMSE_REF = {snr: wmmse for snr, wmmse in zip(_ref['snr'], _ref['wmmse'])}


class UMaLockedSystemMassive(ConfigurableMIMOSystem):
    cluster_radius_m   = R
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
                             force_los=False, indoor_probability=self.indoor_probability)


def eval_perfect(system, dataset, snr_db, num_batches=EVAL_BATCHES, batch_size=32):
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
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    arch = _args.arch

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=DATASET_SIZE, batch_size=32,   # CIR-gen réduit (M=64, cf. leçon MASSIVE)
        cache_file=CACHE_FILE, cluster_radius_m=R, indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'],
        scenario='uma', seed=42)

    if arch == 'single_sc':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='single_sc',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = SingleSCTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'UMa_MASSIVE_SingleSC_signed_attn_4L_128d' + _args.tag
    elif arch == 'intra_rb':
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='intra_rb',
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = IntraRBTransformerPrecoderSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True, use_abs=True, use_cossin=False)
        run_name = 'UMa_MASSIVE_IntraRB_signed_attn_4L_128d' + _args.tag
    else:
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='ta_rb_residual',
                                 embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=6)
        system.precoder = TransformerPrecoderCleanResidualSignedAttn(
            num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
            rb_size=12, tokens_per_rb=6, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
        run_name = 'UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d' + _args.tag

    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=_args.warmup_epochs, finetune_epochs=_args.finetune_epochs,
        batch_size=BATCH_SIZE, learning_rate=LR,
        steps_per_epoch=STEPS_PER_EPOCH, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=STEPS_PER_EPOCH, patience=25)
    train_time = time.time() - t0
    print(f'\n✅ Entraînement fini en {train_time/60:.1f}min, meilleur checkpoint: {trainer.ckpt.best_ckpt_path}',
          flush=True)

    evals_perfect = {snr: eval_perfect(system, dataset, snr) for snr in EVAL_SNRS}
    pct_perfect = {snr: 100.0 * evals_perfect[snr] / WMMSE_REF[snr] for snr in EVAL_SNRS}
    print(f'\n{"="*70}\nCSI PARFAIT (UMa MASSIVE) -- {arch}\n{"="*70}')
    for snr in EVAL_SNRS:
        print(f'  SNR={snr:5.1f}dB | rate={evals_perfect[snr]:7.2f} | {pct_perfect[snr]:5.1f}% WMMSE', flush=True)

    fresh_system = UMaLockedSystemMassive(M, K)
    rng = np.random.RandomState(SEED)

    def precoder_fn(h_est, no):
        return system._call_precoder(h_est, no, training=False)

    print(f'\n{"="*70}\nCSI IMPARFAIT (UMa MASSIVE) -- {arch}, pilote {PILOT_SNR_DB}dB\n{"="*70}')
    csi_imperfect = {}
    for snr in EVAL_SNRS:
        no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_perfect, r_pilot = [], []
        for _ in range(NUM_DRAWS):
            fresh_system.new_topology(CSI_BATCH)
            h_true, _ = fresh_system.channel_and_no(tf.constant(CSI_BATCH, tf.int32), tf.constant(snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)
            g_p = precoder_fn(h_true, no)
            r_perfect.append(float(fresh_system._sum_rate(h_true, g_p, no)))
            g_n = precoder_fn(h_est, no)
            r_pilot.append(float(fresh_system._sum_rate(h_true, g_n, no)))
        perf = float(np.mean(r_perfect))
        pil = float(np.mean(r_pilot))
        pct = 100.0 * pil / perf if perf > 0 else float('nan')
        csi_imperfect[str(snr)] = {'perfect': perf, 'pilot20dB': pil, 'pct_retained': pct}
        print(f'  SNR={snr:5.1f}dB | perfect={perf:7.2f} | pilot20dB={pil:7.2f} | {pct:5.1f}% retenu', flush=True)

    flops, params, acts = system.precoder.complexity(system.rg.num_ofdm_symbols, convention_x_ofdm=False)
    real_params = int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables))

    out = {'arch': arch, 'evals_perfect': evals_perfect, 'pct_of_wmmse_perfect': pct_perfect,
           'csi_imperfect': csi_imperfect, 'flops_real_M': flops / 1e6, 'params': real_params,
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': _args.finetune_epochs, 'warmup_epochs': _args.warmup_epochs, 'M': M, 'K': K}
    with open(f'results/diag_uma_massive_{arch}{_args.tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Sauvé -> results/diag_uma_massive_{arch}{_args.tag}.json')
