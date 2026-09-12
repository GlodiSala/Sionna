"""
train.py — unified training + evaluation entrypoint for the signed-
attention precoders (SC, IntraRB, TA-RB residual) across the 4 regimes
covered by the thesis: {UMi, UMa} channel x {standard (M=8,K=4),
massive (M=64,K=8)} antenna scale.

Consolidates what was, during development, three near-duplicate scripts
(one per channel/scale combination that needed its own dataset cache and
batch-size tuning). The only combination this script does NOT reduce to
a single code path is the per-scale batch size: M=64 needs a much
smaller CIR-generation and training batch than M=8 to avoid OOM (the
`_call_precoder`/effective-channel tensors scale with M), so those stay
as an explicit lookup table rather than a single constant.

Two-phase training (SupervisedTrainer, see system.py): MSE warmup
against RZF, then direct sum-rate finetuning. TA-RB residual uses
T=4 tokens/RB at standard scale (identified as the best throughput/
energy trade-off of the T-sweep, see the thesis chapter and
figures/umi_standard/figB_pareto_energy_T.py) and T=6 at massive scale
(no T-sweep was run there; T=6 is the value that was trained and kept).

A classical RZF/WMMSE reference (classical_reference.py) must be run
first for the chosen channel/scale combination -- its .npy output is
the %WMMSE normalization used in the printed/saved report.

Usage:
    python3 train.py --arch single_sc --channel umi --scale standard
    python3 train.py --arch ta_rb_residual --channel uma --scale massive \\
        --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel.tr38901 import UMa
from sionna.phy.utils import ebnodb2no

from system import MU_MIMO_System, SupervisedTrainer
from datasets import CachedSionnaDataset
from eval_system import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, MASSIVE_TRUE_CONFIG, set_locked_topology
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                         IntraRBTransformerPrecoderSignedAttn,
                                         TransformerPrecoderCleanResidualSignedAttn)

SCALE_CONFIG = {'standard': STANDARD_CONFIG, 'massive': MASSIVE_TRUE_CONFIG}
# CIR-gen / training batch size, and dataset pool size: M=64 needs both cut
# down hard vs M=8 to fit on a shared GPU (empirically OOM's otherwise --
# see SESSION_LOG_20260808.md "MASSIVE dataset/training OOM").
DATASET_SIZE  = {'standard': 20000, 'massive': 4000}
CIRGEN_BATCH  = {'standard': 128, 'massive': 32}
TRAIN_BATCH   = {'standard': {'single_sc': 256, 'intra_rb': 256, 'ta_rb_residual': 256},
                  'massive':  {'single_sc': 16,  'intra_rb': 16,  'ta_rb_residual': 32}}
TOKENS_PER_RB = {'standard': 4, 'massive': 6}   # cf. docstring above

EVAL_SNRS      = [0.0, 5.0, 10.0, 15.0, 17.5, 20.0]
EVAL_BATCHES   = 20
PILOT_SNR_DB   = 20.0
CSI_NUM_DRAWS  = {'standard': 20, 'massive': 12}
CSI_BATCH      = {'standard': 32, 'massive': 16}
SEED           = 2026


def build_precoder(arch, M, K, num_ofdm, fft_size, tokens_per_rb):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=num_ofdm, fft_size=fft_size,
                  embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    if arch == 'single_sc':
        return SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
    if arch == 'intra_rb':
        return IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
    if arch == 'ta_rb_residual':
        return TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=tokens_per_rb, **kwargs)
    raise ValueError(arch)


class LockedEvalSystem(ConfigurableMIMOSystem):
    """Fresh (uncached) topology draws for the CSI-imperfect sweep --
    UMi uses the base ConfigurableMIMOSystem channel model as-is, UMa
    swaps it for tr38901.UMa post-__init__."""

    def __init__(self, num_tx, num_rx, channel, cluster_radius_m, indoor_probability):
        super().__init__(num_tx, num_rx)
        self.cluster_radius_m = cluster_radius_m
        self.indoor_probability = indoor_probability
        if channel == 'uma':
            self.channel_model = UMa(
                carrier_frequency=2.6e9, o2i_model='low',
                ut_array=self.ut_array, bs_array=self.bs_array,
                direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def eval_perfect(system, dataset, snr_db, num_batches, batch_size):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    rates = []
    for _ in range(num_batches):
        h = dataset.get_batch(batch_size)
        g = system._call_precoder(h, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))
        rates.append(float(rate))
    return float(np.mean(rates))


def noisy_channel(h_freq, pilot_snr_db, rng):
    sigma2 = 1.0 / (10.0 ** (pilot_snr_db / 10.0))
    shape = h_freq.shape
    noise = rng.normal(0, np.sqrt(sigma2 / 2), size=(2,) + tuple(shape)).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise[0], noise[1]), h_freq.dtype)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['single_sc', 'intra_rb', 'ta_rb_residual'])
    p.add_argument('--channel', required=True, choices=['umi', 'uma'])
    p.add_argument('--scale', required=True, choices=['standard', 'massive'])
    p.add_argument('--finetune_epochs', type=int, default=80)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--tag', type=str, default='')
    args = p.parse_args()

    cfg = SCALE_CONFIG[args.scale]
    M, K, R = cfg['NUM_TX'], cfg['NUM_RX'], cfg['CLUSTER_RADIUS_M']
    dataset_size = DATASET_SIZE[args.scale]
    train_batch = TRAIN_BATCH[args.scale][args.arch]
    tokens_per_rb = TOKENS_PER_RB[args.scale]
    cache_file = f'/export/tmp/sala/sionna_joint_{args.channel}_{args.scale}_{dataset_size // 1000}k_{M}x{K}.npz'
    wmmse_ref_path = f'results/classical_reference_{args.channel}_{args.scale}.npy'

    os.makedirs('results', exist_ok=True)
    wmmse_ref = None
    if os.path.exists(wmmse_ref_path):
        _ref = np.load(wmmse_ref_path, allow_pickle=True).item()
        wmmse_ref = {snr: w for snr, w in zip(_ref['snr'], _ref['wmmse'])}
    else:
        print(f'⚠️  {wmmse_ref_path} introuvable -- exécuter classical_reference.py '
              f'd\'abord pour ce régime. %WMMSE non calculé, débit brut seulement.')

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=dataset_size, batch_size=CIRGEN_BATCH[args.scale],
        cache_file=cache_file, cluster_radius_m=R, indoor_probability=cfg['INDOOR_PROBABILITY'],
        scenario=args.channel, seed=42)

    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=args.arch,
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=tokens_per_rb)
    system.precoder = build_precoder(args.arch, M, K, system.rg.num_ofdm_symbols, system.rg.fft_size, tokens_per_rb)
    run_name = f'{args.channel}_{args.scale}_{args.arch}_signed_attn' + args.tag

    trainer = SupervisedTrainer(
        system, dataset, run_name=run_name,
        warmup_epochs=args.warmup_epochs, finetune_epochs=args.finetune_epochs,
        batch_size=train_batch, learning_rate=1e-3,
        steps_per_epoch=150, lr_schedule='cosine_long')

    t0 = time.time()
    trainer.train(print_every=150, patience=25)
    train_time = time.time() - t0
    print(f'\n✅ Entraîné en {train_time/60:.1f}min -> {trainer.ckpt.best_ckpt_path}', flush=True)

    evals = {snr: eval_perfect(system, dataset, snr, EVAL_BATCHES, batch_size=min(train_batch, 128))
             for snr in EVAL_SNRS}
    pct = ({snr: 100.0 * evals[snr] / wmmse_ref[snr] for snr in EVAL_SNRS} if wmmse_ref else None)

    print(f'\n{"="*70}\nCSI PARFAIT -- {run_name}\n{"="*70}')
    for snr in EVAL_SNRS:
        line = f'  SNR={snr:5.1f}dB | rate={evals[snr]:7.2f}'
        if pct:
            line += f' | {pct[snr]:5.1f}% WMMSE'
        print(line, flush=True)

    fresh = LockedEvalSystem(M, K, args.channel, R, cfg['INDOOR_PROBABILITY'])
    rng = np.random.RandomState(SEED)
    csi_batch, num_draws = CSI_BATCH[args.scale], CSI_NUM_DRAWS[args.scale]
    csi_imperfect = {}
    print(f'\n{"="*70}\nCSI IMPARFAIT (pilote {PILOT_SNR_DB}dB) -- {run_name}\n{"="*70}')
    for snr in EVAL_SNRS:
        no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        r_perfect, r_pilot = [], []
        for _ in range(num_draws):
            fresh.new_topology(csi_batch)
            h_true, _ = fresh.channel_and_no(tf.constant(csi_batch, tf.int32), tf.constant(snr, tf.float32))
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)
            r_perfect.append(float(fresh._sum_rate(h_true, system._call_precoder(h_true, no, training=False), no)))
            r_pilot.append(float(fresh._sum_rate(h_true, system._call_precoder(h_est, no, training=False), no)))
        perf, pil = float(np.mean(r_perfect)), float(np.mean(r_pilot))
        csi_imperfect[str(snr)] = {'perfect': perf, 'pilot20dB': pil,
                                    'pct_retained': 100.0 * pil / perf if perf > 0 else float('nan')}
        print(f'  SNR={snr:5.1f}dB | perfect={perf:7.2f} | pilot20dB={pil:7.2f} | '
              f'{csi_imperfect[str(snr)]["pct_retained"]:5.1f}% retenu', flush=True)

    flops, params, acts = system.precoder.complexity(system.rg.num_ofdm_symbols, convention_x_ofdm=False)
    out = {'arch': args.arch, 'channel': args.channel, 'scale': args.scale,
           'evals_perfect': evals, 'pct_of_wmmse_perfect': pct, 'csi_imperfect': csi_imperfect,
           'flops_real_M': flops / 1e6, 'params': int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables)),
           'train_time_min': train_time / 60, 'best_ckpt': trainer.ckpt.best_ckpt_path,
           'finetune_epochs': args.finetune_epochs, 'warmup_epochs': args.warmup_epochs, 'M': M, 'K': K}
    out_path = f'results/{args.channel}_{args.scale}_{args.arch}{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Sauvé -> {out_path}')
