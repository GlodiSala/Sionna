"""
Time actual training steps (forward + backward + optimizer.apply_gradients)
for both architectures at both locked configs, to estimate real training
duration (distinct from the earlier dataset-generation timing, which only
covers channel generation, not the gradient step itself).

Reuses main_finall.MU_MIMO_System + SupervisedTrainer's real _warmup_step/
_finetune_step (@tf.function-decorated, identical to what actual training
runs), fed with a real generated channel batch (safe batch size 32,
concatenated up to the real training BATCH_SIZE to avoid the known GPU
ComplexAbs kernel crash at large batch*K*M*fft -- that crash is specific to
channel GENERATION via cir_to_ofdm_channel, not the training step itself,
which only consumes an already-materialized h_freq tensor).
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG

import main_finall as mf

SAFE_GEN_BATCH = 32
TRAIN_BATCH_SIZE = 128   # matches main_finall.BATCH_SIZE
N_TIMED_STEPS = 10


class DummyDataset:
    effective_dataset_size = 250_000   # matches DATASET_SIZE=5000 x aug=50


def make_batch(M, K, cfg, target_batch):
    gen_sys = ConfigurableMIMOSystem(M, K, half_angle_deg=cfg['HALF_ANGLE_DEG'],
                                      los=cfg['FORCE_LOS'],
                                      indoor_probability=cfg['INDOOR_PROBABILITY'])
    chunks = []
    remaining = target_batch
    while remaining > 0:
        b = min(SAFE_GEN_BATCH, remaining)
        gen_sys.new_topology(b)
        chunks.append(gen_sys._gen_channel(b))
        remaining -= b
    return tf.concat(chunks, axis=0)


def time_steps(trainer, h_freq, step_fn_name, n=N_TIMED_STEPS):
    step_fn = getattr(trainer, step_fn_name)
    # Warmup call (tf.function tracing, excluded from timing)
    step_fn(h_freq)
    if tf.config.list_physical_devices('GPU'):
        tf.test.experimental.sync_devices()
    t0 = time.time()
    for _ in range(n):
        step_fn(h_freq)
    if tf.config.list_physical_devices('GPU'):
        tf.test.experimental.sync_devices()
    elapsed = time.time() - t0
    return elapsed / n


def run(name, cfg, precoder_type, version='v4.0'):
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    print(f"\n{'='*70}\n{name} M={M} K={K} precoder_type={precoder_type} version={version}\n{'='*70}")

    for batch_size in [TRAIN_BATCH_SIZE, 64, 32, 16]:
        try:
            system = mf.MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=precoder_type,
                                        version=version)
            trainer = mf.SupervisedTrainer(system, DummyDataset(),
                                            run_name=f'{name}_{precoder_type}',
                                            batch_size=batch_size)
            h_freq = make_batch(M, K, cfg, batch_size)
            h_freq = tf.cast(h_freq, tf.complex64)

            t_warmup = time_steps(trainer, h_freq, '_warmup_step')
            t_finetune = time_steps(trainer, h_freq, '_finetune_step')
            n_params = sum(int(tf.size(v)) for v in trainer.vars)
            if batch_size != TRAIN_BATCH_SIZE:
                print(f"  NOTE: BATCH_SIZE={TRAIN_BATCH_SIZE} (main_finall.py default) "
                      f"OOM'd -- fell back to batch_size={batch_size}")
            break
        except tf.errors.ResourceExhaustedError:
            print(f"  OOM at batch_size={batch_size}, trying smaller...")
            tf.keras.backend.clear_session()
            continue
    else:
        print("  FAILED at all batch sizes tried")
        return dict(name=name, precoder_type=precoder_type, M=M, K=K, failed=True)

    print(f"  params={n_params:,}  (batch_size used: {batch_size})")
    print(f"  warmup_step:   {t_warmup*1000:7.1f} ms/step")
    print(f"  finetune_step: {t_finetune*1000:7.1f} ms/step")

    iters_per_epoch = max(DummyDataset.effective_dataset_size // batch_size, 1)
    warmup_epoch_s = iters_per_epoch * t_warmup
    finetune_epoch_s = iters_per_epoch * t_finetune
    print(f"  iters/epoch (at effective_dataset_size={DummyDataset.effective_dataset_size:,}, "
          f"batch={batch_size}): {iters_per_epoch}")
    print(f"  ~time/epoch: warmup={warmup_epoch_s:.1f}s  finetune={finetune_epoch_s:.1f}s")

    return dict(name=name, precoder_type=precoder_type, version=version, M=M, K=K,
                n_params=n_params, batch_size_used=batch_size,
                t_warmup_ms=t_warmup*1000, t_finetune_ms=t_finetune*1000,
                iters_per_epoch=iters_per_epoch,
                warmup_epoch_s=warmup_epoch_s, finetune_epoch_s=finetune_epoch_s)


if __name__ == '__main__':
    all_results = []
    for name, cfg in [('STANDARD', STANDARD_CONFIG), ('MASSIVE', MASSIVE_CONFIG)]:
        for precoder_type in ['intra_rb', 'transformer_rb']:
            r = run(name, cfg, precoder_type)
            all_results.append(r)

    os.makedirs('./results', exist_ok=True)
    np.save('./results/training_step_timing.npy', all_results, allow_pickle=True)

    print(f"\n{'='*90}\nSUMMARY\n{'='*90}")
    wu, ft = mf.TRAINING_CONFIG['warmup_epochs'], mf.TRAINING_CONFIG['finetune_epochs']
    for r in all_results:
        if r.get('failed'):
            print(f"  {r['name']:9s} {r['precoder_type']:15s} M={r['M']:3d} K={r['K']:3d}  FAILED (OOM at all batch sizes tried)")
            continue
        total_s = wu * r['warmup_epoch_s'] + ft * r['finetune_epoch_s']
        print(f"  {r['name']:9s} {r['precoder_type']:15s} M={r['M']:3d} K={r['K']:3d}  "
              f"batch={r['batch_size_used']:3d}  "
              f"params={r['n_params']:>12,}  "
              f"warmup={r['t_warmup_ms']:6.1f}ms/step  finetune={r['t_finetune_ms']:6.1f}ms/step  "
              f"~total ({wu}+{ft} epochs)={total_s/60:.1f} min")
