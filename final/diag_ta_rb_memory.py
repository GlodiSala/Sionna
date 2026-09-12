"""
diag_ta_rb_memory.py — Bloc A.1 : profiling mémoire GPU par étape de
TransformerPrecoderV4 (v4.3, T=3) pendant forward+backward, à batch
croissant (32/64/128/256), pour identifier l'opération responsable de
l'OOM à batch>32.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_ta_rb_memory.py
"""
import os, sys, gc, traceback
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
from datasets import CachedSionnaDataset
from precoders_w import TransformerPrecoderV4
from channel_config import STANDARD_CONFIG, FFT_SIZE
from sionna.phy.utils import ebnodb2no

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
DEV = '/GPU:0'
print(f'GPUs visible: {gpus}')

NUM_TX = STANDARD_CONFIG['NUM_TX']
NUM_RX = STANDARD_CONFIG['NUM_RX']

# ── petit système bidon juste pour générer des h_freq de la bonne forme ──
from main_finall import MU_MIMO_System

print('Construction système (rzf, juste pour dataset)...')
dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
dataset = CachedSionnaDataset(
    dummy_sys, dataset_size=5000, batch_size=512,
    cache_file=f'/export/tmp/sala/sionna_base_5k_{NUM_TX}x{NUM_RX}.npz',
    augmentation_multiplier=50, seed=42)


def mem_mb():
    info = tf.config.experimental.get_memory_info('GPU:0')
    return info['current'] / 1e6, info['peak'] / 1e6


def reset():
    tf.config.experimental.reset_memory_stats('GPU:0')
    gc.collect()


def build_model():
    m = TransformerPrecoderV4(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=14, fft_size=FFT_SIZE,
        rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4,
        num_layers=4, version='v4.3')
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, 14, FFT_SIZE], dtype=tf.complex64)
    m(dummy_h, no=tf.constant(1e-3), training=False)
    n_params = sum(tf.size(v).numpy() for v in m.trainable_variables)
    print(f'Params: {n_params:,}')
    return m


def staged_forward_backward(model, h_freq, no):
    """Reproduit call() étape par étape, avec checkpoints mémoire.
    Retourne dict stage -> (current_MB, peak_MB_since_reset)."""
    stages = {}

    def snap(name):
        cur, peak = mem_mb()
        stages[name] = (cur, peak)

    with tf.GradientTape(persistent=False) as tape:
        h_freq_sg = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq_sg)[0]
        h_sq = tf.squeeze(h_freq_sg, axis=[2, 3])
        Bo = B * model.num_ofdm
        log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10)

        feat = model._extract_features(h_sq, B, log_no)
        snap('01_features')

        x = model.input_proj(feat)
        x = model.input_norm(x)
        snr_emb = tf.reshape(
            model.snr_proj(tf.reshape(log_no, [1, 1])), [1, 1, 1, model.embed_dim])
        x = x + snr_emb
        x = x + tf.reshape(model.pos_embed(tf.range(model.total_tokens)),
                            [1, model.total_tokens, 1, model.embed_dim])
        snap('02_input_proj_pos')

        for i, blk in enumerate(model.blocks):
            x = blk(x, training=True)
        snap('03_transformer_blocks')

        x = model.joint_output_proj(x)
        snap('04_joint_output_proj')

        x_user = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                             [Bo * model.num_rx, model.total_tokens, model.embed_dim])
        w_up = model.upsample(x_user)
        snap('05_upsample_conv1dtranspose')

        w_up_final = model.final_proj(w_up)
        snap('06_final_proj')

        h_bo = tf.reshape(tf.transpose(h_sq, [0, 3, 4, 1, 2]),
                           [Bo, model.fft_size, model.num_rx, model.num_tx])
        h_user = tf.reshape(tf.transpose(h_bo, [0, 2, 1, 3]),
                             [Bo * model.num_rx, model.fft_size, model.num_tx])
        h_re, h_im = tf.math.real(h_user), tf.math.imag(h_user)
        h_abs, h_phase = tf.abs(h_user), tf.math.angle(h_user)
        refine_input = tf.concat([w_up, h_re, h_im, h_abs, h_phase], axis=-1)
        snap('07_refine_input_concat')

        delta = model.sc_refine(refine_input, training=True)
        snap('08_sc_refine_conv1d')

        gate = model.sc_gate(w_up_final)
        w_final_flat = w_up_final + model.alpha * gate * delta
        snap('09_gate_sigmoid')

        w_final = tf.reshape(w_final_flat, [Bo, model.num_rx, model.fft_size, model.num_tx, 2])
        w_final = tf.transpose(w_final, [0, 2, 3, 1, 4])
        w_final = tf.reshape(w_final, [B, model.num_ofdm, model.fft_size, model.num_tx, model.num_rx, 2])
        w_re, w_im = w_final[..., 0], w_final[..., 1]
        s = tf.sqrt(1.0 / (tf.reduce_sum(w_re**2 + w_im**2, axis=3, keepdims=True) + 1e-12))
        w_re, w_im = w_re * s, w_im * s
        snap('10_power_norm')

        loss = tf.reduce_mean(w_re**2 + w_im**2)  # loss factice, juste pour le backward

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g for g in grads if g is not None]
    _ = [tf.reduce_sum(g) for g in grads]  # force réalisation
    snap('11_after_backward')

    return stages


results = {}
for bs in [32, 64, 128, 256]:
    print(f'\n{"="*60}\nBATCH = {bs}\n{"="*60}')
    reset()
    try:
        with tf.device(DEV):
            model = build_model()
            h = dataset.get_batch(bs)
            no = ebnodb2no(tf.constant(15.0), 2, 0.5, dummy_sys.rg)
            reset()
            stages = staged_forward_backward(model, h, no)
        for name, (cur, peak) in stages.items():
            print(f'  {name:<32} current={cur:8.1f} MB   peak={peak:8.1f} MB')
        results[bs] = stages
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    except tf.errors.ResourceExhaustedError as e:
        print(f'  ❌ OOM à batch={bs}')
        print(str(e)[:500])
        results[bs] = 'OOM'
        gc.collect()
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f'  ❌ Erreur à batch={bs}: {e}')
        traceback.print_exc()
        results[bs] = f'ERROR: {e}'
        gc.collect()
        tf.keras.backend.clear_session()

print('\n\n=== RÉSUMÉ : delta de peak mémoire par étape (MB), batch le plus grand réussi ===')
ok_batches = [b for b, r in results.items() if isinstance(r, dict)]
if ok_batches:
    b_max = max(ok_batches)
    st = results[b_max]
    names = list(st.keys())
    prev_peak = 0.0
    for n in names:
        cur, peak = st[n]
        print(f'  batch={b_max:4d}  {n:<32} peak_total={peak:9.1f} MB   Δpeak_vs_prev={peak-prev_peak:9.1f} MB')
        prev_peak = peak

np.save('results/diag_ta_rb_memory.npy', results, allow_pickle=True)
print('\nSauvé -> results/diag_ta_rb_memory.npy')
