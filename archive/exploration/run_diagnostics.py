#!/usr/bin/env python3
# run_diagnostics.py
# Usage: python final/run_diagnostics.py --gpu 0 --n_batches 6 2>&1 | tee diag_output.log

import os, sys, json, argparse, warnings, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import matplotlib as plt
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--weights_root',
    default='/export/tmp/sala/test_projet/Trans/freq/Sionna/final/weights')
parser.add_argument('--n_batches', type=int, default=6)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

sys.path.insert(0, '/export/tmp/sala/test_projet/Trans/freq/Sionna')
sys.path.insert(0, '/export/tmp/sala/test_projet/Trans/freq/Sionna/final')

from precoders_w import (TransformerPrecoderV4, TransformerPrecoderV5,
                          wmmse_precoder, rzf_precoder)

import sionna
from sionna.phy.channel import (cir_to_ofdm_channel, subcarrier_frequencies,
                                 gen_single_sector_topology as gen_topology)
from sionna.phy.channel.tr38901 import AntennaArray, UMi
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement

# ═══════════════════════════════════════════════════════════════════════════════
# Config système — identique à l'entraînement
# ═══════════════════════════════════════════════════════════════════════════════
NUM_TX   = 8
NUM_RX   = 4
FFT_SIZE = 72
NUM_OFDM = 14
CARRIER  = 2.6e9
SC_SPACE = 30e3

rg = ResourceGrid(
    num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
    subcarrier_spacing=SC_SPACE, num_tx=1,
    num_streams_per_tx=NUM_RX, cyclic_prefix_length=6,
    pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

ut_array = AntennaArray(
    num_rows=1, num_cols=1, polarization="single",
    polarization_type="V", antenna_pattern="omni",
    carrier_frequency=CARRIER)
bs_array = AntennaArray(
    num_rows=1, num_cols=int(NUM_TX/2), polarization="dual",
    polarization_type="cross", antenna_pattern="38.901",
    carrier_frequency=CARRIER)

channel_model = UMi(
    carrier_frequency=CARRIER, o2i_model="low",
    ut_array=ut_array, bs_array=bs_array,
    direction='downlink', enable_pathloss=False, enable_shadow_fading=False)

rx_tx_assoc = np.ones([NUM_RX, 1])
sm = StreamManagement(rx_tx_assoc, num_streams_per_tx=NUM_RX)

NO_15DB = tf.cast(10.0 ** (-15.0 / 10.0), tf.float32)

# ── COMMON kwargs par famille ──────────────────────────────────────────────────
COMMON_V4 = dict(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=NUM_OFDM,
                 fft_size=FFT_SIZE, rb_size=12)
COMMON_V5 = dict(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=NUM_OFDM,
                 fft_size=FFT_SIZE)

# ── Checkpoints : (run_name, cls, common_kw, extra_kw, subdir) ────────────────
W = args.weights_root
CHECKPOINTS = [
    ('V4.2_1tok', TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=1, version='v4.2', embed_dim=128, num_heads=4, num_layers=4),
     'V4.2_1tok-RB'),
    ('V4.2_2tok', TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=2, version='v4.2', embed_dim=128, num_heads=4, num_layers=4),
     'V4.2_2tok-RB'),
    ('V4.2_3tok', TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=3, version='v4.2', embed_dim=128, num_heads=4, num_layers=4),
     'V4.2_3tok-RB'),
    ('V4.2_4tok', TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=4, version='v4.2', embed_dim=128, num_heads=4, num_layers=4),
     'V4.2_4tok-RB'),
    ('V4.2_6tok', TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=6, version='v4.2', embed_dim=128, num_heads=4, num_layers=4),
     'V4.2_6tok-RB'),
    ('V4_4tok',   TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=4, version='v4.0', embed_dim=128, num_heads=4, num_layers=4),
     'V4_4tok-RB'),
    ('V4_6tok',   TransformerPrecoderV4, COMMON_V4,
     dict(tokens_per_rb=6, version='v4.0', embed_dim=128, num_heads=4, num_layers=4),
     'V4_6tok-RB'),
    ('V5.2_2L',   TransformerPrecoderV5, COMMON_V5,
     dict(embed_dim=128, num_heads=4, num_intra_layers=2, num_inter_layers=2,
          use_swin_shift=False),
     'V5.2_2L_128d'),
    ('V5Base_2L', TransformerPrecoderV5, COMMON_V5,
     dict(embed_dim=128, num_heads=4, num_intra_layers=2, num_inter_layers=2,
          use_swin_shift=True),
     'V5-Base_2L_128d'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Génération des canaux
# ═══════════════════════════════════════════════════════════════════════════════
def make_channel_batch(batch_size=32, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    topology = gen_topology(batch_size, NUM_RX, 'umi',
                            min_ut_velocity=0.0, max_ut_velocity=0.0)
    channel_model.set_topology(*topology)
    cir = channel_model(batch_size, rg.num_ofdm_symbols,
                        1.0 / rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    return h_freq   # [B, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE]

print("Génération des batches de canaux...")
batches = [make_channel_batch(32, seed=i*7) for i in range(args.n_batches)]
print(f"  {args.n_batches} batches — shape: {batches[0].shape}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des poids (format SimpleCheckpoint → weights.pkl)
# ═══════════════════════════════════════════════════════════════════════════════
def find_best_pkl(ckpt_dir):
    """
    Cherche parmi les sous-dossiers best_YYYYMMDD_HHMMSS celui qui a
    le meilleur sum_rate selon metrics.json.
    Retourne (pkl_path, best_rate) ou (None, -1).
    """
    best_dirs = sorted(
        [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir)
         if d.startswith('best_') and
         os.path.isfile(os.path.join(ckpt_dir, d, 'weights.pkl'))],
        reverse=True)   # plus récent en premier (fallback)

    if not best_dirs:
        return None, -1.0

    best_path = os.path.join(best_dirs[0], 'weights.pkl')  # fallback = plus récent
    best_rate = -1.0

    for d in best_dirs:
        mfile = os.path.join(d, 'metrics.json')
        try:
            with open(mfile) as f:
                m = json.load(f)
            # metrics.json peut utiliser différentes clés selon la version
            rate = float(m.get('sum_rate',
                         m.get('val_sum_rate',
                         m.get('best_sum_rate', -1.0))))
            if rate > best_rate:
                best_rate = rate
                best_path = os.path.join(d, 'weights.pkl')
        except Exception:
            pass   # pas de metrics.json lisible → garde le plus récent

    return best_path, best_rate


def load_model(cls, common_kw, extra_kw, ckpt_dir):
    """Construit le modèle, trouve le meilleur checkpoint, charge les poids."""
    model = cls(**common_kw, **extra_kw)
    dummy = tf.zeros([2, NUM_RX, 1, 1, NUM_TX, NUM_OFDM, FFT_SIZE],
                     dtype=tf.complex64)
    _ = model(dummy, training=False)

    pkl_path, best_rate = find_best_pkl(ckpt_dir)
    if pkl_path is None:
        raise RuntimeError(f"Aucun weights.pkl trouvé dans {ckpt_dir}")

    rel = os.path.relpath(pkl_path, ckpt_dir)
    print(f"    ✓ {rel}  (sum_rate={best_rate:.3f})")

    with open(pkl_path, 'rb') as f:
        saved_weights = pickle.load(f)

    model_vars = model.trainable_variables
    if len(saved_weights) != len(model_vars):
        raise RuntimeError(
            f"Mismatch poids: sauvés={len(saved_weights)} "
            f"vs modèle={len(model_vars)}")

    for var, w in zip(model_vars, saved_weights):
        var.assign(w)

    return model

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
def attn_entropy(w_np):
    """w_np: [B, heads, seq_q, seq_k] → entropie moyenne par head [heads]"""
    p = np.clip(w_np, 1e-9, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)   # [B, heads, seq_q]
    return H.mean(axis=(0, 2)).tolist()

def inter_user_corr(W_np):
    """W_np: [B, ofdm, fft, tx, rx] → corrélation inter-user moyenne scalaire"""
    K = W_np.shape[-1]
    corrs = []
    for i in range(K):
        for j in range(i+1, K):
            wi = W_np[..., i]   # [B, ofdm, fft, tx]
            wj = W_np[..., j]
            dot = np.abs(np.sum(np.conj(wi) * wj, axis=-1))   # [B, ofdm, fft]
            ni  = np.linalg.norm(wi, axis=-1)
            nj  = np.linalg.norm(wj, axis=-1)
            corrs.append((dot / (ni * nj + 1e-12)).mean())
    return float(np.mean(corrs))

def cosine_sim(Wa, Wb):
    """Similarité cosine moyenne des vecteurs de précodage par user"""
    K = Wa.shape[-1]
    sims = []
    for k in range(K):
        wa = Wa[..., k].reshape(-1, Wa.shape[-2])
        wb = Wb[..., k].reshape(-1, Wb.shape[-2])
        dot = np.abs(np.sum(np.conj(wa) * wb, axis=-1))
        na  = np.linalg.norm(wa, axis=-1)
        nb  = np.linalg.norm(wb, axis=-1)
        sims.append((dot / (na * nb + 1e-12)).mean())
    return float(np.mean(sims))

def get_W(model, h):
    """Sortie précoder → [B, ofdm, fft, tx, rx]"""
    return tf.squeeze(model(h, training=False), axis=1).numpy()

def get_wmmse(h):
    return tf.squeeze(
        wmmse_precoder(h, NO_15DB, sm, num_iterations=10), axis=1).numpy()

def get_rzf(h):
    return tf.squeeze(
        rzf_precoder(h, sm, no=NO_15DB), axis=1).numpy()

def scale_weights(model):
    """Extrait les scalaires appris (s_sc, s_user, alpha, etc.)"""
    out = {}
    keywords = {'s_sc', 's_user', 's_ffn', 's_rb', 's_cross',
                's_freq', 's_attn', 'alpha'}
    for v in model.trainable_variables:
        short = v.name.split('/')[-1].replace(':0', '')
        if short in keywords:
            out[v.name] = float(v.numpy())
    return out

def diag_v5_attention(model, h_sample):
    """Parcourt intra + inter blocks et retourne les entropies d'attention."""
    h = tf.stop_gradient(h_sample[:16])
    B  = tf.shape(h)[0]
    h_sq = tf.squeeze(h, axis=[2, 3])
    Bo   = B * model.num_ofdm

    feat = model._extract_features(h_sq, B)
    x    = model.input_norm(model.input_proj(feat))

    sc_pos = tf.tile(tf.range(model.rb_size), [model.num_rb])
    x = x + tf.reshape(model.pos_embed_sc(sc_pos),
                        [1, model.fft_size, 1, model.embed_dim])
    bias = tf.broadcast_to(
        tf.reshape(model.rb_pos_bias,
                   [1, model.num_rb, model.rb_size, 1, model.embed_dim]),
        [Bo, model.num_rb, model.rb_size, model.num_rx, model.embed_dim])
    x = x + tf.reshape(bias, [Bo, model.fft_size, model.num_rx, model.embed_dim])

    intra_ent = []
    for blk in model.intra_blocks:
        x_rb = tf.reshape(x, [Bo * model.num_rb,
                               model.rb_size, model.num_rx, model.embed_dim])

        # SC attention entropy
        xf   = tf.reshape(tf.transpose(x_rb, [0, 2, 1, 3]),
                          [int(Bo) * model.num_rb * model.num_rx,
                           model.rb_size, model.embed_dim])
        xf_n = blk.norm_sc(xf)
        _, sc_w = blk.sc_attn(xf_n, xf_n,
                               return_attention_scores=True, training=False)

        # User attention entropy
        xu   = tf.reshape(x_rb, [int(Bo) * model.num_rb * model.rb_size,
                                  model.num_rx, model.embed_dim])
        xu_n = blk.norm_user(xu)
        _, u_w = blk.user_attn(xu_n, xu_n,
                                return_attention_scores=True, training=False)

        sc_e = attn_entropy(sc_w.numpy())
        u_e  = attn_entropy(u_w.numpy())
        intra_ent.append({'sc': sc_e, 'user': u_e,
                          'sc_mean':   float(np.mean(sc_e)),
                          'user_mean': float(np.mean(u_e))})

        x_rb = blk(x_rb, training=False)
        x = tf.reshape(x_rb, [Bo, model.fft_size, model.num_rx, model.embed_dim])

    # RB pooling — gère mean+max (V5.2) et mean-only (legacy)
    x_rb2   = tf.reshape(x, [int(Bo) * model.num_rb,
                               model.rb_size, model.num_rx, model.embed_dim])
    rb_mean = tf.reduce_mean(x_rb2, axis=1)
    rb_max  = tf.reduce_max(x_rb2,  axis=1)
    try:
        rb_tok = model.rb_pool_proj(tf.concat([rb_mean, rb_max], axis=-1))
    except Exception:
        rb_tok = model.rb_pool_proj(rb_mean)   # legacy mean-only

    rb_tokens = tf.reshape(rb_tok,
                           [Bo, model.num_rb, model.num_rx, model.embed_dim])
    rb_tokens = rb_tokens + tf.reshape(
        model.pos_embed_rb(tf.range(model.num_rb)),
        [1, model.num_rb, 1, model.embed_dim])

    inter_ent = []
    for blk in model.inter_blocks:
        # RB attention entropy
        xr   = tf.reshape(tf.transpose(rb_tokens, [0, 2, 1, 3]),
                          [int(Bo) * model.num_rx, model.num_rb, model.embed_dim])
        xr_n = blk.norm_rb(xr)
        _, rb_w = blk.rb_attn(xr_n, xr_n,
                               return_attention_scores=True, training=False)

        # User attention entropy
        xu   = tf.reshape(rb_tokens,
                          [int(Bo) * model.num_rb, model.num_rx, model.embed_dim])
        xu_n = blk.norm_user(xu)
        _, u_w = blk.user_attn(xu_n, xu_n,
                                return_attention_scores=True, training=False)

        rb_e = attn_entropy(rb_w.numpy())
        u_e  = attn_entropy(u_w.numpy())
        inter_ent.append({'rb': rb_e, 'user': u_e,
                          'rb_mean':   float(np.mean(rb_e)),
                          'user_mean': float(np.mean(u_e))})

        rb_tokens = blk(rb_tokens, training=False)

    return intra_ent, inter_ent

# ═══════════════════════════════════════════════════════════════════════════════
# Références WMMSE / RZF
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Références baselines (batch 0)")
print("=" * 65)
Ww0 = get_wmmse(batches[0])
Wr0 = get_rzf(batches[0])
ref_corr_wmmse = inter_user_corr(Ww0)
ref_corr_rzf   = inter_user_corr(Wr0)
print(f"  WMMSE corr inter-user : {ref_corr_wmmse:.4f}")
print(f"  RZF   corr inter-user : {ref_corr_rzf:.4f}")
print(f"  Entropie uniforme SC  : log(12) = {np.log(12):.3f}")
print(f"  Entropie uniforme user: log(4)  = {np.log(4):.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Boucle principale
# ═══════════════════════════════════════════════════════════════════════════════
all_results = {}

for (run_name, cls, common_kw, extra_kw, subdir) in CHECKPOINTS:
    ckpt_dir = os.path.join(W, subdir)
    print(f"\n{'=' * 65}")
    print(f"  {run_name}  [{subdir}]")
    print(f"{'=' * 65}")

    try:
        model = load_model(cls, common_kw, extra_kw, ckpt_dir)
    except Exception as e:
        print(f"  ✗ SKIP: {e}")
        all_results[run_name] = {'error': str(e)}
        continue

    res = {'run': run_name}

    # ── D1: Poids d'échelle ─────────────────────────────────────────────────
    scales = scale_weights(model)
    res['scale_weights'] = scales
    print(f"\n  D1 — Scale weights:")
    if not scales:
        print(f"    (aucun trouvé)")
    for name, val in scales.items():
        short = name.split('/')[-1].replace(':0', '')
        flag = ("  ← MORT"     if abs(val) < 0.01 else
                "  ← DOMINANT" if abs(val) > 1.0  else "")
        print(f"    {short:<35} {val:+.4f}{flag}")

    # ── D2: Entropie attention (V5 seulement) ───────────────────────────────
    if cls is TransformerPrecoderV5:
        print(f"\n  D2 — Entropie attention:")
        print(f"    Seuil uniforme : SC=log(12)={np.log(12):.3f} | "
              f"user=log(4)={np.log(4):.3f} | RB=log(6)={np.log(6):.3f}")
        try:
            intra_e, inter_e = diag_v5_attention(model, batches[0])
            res['intra_entropy'] = intra_e
            res['inter_entropy'] = inter_e
            for i, e in enumerate(intra_e):
                sc_flag   = "  ← UNIFORME (mort)" if e['sc_mean']   > 2.3 else ""
                user_flag = "  ← UNIFORME (mort)" if e['user_mean'] > 1.3 else ""
                print(f"    IntraRB[{i}] SC   "
                      f"{[f'{v:.3f}' for v in e['sc']]}  "
                      f"moy={e['sc_mean']:.3f}{sc_flag}")
                print(f"    IntraRB[{i}] User "
                      f"{[f'{v:.3f}' for v in e['user']]}  "
                      f"moy={e['user_mean']:.3f}{user_flag}")
            for i, e in enumerate(inter_e):
                rb_flag   = "  ← UNIFORME (mort)" if e['rb_mean']   > 1.7 else ""
                user_flag = "  ← UNIFORME (mort)" if e['user_mean'] > 1.3 else ""
                print(f"    InterRB[{i}] RB   "
                      f"{[f'{v:.3f}' for v in e['rb']]}  "
                      f"moy={e['rb_mean']:.3f}{rb_flag}")
                print(f"    InterRB[{i}] User "
                      f"{[f'{v:.3f}' for v in e['user']]}  "
                      f"moy={e['user_mean']:.3f}{user_flag}")
        except Exception as e:
            print(f"    ✗ Erreur: {e}")
            res['attn_error'] = str(e)

    # ── D3/D4: Corrélation inter-user + cosine vers WMMSE/RZF ──────────────
    print(f"\n  D3/D4 — Séparation spatiale & alignement directionnel:")
    corrs, sims_w, sims_r = [], [], []
    for h in batches:
        try:
            Wm = get_W(model, h)
            Ww = get_wmmse(h)
            Wr = get_rzf(h)
            corrs.append(inter_user_corr(Wm))
            sims_w.append(cosine_sim(Wm, Ww))
            sims_r.append(cosine_sim(Wm, Wr))
        except Exception as e:
            print(f"    ✗ batch: {e}")

    if corrs:
        cm = float(np.mean(corrs))
        sw = float(np.mean(sims_w))
        sr = float(np.mean(sims_r))
        res.update({'inter_user_corr': cm,
                    'cosine_wmmse':    sw,
                    'cosine_rzf':      sr})

        print(f"    Corr inter-user : {cm:.4f}  "
              f"[ref WMMSE={ref_corr_wmmse:.4f}, RZF={ref_corr_rzf:.4f}]")
        print(f"    Cosine → WMMSE  : {sw:.4f}")
        print(f"    Cosine → RZF    : {sr:.4f}")

        if sr > sw + 0.05:
            diag = "⚠️  BLOQUÉ PRIOR RZF — réentraîner from scratch"
        elif sw > 0.85:
            diag = "✅ Bonne direction WMMSE — gap résiduel = puissance/amplitude"
        elif cm > ref_corr_rzf + 0.05:
            diag = "⚠️  SÉPARATION SPATIALE INSUFFISANTE — interférence dominante"
        else:
            diag = "→ Structure propre (ni RZF ni WMMSE pur)"
        print(f"    Diagnostic : {diag}")
        res['diagnostic'] = diag

    all_results[run_name] = res
    tf.keras.backend.clear_session()

# ═══════════════════════════════════════════════════════════════════════════════
# Rapport consolidé
# ═══════════════════════════════════════════════════════════════════════════════
header = (f"\n\n{'=' * 65}\n"
          f"RAPPORT CONSOLIDÉ\n"
          f"{'=' * 65}\n"
          f"Réf: WMMSE corr={ref_corr_wmmse:.4f} | "
          f"RZF corr={ref_corr_rzf:.4f}\n")
print(header)

table_header = f"{'Modèle':<20} {'Corr↓':>8} {'→WMMSE↑':>9} {'→RZF↑':>8}  Diagnostic"
print(table_header)
print("-" * 65)

lines = [header, table_header, "-" * 65]
for name, res in all_results.items():
    if 'error' in res:
        line = f"{name:<20}  ERREUR: {res['error'][:42]}"
    else:
        cm   = res.get('inter_user_corr', float('nan'))
        sw   = res.get('cosine_wmmse',    float('nan'))
        sr   = res.get('cosine_rzf',      float('nan'))
        diag = res.get('diagnostic', '')
        line = (f"{name:<20} {cm:>8.4f} {sw:>9.4f} {sr:>8.4f}  {diag}")
    print(line)
    lines.append(line)

# Résumé poids d'échelle notables
print("\nPoids d'échelle notables:")
lines.append("\nPoids d'échelle notables:")
for name, res in all_results.items():
    if 'scale_weights' not in res:
        continue
    dead = [k.split('/')[-1].replace(':0', '')
            for k, v in res['scale_weights'].items() if abs(v) < 0.01]
    dom  = [k.split('/')[-1].replace(':0', '')
            for k, v in res['scale_weights'].items() if abs(v) > 1.0]
    if dead:
        l = f"  {name}: MORTS    = {dead}"
        print(l); lines.append(l)
    if dom:
        l = f"  {name}: DOMINANTS = {dom}"
        print(l); lines.append(l)

# Sauvegarde
with open('diag_report.txt', 'w') as f:
    f.write('\n'.join(lines))
with open('diag_report.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n✅ diag_report.txt + diag_report.json sauvegardés")