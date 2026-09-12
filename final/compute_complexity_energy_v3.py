"""
compute_complexity_energy_v3.py — ÉTAPE 2 (SESSION_NUIT_RESUME.md).

Recalcule FLOPs/params/énergie pour les 3 architectures nettoyées
(single_sc, intra_rb, ta_rb) aux hyperparamètres de production
(embed_dim=128, num_layers=4, TA-RB à 3 tok/RB), sur les DEUX configs
verrouillées (STANDARD M8K4, MASSIVE M32K8 -- remplace M64K32, obsolète
depuis la révision clustering §1).

Succède à complexity_energy_results_v2.csv : cette fois les 3 méthodes de
calcul de complexité elles-mêmes ont été revalidées (pas juste
réappliquées) -- voir precoders_v2.py/precoder_intra_rb.py, chaque
complexity() a été vérifiée EXACTEMENT contre
sum(tf.size(v) for v in trainable_variables) avant ce calcul (script de
vérification : voir SESSION_LOG_20260807.md, ÉTAPE 2).

Usage: CUDA_VISIBLE_DEVICES=0 python3 compute_complexity_energy_v3.py
"""
import os, sys, csv
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

from precoders_v2 import TransformerPrecoderClean
from precoder_intra_rb import IntraRBTransformerPrecoder, SingleSCTransformerPrecoder
from main_finall import compute_energy_uJ, compute_classical_complexity, Q_W, Q_A
from channel_config import STANDARD_CONFIG, MASSIVE_CONFIG, FFT_SIZE

NUM_OFDM = 14
D, L, H = 128, 4, 4          # production hyperparams (MODELS_TO_TRAIN)
TOKENS_PER_RB = 3

rows = []


def add_row(config_name, method, flops_real, weights, acts_real,
            flops_conv, acts_conv, q_w, q_a, note=''):
    e_real = compute_energy_uJ(flops_real, weights, acts_real, q_w, q_a)
    e_conv = (compute_energy_uJ(flops_conv, weights, acts_conv, q_w, q_a)
              if flops_conv is not None else None)
    rows.append(dict(config=config_name, method=method,
                      params_k=weights / 1e3,
                      flops_real_m=flops_real / 1e6,
                      energy_real_uj=e_real,
                      flops_conv_m=(flops_conv / 1e6) if flops_conv is not None else None,
                      energy_conv_uj=e_conv,
                      note=note))


for cfg_name, cfg in [('STANDARD (8x4, R=20m)', STANDARD_CONFIG),
                       ('MASSIVE (32x8, R=5m)', MASSIVE_CONFIG)]:
    M, K = cfg['NUM_TX'], cfg['NUM_RX']
    print(f'\n{"="*70}\n{cfg_name}\n{"="*70}', flush=True)

    h_dummy = tf.zeros([1, K, 1, 1, M, NUM_OFDM, FFT_SIZE], dtype=tf.complex64)

    # ── Classiques (FP32, Q=32) ────────────────────────────────────────────
    f_rzf, w_rzf, a_rzf = compute_classical_complexity('RZF', M, K, FFT_SIZE, NUM_OFDM)
    add_row(cfg_name, 'RZF', f_rzf, w_rzf, a_rzf, None, None, 32, 32)

    f_wmmse, w_wmmse, a_wmmse = compute_classical_complexity('WMMSE', M, K, FFT_SIZE, NUM_OFDM, I_wmmse=10)
    add_row(cfg_name, 'WMMSE (I=10)', f_wmmse, w_wmmse, a_wmmse, None, None, 32, 32)

    # ── Single-SC (pas de fix OFDM à proprement parler : déjà 1-symbole natif,
    #    mais complexity() applique quand même le multiplicateur x14 "historique") ──
    m_single = SingleSCTransformerPrecoder(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                            fft_size=FFT_SIZE, embed_dim=D,
                                            num_heads=H, num_layers=L)
    _ = m_single(h_dummy, no=tf.constant(1e-3))
    f_r, w_s, a_r = m_single.complexity(NUM_OFDM, convention_x_ofdm=False)
    f_c, _,   a_c = m_single.complexity(NUM_OFDM, convention_x_ofdm=True)
    add_row(cfg_name, 'Single-SC (feat réduites, 3M+1)', f_r, w_s, a_r, f_c, a_c, Q_W, Q_A)

    # ── IntraRB nettoyé ─────────────────────────────────────────────────────
    m_intra = IntraRBTransformerPrecoder(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                          fft_size=FFT_SIZE, rb_size=12,
                                          embed_dim=D, num_heads=H, num_layers=L)
    _ = m_intra(h_dummy, no=tf.constant(1e-3))
    f_r, w_i, a_r = m_intra.complexity(NUM_OFDM, convention_x_ofdm=False)
    f_c, _,   a_c = m_intra.complexity(NUM_OFDM, convention_x_ofdm=True)
    add_row(cfg_name, 'IntraRB nettoyé (feat réduites, 3M+1, RB=12)', f_r, w_i, a_r, f_c, a_c, Q_W, Q_A)

    # ── TA-RB nettoyé ───────────────────────────────────────────────────────
    m_tarb = TransformerPrecoderClean(num_tx=M, num_rx=K, num_ofdm=NUM_OFDM,
                                       fft_size=FFT_SIZE, rb_size=12,
                                       tokens_per_rb=TOKENS_PER_RB,
                                       embed_dim=D, num_heads=H, num_layers=L)
    _ = m_tarb(h_dummy, no=tf.constant(1e-3))
    f_r, w_t, a_r = m_tarb.complexity(NUM_OFDM, convention_x_ofdm=False)
    f_c, _,   a_c = m_tarb.complexity(NUM_OFDM, convention_x_ofdm=True)
    add_row(cfg_name, f'TA-RB nettoyé T={TOKENS_PER_RB} (fix OFDM+déc. v4.2 sans gate, feat mean+var)',
            f_r, w_t, a_r, f_c, a_c, Q_W, Q_A)

    tf.keras.backend.clear_session()

    # ── Impression tableau lisible ──────────────────────────────────────────
    print(f'\n{"Méthode":<48} {"Params(K)":>10} {"FLOPs réel(M)":>14} {"Energie réelle(µJ)":>19} '
          f'{"vs WMMSE F":>11} {"vs WMMSE E":>11}')
    wmmse_row = next(r for r in rows if r['config'] == cfg_name and r['method'] == 'WMMSE (I=10)')
    for r in rows:
        if r['config'] != cfg_name:
            continue
        vs_f = wmmse_row['flops_real_m'] / r['flops_real_m'] if r['flops_real_m'] > 0 else float('nan')
        vs_e = wmmse_row['energy_real_uj'] / r['energy_real_uj'] if r['energy_real_uj'] > 0 else float('nan')
        print(f'{r["method"]:<48} {r["params_k"]:>10.1f} {r["flops_real_m"]:>14.2f} '
              f'{r["energy_real_uj"]:>19.4f} {vs_f:>10.2f}× {vs_e:>10.2f}×')


# ── Sauvegarde CSV ──────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'complexity_energy_results_v3.csv')
with open(out_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['Config', 'Method', 'Params (K)', 'FLOPs réel (M)', 'Énergie réelle (µJ)',
                 'FLOPs convention x14 (M)', 'Énergie convention (µJ)', 'Note'])
    for r in rows:
        wr.writerow([r['config'], r['method'], f"{r['params_k']:.1f}",
                     f"{r['flops_real_m']:.3f}", f"{r['energy_real_uj']:.4f}",
                     f"{r['flops_conv_m']:.1f}" if r['flops_conv_m'] is not None else '-',
                     f"{r['energy_conv_uj']:.4f}" if r['energy_conv_uj'] is not None else '-',
                     r['note']])
print(f'\n✅ Sauvé -> {out_path}')
