"""
diag_ber_mechanism_investigation.py — deux investigations demandées en
parallèle de P1 (poids figés, pas d'entraînement) :

INVESTIGATION 1 (SingleSC vs RZF/WMMSE) : le sum-rate (moyenne de
log(1+SINR)) écrase-t-il une queue basse de SINR invisible en moyenne
mais dominante pour le BER ? Distribution complète du SINR par (RE,user),
percentiles bas, corrélation erreur<->SINR bas, sur les MÊMES réalisations
de canal (paired via call_cached) pour comparaison directe.

INVESTIGATION 2 (IntraRB vs SingleSC) : les erreurs d'IntraRB sont-elles
groupées en rafales dans le codeword (Fano factor / indice de dispersion
par bloc-SC) et/ou corrélées entre users à une même RE (lift de
co-occurrence), plus que SingleSC ?

Vérification pipeline (partie 3 des deux investigations) : `_forward_
from_precoder` (main_finall.py) est un SEUL chemin de code partagé par
TOUS les précodeurs (RZF/WMMSE/SingleSC/IntraRB) -- aucun branchement
après le calcul de `g`, déjà vérifié ligne par ligne dans une
investigation précédente (cf. SESSION_LOG_20260808.md). Pas de raccourci
possible : mapper/canal/égaliseur/démappeur/décodeur sont le MÊME objet
de code exécuté pour toutes les méthodes ici aussi (call_cached suit la
même logique que call()).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_ber_mechanism_investigation.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, NUM_TX, NUM_RX
from sionna.phy.channel import cir_to_ofdm_channel
from sionna.phy.utils import ebnodb2no

K            = NUM_RX
SNR_POINTS   = [15.0, 20.0]
NUM_BATCHES  = 25
BATCH_SIZE   = 128
RB_SIZE      = 12
NUM_SC       = 96

NEURAL_CKPTS = {
    'SingleSC': ('single_sc', 'weights/SingleSC_4L_128d/best_20260807_125302'),
    'IntraRB':  ('intra_rb',  'weights/IntraRB_4L_128d/best_20260807_141928'),
}
METHODS = ['RZF', 'WMMSE', 'SingleSC', 'IntraRB']


def build_re_map(system):
    mask = system.rg.pilot_pattern.mask.numpy()[0, 0]
    ofdm_idx, sc_idx = np.where(mask == 0)   # ordre canonique : ofdm outer, sc inner
    assert len(sc_idx) == int(system.rg.num_data_symbols)
    return ofdm_idx.astype(np.int32), sc_idx.astype(np.int32)


def gen_h_freq(system, batch_size):
    system.new_topology(batch_size)
    cir = system.channel_model(batch_size, system.rg.num_ofdm_symbols,
                               1.0 / system.rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
    return system.remove_nulled(h_freq)


def build_neural_system(kind):
    sys_ = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=kind,
                           embed_dim=128, num_heads=4, num_layers=4)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, sys_.rg.num_ofdm_symbols,
                         sys_.rg.fft_size], dtype=tf.complex64)
    _ = sys_.precoder(dummy_h, no=tf.constant(0.01), training=False)
    return sys_


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    systems = {
        'RZF':    MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf'),
        'WMMSE':  MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='wmmse'),
    }
    for name, (kind, ckpt) in NEURAL_CKPTS.items():
        s = build_neural_system(kind)
        ok = s.load_weights_from(ckpt)
        assert ok, f"chargement échoué pour {name}"
        systems[name] = s

    ofdm_idx, sc_idx = build_re_map(systems['RZF'])
    n_re = len(ofdm_idx)
    ofdm_idx_t = tf.constant(ofdm_idx, dtype=tf.int32)
    sc_idx_t   = tf.constant(sc_idx, dtype=tf.int32)
    idx_pairs  = tf.stack([ofdm_idx_t, sc_idx_t], axis=1)   # [n_re, 2]

    out = {}
    for snr in SNR_POINTS:
        print(f'\n{"="*70}\nSNR={snr}dB\n{"="*70}')
        sinr_acc = {m: [] for m in METHODS}
        err_acc  = {m: [] for m in METHODS}          # symbole erroné (OR des 2 bits), aligné avec sinr_acc
        block_acc = {m: [] for m in ('SingleSC', 'IntraRB')}   # erreurs/bloc-SC (24 essais/bloc)

        for bi in range(NUM_BATCHES):
            h_freq = gen_h_freq(systems['RZF'], BATCH_SIZE)   # MÊME canal pour toutes les méthodes ce batch

            per_method_symbol_err = {}
            for name in METHODS:
                sys_ = systems[name]
                b, b_hat, c, llr, h_eff, no_ret, h_freq_ret, x_rg, x_pre, g = sys_.call_cached(
                    tf.constant(BATCH_SIZE, tf.int32), tf.constant(snr, tf.float32), h_freq, training=False)

                sinr = sys_.lmmse_sinr(h_eff, no=no_ret, interference_whitening=True)  # [B,14,96,K,1]
                sinr_t = tf.transpose(sinr[..., 0], [1, 2, 0, 3])     # [14,96,B,K]
                sinr_data = tf.gather_nd(sinr_t, idx_pairs)           # [n_re,B,K]
                sinr_data = tf.transpose(sinr_data, [1, 0, 2])        # [B,n_re,K]

                hard = tf.cast(llr > 0, c.dtype)         # LLR>0 => bit=1 (vérifié, cf. investigation précédente)
                err_bits = tf.cast(hard != c, tf.int32)  # [B,1,K,n=2*n_re]
                err_bits2 = tf.reshape(tf.squeeze(err_bits, axis=1), [BATCH_SIZE, K, n_re, 2])
                symbol_err = tf.reduce_max(err_bits2, axis=-1)        # [B,K,n_re] -- 1 si au moins un bit faux
                symbol_err_t = tf.transpose(symbol_err, [0, 2, 1])    # [B,n_re,K] -- aligné avec sinr_data

                sinr_acc[name].append(sinr_data.numpy().reshape(-1))
                err_acc[name].append(symbol_err_t.numpy().reshape(-1))
                per_method_symbol_err[name] = symbol_err_t.numpy()    # [B,n_re,K]

                if name in block_acc:
                    err_sc = tf.reshape(err_bits2, [BATCH_SIZE, K, n_re // NUM_SC, NUM_SC, 2])
                    block_err = tf.reduce_sum(err_sc, axis=[2, 4])    # [B,K,96] -- erreurs/bloc-SC (sur 24 essais)
                    block_acc[name].append(block_err.numpy().reshape(-1, NUM_SC))

            if bi == 0:
                # cross-user co-occurrence -- accumulé séparément (a besoin des 4 users ensemble)
                cross_user_acc = {m: [] for m in ('SingleSC', 'IntraRB', 'RZF', 'WMMSE')}
            for name in ('SingleSC', 'IntraRB', 'RZF', 'WMMSE'):
                cross_user_acc[name].append(per_method_symbol_err[name].reshape(-1, K))

            if (bi + 1) % 5 == 0:
                print(f'  batch {bi+1}/{NUM_BATCHES}', flush=True)

        # ── Investigation 1 : percentiles SINR + corrélation erreur/SINR bas ──
        print(f'\n--- Investigation 1 : distribution SINR (dB) ---')
        inv1 = {}
        for name in ('RZF', 'WMMSE', 'SingleSC'):
            s = np.concatenate(sinr_acc[name])
            s_db = 10 * np.log10(np.maximum(s, 1e-12))
            e = np.concatenate(err_acc[name]).astype(bool)
            p1, p5, p50 = np.percentile(s_db, [1, 5, 50])
            p_err_overall = e.mean()
            low5_mask = s_db <= p5
            p_err_low5 = e[low5_mask].mean() if low5_mask.any() else float('nan')
            lift = p_err_low5 / p_err_overall if p_err_overall > 0 else float('nan')
            print(f'  {name:<10} SINR p1={p1:6.2f}dB p5={p5:6.2f}dB p50={p50:6.2f}dB | '
                  f'P(err)={p_err_overall:.2e} | P(err|SINR<=p5)={p_err_low5:.2e} | lift={lift:.1f}x')
            inv1[name] = {'sinr_p1_db': float(p1), 'sinr_p5_db': float(p5), 'sinr_p50_db': float(p50),
                          'p_err_overall': float(p_err_overall), 'p_err_low5pct': float(p_err_low5),
                          'lift_low5pct': float(lift), 'n_samples': int(len(s))}

        # ── Investigation 2.1 : Fano factor (burstiness) par bloc-SC ──
        print(f'\n--- Investigation 2.1 : Fano factor (Var/Mean) par bloc-SC (24 essais/bloc) ---')
        inv2a = {}
        for name in ('SingleSC', 'IntraRB'):
            blocks = np.concatenate(block_acc[name], axis=0)   # [N, 96]
            flat = blocks.reshape(-1)
            fano = flat.var() / flat.mean() if flat.mean() > 0 else float('nan')
            print(f'  {name:<10} Fano factor={fano:.2f}  (1.0 = pas de rafale, >>1 = erreurs groupées) | '
                  f'mean_err/block={flat.mean():.4f}')
            inv2a[name] = {'fano_factor': float(fano), 'mean_err_per_block': float(flat.mean())}

        # ── Investigation 2.2 : corrélation d'erreur entre users à une même RE ──
        print(f'\n--- Investigation 2.2 : lift de co-occurrence d\'erreur ENTRE users (même RE) ---')
        inv2b = {}
        for name in ('RZF', 'WMMSE', 'SingleSC', 'IntraRB'):
            arr = np.concatenate(cross_user_acc[name], axis=0).astype(bool)   # [N, K]
            p_k = arr.mean(axis=0)   # [K]
            lifts = []
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    p_both = (arr[:, k1] & arr[:, k2]).mean()
                    p_indep = p_k[k1] * p_k[k2]
                    if p_indep > 0:
                        lifts.append(p_both / p_indep)
            mean_lift = float(np.mean(lifts)) if lifts else float('nan')
            print(f'  {name:<10} lift co-occurrence moyen (paires users)={mean_lift:.2f}x  '
                  f'(1.0 = indépendant, >1 = erreurs corrélées entre users)')
            inv2b[name] = {'mean_cross_user_lift': mean_lift}

        out[str(snr)] = {'inv1_sinr': inv1, 'inv2a_fano': inv2a, 'inv2b_cross_user': inv2b}

    with open('results/diag_ber_mechanism_investigation.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_ber_mechanism_investigation.json')
