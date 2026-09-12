"""
diag_intrarb_ber_edge_vs_center.py — Investigation hypothèse (c) (demande
utilisateur) : le plancher BER d'IntraRB (~1.2e-4, quasi plat 10-20dB,
cf. SESSION_LOG_20260808.md) est-il concentré sur les sous-porteuses en
bord de RB (SC0, SC11 de chaque RB de 12) plutôt qu'au centre (SC1-10) ?
Cohérent avec l'effet de bord déjà observé dans le pattern de SC-attention
appris (SESSION_LOG_20260807.md, inspection des poids).

Méthode : BER NON CODÉ (avant décodage LDPC), pas le BER post-décodage
utilisé dans evaluate_system() -- le décodage LDPC itératif MÉLANGE les
bits de TOUT le codeword (bords + centre indifféremment), donc un bit
d'erreur post-décodage ne peut PAS être attribué à une sous-porteuse
précise. Le BER non codé (décision dure sur le LLR du démappeur, comparé
au bit codé transmis `c`) reste directement attribuable à sa RE d'origine
via le mapping canonique du ResourceGridMapper (extrait empiriquement du
masque pilote, PAS supposé en dur) -- c'est la bonne quantité pour tester
un effet localisé en fréquence.

Comparaison croisée avec SingleSC (même mesure, checkpoint ÉTAPE 4) comme
contrôle -- si l'effet de bord est spécifique à IntraRB (qui a de la
SC-attention) et absent de SingleSC (qui n'en a pas), ça renforce
directement la causalité.

Checkpoints (ÉTAPE 4, confirmés via diag_intrarb_gradient_layers.py) :
  IntraRB  : weights/IntraRB_4L_128d/best_20260807_141928
  SingleSC : weights/SingleSC_4L_128d/best_20260807_125302

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_intrarb_ber_edge_vs_center.py
"""
import os, sys, json, pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, NUM_TX, NUM_RX

CHECKPOINTS = {
    'IntraRB':  ('intra_rb',  'weights/IntraRB_4L_128d/best_20260807_141928'),
    'SingleSC': ('single_sc', 'weights/SingleSC_4L_128d/best_20260807_125302'),
}

RB_SIZE      = 12
FFT_SIZE     = 96
SNR_POINTS   = [15.0, 17.5, 20.0]
NUM_BATCHES  = 300     # ~29.5M bits sur le groupe EDGE (le plus petit, 16/96 SC)
BATCH_SIZE   = 128


def build_sc_position_map(system):
    """Construit la correspondance (indice symbole data linéaire -> indice
    sous-porteuse 0..95), extraite du VRAI masque pilote (pas supposée),
    pour un seul stream (identique pour tous les streams -- pilotes en
    bloc symbole OFDM entier, vérifié). Retourne un tf.Tensor[num_data_symbols]
    d'indices SC, et un masque booléen 'is_edge' (SC%12 in {0,11})."""
    mask = system.rg.pilot_pattern.mask.numpy()   # [1, num_streams, ofdm, fft]
    m0 = mask[0, 0]                                # [ofdm, fft], 0=data 1=pilot
    # vérif : masque identique pour tous les streams (pilotes en bloc symbole)
    for s in range(mask.shape[1]):
        assert np.array_equal(mask[0, s], m0), "masque pilote différent par stream -- hypothèse à revoir"

    ofdm_idx, sc_idx = np.where(m0 == 0)   # row-major : ofdm outer, sc inner -> ordre canonique data
    assert len(sc_idx) == int(system.rg.num_data_symbols)
    sc_idx = sc_idx.astype(np.int32)
    is_edge = np.isin(sc_idx % RB_SIZE, [0, RB_SIZE - 1])
    print(f"  Vérif mapping : {len(sc_idx)} symboles data, "
          f"{is_edge.sum()} 'edge' ({is_edge.sum()}/{len(sc_idx)}={is_edge.mean()*100:.1f}%, "
          f"attendu {16}/{96}={16/96*100:.1f}%)")
    assert abs(is_edge.mean() - 16/96) < 1e-9
    return tf.constant(sc_idx), tf.constant(is_edge)


def uncoded_ber_by_position(system, snr_db, sc_idx, is_edge, num_batches, batch_size):
    """BER non codé (LLR hard-decision vs bit codé transmis c), ventilé
    edge/centre. c et llr ont la même longueur/ordre linéaire (domaine
    bits codés, n=2*num_data_symbols par user) -- indexé via sc_idx
    (répété x2 car 2 bits codés par symbole QPSK, dans l'ordre
    [bit0_sym0, bit1_sym0, bit0_sym1, bit1_sym1, ...] -- convention
    Sionna Mapper QAM)."""
    is_edge_bits = tf.repeat(is_edge, 2)   # [n] -- 2 bits codés par symbole QPSK

    err_edge = err_center = n_edge = n_center = 0
    for _ in range(num_batches):
        system.new_topology(batch_size)
        b, b_hat, c, llr, h_eff, no, h_freq, x_rg, x_pre, g = system(
            tf.constant(batch_size, tf.int32), tf.constant(snr_db, tf.float32), training=False)

        # Convention Sionna vérifiée (docstring Demapper) : LLR = ln(P(b=1)/P(b=0))
        # => LLR>0 signifie bit=1 plus probable.
        hard = tf.cast(llr > 0, c.dtype)
        err = tf.cast(hard != c, tf.int32)   # [B,1,K,n]
        err_np = err.numpy().reshape(-1, err.shape[-1])   # [B*1*K, n]

        edge_mask_np = is_edge_bits.numpy()
        err_edge   += int(err_np[:, edge_mask_np].sum())
        err_center += int(err_np[:, ~edge_mask_np].sum())
        n_edge      += err_np[:, edge_mask_np].size
        n_center    += err_np[:, ~edge_mask_np].size

    return err_edge / n_edge, err_edge, n_edge, err_center / n_center, err_center, n_center


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    out = {}

    for name, (ptype, ckpt) in CHECKPOINTS.items():
        print(f'\n{"="*70}\n{name} ({ckpt})\n{"="*70}')
        system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ptype,
                                 embed_dim=128, num_heads=4, num_layers=4)
        # Le modèle doit être "built" (variables créées) via un forward pass
        # AVANT le chargement -- sinon trainable_variables est incomplet/mal
        # ordonné vs le pickle (trouvé au dry-run : shape mismatch sur s_sc).
        dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, system.rg.num_ofdm_symbols,
                             system.rg.fft_size], dtype=tf.complex64)
        _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
        ok = system.load_weights_from(ckpt)
        assert ok, f"chargement poids échoué pour {name}"

        sc_idx, is_edge = build_sc_position_map(system)
        out[name] = {}

        for snr in SNR_POINTS:
            ber_edge, ne, Ne, ber_center, nc, Nc = uncoded_ber_by_position(
                system, snr, sc_idx, is_edge, NUM_BATCHES, BATCH_SIZE)
            ratio = ber_edge / ber_center if ber_center > 0 else float('inf')
            print(f'  SNR={snr:5.1f}dB | BER_edge={ber_edge:.3e} ({ne} err / {Ne:,} bits) | '
                  f'BER_center={ber_center:.3e} ({nc} err / {Nc:,} bits) | ratio={ratio:.2f}x')
            out[name][snr] = {'ber_edge': ber_edge, 'n_err_edge': ne, 'n_bits_edge': Ne,
                               'ber_center': ber_center, 'n_err_center': nc, 'n_bits_center': Nc,
                               'ratio': ratio}

    with open('results/diag_intrarb_ber_edge_vs_center.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSauvé -> results/diag_intrarb_ber_edge_vs_center.json')
