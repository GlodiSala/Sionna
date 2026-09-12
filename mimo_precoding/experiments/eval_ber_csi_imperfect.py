"""
experiments/eval_ber_csi_imperfect.py — Figure D double panneau (demande utilisateur) :
BER sous CSI imparfait (pilote 20dB), même principe que le sweep débit-
somme CSI imparfait mais avec le pipeline LDPC Monte-Carlo complet
(pas seulement le débit LMMSE). RZF, WMMSE, SC, IB, TA-RB (T=4).

Méthode : le précodeur voit le canal BRUITÉ (h_est, bruit LS gaussien,
pilote 20dB) -- exactement comme un système réel où seule l'estimée de
canal au TX est imparfaite -- mais la propagation réelle et la
détection utilisent le VRAI canal (h_true), via
MU_MIMO_System._forward_from_precoder(b, c, x_rg, h_true, g_from_h_est,
no) -- réutilise directement la méthode interne du pipeline déjà
validé (system.py), pas de nouvelle logique de détection/LDPC.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/eval_ber_csi_imperfect.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from sionna.phy.channel import cir_to_ofdm_channel
from sionna.phy.utils import ebnodb2no
from system import MU_MIMO_System, NUM_TX, NUM_RX
from precoders.classical import rzf_precoder, wmmse_precoder
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

DATA_SNRS_DB = [0.0, 5.0, 10.0, 15.0, 20.0]
PILOT_SNR_DB = 20.0
NUM_BATCHES = 15   # réduit de 50 -- génération de canal (UMi CIR) domine le
                   # coût par batch en exécution eager (pas de @tf.function
                   # global comme dans evaluate_system()) ; 50 aurait pris
                   # >1h pour les 25 combinaisons SNR x méthode, pas réaliste
                   # dans le budget de cette session -- 15 reste statistiquement
                   # raisonnable vu que le BER sous CSI imparfait est nettement
                   # plus élevé (moins de batches nécessaires pour un
                   # échantillon stable) que sous CSI parfait
BATCH_SIZE = 256
SEED = 2026
OUT_JSON = 'results/diag_ber_csi_imperfect.json'

RESULT_JSONS = {
    'SingleSC-signed_attn': ('single_sc', 'results/diag_front_a_signed_attn.json',
                              lambda **kw: SingleSCTransformerPrecoderSignedAttn(**kw)),
    'IntraRB-signed_attn': ('intra_rb', 'results/diag_front_a_signed_attn_intra_rb.json',
                             lambda **kw: IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kw)),
    'TA_RB_residual-signed_attn': ('ta_rb_residual', 'results/diag_tarb_residual_signed_attn_T4_train.json',
                                    lambda **kw: TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=4, **kw)),
}


def gen_h_true(system, batch_size):
    cir = system.channel_model(batch_size, system.rg.num_ofdm_symbols,
                                1.0 / system.rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(system.frequencies, *cir, normalize=True)
    return system.remove_nulled(h_freq)


def noisy_channel(h_freq, pilot_snr_db, rng):
    snr_lin = 10.0 ** (pilot_snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    shape = h_freq.shape
    noise_re = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    noise_im = rng.normal(0, np.sqrt(sigma2 / 2), size=shape).astype(np.float32)
    return h_freq + tf.cast(tf.complex(noise_re, noise_im), h_freq.dtype)


def eval_ber_imperfect(system, precoder_fn, rng, name):
    print(f'\n📊 CSI imparfait (pilote {PILOT_SNR_DB}dB) : {name}', flush=True)
    out = {}
    for snr in DATA_SNRS_DB:
        no = ebnodb2no(tf.constant(snr, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
        tot_bits = tot_err = 0
        for _ in range(NUM_BATCHES):
            system.new_topology(BATCH_SIZE)
            h_true = gen_h_true(system, BATCH_SIZE)
            h_est = noisy_channel(h_true, PILOT_SNR_DB, rng)

            b = system.binary_source([BATCH_SIZE, 1, system.num_users, int(system.rg.num_data_symbols)])
            c = system.encoder(b)
            x = system.mapper(c)
            x_rg = system.rg_mapper(x)

            g = precoder_fn(h_est, no)
            b, b_hat, *_ = system._forward_from_precoder(b, c, x_rg, h_true, g, no)

            tot_err += int(tf.reduce_sum(tf.cast(b != b_hat, tf.int32)).numpy())
            tot_bits += int(b.numpy().size)

        ber = tot_err / max(tot_bits, 1)
        out[str(snr)] = ber
        print(f'  SNR={snr:5.1f}dB | BER={ber:.2e}', flush=True)
    return out


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    rng = np.random.RandomState(SEED)
    results = {}

    for bname, ptype in [('RZF', 'rzf'), ('WMMSE', 'wmmse')]:
        system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ptype)
        pfn = ((lambda h, no: rzf_precoder(h, stream_management=system.sm, no=no)) if ptype == 'rzf'
               else (lambda h, no: wmmse_precoder(h, no=no, stream_management=system.sm, num_iterations=10)))
        results[bname] = eval_ber_imperfect(system, pfn, rng, bname)
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=2)   # sauvé après chaque méthode -- reprenable

    for name, (base_ptype, json_path, builder) in RESULT_JSONS.items():
        with open(json_path) as f:
            ckpt = json.load(f)['best_ckpt']
        assert os.path.isdir(ckpt), f"checkpoint introuvable pour {name}: {ckpt}"
        system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=base_ptype,
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = builder(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols,
                                   fft_size=system.rg.fft_size, embed_dim=128, num_heads=4, num_layers=4,
                                   snr_aware=True)
        dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, system.rg.num_ofdm_symbols,
                             system.rg.fft_size], dtype=tf.complex64)
        _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
        ok = system.load_weights_from(ckpt)
        assert ok, f"chargement échoué {name}"

        pfn = lambda h, no: system._call_precoder(h, no, training=False)
        results[name] = eval_ber_imperfect(system, pfn, rng, name)
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\n✅ Sauvé -> {OUT_JSON}')
