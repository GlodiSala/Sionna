"""
experiments/mechanism_ber_error_floor.py — pourquoi les précodeurs appris ont
un plancher de BER (~1e-5..1e-4) à haut SNR alors que RZF descend sous la
résolution de mesure ?

La chaîne transmet du QPSK + LDPC rate 1/2 (1 bit/symbole/utilisateur), alors
que le SINR post-égalisation à 20 dB autorise ~10 bps/Hz/utilisateur : la
marge est énorme, donc un plancher qui ne s'améliore plus avec le SNR ne peut
pas venir du bruit. Deux explications possibles, qui se distinguent par la
STRUCTURE des erreurs :

  (a) outage / plancher d'interférence résiduelle — les erreurs sont
      CONCENTREES dans une petite fraction de mots de code dont le SINR
      effectif tombe sous le seuil de décodage LDPC (~0-3 dB pour QPSK r=1/2).
      Attendu si le précodeur appris n'annule pas exactement l'interférence
      inter-utilisateurs : le SINR sature alors en SNR (même mécanisme que
      experiments/mechanism_interference_floor.py pour RZF-RB).
  (b) dégradation systématique (bug) — les erreurs sont ETALEES sur tous les
      mots de code, ce qui signalerait un problème d'échelle de LLR, de
      normalisation de puissance ou de désalignement de pilotes, pas un
      phénomène de canal.

Le script mesure donc, par mot de code (un mot = un couple (réalisation,
utilisateur), k=1152 bits d'information) :
  - le nombre de bits erronés,
  - l'efficacité spectrale moyenne du mot (moyenne sur sous-porteuses de
    log2(1+SINR_LMMSE)), pour voir si les mots qui échouent sont bien ceux
    dont le SINR est le plus bas.

Sorties : taux d'erreur trame (FER), bits erronés par trame en échec,
distribution, et comparaison SINR trames en échec vs trames correctes.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/mechanism_ber_error_floor.py
       [--num_batches N] [--snrs 10 15 20]
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from system import MU_MIMO_System, NUM_TX, NUM_RX
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from sionna.phy.config import config as sionna_config

_p = argparse.ArgumentParser()
_p.add_argument('--num_batches', type=int, default=4)
_p.add_argument('--batch_size', type=int, default=256)
_p.add_argument('--snrs', type=float, nargs='+', default=[10.0, 15.0, 20.0])
_a = _p.parse_args()
SEED = 42

NEURAL = {
    'SC':    ('single_sc',      'results/diag_front_a_signed_attn.json',
              lambda **kw: SingleSCTransformerPrecoderSignedAttn(**kw)),
    'IB':    ('intra_rb',       'results/diag_front_a_signed_attn_intra_rb.json',
              lambda **kw: IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kw)),
    'TA-RB': ('ta_rb_residual', 'results/diag_front_a_signed_attn_ta_rb_residual.json',
              lambda **kw: TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kw)),
}


def reset_seed():
    tf.random.set_seed(SEED); np.random.seed(SEED); sionna_config.seed = SEED


def build(name):
    if name in ('RZF', 'WMMSE'):
        return MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=name.lower())
    base, jpath, builder = NEURAL[name]
    ckpt = json.load(open(jpath))['best_ckpt']
    assert os.path.isdir(ckpt), f'checkpoint introuvable : {ckpt}'
    s = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=base,
                       embed_dim=128, num_heads=4, num_layers=4)
    s.precoder = builder(num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=s.rg.num_ofdm_symbols,
                         fft_size=s.rg.fft_size, embed_dim=128, num_heads=4,
                         num_layers=4, snr_aware=True)
    dummy = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, s.rg.num_ofdm_symbols, s.rg.fft_size],
                     dtype=tf.complex64)
    _ = s.precoder(dummy, no=tf.constant(0.01), training=False)
    assert s.load_weights_from(ckpt), f'chargement echoue : {name}'
    return s


def probe(system, snr):
    """Retourne (erreurs par mot de code, efficacite spectrale par mot de code)."""
    errs, rates = [], []
    for _ in range(_a.num_batches):
        system.new_topology(_a.batch_size)
        b, b_hat, _, _, h_eff, no, _, _, _, _ = system(
            tf.constant(_a.batch_size, tf.int32), tf.constant(float(snr), tf.float32),
            training=False)
        e = tf.reduce_sum(tf.cast(b != b_hat, tf.int32), axis=-1)      # [B,1,K]
        errs.append(tf.reshape(e, [-1]).numpy())
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate = tf.math.log(1.0 + sinr) / tf.math.log(2.0)
        rates.append(tf.reshape(tf.reduce_mean(rate, axis=[1, 2, 4]), [-1]).numpy())
    return np.concatenate(errs), np.concatenate(rates)


if __name__ == '__main__':
    K_BITS = None
    out = {'config': {'seed': SEED, 'num_batches': _a.num_batches,
                      'batch_size': _a.batch_size, 'snrs': _a.snrs,
                      'modulation': 'QPSK (2 bits/symbole)', 'coderate': 0.5,
                      'note': 'SNR = Eb/N0 (ebnodb2no avec coderate=0.5)'},
           'methods': {}}
    for name in ['RZF', 'WMMSE', 'SC', 'IB', 'TA-RB']:
        system = build(name)
        K_BITS = int(system.rg.num_data_symbols)
        out['config']['k_bits_per_codeword'] = K_BITS
        out['methods'][name] = {}
        print(f'\n{"="*78}\n{name}\n{"="*78}', flush=True)
        for snr in _a.snrs:
            reset_seed()
            e, r = probe(system, snr)
            nfr = e.size
            bad = e > 0
            tot_bits = nfr * K_BITS
            rec = {
                'codewords': int(nfr),
                'bits': int(tot_bits),
                'bit_errors': int(e.sum()),
                'ber': float(e.sum() / tot_bits),
                'frame_errors': int(bad.sum()),
                'fer': float(bad.sum() / nfr),
                'errors_per_failed_frame_mean': float(e[bad].mean()) if bad.any() else 0.0,
                'errors_per_failed_frame_max': int(e[bad].max()) if bad.any() else 0,
                'frac_of_errors_in_worst_1pct_frames': float(
                    np.sort(e)[::-1][:max(1, nfr // 100)].sum() / max(e.sum(), 1)),
                'rate_mean_all': float(r.mean()),
                'rate_p1': float(np.percentile(r, 1)),
                'rate_min': float(r.min()),
                'rate_mean_failed': float(r[bad].mean()) if bad.any() else None,
                'rate_mean_ok': float(r[~bad].mean()),
            }
            out['methods'][name][f'{snr:.1f}'] = rec
            print(f'  SNR={snr:5.1f} dB | BER={rec["ber"]:.2e} | FER={rec["fer"]:.2e} '
                  f'({rec["frame_errors"]}/{nfr} mots) | err/trame={rec["errors_per_failed_frame_mean"]:6.1f} '
                  f'| part des erreurs dans le 1% pire={rec["frac_of_errors_in_worst_1pct_frames"]:.2f} '
                  f'| SE moy={rec["rate_mean_all"]:5.2f} (echec {rec["rate_mean_failed"] if rec["rate_mean_failed"] else float("nan"):5.2f} '
                  f'vs ok {rec["rate_mean_ok"]:5.2f})', flush=True)
        del system

    dst = f"results/mechanism_ber_error_floor_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out['config']['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSauve -> {dst}')
