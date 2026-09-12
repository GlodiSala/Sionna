"""
experiments/eval_ber_umi_standard.py — Figure D (demande utilisateur) : BER
sur la plage SNR la plus large possible avec le pipeline déjà validé
(evaluate_system(), system.py) -- étend diag_ber_signed_attn.py
(qui ne couvrait que 15/17.5/20dB) à EVALUATION_SNR_RANGE complète
(0 à 20dB, 9 points), pour RZF/WMMSE + les 3 architectures signed_attn.

Protocole de graine (ajout du 12/09, aligné sur la campagne seedfix) :
`sionna_config.seed = SEED` en plus de `tf.random.set_seed`, et la graine est
REMISE A ZERO juste avant l'évaluation de chaque méthode (après construction
du modèle et chargement des poids, pour que l'initialisation des poids ne
désynchronise pas le flux). Chaque méthode voit donc la MEME séquence de
topologies/canaux/bits/bruit : les BER deviennent comparables par paires, ce
qui n'était pas le cas de la version d'origine (tirages indépendants par
méthode). Le nombre de tirages par point est identique pour toutes les
méthodes, donc l'appariement tient.

Sortie : results/diag_ber_umi_standard_seedfix_<horodatage>.json. Le fichier
historique results/diag_ber_signed_attn_full_range.json (tirages non
appariés) n'est PAS écrasé -- c'est lui qui alimente encore figD_ber_snr.py.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/eval_ber_umi_standard.py [--num_batches N]
"""
import os, sys, json, time, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from system import MU_MIMO_System, NUM_TX, NUM_RX, evaluate_system, EVALUATION_SNR_RANGE
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

_p = argparse.ArgumentParser()
_p.add_argument('--num_batches', type=int, default=50,
                help='tirages Monte-Carlo par point SNR et par méthode (défaut 50)')
_p.add_argument('--batch_size', type=int, default=256)
_args = _p.parse_args()

NUM_BATCHES = _args.num_batches
BATCH_SIZE = _args.batch_size
SEED = 42

from sionna.phy.config import config as sionna_config


def reset_seed():
    """Remet le flux aléatoire à l'état initial -- appelé avant CHAQUE méthode
    pour que toutes voient la même séquence de canaux (appariement)."""
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    sionna_config.seed = SEED

RESULT_JSONS = {
    'SingleSC-signed_attn': ('single_sc', 'results/diag_front_a_signed_attn.json',
                              lambda **kw: SingleSCTransformerPrecoderSignedAttn(**kw)),
    'IntraRB-signed_attn': ('intra_rb', 'results/diag_front_a_signed_attn_intra_rb.json',
                             lambda **kw: IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kw)),
    'TA_RB_residual-signed_attn': ('ta_rb_residual', 'results/diag_front_a_signed_attn_ta_rb_residual.json',
                                    lambda **kw: TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=6, **kw)),
}


def resolve_ckpt(path):
    with open(path) as f:
        d = json.load(f)
    ckpt = d['best_ckpt']
    assert ckpt and os.path.isdir(ckpt), f"checkpoint introuvable: {ckpt}"
    return ckpt


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results_all = {}

    for bname, ptype in [('RZF', 'rzf'), ('WMMSE', 'wmmse')]:
        sys_ = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type=ptype)
        reset_seed()
        results_all[bname] = evaluate_system(sys_, EVALUATION_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, bname)

    for name, (base_ptype, json_path, builder) in RESULT_JSONS.items():
        ckpt = resolve_ckpt(json_path)
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
        reset_seed()
        results_all[name] = evaluate_system(system, EVALUATION_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, name)

    print(f'\n{"="*110}\nBER comparatif complet -- signed_attn vs classiques, UMi (M8K4)\n{"="*110}')
    print(f'{"Méthode":<28} ' + ' '.join(f'{"%4.1fdB" % s:>11}' for s in EVALUATION_SNR_RANGE))
    for name, r in results_all.items():
        print(f'{name:<28} ' + ' '.join(f'{b:>11.2e}' for b in r['ber']))

    out = {'metadata': {
        'seed': SEED,
        'seed_fix': 'sionna_config.seed = SEED, remis a zero avant chaque methode (tirages apparies)',
        'num_batches': NUM_BATCHES, 'batch_size': BATCH_SIZE,
        'tokens_per_rb_ta_rb': 6,
        'checkpoints': {n: resolve_ckpt(j) for n, (_, j, _) in RESULT_JSONS.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
    }}
    for name, r in results_all.items():
        out[name] = {'snr': EVALUATION_SNR_RANGE.tolist(), 'sum_rate': r['sum_rate'], 'ber': r['ber'],
                      'flops_M': r['flops_M'], 'params_K': r['params_K']}
    dst = f"results/diag_ber_umi_standard_seedfix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Sauvé -> {dst}')
