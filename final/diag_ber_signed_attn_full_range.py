"""
diag_ber_signed_attn_full_range.py — Figure D (demande utilisateur) : BER
sur la plage SNR la plus large possible avec le pipeline déjà validé
(evaluate_system(), main_finall.py) -- étend diag_ber_signed_attn.py
(qui ne couvrait que 15/17.5/20dB) à EVALUATION_SNR_RANGE complète
(0 à 20dB, 9 points), pour RZF/WMMSE + les 3 architectures signed_attn.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_ber_signed_attn_full_range.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, NUM_TX, NUM_RX, evaluate_system, EVALUATION_SNR_RANGE
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

NUM_BATCHES = 50
BATCH_SIZE = 256

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
        results_all[name] = evaluate_system(system, EVALUATION_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, name)

    print(f'\n{"="*110}\nBER comparatif complet -- signed_attn vs classiques, UMi (M8K4)\n{"="*110}')
    print(f'{"Méthode":<28} ' + ' '.join(f'{"%4.1fdB" % s:>11}' for s in EVALUATION_SNR_RANGE))
    for name, r in results_all.items():
        print(f'{name:<28} ' + ' '.join(f'{b:>11.2e}' for b in r['ber']))

    out = {}
    for name, r in results_all.items():
        out[name] = {'snr': EVALUATION_SNR_RANGE.tolist(), 'sum_rate': r['sum_rate'], 'ber': r['ber'],
                      'flops_M': r['flops_M'], 'params_K': r['params_K']}
    with open('results/diag_ber_signed_attn_full_range.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\n✅ Sauvé -> results/diag_ber_signed_attn_full_range.json')
