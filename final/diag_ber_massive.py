"""
diag_ber_massive.py — Figure D, UMi MASSIVE (M=64, K=8), demande
utilisateur (nuit) : BER vs SNR pour RZF, WMMSE, SC, IB, TA-RB
(checkpoints budget étendu, cohérents). CSI parfait uniquement (le
CSI imparfait est déjà couvert par le débit -- Figure A -- ajouter
le BER-imparfait à cette échelle serait un calcul lourd de plus,
priorité donnée à couvrir d'abord tous les régimes en CSI parfait).

Batch réduit (comme le training/eval MASSIVE, cf. diag_massive_true_
train.py) -- la génération de canal CIR à M=64 est nettement plus
coûteuse en mémoire qu'à M=8.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_ber_massive.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, evaluate_system
from channel_config import MASSIVE_TRUE_CONFIG
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
EVAL_SNR_RANGE = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
NUM_BATCHES = 15
BATCH_SIZE = 32
OUT_JSON = 'results/diag_ber_massive.json'

RESULT_JSONS = {
    'SingleSC-signed_attn': ('single_sc', 'results/diag_massive_true_single_sc_extbudget.json',
                              lambda **kw: SingleSCTransformerPrecoderSignedAttn(**kw)),
    'IntraRB-signed_attn': ('intra_rb', 'results/diag_massive_true_intra_rb_extbudget.json',
                             lambda **kw: IntraRBTransformerPrecoderSignedAttn(rb_size=12, **kw)),
    'TA_RB_residual-signed_attn': ('ta_rb_residual', 'results/diag_massive_true_ta_rb_residual_extbudget.json',
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
        sys_ = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=ptype)
        results_all[bname] = evaluate_system(sys_, EVAL_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, bname)
        with open(OUT_JSON, 'w') as f:
            json.dump({k: {'snr': EVAL_SNR_RANGE.tolist(), 'ber': v['ber']} for k, v in results_all.items()}, f, indent=2)

    for name, (base_ptype, json_path, builder) in RESULT_JSONS.items():
        ckpt = resolve_ckpt(json_path)
        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=base_ptype,
                                 embed_dim=128, num_heads=4, num_layers=4)
        system.precoder = builder(num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols,
                                   fft_size=system.rg.fft_size, embed_dim=128, num_heads=4, num_layers=4,
                                   snr_aware=True)
        dummy_h = tf.zeros([1, K, 1, 1, M, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
        _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
        ok = system.load_weights_from(ckpt)
        assert ok, f"chargement échoué {name}"
        results_all[name] = evaluate_system(system, EVAL_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, name)
        with open(OUT_JSON, 'w') as f:
            json.dump({k: {'snr': EVAL_SNR_RANGE.tolist(), 'ber': v['ber']} for k, v in results_all.items()}, f, indent=2)

    print(f'\n✅ Sauvé -> {OUT_JSON}')
