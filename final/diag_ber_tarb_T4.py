"""
diag_ber_tarb_T4.py — Figure D (demande utilisateur) : T=4 devient la
référence TA-RB (meilleur %WMMSE moyen du T-sweep ET moins cher que
T=6/T=12, cf. Figure B). Recalcule UNIQUEMENT la courbe BER TA-RB avec
le checkpoint T=4 -- RZF/WMMSE/SC/IB restent ceux déjà calculés dans
diag_ber_signed_attn_full_range.json (inchangés, pas de raison de les
refaire). Le résultat est fusionné dans le même JSON (clé
'TA_RB_residual-signed_attn' remplacée, T=6 -> T=4).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_ber_tarb_T4.py
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
from precoder_experimental import TransformerPrecoderCleanResidualSignedAttn

NUM_BATCHES = 50
BATCH_SIZE = 256
JSON_PATH = 'results/diag_ber_signed_attn_full_range.json'
T4_RESULT_JSON = 'results/diag_tarb_residual_signed_attn_T4_train.json'

if __name__ == '__main__':
    with open(T4_RESULT_JSON) as f:
        ckpt = json.load(f)['best_ckpt']
    assert os.path.isdir(ckpt), f"checkpoint introuvable: {ckpt}"

    system = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='ta_rb_residual',
                             embed_dim=128, num_heads=4, num_layers=4, tokens_per_rb=4)
    system.precoder = TransformerPrecoderCleanResidualSignedAttn(
        num_tx=NUM_TX, num_rx=NUM_RX, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
        rb_size=12, tokens_per_rb=4, embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
    dummy_h = tf.zeros([1, NUM_RX, 1, 1, NUM_TX, system.rg.num_ofdm_symbols,
                         system.rg.fft_size], dtype=tf.complex64)
    _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
    ok = system.load_weights_from(ckpt)
    assert ok, "chargement échoué TA-RB T=4"

    r = evaluate_system(system, EVALUATION_SNR_RANGE, NUM_BATCHES, BATCH_SIZE, 'TA_RB_residual-signed_attn T=4')

    print(f'\n{"="*70}\nBER TA-RB T=4 (référence, remplace T=6)\n{"="*70}')
    print(f'{"SNR":>8} ' + ' '.join(f'{"%4.1fdB" % s:>11}' for s in EVALUATION_SNR_RANGE))
    print(f'{"BER":>8} ' + ' '.join(f'{b:>11.2e}' for b in r['ber']))

    with open(JSON_PATH) as f:
        out = json.load(f)
    out['TA_RB_residual-signed_attn'] = {'snr': EVALUATION_SNR_RANGE.tolist(), 'sum_rate': r['sum_rate'],
                                          'ber': r['ber'], 'flops_M': r['flops_M'], 'params_K': r['params_K'],
                                          'T': 4}
    with open(JSON_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Fusionné (TA-RB T=4) -> {JSON_PATH}')
