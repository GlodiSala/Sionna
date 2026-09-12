"""
experiments/intermediate_snr_annex_b.py — Annexe B (demande utilisateur) :
évalue les checkpoints signed_attn DÉJÀ ENTRAÎNÉS aux SNR intermédiaires
{2.5, 7.5, 12.5, 17.5}dB, absents des tableaux principaux (EVAL_SNRS
n'a jamais inclus ces points dans aucun script d'entraînement, cf.
vérification demandée par l'utilisateur). Pas de nouvel entraînement --
chargement de poids + passes forward seulement, même méthodologie
(eval_perfect) que les scripts d'entraînement d'origine.

4 régimes x 3 architectures = 12 combinaisons, un régime par invocation
(les 3 architectures d'un régime dans la même invocation, dataset
chargé une seule fois).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 experiments/intermediate_snr_annex_b.py --regime umi_standard
"""
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # results/, weights/ relatifs a la racine du paquet
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from system import MU_MIMO_System
from datasets import CachedSionnaDataset
from precoders.signed_attention import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)
from channel_config import STANDARD_CONFIG, MASSIVE_TRUE_CONFIG
from sionna.phy.utils import ebnodb2no

INTERMEDIATE_SNRS = [2.5, 7.5, 12.5, 17.5]
EVAL_BATCHES = 20

# -- régimes : (config, channel, dataset_size, cache_file, tokens_per_rb, eval_batch_size, {arch: (ckpt_json, cirgen_batch_size_for_dataset_open)}) --
REGIMES = {
    'umi_standard': dict(
        cfg=STANDARD_CONFIG, channel='umi', dataset_size=20000, tokens_per_rb=4, eval_batch=128,
        cache=lambda M, K, ds: f'/export/tmp/sala/sionna_joint_{ds//1000}k_{M}x{K}.npz',
        ckpts={'single_sc': 'results/diag_front_a_signed_attn.json',
               'intra_rb': 'results/diag_front_a_signed_attn_intra_rb.json',
               'ta_rb_residual': 'results/diag_tarb_residual_signed_attn_T4_train.json'}),
    'uma_standard': dict(
        cfg=STANDARD_CONFIG, channel='uma', dataset_size=8000, tokens_per_rb=4, eval_batch=128,
        cache=lambda M, K, ds: f'/export/tmp/sala/sionna_joint_uma_{ds//1000}k_{M}x{K}.npz',
        ckpts={'single_sc': 'results/diag_uma_signed_attn_single_sc.json',
               'intra_rb': 'results/diag_uma_signed_attn_intra_rb.json',
               'ta_rb_residual': 'results/diag_uma_signed_attn_ta_rb_residual.json'}),
    'umi_massive': dict(
        cfg=MASSIVE_TRUE_CONFIG, channel='umi', dataset_size=4000, tokens_per_rb=6, eval_batch=32,
        cache=lambda M, K, ds: f'/export/tmp/sala/sionna_joint_massive_true_{ds//1000}k_{M}x{K}.npz',
        ckpts={'single_sc': 'results/diag_massive_true_single_sc_extbudget.json',
               'intra_rb': 'results/diag_massive_true_intra_rb_extbudget.json',
               'ta_rb_residual': 'results/diag_massive_true_ta_rb_residual_extbudget.json'}),
    'uma_massive': dict(
        cfg=MASSIVE_TRUE_CONFIG, channel='uma', dataset_size=4000, tokens_per_rb=6, eval_batch=32,
        cache=lambda M, K, ds: f'/export/tmp/sala/sionna_joint_uma_massive_{ds//1000}k_{M}x{K}.npz',
        ckpts={'single_sc': 'results/diag_uma_massive_single_sc.json',
               'intra_rb': 'results/diag_uma_massive_intra_rb.json',
               'ta_rb_residual': 'results/diag_uma_massive_ta_rb_residual.json'}),
}


def eval_perfect(system, dataset, snr_db, num_batches, batch_size):
    no = ebnodb2no(tf.constant(snr_db, tf.float32), system.num_bits_per_symbol, 0.5, system.rg)
    rates = []
    for _ in range(num_batches):
        h = dataset.get_batch(batch_size)
        g = system._call_precoder(h, no, training=False)
        h_eff = system.ch_helper.compute_effective_channel(h, g)
        sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
        rate = tf.reduce_sum(tf.reduce_mean(
            tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
            axis=[0, 1, 2, 4]))
        rates.append(float(rate))
    return float(np.mean(rates))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', required=True, choices=list(REGIMES.keys()))
    args = p.parse_args()
    os.makedirs('results', exist_ok=True)

    r = REGIMES[args.regime]
    M, K = r['cfg']['NUM_TX'], r['cfg']['NUM_RX']
    T = r['tokens_per_rb']
    cache_file = r['cache'](M, K, r['dataset_size'])
    assert os.path.exists(cache_file), f"cache introuvable: {cache_file}"

    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    dataset = CachedSionnaDataset(
        dummy, dataset_size=r['dataset_size'], batch_size=32,
        cache_file=cache_file, cluster_radius_m=r['cfg']['CLUSTER_RADIUS_M'],
        indoor_probability=r['cfg']['INDOOR_PROBABILITY'], scenario=r['channel'], seed=42)

    out = {}
    for arch, ckpt_json in r['ckpts'].items():
        with open(ckpt_json) as f:
            ckpt = json.load(f)['best_ckpt']
        assert os.path.isdir(ckpt), f"checkpoint introuvable: {ckpt}"

        system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=arch, embed_dim=128, num_heads=4,
                                 num_layers=4, tokens_per_rb=T)
        kwargs = dict(num_tx=M, num_rx=K, num_ofdm=system.rg.num_ofdm_symbols, fft_size=system.rg.fft_size,
                      embed_dim=128, num_heads=4, num_layers=4, snr_aware=True)
        if arch == 'single_sc':
            system.precoder = SingleSCTransformerPrecoderSignedAttn(use_abs=True, use_cossin=False, **kwargs)
        elif arch == 'intra_rb':
            system.precoder = IntraRBTransformerPrecoderSignedAttn(rb_size=12, use_abs=True, use_cossin=False, **kwargs)
        else:
            system.precoder = TransformerPrecoderCleanResidualSignedAttn(rb_size=12, tokens_per_rb=T, **kwargs)

        dummy_h = tf.zeros([1, K, 1, 1, M, system.rg.num_ofdm_symbols, system.rg.fft_size], dtype=tf.complex64)
        _ = system.precoder(dummy_h, no=tf.constant(0.01), training=False)
        ok = system.load_weights_from(ckpt)
        assert ok, f"chargement échoué {arch}"
        print(f"✅ {arch} chargé depuis {ckpt}", flush=True)

        evals = {}
        for snr in INTERMEDIATE_SNRS:
            rate = eval_perfect(system, dataset, snr, EVAL_BATCHES, r['eval_batch'])
            evals[str(snr)] = rate
            print(f"  {args.regime}/{arch} | SNR={snr:5.1f}dB | rate={rate:7.2f}", flush=True)
        out[arch] = {'ckpt': ckpt, 'evals_intermediate': evals}

        out_path = f'results/diag_intermediate_snr_{args.regime}.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)   # sauvé après chaque archi -- reprenable

    print(f"\n✅ Sauvé -> results/diag_intermediate_snr_{args.regime}.json")
