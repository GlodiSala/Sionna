"""
diag_etape1_smoke.py — Smoke test ÉTAPE 1 (SESSION_NUIT_RESUME.md) : valide
que les 3 architectures nettoyées (single_sc, intra_rb, ta_rb) tournent
sans erreur sur le pipeline dataset corrigé (§0) -- build, forward pass,
1 pas warmup + 1 pas finetune, pas de NaN.

Pas un test de qualité d'entraînement (voir ÉTAPE 3 pour ça) -- juste
"ça tourne, les nombres sont finis, les gradients ne sont pas nuls".

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_etape1_smoke.py
"""
import os, sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer, STANDARD_CONFIG
from datasets import CachedSionnaDataset

SCRATCH_DIR = '/tmp/claude-4849/-users-sala-Documents-test-projet-Trans-freq-Sionna/7ea26a8d-5dda-4f6b-9915-78d4e8816314/scratchpad'

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']

ARCHS = [
    ('single_sc', {'embed_dim': 64, 'num_heads': 4, 'num_layers': 2}),
    ('intra_rb',  {'embed_dim': 64, 'num_heads': 4, 'num_layers': 2}),
    ('ta_rb',     {'embed_dim': 64, 'num_heads': 4, 'num_layers': 2, 'tokens_per_rb': 3}),
]


def build_smoke_dataset():
    dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
    cache_file = os.path.join(SCRATCH_DIR, 'smoke_etape1_dataset.npz')
    if os.path.exists(cache_file):
        os.remove(cache_file)
    return CachedSionnaDataset(
        dummy, dataset_size=300, batch_size=128,
        cache_file=cache_file,
        cluster_radius_m=STANDARD_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=STANDARD_CONFIG['INDOOR_PROBABILITY'],
        seed=7)


def run_smoke(name, kwargs, dataset):
    print(f'\n{"="*70}\n{name}\n{"="*70}', flush=True)
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=name, **kwargs)
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'smoke_{name}',
        warmup_epochs=1, finetune_epochs=1,
        batch_size=16, learning_rate=1e-3,
        steps_per_epoch=3)   # juste assez pour vérifier que ça tourne
    trainer.train(print_every=1, patience=99)

    n_params = sum(tf.size(v).numpy() for v in trainer.vars)
    losses_ok = all(np.isfinite(h['loss']) for h in trainer.history)
    rates_ok  = all(np.isfinite(h['sum_rate']) for h in trainer.history)
    flops, weights, acts = system.precoder.complexity(system.rg.num_ofdm_symbols)

    print(f'  params(trainable)={n_params:,} | complexity.weights={weights:,.0f} | '
          f'FLOPs={flops:,.0f} | acts={acts:,.0f}')
    status = 'OK' if (losses_ok and rates_ok) else 'FAIL (NaN/Inf detected)'
    print(f'  -> {status}')
    return losses_ok and rates_ok


if __name__ == '__main__':
    dataset = build_smoke_dataset()

    results = {}
    for name, kwargs in ARCHS:
        try:
            results[name] = run_smoke(name, kwargs, dataset)
        except Exception as e:
            print(f'  -> EXCEPTION: {type(e).__name__}: {e}', flush=True)
            results[name] = False
        tf.keras.backend.clear_session()

    print(f'\n{"="*70}\nRÉSUMÉ SMOKE TEST ÉTAPE 1\n{"="*70}')
    for name, ok in results.items():
        print(f'  {name:12s} : {"OK" if ok else "FAIL"}')

    if all(results.values()):
        print('\n✅ Les 3 architectures nettoyées passent le smoke test.')
    else:
        print('\n❌ Au moins une architecture a échoué -- voir ci-dessus.')
        sys.exit(1)
