"""
diag_massive_true_smoke.py — Priorité 3.1 (demande utilisateur) : smoke
test de stabilité pour MASSIVE_TRUE_CONFIG (M=64, K=8, canal STANDARD
R=20m, PAS de clustering serré artificiel -- channel_config.py) AVANT
tout run long. Génère (ou réutilise) le cache dataset à cette échelle,
construit les 3 architectures signed_attn (SC/IB/TA-RB T=6), fait
quelques pas d'entraînement réels (warmup + finetune) pour détecter tout
OOM/erreur de shape avant de lancer les runs complets 83ep.

dataset_size réduit (4000, vs 8000 pour l'ancien MASSIVE_CONFIG M32K8) --
"charge de calcul raisonnable" (consigne) : M=64 double le coût de
génération CIR par échantillon vs M=32.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 diag_massive_true_smoke.py
"""
import os, sys, time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from main_finall import MU_MIMO_System, SupervisedTrainer
from datasets import CachedSionnaDataset
from channel_config import MASSIVE_TRUE_CONFIG
from precoder_experimental import (SingleSCTransformerPrecoderSignedAttn,
                                    IntraRBTransformerPrecoderSignedAttn,
                                    TransformerPrecoderCleanResidualSignedAttn)

M, K = MASSIVE_TRUE_CONFIG['NUM_TX'], MASSIVE_TRUE_CONFIG['NUM_RX']
DATASET_SIZE = 4000
CACHE_FILE = f'/export/tmp/sala/sionna_joint_massive_true_{DATASET_SIZE//1000}k_{M}x{K}.npz'

print(f'{"="*70}\nMASSIVE_TRUE smoke test : M={M} K={K} R={MASSIVE_TRUE_CONFIG["CLUSTER_RADIUS_M"]}m\n{"="*70}')

t0 = time.time()
dummy = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type='rzf')
# batch_size (génération CIR) réduit 128->32 : M=64 double la taille du
# tenseur intermédiaire cir_to_ofdm_channel vs M=32 (déjà documenté dans
# datasets.py comme sensible à M) -- 128 a OOM sur un GPU partagé, trouvé
# au premier essai de ce smoke test.
dataset = CachedSionnaDataset(
    dummy, dataset_size=DATASET_SIZE, batch_size=32,
    cache_file=CACHE_FILE,
    cluster_radius_m=MASSIVE_TRUE_CONFIG['CLUSTER_RADIUS_M'],
    indoor_probability=MASSIVE_TRUE_CONFIG['INDOOR_PROBABILITY'], seed=42)
print(f'Dataset prêt en {(time.time()-t0)/60:.1f}min -> {CACHE_FILE}')

builders = {
    'single_sc': lambda: SingleSCTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, embed_dim=128, num_heads=4, num_layers=4,
        snr_aware=True, use_abs=True, use_cossin=False),
    'intra_rb': lambda: IntraRBTransformerPrecoderSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12, embed_dim=128, num_heads=4, num_layers=4,
        snr_aware=True, use_abs=True, use_cossin=False),
    'ta_rb_residual': lambda: TransformerPrecoderCleanResidualSignedAttn(
        num_tx=M, num_rx=K, num_ofdm=14, fft_size=96, rb_size=12, tokens_per_rb=6,
        embed_dim=128, num_heads=4, num_layers=4, snr_aware=True),
}

for name, builder in builders.items():
    print(f'\n--- {name} ---')
    base_ptype = name
    system = MU_MIMO_System(num_tx=M, num_rx=K, precoder_type=base_ptype,
                             embed_dim=128, num_heads=4, num_layers=4,
                             tokens_per_rb=6 if name == 'ta_rb_residual' else 1)
    system.precoder = builder()

    batch_size = 16 if name != 'ta_rb_residual' else 32   # réduit depuis 128/256 -- OOM
    # au premier essai (finetune, tenseur [B,K,1,ofdm,fft,M,K] ~5.6GB à B=128,
    # M=64) -- M=64 alourdit ce tenseur x8 vs M=8 (STANDARD), pas juste x8 en
    # linéaire à cause du produit M*K dans la dernière dimension du calcul SINR
    trainer = SupervisedTrainer(
        system, dataset, run_name=f'massive_true_smoke_{name}',
        warmup_epochs=1, finetune_epochs=1,
        batch_size=batch_size, learning_rate=1e-3,
        steps_per_epoch=5, lr_schedule='cosine_long')

    t1 = time.time()
    trainer.train(print_every=5, patience=999)
    print(f'  -> OK, {name} : {(time.time()-t1):.1f}s pour 2x5 pas (batch={batch_size})')

    real_params = int(sum(tf.size(v).numpy() for v in system.precoder.trainable_variables))
    print(f'  params={real_params:,}')
    tf.keras.backend.clear_session()

print(f'\n{"="*70}\n✅ SMOKE TEST MASSIVE_TRUE COMPLET -- les 3 architectures tournent sans erreur\n{"="*70}')
