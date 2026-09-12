"""
diag_inspect_intrarb_weights.py — Investigation Partie 1, hypothèse #3 :
le réseau IntraRB entraîné (STANDARD, run complet ÉTAPE 4, checkpoint
best_20260807_141928) utilise-t-il vraiment la SC-attention, ou les
scalaires résiduels s_sc ont-ils convergé vers ~0 (SC-attention
neutralisée en pratique) ?

Inspecte : s_sc/s_usr/s_ffn par bloc, normes des poids de attn_sc vs
attn_usr, et (si possible) un pattern d'attention réel sur un batch de
canaux du cache STANDARD.

N'entraîne rien -- juste un chargement de poids + inspection numpy.
Léger, sûr à faire tourner à côté de MASSIVE.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_inspect_intrarb_weights.py
"""
import os, sys, pickle, json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from precoder_intra_rb import IntraRBTransformerPrecoder

CKPT = 'weights/IntraRB_4L_128d/best_20260807_141928'

model = IntraRBTransformerPrecoder(num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                                    rb_size=12, embed_dim=128, num_heads=4,
                                    num_layers=4, snr_aware=True)

# Build (comme load_weights_from le suppose implicitement -- les variables
# doivent exister avant l'assign)
dummy_h = tf.zeros([1, 4, 1, 1, 8, 14, 96], dtype=tf.complex64)
_ = model(dummy_h, no=tf.constant(1e-3), training=False)

with open(os.path.join(CKPT, 'weights.pkl'), 'rb') as f:
    ws = pickle.load(f)
for v, w in zip(model.trainable_variables, ws):
    v.assign(w)
print(f'✅ {len(ws)} poids chargés depuis {CKPT}')

with open(os.path.join(CKPT, 'metrics.json')) as f:
    print('metrics.json:', json.load(f))
with open(os.path.join(CKPT, 'config.json')) as f:
    print('config.json:', json.load(f))

print(f'\n{"="*70}\nScalaires résiduels par bloc (s_sc / s_usr / s_ffn)\n{"="*70}')
print('(initialisés à 0 -- une valeur proche de 0 après entraînement veut dire')
print(' que le réseau a appris à NE PAS utiliser cette branche)\n')
for v in model.trainable_variables:
    if '/s_sc:' in v.name or '/s_usr:' in v.name or '/s_ffn:' in v.name:
        print(f'  {v.name:45s} = {float(v.numpy()):+.5f}')

print(f'\n{"="*70}\nNormes des poids attn_sc vs attn_usr par bloc (kernel query)\n{"="*70}')
for i, blk in enumerate(model.blocks):
    q_sc  = blk.attn_sc._query_dense.kernel.numpy()
    q_usr = blk.attn_usr._query_dense.kernel.numpy()
    o_sc  = blk.attn_sc._output_dense.kernel.numpy()
    o_usr = blk.attn_usr._output_dense.kernel.numpy()
    print(f'  Bloc {i}: ||Q_sc||={np.linalg.norm(q_sc):.3f}  ||Q_usr||={np.linalg.norm(q_usr):.3f}  '
          f'||O_sc||={np.linalg.norm(o_sc):.3f}  ||O_usr||={np.linalg.norm(o_usr):.3f}  '
          f's_sc={float(blk.s_sc.numpy()):+.5f}  s_usr={float(blk.s_usr.numpy()):+.5f}')

print(f'\n{"="*70}\nPatterns d\'attention réels (batch du cache STANDARD, SNR=15dB)\n{"="*70}')
try:
    from datasets import CachedSionnaDataset
    from main_finall import MU_MIMO_System, NUM_TX, NUM_RX, DATASET_SIZE, CHOSEN_CONFIG
    dummy_sys = MU_MIMO_System(num_tx=NUM_TX, num_rx=NUM_RX, precoder_type='rzf')
    ds = CachedSionnaDataset(
        dummy_sys, dataset_size=DATASET_SIZE, batch_size=128,
        cache_file=f'/export/tmp/sala/sionna_joint_{DATASET_SIZE//1000}k_{NUM_TX}x{NUM_RX}.npz',
        cluster_radius_m=CHOSEN_CONFIG['CLUSTER_RADIUS_M'],
        indoor_probability=CHOSEN_CONFIG['INDOOR_PROBABILITY'], seed=999)
    h = ds.get_batch(8)   # petit batch, léger sur le GPU partagé avec MASSIVE

    # Reproduit le début de call() pour intercepter l'entrée du bloc 0
    B = tf.shape(h)[0]
    h_sq = tf.squeeze(h, axis=[2, 3])
    h_pilot = h_sq[:, :, :, model.pilot_idx, :]
    feats = model._extract_features(h_pilot, tf.constant(1e-3))
    x = model.input_norm(model.input_embed(feats))
    sc_idx = tf.range(model.rb_size)
    pos = model.pos_enc_sc(sc_idx)
    pos = tf.reshape(pos, [1, 1, model.rb_size, 1, model.D])
    x = x + pos
    x = tf.reshape(x, [B * model.N_RB, model.rb_size, model.K, model.D])

    blk = model.blocks[0]
    B_rb, S, K, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    xsc = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B_rb * K, S, D])
    xsc_n = blk.norm_sc(xsc)
    _, attn_scores = blk.attn_sc(xsc_n, xsc_n, return_attention_scores=True, training=False)
    # attn_scores: [B_rb*K, num_heads, S, S]
    mean_attn = tf.reduce_mean(attn_scores, axis=[0, 1]).numpy()   # [S, S]
    uniform = 1.0 / model.rb_size
    print(f'  Attention SC bloc 0, moyenne sur le batch/heads (uniforme = {uniform:.4f}) :')
    print(f'  Diagonale (self) moy={np.diag(mean_attn).mean():.4f} | '
          f'hors-diag moy={(mean_attn.sum()-np.trace(mean_attn))/(model.rb_size**2-model.rb_size):.4f}')
    print(f'  Écart-type spatial (sur l\'axe clé, moyenné sur les requêtes) : {mean_attn.std(axis=1).mean():.5f}')
    print(f'  Matrice complète [S,S] :')
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(mean_attn)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'⚠️  Inspection attention réelle échouée : {e}')
