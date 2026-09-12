"""
diag_intrarb_gradient_layers.py — Investigation IntraRB (demande
utilisateur) : où EXACTEMENT dans l'architecture le gradient s'amortit
par rapport à SingleSC (déjà su : gradient global d'IntraRB
systématiquement plus faible que SingleSC à TOUS les SNR, cf. diag_
high_snr_disadvantage.py) ? Couche par couche, pas juste en bout de
chaîne.

Les deux modèles ont un point de comparaison DIRECT et honnête :
input_embed (Dense(25->128), MÊME feat_dim=25 aux deux, MÊME shape) et
output_proj (Dense(128->16), MÊME shape) sont structurellement
identiques entre les deux architectures -- si LEUR gradient diffère
déjà, la cause est en amont/aval de ces couches partagées, pas
spécifique à la présence de SC-attention. Le bloc user-attention
(attn_usr) existe aussi dans les deux (SingleSCBlock ET IntraRBBlock)
avec la même shape -- deuxième point de comparaison direct.

Usage: CUDA_VISIBLE_DEVICES=0 python3 diag_intrarb_gradient_layers.py
"""
import os, sys, re, pickle
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(__file__))
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from wmmse_convergence_check import ConfigurableMIMOSystem
from channel_config import STANDARD_CONFIG, set_locked_topology
from precoder_intra_rb import SingleSCTransformerPrecoder, IntraRBTransformerPrecoder
from sionna.phy.utils import ebnodb2no

M, K = STANDARD_CONFIG['NUM_TX'], STANDARD_CONFIG['NUM_RX']
SNR_POINTS = [5.0, 15.0, 20.0]
NUM_DRAWS  = 15
BATCH      = 32

CHECKPOINTS = {
    'SingleSC': ('single_sc', 'weights/SingleSC_4L_128d/best_20260807_125302'),
    'IntraRB':  ('intra_rb',  'weights/IntraRB_4L_128d/best_20260807_141928'),
}


class LockedSystem(ConfigurableMIMOSystem):
    cluster_radius_m = STANDARD_CONFIG['CLUSTER_RADIUS_M']
    indoor_probability = 0.0

    def new_topology(self, batch_size):
        set_locked_topology(self.channel_model, batch_size, num_ut=self.num_rx,
                             cluster_radius_m=self.cluster_radius_m,
                             force_los=False, indoor_probability=self.indoor_probability)


def build_neural(kind):
    kwargs = dict(num_tx=M, num_rx=K, num_ofdm=14, fft_size=96,
                  embed_dim=128, num_heads=4, num_layers=4)
    if kind == 'single_sc':
        return SingleSCTransformerPrecoder(**kwargs)
    if kind == 'intra_rb':
        return IntraRBTransformerPrecoder(rb_size=12, **kwargs)
    raise ValueError(kind)


def load_weights(model, ckpt_dir, dummy_h, no):
    _ = model(dummy_h, no=no, training=False)
    with open(os.path.join(ckpt_dir, 'weights.pkl'), 'rb') as f:
        ws = pickle.load(f)
    for v, w in zip(model.trainable_variables, ws):
        v.assign(w)
    return model


def build_var_to_label(model, kind):
    """Mappe chaque variable (par IDENTITÉ objet, pas par nom -- les noms
    Keras auto-générés type 'layer_normalization_7' ne reflètent PAS quel
    sous-composant sémantique c'est) vers un label de couche lisible, en
    parcourant directement la structure connue du modèle (input_embed,
    input_norm, blocks[i].{norm_sc,attn_sc,norm_usr,attn_usr,norm_ffn,ffn,
    s_sc,s_usr,s_ffn}, output_proj)."""
    label_of = {}

    def add(layer_or_var, label):
        vs = layer_or_var.trainable_variables if hasattr(layer_or_var, 'trainable_variables') else [layer_or_var]
        for v in vs:
            label_of[id(v)] = label

    add(model.input_embed, '1_input_embed')
    add(model.input_norm, '2_input_norm')
    if kind == 'intra_rb':
        add(model.pos_enc_sc, '3_pos_enc_sc')

    for i, blk in enumerate(model.blocks):
        if kind == 'intra_rb':
            add(blk.norm_sc, f'4_blk{i}_norm_sc')
            add(blk.attn_sc, f'5_blk{i}_attn_sc')
            add(blk.s_sc, f'A_blk{i}_s_sc(scalaire)')
        add(blk.norm_usr, f'6_blk{i}_norm_usr')
        add(blk.attn_usr, f'7_blk{i}_attn_usr')
        add(blk.norm_ffn, f'8_blk{i}_norm_ffn')
        add(blk.ffn_up, f'9_blk{i}_ffn_up')
        add(blk.ffn_out, f'9_blk{i}_ffn_out')
        add(blk.s_usr, f'B_blk{i}_s_usr(scalaire)')
        add(blk.s_ffn, f'C_blk{i}_s_ffn(scalaire)')

    add(model.output_proj, 'D_output_proj')
    return label_of


if __name__ == '__main__':
    fresh_system = LockedSystem(M, K)
    dummy_h = tf.zeros([1, K, 1, 1, M, 14, 96], dtype=tf.complex64)
    dummy_no = ebnodb2no(tf.constant(15.0, tf.float32), 2, 0.5, fresh_system.rg)

    models = {}
    for name, (kind, ckpt) in CHECKPOINTS.items():
        m = build_neural(kind)
        load_weights(m, ckpt, dummy_h, dummy_no)
        models[name] = m

    # -- valeurs s_sc/s_usr/s_ffn actuelles (vérif rapide demandée) --
    print(f'\n{"="*80}\nScalaires résiduels appris (IntraRB, checkpoint STANDARD final)\n{"="*80}')
    for v in models['IntraRB'].trainable_variables:
        if re.search(r's_(sc|usr|ffn):', v.name):
            print(f'  {v.name:45s} = {float(v.numpy()):+.5f}')

    rate_norm = float(K) * 9.0
    label_maps = {name: build_var_to_label(models[name], kind)
                  for name, (kind, _) in CHECKPOINTS.items()}

    def grad_by_group(model, h, no, system, label_of):
        with tf.GradientTape() as tape:
            g_re, g_im = model(h, no=no, training=True, return_real_imag=True)
            g = tf.complex(g_re, g_im)
            h_eff = system.precoded_channel_helper.compute_effective_channel(h, g)
            sinr = system.lmmse_sinr(h_eff, no=no, interference_whitening=True)
            rate = tf.reduce_sum(tf.reduce_mean(
                tf.math.log(1.0 + tf.clip_by_value(sinr, 1e-9, 1e4)) / tf.math.log(2.0),
                axis=[0, 1, 2, 4]))
            loss = -rate / rate_norm
        grads = tape.gradient(loss, model.trainable_variables)
        groups = {}
        for v, gr in zip(model.trainable_variables, grads):
            if gr is None:
                continue
            grp = label_of.get(id(v), f'Z_unmapped({v.name})')
            gn = float(tf.norm(gr))
            groups.setdefault(grp, []).append(gn)
        return {k: float(np.mean(v)) for k, v in groups.items()}

    print(f'\n{"="*100}\nNorme de gradient PAR GROUPE DE COUCHE, moyennée sur {NUM_DRAWS} tirages, par SNR\n{"="*100}')
    all_results = {}
    for snr in SNR_POINTS:
        no = ebnodb2no(tf.constant(snr, tf.float32), 2, 0.5, fresh_system.rg)
        acc = {name: {} for name in models}
        for _ in range(NUM_DRAWS):
            fresh_system.new_topology(BATCH)
            h, _ = fresh_system.channel_and_no(tf.constant(BATCH, tf.int32), tf.constant(snr, tf.float32))
            for name, model in models.items():
                gg = grad_by_group(model, h, no, fresh_system, label_maps[name])
                for k, v in gg.items():
                    acc[name].setdefault(k, []).append(v)
        all_results[snr] = {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in acc.items()}

        print(f'\n--- SNR={snr}dB ---')
        all_groups = sorted(set(all_results[snr]['SingleSC']) | set(all_results[snr]['IntraRB']))
        print(f'{"Groupe":<35} {"SingleSC":>12} {"IntraRB":>12} {"ratio I/S":>10}')
        for grp in all_groups:
            s = all_results[snr]['SingleSC'].get(grp, float('nan'))
            i = all_results[snr]['IntraRB'].get(grp, float('nan'))
            ratio = i / s if (s and not np.isnan(s) and not np.isnan(i)) else float('nan')
            print(f'{grp:<35} {s:>12.5f} {i:>12.5f} {ratio:>10.3f}')

    import json
    os.makedirs('results', exist_ok=True)
    with open('results/diag_intrarb_gradient_layers.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print('\nSauvé -> results/diag_intrarb_gradient_layers.json')
