"""
precoder_experimental.py — Front A (demande utilisateur) : pistes pour
combler le décrochage à haut SNR (canal-indépendant, cf. SESSION_LOG_
20260808.md), motivé par arXiv:2211.14775 (le précodeur optimal à haut SNR
approche ZFBF, qui nécessite une inversion précise de la matrice de Gram
H^H·H -- opération structurellement difficile pour un réseau générique).

PAS de deep unfolding (aucune itération d'un algorithme existant n'est
déroulée dans l'architecture) -- toutes les variantes ci-dessous restent
des couches apprises génériques, juste avec une forme fonctionnelle ou une
feature d'entrée mieux adaptée au problème.

Variantes (sous-classes de SingleSCTransformerPrecoder, backward-compatible
-- la classe de base n'est PAS modifiée) :

1. SingleSCTransformerPrecoderGram : feature d'entrée augmentée de la ligne
   de la matrice de Gram H^H·H (produit scalaire complexe entre chaque
   paire d'utilisateurs, par SC) -- donne explicitement au réseau
   l'information d'interférence croisée que RZF/WMMSE utilisent
   directement via (H^H·H + alpha·I)^-1, au lieu de la laisser reconstruire
   implicitement depuis re/im/abs bruts.

3. SingleSCTransformerPrecoderBilinearOut : constat de code -- attn_usr
   (SingleSCBlock) est une MultiHeadAttention SOFTMAX standard, qui ne peut
   mathématiquement produire qu'une COMBINAISON CONVEXE des value vectors
   (poids >=0, somme=1). Le zero-forcing/RZF a structurellement besoin de
   coefficients SIGNÉS (inversion de matrice) pour annuler l'interférence
   -- une combinaison convexe ne peut PAS représenter une soustraction.
   Ajoute une couche de mixing bilinéaire SANS softmax (scores signés,
   pas de normalisation positive) juste avant la projection de sortie,
   donnant au réseau la capacité algébrique de représenter des
   combinaisons à coefficients négatifs entre users -- toujours une
   couche générique apprise, aucune itération d'algorithme déroulée.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import RMSNormalization

from precoder_intra_rb import (SingleSCTransformerPrecoder, SingleSCBlock,
                                IntraRBTransformerPrecoder, IntraRBBlock)
from precoders_v2 import (TransformerPrecoderCleanResidual,
                           SeparableAttentionBlock)


# =============================================================================
# 1. FEATURE ENGINEERING — Gram matrix (H^H . H) par SC
# =============================================================================

class SingleSCTransformerPrecoderGram(SingleSCTransformerPrecoder):
    """Ajoute 2K features (re+im de la ligne de Gram du user courant) aux
    features de base (re/im/abs/log_no). Le reste de l'architecture
    (blocks, output_proj) est inchangé -- seul input_embed est reconstruit
    pour le nouveau feat_dim."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_dim_base = self.feat_dim
        self.feat_dim = self.feat_dim_base + 2 * self.K
        self.input_embed = layers.Dense(self.D, name='input_embed_gram')
        print(f"  -> +Gram features : feat_dim {self.feat_dim_base} -> {self.feat_dim} "
              f"(+{2 * self.K} = 2K, K={self.K})")

    def _extract_features(self, h, no=None):
        """h : [B, K, M, fft]"""
        base_feats = super()._extract_features(h, no)   # [B, fft, K, feat_dim_base]

        # Gram matrix par SC : G[b,f,k,l] = sum_m h[b,k,m,f] * conj(h[b,l,m,f])
        h_p  = tf.transpose(h, [0, 3, 1, 2])              # [B, fft, K, M]
        gram = tf.einsum('bfkm,bflm->bfkl', h_p, tf.math.conj(h_p))  # [B,fft,K,K] complex
        # Normalisation par M (diagonale = ||h_k||^2, somme de M termes
        # unit-power -> échelle ~M, ramené à ~1 comme les autres features)
        gram_re = tf.math.real(gram) / tf.cast(self.M, tf.float32)
        gram_im = tf.math.imag(gram) / tf.cast(self.M, tf.float32)
        gram_feats = tf.concat([gram_re, gram_im], axis=-1)   # [B, fft, K, 2K]

        return tf.concat([base_feats, gram_feats], axis=-1)   # [B, fft, K, feat_dim]


# =============================================================================
# 3. CAPACITÉ CIBLÉE — mixing bilinéaire SIGNÉ (pas de softmax) en sortie
# =============================================================================

class BilinearMixBlock(layers.Layer):
    """Mixing appris entre les K users, SANS softmax (scores signés) --
    contrairement à MultiHeadAttention (combinaison convexe uniquement),
    peut représenter une combinaison à coefficients négatifs, nécessaire
    pour l'annulation d'interférence (structure zero-forcing).
    Init à 0 (mix_scale) -> identité au démarrage, apprentissage progressif
    (même convention que s_sc/s_usr/s_ffn du reste du codebase)."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.q_proj = layers.Dense(embed_dim, use_bias=False, name='bilinear_q')
        self.k_proj = layers.Dense(embed_dim, use_bias=False, name='bilinear_k')
        self.v_proj = layers.Dense(embed_dim, name='bilinear_v')
        self.scale  = embed_dim ** -0.5
        self.mix_scale = self.add_weight(shape=(), dtype='float32',
                            initializer='zeros', trainable=True, name='mix_scale')

    def call(self, x, training=False):
        """x : [N, K, D]"""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = tf.einsum('nkd,nld->nkl', q, k) * self.scale   # [N,K,K] -- PAS de softmax
        mix    = tf.einsum('nkl,nld->nkd', scores, v)
        return x + self.mix_scale * mix


class SingleSCTransformerPrecoderBilinearOut(SingleSCTransformerPrecoder):
    """SingleSCTransformerPrecoder + BilinearMixBlock inséré juste avant
    output_proj. call() dupliqué depuis la classe de base (une seule ligne
    ajoutée) -- le reste (features, blocks, normalisation de sortie) est
    identique."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bilinear_mix = BilinearMixBlock(self.D, name='bilinear_mix')
        print(f"  -> +BilinearMixBlock (signé, pas de softmax) avant output_proj")

    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        B = tf.shape(h_freq)[0]
        K, M, D, F = self.K, self.M, self.D, self.fft_size

        h = tf.squeeze(h_freq, axis=[2, 3])
        h_pilot = h[:, :, :, self.pilot_idx, :]

        feats = self._extract_features(h_pilot, no)
        x = self.input_norm(self.input_embed(feats))

        x = tf.reshape(x, [B * F, K, D])
        for blk in self.blocks:
            x = blk(x, training=training)

        # ── Seule différence vs la classe de base : mixing bilinéaire signé ──
        x = self.bilinear_mix(x, training=training)

        out = self.output_proj(x)
        out_r, out_i = tf.split(out, 2, axis=-1)

        out_r = tf.transpose(out_r, [0, 2, 1])
        out_i = tf.transpose(out_i, [0, 2, 1])

        norm = tf.sqrt(out_r**2 + out_i**2)
        norm = tf.sqrt(tf.reduce_sum(norm**2, axis=1, keepdims=True) + 1e-12)
        out_r = out_r / norm
        out_i = out_i / norm

        out_r = tf.reshape(out_r, [B, F, M, K])
        out_i = tf.reshape(out_i, [B, F, M, K])

        def to_sionna(t):
            t = tf.expand_dims(t, axis=1)
            t = tf.expand_dims(t, axis=2)
            return tf.tile(t, [1, 1, self.num_ofdm, 1, 1, 1])

        if return_real_imag:
            return to_sionna(out_r), to_sionna(out_i)
        w = tf.complex(out_r, out_i)
        return to_sionna(w)


# =============================================================================
# 5. GNN permutation-équivariante — "sign-aware gating" (demande utilisateur,
#    littérature Zhang/Han/Yang 2025 arXiv:2503.06077 ; GESC 2026
#    arXiv:2511.16062) : remplace attn_usr (MultiHeadAttention softmax
#    standard) par un mécanisme à poids SIGNÉS, dans TOUS les blocs (pas
#    juste une correction finale comme BilinearMixBlock -- structurellement
#    plus proche de la littérature GNN citée, qui bâtit l'équivariance de
#    permutation ET l'annulation signée directement dans le mécanisme
#    d'agrégation entre users, répété à chaque couche).
# =============================================================================

class SignedGateMHA(layers.Layer):
    """User-attention avec porte de signe : poids_final = softmax(scores)
    [magnitude, normalisée -- garde la stabilité d'entraînement du softmax
    standard] * tanh(scores) [porte de signe, [-1,1] -- permet au réseau
    d'INVERSER le signe de la contribution d'un user à un autre, ce que
    softmax seul ne peut jamais faire (poids toujours >=0)]. Permutation-
    équivariant par construction (comme MultiHeadAttention standard --
    aucun poids ne dépend de l'ORDRE des K users, seulement de leur
    contenu). Interface drop-in compatible avec layers.MultiHeadAttention
    pour l'appel self-attention `attn(query, value, training=...)` déjà
    utilisé dans SingleSCBlock.call -- aucune autre modification needed."""

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.h, self.kd = num_heads, key_dim
        d = num_heads * key_dim
        self.q_proj   = layers.Dense(d, name='sg_q')
        self.k_proj   = layers.Dense(d, name='sg_k')
        self.v_proj   = layers.Dense(d, name='sg_v')
        self.out_proj = layers.Dense(d, name='sg_out')
        self.scale    = key_dim ** -0.5

    def call(self, query, value, training=False):
        """query, value : [N, K, D] (self-attention : value == query)"""
        N = tf.shape(query)[0]
        K = tf.shape(query)[1]

        def split_heads(t):
            t = tf.reshape(t, [N, K, self.h, self.kd])
            return tf.transpose(t, [0, 2, 1, 3])          # [N,h,K,kd]

        q = split_heads(self.q_proj(query))
        k = split_heads(self.k_proj(value))
        v = split_heads(self.v_proj(value))

        raw  = tf.einsum('nhkd,nhld->nhkl', q, k) * self.scale   # [N,h,K,K]
        mag  = tf.nn.softmax(raw, axis=-1)     # magnitude >=0, somme=1 (stabilité)
        sign = tf.tanh(raw)                    # porte de signe [-1,1]
        w    = mag * sign                      # combinaison SIGNÉE, plus une combinaison convexe

        out = tf.einsum('nhkl,nhld->nhkd', w, v)              # [N,h,K,kd]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [N, K, self.h * self.kd])
        return self.out_proj(out)


class SingleSCBlockSignedAttn(SingleSCBlock):
    """SingleSCBlock avec attn_usr remplacé par SignedGateMHA. norm_usr,
    FFN, s_usr/s_ffn inchangés (hérités tels quels)."""

    def __init__(self, embed_dim=128, num_heads=4, ffn_mult=4, dropout=0.0, **kwargs):
        super().__init__(embed_dim, num_heads, ffn_mult, dropout, **kwargs)
        kd = embed_dim // num_heads
        self.attn_usr = SignedGateMHA(num_heads, kd, name='signed_attn_usr')


class SingleSCTransformerPrecoderSignedAttn(SingleSCTransformerPrecoder):
    """SingleSCTransformerPrecoder avec les blocs remplacés par
    SingleSCBlockSignedAttn -- signed gating appliqué à CHAQUE couche
    (4 blocs), pas juste en sortie. Features/output_proj/normalisation
    inchangés.

    N'appelle PAS SingleSCTransformerPrecoder.__init__ (Model.__init__
    direct + setup dupliqué à la main) : appeler le __init__ de base
    construit d'abord les blocs softmax standard AVANT qu'on les
    remplace -- ces blocs jamais utilisés dans le forward restent
    tracked par Keras (poids jamais appelés -> gradient None, planté à
    l'entraînement, trouvé au smoke test). Bypasser complètement évite
    le problème à la racine plutôt que de le patcher après coup."""

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 pilot_idx=2, embed_dim=128, num_heads=4, num_layers=2,
                 ffn_mult=4, dropout=0.0, snr_aware=True, use_abs=True,
                 use_cossin=False, **kwargs):
        Model.__init__(self, **kwargs)

        self.M, self.K = num_tx, num_rx
        self.num_ofdm, self.fft_size = num_ofdm, fft_size
        self.pilot_idx = pilot_idx
        self.D = embed_dim
        self.snr_aware = snr_aware
        self.use_abs, self.use_cossin = use_abs, use_cossin
        self.feat_dim = (2 * num_tx
                          + (num_tx if use_abs else 0)
                          + (2 * num_tx if use_cossin else 0)
                          + (1 if snr_aware else 0))

        self.input_embed = layers.Dense(embed_dim, name='input_embed')
        self.input_norm  = layers.LayerNormalization(epsilon=1e-6, name='input_norm')
        self.blocks = [
            SingleSCBlockSignedAttn(embed_dim, num_heads, ffn_mult, dropout,
                                     name=f'single_sc_signed_{i}')
            for i in range(num_layers)
        ]
        self.output_proj = layers.Dense(2 * num_tx, name='output_proj')

        print(
            f"SingleSCTransformerPrecoderSignedAttn | M={num_tx} K={num_rx} | "
            f"D={embed_dim} L={num_layers} H={num_heads} | feat_dim={self.feat_dim} | "
            f"SignedGateMHA dans tous les blocs (poids signés, pas de contrainte de convexité)"
        )


# =============================================================================
# 5bis. signed_attn étendu — IntraRB et TA-RB résiduel (Priorité 2, demande
#    utilisateur) : même mécanisme SignedGateMHA, appliqué à l'attention
#    user de chaque architecture (attn_usr / user_attn) -- la SC-attention
#    / freq-attn (mélange fréquentiel, sans rapport avec l'annulation
#    d'interférence entre users) reste INCHANGÉE dans les deux cas.
# =============================================================================

class IntraRBBlockSignedAttn(IntraRBBlock):
    """IntraRBBlock avec attn_usr remplacé par SignedGateMHA. attn_sc
    (SC-attention), norm_sc/s_sc, FFN inchangés (hérités tels quels)."""

    def __init__(self, embed_dim=128, num_heads=4, ffn_mult=4, dropout=0.0, **kwargs):
        super().__init__(embed_dim, num_heads, ffn_mult, dropout, **kwargs)
        kd = embed_dim // num_heads
        self.attn_usr = SignedGateMHA(num_heads, kd, name='signed_attn_usr')


class IntraRBTransformerPrecoderSignedAttn(IntraRBTransformerPrecoder):
    """IntraRBTransformerPrecoder avec les blocs remplacés par
    IntraRBBlockSignedAttn. Bypass complet de IntraRBTransformerPrecoder.
    __init__ (même raison que SingleSCTransformerPrecoderSignedAttn :
    éviter le tracking Keras des blocs softmax standard jamais utilisés)."""

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 rb_size=12, pilot_idx=2, embed_dim=128, num_heads=4,
                 num_layers=4, ffn_mult=4, dropout=0.0, snr_aware=True,
                 use_abs=True, use_cossin=False, **kwargs):
        Model.__init__(self, **kwargs)

        assert fft_size % rb_size == 0

        self.M, self.K = num_tx, num_rx
        self.num_ofdm, self.fft_size = num_ofdm, fft_size
        self.rb_size = rb_size
        self.N_RB = fft_size // rb_size
        self.pilot_idx = pilot_idx
        self.D = embed_dim
        self.snr_aware = snr_aware
        self.use_abs, self.use_cossin = use_abs, use_cossin
        self.feat_dim = (2 * num_tx
                          + (num_tx if use_abs else 0)
                          + (2 * num_tx if use_cossin else 0)
                          + (1 if snr_aware else 0))

        self.pos_enc_sc  = layers.Embedding(rb_size, embed_dim, name='pos_sc')
        self.input_embed = layers.Dense(embed_dim, name='input_embed')
        self.input_norm  = layers.LayerNormalization(epsilon=1e-6, name='input_norm')
        self.blocks = [
            IntraRBBlockSignedAttn(embed_dim, num_heads, ffn_mult, dropout,
                                    name=f'intra_rb_signed_{i}')
            for i in range(num_layers)
        ]
        self.output_proj = layers.Dense(2 * num_tx, name='output_proj')

        print(
            f"IntraRBTransformerPrecoderSignedAttn | M={num_tx} K={num_rx} | "
            f"rb_size={rb_size} N_RB={self.N_RB} | D={embed_dim} L={num_layers} H={num_heads} | "
            f"feat_dim={self.feat_dim} | SignedGateMHA dans attn_usr (attn_sc inchangée)"
        )


class SeparableAttentionBlockSignedAttn(SeparableAttentionBlock):
    """SeparableAttentionBlock (TA-RB) avec user_attn remplacé par
    SignedGateMHA. freq_attn (mélange fréquentiel), FFN inchangés."""

    def __init__(self, num_tokens, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(num_tokens, num_users, embed_dim, num_heads, dropout, **kwargs)
        kd = embed_dim // num_heads
        self.user_attn = SignedGateMHA(num_heads, kd, name='signed_user_attn')


class TransformerPrecoderCleanResidualSignedAttn(TransformerPrecoderCleanResidual):
    """TransformerPrecoderCleanResidual (TA-RB résiduel, standard actuel)
    avec les blocs remplacés par SeparableAttentionBlockSignedAttn.
    Bypass complet de TransformerPrecoderCleanResidual.__init__ (même
    raison que les deux classes ci-dessus) -- _extract_features/call/
    complexity hérités tels quels (agnostiques au contenu interne des
    blocs, cf. precoders_v2.py)."""

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4,
                 num_layers=4, dropout=0.0, feature_flags=None,
                 snr_aware=True, alpha_init=0.1, **kwargs):
        Model.__init__(self, **kwargs)
        self.num_tx, self.num_rx = num_tx, num_rx
        self.num_ofdm = num_ofdm
        self.fft_size, self.rb_size = fft_size, rb_size
        self.num_rb = fft_size // rb_size
        self.tokens_per_rb = tokens_per_rb
        self.sc_per_token = rb_size // tokens_per_rb
        self.total_tokens = self.num_rb * tokens_per_rb
        self.embed_dim = embed_dim
        self.snr_aware = snr_aware

        self.feature_flags = feature_flags or {'mean': True, 'slope': False, 'var': True, 'cov': False}
        ff = self.feature_flags
        self.feat_dim = (2*num_tx if ff['mean'] else 0) + (2*num_tx if ff['slope'] else 0) \
                       + (num_tx if ff['var'] else 0) + (2*num_rx if ff['cov'] else 0) \
                       + num_rx + (1 if snr_aware else 0)
        assert self.feat_dim > num_rx, "au moins une feature canal doit être active"

        self.input_proj = layers.Dense(embed_dim, activation='gelu', name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')
        if snr_aware:
            self.snr_proj = layers.Dense(embed_dim, activation='gelu', name='snr_proj')
        self.pos_embed = layers.Embedding(self.total_tokens, embed_dim, name='pos_embed')
        self.blocks = [
            SeparableAttentionBlockSignedAttn(self.total_tokens, num_rx, embed_dim, num_heads,
                                               dropout=dropout, name=f'block_signed_{i}')
            for i in range(num_layers)
        ]
        self.joint_output_proj = layers.Dense(embed_dim, name='joint_output_proj')

        self.token_to_precoder = layers.Dense(2 * num_tx, name='token_to_precoder')
        hidden = 4 * num_tx
        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(hidden, kernel_size=rb_size, padding='same',
                          activation='gelu', name='sc_r1_res'),
            layers.Conv1D(2 * num_tx, kernel_size=1, padding='same', name='sc_r2_res'),
        ], name='sc_refine_residual')
        self.alpha = self.add_weight(name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(alpha_init), trainable=True)

        print(f"TransformerPrecoderCleanResidualSignedAttn | T={tokens_per_rb} tok/RB | D={embed_dim} | "
              f"feat_dim={self.feat_dim} | SignedGateMHA dans user_attn (freq_attn inchangée)")
