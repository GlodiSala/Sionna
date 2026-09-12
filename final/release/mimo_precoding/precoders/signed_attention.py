"""
signed_attention.py — SignedGateMHA and the three signed-attention
precoder architectures (SC, IntraRB, TA-RB residual). This is the
winning architecture family of the thesis: user-attention is replaced,
in every attention block, by a magnitude/sign-gated mechanism instead
of standard softmax multi-head attention.

Motivation: the optimal high-SNR precoder approaches ZFBF, which
requires signed (not just convex-combination) coefficients between
users to null interference -- a plain softmax attention can only
produce a convex combination of value vectors (weights >=0, sum=1),
so it cannot represent the sign flips zero-forcing needs. SignedGateMHA
multiplies the softmax magnitude by a tanh(scores) sign gate, giving
the network that degree of freedom while keeping softmax's training
stability. Permutation-equivariant by construction (verified
numerically, max abs diff 0.0 under permutation of the K users) --
same drop-in interface as tf.keras.layers.MultiHeadAttention.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from precoders.base_intra_rb import (SingleSCTransformerPrecoder, SingleSCBlock,
                                      IntraRBTransformerPrecoder, IntraRBBlock)
from precoders.base_residual import (TransformerPrecoderCleanResidual,
                                      SeparableAttentionBlock)


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
