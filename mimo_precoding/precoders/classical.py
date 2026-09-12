# =============================================================================
# precoders/classical.py  —  Transformer Precoding for MU-MIMO
# Architectures : TransformerPrecoderV4 (v4.0/v4.1/v4.2/v4.3)
#                 TransformerPrecoderV5
# =============================================================================
import tensorflow as tf
import numpy as np
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import RMSNormalization


# =============================================================================
# HELPERS — canal Sionna
# =============================================================================

def _get_desired_channels(h_hat, stream_management):
    """
    Réplique de PrecodedChannel.get_desired_channels() de Sionna.
    Input:  h_hat [B, num_rx, num_rx_ant, num_tx, num_tx_ant, ofdm, fft]
    Output: h_pc  [B, num_tx, ofdm, fft, num_streams_per_tx, num_tx_ant]
    """
    h_pc = tf.transpose(h_hat, [3, 1, 2, 4, 5, 6, 0])
    h_pc = tf.gather(h_pc, stream_management.precoding_ind,
                     axis=1, batch_dims=1)
    s    = tf.shape(h_pc)
    h_pc = tf.reshape(h_pc, [s[0], s[1]*s[2], s[3], s[4], s[5], s[6]])
    h_pc = tf.transpose(h_pc, [5, 0, 3, 4, 1, 2])
    return h_pc


# =============================================================================
# PRÉCODEURS CLASSIQUES
# =============================================================================

def rzf_precoder(h_freq, stream_management, no=None, alpha=None):
    """RZF — régularisation α = no si non spécifié."""
    h_pc = _get_desired_channels(h_freq, stream_management)
    if alpha is None:
        alpha_val = tf.reshape(tf.cast(no/2, tf.float32), [])
    else:
        alpha_val = tf.cast(alpha, tf.float32)
    return rzf_precoding_matrix(h_pc, alpha=alpha_val)


def wmmse_precoder(h_freq, no, stream_management, num_iterations=10,
                    bisection_iters=30, bisection_doublings=30):
    """
    WMMSE itératif — initialisé depuis RZF, α = no (optimal i.i.d.).
    Sortie : [B, 1, ofdm, fft, M, K]

    A_reg devient sévèrement mal conditionnée à haut SNR (no -> 0 alors que
    les poids W peuvent atteindre ~1e4), ce que complex64/float32 ne peut
    résoudre avec précision (cond(A_reg) observé jusqu'à ~1e11). La boucle
    est donc exécutée en complex128/float64 ; seule l'entrée/sortie reste
    complex64 pour ne pas changer l'interface externe.

    Mise à jour de V — multiplicateur de Lagrange μ exact (Shi et al. 2011,
    "An Iteratively Weighted MMSE Approach...", Algorithm 1) :
    v_k = (Σ_j w_j|u_j|² h_j h_j^H + μI)⁻¹ w_k u_k* h_k, où μ≥0 est choisi
    pour que Σ_k‖v_k‖² = K EXACTEMENT (contrainte de puissance active).
    Précédemment, μ était fixé arbitrairement à `no` (par analogie avec la
    régularisation RZF) puis la puissance totale était renormalisée a
    posteriori — une renormalisation scalaire globale ne peut pas
    reproduire l'effet d'un μ différent (qui change la DIRECTION de chaque
    v_k, pas seulement sa norme), ce qui cassait la garantie de
    non-décroissance monotone du WSR de l'algorithme (Théorème 1 de Shi et
    al.) : diagnostic instrumenté = WSR interne strictement DÉCROISSANT à
    chaque itération (14/14 violations sur 3 batches testés), et le gap
    RZF-WMMSE qui grandissait avec plus d'itérations (+0.54% à 10 iters,
    +1.84% à 50 iters) au lieu de se refermer.

    μ est résolu par bisection vectorisée, sans re-résoudre le système
    linéaire à chaque étape de bisection : A = Σ_k w_k|u_k|² h_k h_k^H est
    Hermitienne semi-définie positive, donc A = QΣQ^H (eigh), et
    v_k(μ) = Q(Σ+μI)⁻¹Q^H b_k implique ‖V(μ)‖²_F = Σ_m p_m/(σ_m+μ)² avec
    p_m = Σ_k|c_k[m]|², c_k = Q^H b_k — fonction scalaire décroissante de
    μ, bisectée en ~30+30 itérations d'arithmétique bon marché (un seul
    eigh() par itération WMMSE, pas un solve() par étape de bisection).

    Validé (voir diagnostic) : WSR interne désormais non-décroissant à
    chaque itération (0/14 violations sur les mêmes 3 batches), et le gap
    RZF-WMMSE se referme avec plus d'itérations (+0.11% à 10 iters,
    -0.08% à 50 iters — WMMSE dépasse RZF, comme attendu théoriquement).
    """
    input_dtype = h_freq.dtype
    h_pc = _get_desired_channels(h_freq, stream_management)
    H    = tf.cast(tf.squeeze(h_pc, axis=1), tf.complex128)   # [B, ofdm, fft, K, M]

    s       = tf.shape(H)
    K       = s[3]; M = s[4]
    K_float = tf.cast(K, tf.float64)

    no_scalar  = tf.cast(tf.reshape(no, []), tf.float64)
    no_val     = tf.reshape(no_scalar,  [1, 1, 1, 1])

    V = tf.cast(tf.squeeze(
        rzf_precoding_matrix(h_pc, alpha=tf.cast(no_scalar, tf.float32)), axis=1),
        tf.complex128)

    for _ in range(num_iterations):
        HV      = tf.matmul(H, V)
        signal  = tf.linalg.diag_part(HV)
        tot_pwr = tf.reduce_sum(tf.abs(HV)**2, axis=-1)
        denom   = tf.cast(tot_pwr + no_val, H.dtype)
        U       = tf.math.conj(signal) / denom

        mse = 1.0 - tf.math.real(U * signal)
        W   = 1.0 / tf.maximum(mse, 1e-7)

        weights     = tf.cast(W * tf.abs(U)**2, H.dtype)
        H_H         = tf.linalg.adjoint(H)
        weighted_HH = H_H * weights[:, :, :, None, :]
        A           = tf.matmul(weighted_HH, H)          # [...,M,M] Hermitienne PSD (pas de +no*I ici)

        target = tf.cast(W, H.dtype) * tf.math.conj(U)
        B_rhs  = H_H * target[:, :, :, None, :]           # [...,M,K]

        # --- V(μ) exact sous contrainte de puissance, via eigh + bisection sur μ ---
        # tf.linalg.eigh batché sur beaucoup de petites matrices (M<=64) est
        # extrêmement lent/instable sur GPU dans cette version de TF (mesuré :
        # ~9s/appel pour M=8, hang/OOM total pour M=64, sur un batch B×ofdm×fft
        # ~43k matrices) -- LAPACK CPU gère ce cas ~35-1000x plus vite (<0.3s
        # même à M=64). Forcé sur CPU ; les tenseurs (quelques dizaines de MB)
        # traversent le bus au besoin, coût négligeable face au gain.
        with tf.device('/CPU:0'):
            sigma, Q = tf.linalg.eigh(A)                    # eigh(complex128) -> (complex128, complex128) sur ce TF
        sigma    = tf.maximum(tf.math.real(sigma), 0.0)    # valeurs propres d'une matrice Hermitienne : réelles
        C = tf.matmul(Q, B_rhs, adjoint_a=True)             # Q^H @ B_rhs -> [...,M,K]
        p = tf.reduce_sum(tf.abs(C) ** 2, axis=-1)          # p_m = Σ_k |c_k[m]|² -> [...,M]

        def power_at(mu):
            """‖V(μ)‖²_F = Σ_m p_m/(σ_m+μ)² -- décroissante en μ."""
            denom_m = tf.maximum(sigma + mu, 1e-30)
            return tf.reduce_sum(p / (denom_m ** 2), axis=-1, keepdims=True)

        lo = tf.zeros_like(sigma[..., :1])
        hi = tf.fill(tf.shape(sigma[..., :1]), tf.constant(1.0, tf.float64))
        for _ in range(bisection_doublings):    # cadrage : agrandit hi tant que power(hi) > K
            hi = tf.where(power_at(hi) > K_float, hi * 2.0, hi)
        for _ in range(bisection_iters):        # bisection standard
            mid   = 0.5 * (lo + hi)
            too_much = power_at(mid) > K_float
            lo = tf.where(too_much, mid, lo)
            hi = tf.where(too_much, hi, mid)
        mu_star = 0.5 * (lo + hi)                # [...,1]

        inv_diag = 1.0 / tf.maximum(sigma + mu_star, 1e-30)         # [...,M]
        V = tf.matmul(Q, tf.cast(inv_diag, H.dtype)[..., None] * C)  # Q @ diag(inv) @ C
        # Contrainte de puissance Σ_k‖v_k‖²=K satisfaite par construction (μ exact)
        # -- plus besoin de renormalisation post-hoc.

    V = tf.cast(V, input_dtype)
    return tf.expand_dims(V, axis=1)   # [B, 1, ofdm, fft, M, K]


# =============================================================================
# BLOCS D'ATTENTION PARTAGÉS
# =============================================================================

class SeparableAttentionBlock(layers.Layer):
    """
    Attention séparable fréquence → user → FFN.
    Utilisé dans V4 (sur les tokens RB).
    Input/output : [B, num_tokens, num_users, embed_dim]
    """
    def __init__(self, num_tokens, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.num_users  = num_users
        self.embed_dim  = embed_dim
        kd = embed_dim // num_heads

        self.norm_freq = RMSNormalization(epsilon=1e-6, name='norm_freq')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')

        self.freq_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='freq_attn')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='user_attn')

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        self.s_freq = self.add_weight(name='s_freq', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_user = self.add_weight(name='s_user', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_ffn  = self.add_weight(name='s_ffn',  shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

    def call(self, x, training=False):
        B = tf.shape(x)[0]

        # Freq attention — chaque user indépendamment
        xf = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                        [B * self.num_users, self.num_tokens, self.embed_dim])
        xf = xf + self.s_freq * self.freq_attn(
            self.norm_freq(xf), self.norm_freq(xf), training=training)
        x  = tf.transpose(
            tf.reshape(xf, [B, self.num_users, self.num_tokens, self.embed_dim]),
            [0, 2, 1, 3])

        # User attention — chaque token indépendamment
        xu = tf.reshape(x, [B * self.num_tokens, self.num_users, self.embed_dim])
        xu = xu + self.s_user * self.user_attn(
            self.norm_user(xu), self.norm_user(xu), training=training)
        x  = tf.reshape(xu, [B, self.num_tokens, self.num_users, self.embed_dim])

        # FFN
        xp = tf.reshape(x, [B * self.num_tokens * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(self.norm_ffn(xp), training=training)
        return tf.reshape(xp, [B, self.num_tokens, self.num_users, self.embed_dim])


class IntraRBBlock(layers.Layer):
    """
    Attention intra-RB partagée — SC attention + user attention + FFN.
    Poids partagés entre tous les RBs (inductive bias physique : stationnarité intra-RB).
    Input/output : [B*ofdm*num_rb, sc_per_rb, num_users, embed_dim]
    """
    def __init__(self, sc_per_rb, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.sc_per_rb  = sc_per_rb
        self.num_users  = num_users
        self.embed_dim  = embed_dim
        kd = embed_dim // num_heads

        self.norm_sc   = RMSNormalization(epsilon=1e-6, name='norm_sc')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')

        self.sc_attn   = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='sc_attn')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='user_attn')

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        self.s_sc   = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_sc')
        self.s_user = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_user')
        self.s_ffn  = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_ffn')

    def call(self, x, training=False):
        B = tf.shape(x)[0]

        # SC attention — chaque user indépendamment
        xf   = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                          [B * self.num_users, self.sc_per_rb, self.embed_dim])
        xf_n = self.norm_sc(xf)
        xf   = xf + self.s_sc * self.sc_attn(xf_n, xf_n, training=training)
        x    = tf.transpose(
            tf.reshape(xf, [B, self.num_users, self.sc_per_rb, self.embed_dim]),
            [0, 2, 1, 3])

        # User attention — chaque SC indépendamment
        xu   = tf.reshape(x, [B * self.sc_per_rb, self.num_users, self.embed_dim])
        xu_n = self.norm_user(xu)
        xu   = xu + self.s_user * self.user_attn(xu_n, xu_n, training=training)
        x    = tf.reshape(xu, [B, self.sc_per_rb, self.num_users, self.embed_dim])

        # FFN
        xp = tf.reshape(x, [B * self.sc_per_rb * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(self.norm_ffn(xp), training=training)
        return tf.reshape(xp, [B, self.sc_per_rb, self.num_users, self.embed_dim])


class InterRBBlock(layers.Layer):
    """
    Attention inter-RB — capture les dépendances wideband entre les 6 RBs.
    Input/output : [B*ofdm, num_rb, num_users, embed_dim]
    """
    def __init__(self, num_rb, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_rb    = num_rb
        self.num_users = num_users
        self.embed_dim = embed_dim
        kd = embed_dim // num_heads

        self.norm_rb   = RMSNormalization(epsilon=1e-6, name='norm_rb')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')

        self.rb_attn   = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='rb_attn')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='user_attn')

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        self.s_rb   = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_rb')
        self.s_user = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_user')
        self.s_ffn  = self.add_weight(shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_ffn')

    def call(self, x, training=False):
        B = tf.shape(x)[0]

        # RB attention — chaque user indépendamment
        xr   = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                          [B * self.num_users, self.num_rb, self.embed_dim])
        xr_n = self.norm_rb(xr)
        xr   = xr + self.s_rb * self.rb_attn(xr_n, xr_n, training=training)
        x    = tf.transpose(
            tf.reshape(xr, [B, self.num_users, self.num_rb, self.embed_dim]),
            [0, 2, 1, 3])

        # User attention — chaque RB indépendamment
        xu   = tf.reshape(x, [B * self.num_rb, self.num_users, self.embed_dim])
        xu_n = self.norm_user(xu)
        xu   = xu + self.s_user * self.user_attn(xu_n, xu_n, training=training)
        x    = tf.reshape(xu, [B, self.num_rb, self.num_users, self.embed_dim])

        # FFN
        xp = tf.reshape(x, [B * self.num_rb * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(self.norm_ffn(xp), training=training)
        return tf.reshape(xp, [B, self.num_rb, self.num_users, self.embed_dim])


# =============================================================================
# TransformerPrecoderV4  —  v4.0 / v4.1 / v4.2 / v4.3
# =============================================================================

class TransformerPrecoderV4(Model):
    """
    RB-Aggregated Transformer Precoder.

    Versions :
    ─────────────────────────────────────────────────────────────────────
    v4.0 — Baseline
        Upsampling  : tf.repeat (blocs constants)
        SC refine   : Conv1D(k=3), input=[w_up, h_re, h_im]

    v4.1 — Upsampling appris
        Upsampling  : Conv1DTranspose(k=sc_per_token)
        SC refine   : Conv1D(k=rb_size), input=[w_up]

    v4.2 — Features SC enrichies
        Upsampling  : Conv1DTranspose
        SC refine   : Conv1D(k=rb_size), input=[w_up, h_re, h_im, |h|, ∠h]

    v4.3 — SNR-aware + gated SC refine  ← NOUVEAU
        Upsampling  : Conv1DTranspose
        SC refine   : Conv1D(k=rb_size), input=[w_up, h_re, h_im, |h|, ∠h]
        + log(no) dans les features token ET dans l'embedding
        + gate sigmoid sur le delta SC (contrôle où corriger)
        call() accepte l'argument optionnel `no`
    ─────────────────────────────────────────────────────────────────────

    Architecture commune :
        Features token → input_proj → SeparableAttentionBlocks
        → joint_output_proj → Conv1DTranspose / tf.repeat
        → SC Conv1D refine → power norm
    """

    _VERSION_MAP = {
        'v4.0': (False, 'basic'),
        'v4.1': (True,  'minimal'),
        'v4.2': (True,  'rich'),
        'v4.3': (True,  'rich_snr'),
    }

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.0, version='v4.2', **kwargs):
        super().__init__(**kwargs)

        assert version in self._VERSION_MAP, \
            f"version must be one of {list(self._VERSION_MAP)}, got: {version}"

        self.num_tx        = num_tx
        self.num_rx        = num_rx
        self.num_ofdm      = num_ofdm
        self.fft_size      = fft_size
        self.rb_size       = rb_size
        self.num_rb        = fft_size // rb_size
        self.tokens_per_rb = tokens_per_rb
        self.sc_per_token  = rb_size // tokens_per_rb
        self.total_tokens  = self.num_rb * tokens_per_rb
        self.embed_dim     = embed_dim
        self.version       = version

        self.use_learned_upsample, self.refine_mode = self._VERSION_MAP[version]
        self.snr_aware = (version == 'v4.3')

        sc_out = num_tx * num_rx * 2

        # feat_dim : mean(2tx) + slope(2tx) + var(tx) + cov(2rx) + onehot(rx)
        #            + log_no(1) si v4.3
        self.feat_dim = 5 * num_tx + 3 * num_rx + (1 if self.snr_aware else 0)

        # ── Input projection + norm ───────────────────────────────────────
        self.input_proj = layers.Dense(embed_dim, activation='gelu',
                                        name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')

        # ── SNR embedding (v4.3 seulement) ───────────────────────────────
        if self.snr_aware:
            self.snr_proj = layers.Dense(embed_dim, activation='gelu',
                                          name='snr_proj')

        # ── Positional embedding ──────────────────────────────────────────
        self.pos_embed = layers.Embedding(self.total_tokens, embed_dim,
                                           name='pos_embed')

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = [
            SeparableAttentionBlock(
                self.total_tokens, num_rx, embed_dim, num_heads,
                dropout=dropout, name=f'block_{i}')
            for i in range(num_layers)
        ]

        # ── Joint output projection ───────────────────────────────────────
        # Stays in embedding space (D->D), per (token, user) -- cross-user
        # coupling is already provided by the attention blocks' user-
        # attention sub-layer, so this does NOT need to flatten K users
        # into one M*K*2-wide joint dense (that was the O((M*K)^2) blow-up
        # at K=36). See channel_config.py / conversation history for why
        # this was redesigned.
        self.joint_output_proj = layers.Dense(embed_dim,
                                               name='joint_output_proj')

        # ── Upsampling ────────────────────────────────────────────────────
        # Operates per-user (batch dim folded to Bo*K), D channels wide --
        # NOT sc_out=M*K*2 wide as before.
        if self.use_learned_upsample:
            self.upsample = layers.Conv1DTranspose(
                filters=embed_dim,
                kernel_size=self.sc_per_token,
                strides=self.sc_per_token,
                padding='valid', activation=None,
                name='upsample_learned')

        # ── Final per-antenna projection (D -> 2*M), applied at the very
        # end, per (SC, user) -- this is the only M-scaling dense left, and
        # it scales linearly with M (like IntraRB's output_proj), not with
        # M*K. ─────────────────────────────────────────────────────────────
        self.final_proj = layers.Dense(2 * num_tx, name='final_proj')

        # ── SC refinement ─────────────────────────────────────────────────
        # Per-user (Bo*K batch), raw channel features are M-wide (this
        # user's own antennas), not M*K-wide -- refine hidden width scales
        # with embed_dim, not with M*K.
        sc_in_map = {
            'basic'   : embed_dim + num_tx * 2,   # v4.0: w_up(D) + h_re,h_im(2M)
            'minimal' : embed_dim,                 # v4.1: w_up(D) only
            'rich'    : embed_dim + num_tx * 4,    # v4.2: w_up(D) + h_re,h_im,h_abs,h_phase(4M)
            'rich_snr': embed_dim + num_tx * 4,    # v4.3 — même input que v4.2
        }
        sc_in    = sc_in_map[self.refine_mode]
        k_refine = rb_size if self.use_learned_upsample else 3
        hidden   = 2 * embed_dim

        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(hidden, kernel_size=k_refine,
                          padding='same', activation='gelu', name='sc_r1'),
            layers.Conv1D(2 * num_tx, kernel_size=1,
                          padding='same', name='sc_r2'),
        ], name='sc_refine')

        # ── Gate sigmoid sur le delta (v4.3 seulement) ───────────────────
        if self.snr_aware:
            self.sc_gate = layers.Conv1D(2 * num_tx, kernel_size=1,
                                          padding='same', activation='sigmoid',
                                          name='sc_gate')

        # ── Alpha résiduel ────────────────────────────────────────────────
        self.alpha = self.add_weight(
            name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

        print(f"TransformerPrecoderV4 [{version.upper()}] | "
              f"tokens_per_rb={tokens_per_rb} | sc_per_token={self.sc_per_token} | "
              f"embed_dim={embed_dim} | snr_aware={self.snr_aware}")

    # ── Feature extraction ────────────────────────────────────────────────────
    def _extract_features(self, h_sq, B, log_no=None):
        """
        h_sq   : [B, rx, tx, ofdm, fft]
        log_no : scalaire float32 ou None
        →        [B*ofdm, T, rx, feat_dim]

        Features par (token, user) :
          mean(2tx) + slope(2tx) + var(tx) + cov(2rx) + onehot(rx) [+ log_no(1)]
        One-hot concaténé APRÈS la normalisation d'instance
        (évite que la norm écrase l'identité user).
        """
        T = self.total_tokens
        S = self.sc_per_token
        Bo = B * self.num_ofdm

        h_tok = tf.reshape(h_sq,
            [B, self.num_rx, self.num_tx, self.num_ofdm, T, S])

        h_mean  = tf.reduce_mean(h_tok, axis=-1)
        h_slope = h_tok[..., -1] - h_tok[..., 0]
        h_var   = tf.math.log1p(
            tf.reduce_mean(
                tf.abs(h_tok - tf.expand_dims(h_mean, -1))**2, axis=-1))

        h_mean_p = tf.transpose(h_mean, [0, 3, 4, 1, 2])
        cov = tf.matmul(h_mean_p, h_mean_p, adjoint_b=True) \
              / tf.cast(self.num_tx, h_mean_p.dtype)
        cov_feat = tf.concat(
            [tf.math.real(cov), tf.math.imag(cov)], axis=-1)

        def c2r(h):
            p = tf.transpose(h, [0, 3, 4, 1, 2])
            return tf.concat([tf.math.real(p), tf.math.imag(p)], axis=-1)
        def r2p(h):
            return tf.transpose(h, [0, 3, 4, 1, 2])

        # Features canal — [B, ofdm, T, rx, 5*tx + 2*rx]
        feat_canal = tf.concat([c2r(h_mean), c2r(h_slope), r2p(h_var), cov_feat],
                               axis=-1)
        feat_canal = tf.reshape(feat_canal,
            [Bo, T, self.num_rx, 5*self.num_tx + 2*self.num_rx])

        # Instance norm sur les features canal seulement
        f_mean = tf.reduce_mean(feat_canal, axis=-1, keepdims=True)
        f_std  = tf.math.reduce_std(feat_canal, axis=-1, keepdims=True)
        feat_norm = (feat_canal - f_mean) / (f_std + 1e-6)

        # One-hot user APRÈS norm — identité user préservée
        user_onehot = tf.tile(
            tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                       [1, 1, self.num_rx, self.num_rx]),
            [Bo, T, 1, 1])

        parts = [feat_norm, user_onehot]

        # log(no) broadcasté si v4.3
        if log_no is not None:
            log_no_bc = tf.fill([Bo, T, self.num_rx, 1],
                                tf.cast(log_no, tf.float32))
            parts.append(log_no_bc)

        return tf.concat(parts, axis=-1)

    # ── Complexité ────────────────────────────────────────────────────────────
    def complexity(self, num_ofdm=14):
        M  = self.num_tx;  K  = self.num_rx
        T  = self.total_tokens;  D  = self.embed_dim
        fd = self.feat_dim;      N  = self.fft_size
        S  = self.sc_per_token;  RB = self.rb_size
        sc_out = M * K * 2

        sc_in_map = {'basic': M*K*4, 'minimal': M*K*2,
                     'rich': M*K*6, 'rich_snr': M*K*6}
        sc_in = sc_in_map[self.refine_mode]
        k_r   = RB if self.use_learned_upsample else 3

        def df(n, i, o): return 2.0 * n * i * o
        def dp(i, o):    return i * o + o
        def mf(seq, d):  return 8.0*seq*d**2 + 4.0*seq**2*d
        def mp(d):       return 4 * (d*d + d)
        def ff(n, d):    return df(n, d, 4*d) + df(n, 4*d, d)
        def fp(d):       return dp(d, 4*d) + dp(4*d, d)

        f = df(T*K, fd, D);  w = dp(fd, D);  a = T*K*D
        for _ in self.blocks:
            f += K*mf(T, D) + T*mf(K, D) + ff(T*K, D)
            w += 2*mp(D) + fp(D);  a += 3*T*K*D
        f += df(T, K*D, K*M*2);  w += dp(K*D, K*M*2);  a += T*K*M*2

        if self.use_learned_upsample:
            f += 2.0*N*S*sc_out*sc_out;  w += S*sc_out*sc_out + sc_out
            a += N*sc_out

        hidden = sc_out * 2
        f += 2.0*N*k_r*sc_in*hidden + 2.0*N*1*hidden*sc_out
        w += k_r*sc_in*hidden + hidden + 1*hidden*sc_out + sc_out
        a += N*(hidden + sc_out)

        return f * num_ofdm, w, a * num_ofdm

    # ── Forward pass ──────────────────────────────────────────────────────────
    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        """
        h_freq : [B, rx, 1, 1, tx, ofdm, fft]
        no     : scalaire float32 (bruit). Requis pour v4.3, ignoré sinon.
        Sortie : [B, 1, ofdm, fft, tx, rx]
        """
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]
        h_sq   = tf.squeeze(h_freq, axis=[2, 3])
        Bo     = B * self.num_ofdm

        # log(no) pour v4.3
        if self.snr_aware:
            log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10) \
                     if no is not None else tf.constant(0.0)
        else:
            log_no = None

        # ── Features ─────────────────────────────────────────────────────
        feat = self._extract_features(h_sq, B, log_no)
        x    = self.input_proj(feat)
        x    = self.input_norm(x)

        # SNR embedding injecté dans les tokens (v4.3)
        if self.snr_aware:
            snr_emb = tf.reshape(
                self.snr_proj(tf.reshape(log_no, [1, 1])),
                [1, 1, 1, self.embed_dim])
            x = x + snr_emb

        # Positional embedding
        x = x + tf.reshape(
            self.pos_embed(tf.range(self.total_tokens)),
            [1, self.total_tokens, 1, self.embed_dim])

        # ── Transformer ───────────────────────────────────────────────────
        for blk in self.blocks:
            x = blk(x, training=training)
        # x : [Bo, T, K, D]

        # ── Joint output, stays D-wide (embedding space) ──────────────────
        x = self.joint_output_proj(x)   # [Bo, T, K, D]

        # ── Fold user axis into batch: per-user token sequences ──────────
        # [Bo, T, K, D] -> [Bo, K, T, D] -> [Bo*K, T, D]
        x_user = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                             [Bo * self.num_rx, self.total_tokens, self.embed_dim])

        # ── Upsampling (per user, D channels) → [Bo*K, fft, D] ───────────
        if self.use_learned_upsample:
            w_up = self.upsample(x_user)
        else:
            w_up = tf.repeat(x_user, self.sc_per_token, axis=1)

        # ── Final per-antenna projection, D -> 2M (per user) ─────────────
        w_up_final = self.final_proj(w_up)   # [Bo*K, fft, 2M]

        # ── SC refinement (per user, raw channel features are M-wide) ────
        # h_sq : [B, rx, tx, ofdm, fft] -> per-user [Bo*K, fft, tx]
        h_bo   = tf.reshape(tf.transpose(h_sq, [0, 3, 4, 1, 2]),
                             [Bo, self.fft_size, self.num_rx, self.num_tx])
        h_user = tf.reshape(tf.transpose(h_bo, [0, 2, 1, 3]),
                             [Bo * self.num_rx, self.fft_size, self.num_tx])

        if self.refine_mode == 'minimal':
            refine_input = w_up

        elif self.refine_mode == 'basic':
            h_ri = tf.concat([tf.math.real(h_user), tf.math.imag(h_user)], axis=-1)
            refine_input = tf.concat([w_up, h_ri], axis=-1)

        else:  # 'rich' ou 'rich_snr'
            h_re    = tf.math.real(h_user)
            h_im    = tf.math.imag(h_user)
            h_abs   = tf.abs(h_user)
            h_phase = tf.math.angle(h_user)
            refine_input = tf.concat([w_up, h_re, h_im, h_abs, h_phase], axis=-1)

        delta = self.sc_refine(refine_input, training=training)   # [Bo*K, fft, 2M]

        # Gate sigmoid (v4.3) — contrôle où appliquer la correction
        if self.snr_aware:
            gate         = self.sc_gate(w_up_final)
            w_final_flat = w_up_final + self.alpha * gate * delta
        else:
            w_final_flat = w_up_final + self.alpha * delta
        # w_final_flat : [Bo*K, fft, 2M]

        # ── Unfold user axis, reshape + power norm ────────────────────────
        # [Bo*K, fft, 2M] -> [Bo, K, fft, M, 2] -> [Bo, fft, K, M, 2]
        #                  -> [B, ofdm, fft, M, K, 2]  (M before K, matching
        #                     RZFPrecodedChannel's expected output layout)
        w_final = tf.reshape(w_final_flat,
            [Bo, self.num_rx, self.fft_size, self.num_tx, 2])
        w_final = tf.transpose(w_final, [0, 2, 3, 1, 4])
        w_final = tf.reshape(w_final,
            [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
        w_re = w_final[..., 0]
        w_im = w_final[..., 1]
        s    = tf.sqrt(1.0 / (
            tf.reduce_sum(w_re**2 + w_im**2, axis=3, keepdims=True) + 1e-12))
        w_re, w_im = w_re * s, w_im * s

        if return_real_imag:
            return tf.expand_dims(w_re, 1), tf.expand_dims(w_im, 1)
        return tf.expand_dims(tf.complex(w_re, w_im), 1)


# =============================================================================
# TransformerPrecoderV5  —  Hiérarchique Swin-inspired, SNR-aware
# =============================================================================

class TransformerPrecoderV5(Model):
    """
    Précoder hiérarchique MU-MIMO.

    Architecture :
        Features SC enrichies (5M + 3M + 2K² = 96 dims, normalisées)
        + one-hot user HORS normalisation (préserve l'identité user)
        + log(no) optionnel (SNR-aware)
        → IntraRBBlocks partagés (stationnarité intra-RB)
        → RB pooling mean+max → InterRBBlocks
        → Gate additive wideband → SC  (s_cross init=0.3)
        → Joint output per-SC

    Corrections vs V5.2 :
        1. One-hot hors instance norm  → user attention non-uniforme
        2. s_cross init 0.3            → gate wideband actif dès le début
        3. SNR-aware via log(no)        → adaptation amplitude/waterfilling
        4. feat_dim = 8M + 2K² (sans K) → one-hot géré séparément
    """

    RB_SIZE = 12   # Standard 5G NR — fixe

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 embed_dim=128, num_heads=4,
                 num_intra_layers=2, num_inter_layers=2,
                 dropout=0.0, use_swin_shift=False,
                 snr_aware=True, **kwargs):
        super().__init__(**kwargs)

        assert fft_size % self.RB_SIZE == 0, \
            f"fft_size {fft_size} doit être divisible par {self.RB_SIZE}"

        self.num_tx           = num_tx
        self.num_rx           = num_rx
        self.num_ofdm         = num_ofdm
        self.fft_size         = fft_size
        self.rb_size          = self.RB_SIZE
        self.num_rb           = fft_size // self.RB_SIZE
        self.embed_dim        = embed_dim
        self.num_intra_layers = num_intra_layers
        self.num_inter_layers = num_inter_layers
        self.use_swin_shift   = use_swin_shift
        self.snr_aware        = snr_aware

        # feat_dim : features canal normalisées (sans one-hot, sans log_no)
        #   per-SC  : h_re(M) + h_im(M) + |h|(M) + sin∠h(M) + cos∠h(M) = 5M
        #   per-RB  : slope_re(M) + slope_im(M) + var(M) broadcastés     = 3M
        #   cov     : re(K²) + im(K²) broadcastés                          = 2K²
        # total normalisé = 8M + 2K²
        # + one-hot(K) ajouté après norm
        # + log_no(1) si snr_aware
        self.feat_dim_canal = 8 * num_tx + 2 * num_rx * num_rx
        self.feat_dim       = (self.feat_dim_canal + num_rx
                               + (1 if snr_aware else 0))

        # ── Blocs ─────────────────────────────────────────────────────────
        self.input_proj   = layers.Dense(embed_dim, activation='gelu',
                                          name='input_proj')
        self.input_norm   = RMSNormalization(epsilon=1e-6, name='input_norm')
        self.pos_embed_sc = layers.Embedding(self.rb_size, embed_dim,
                                              name='pos_embed_sc')
        self.pos_embed_rb = layers.Embedding(self.num_rb,  embed_dim,
                                              name='pos_embed_rb')

        self.rb_pos_bias  = self.add_weight(
            shape=(self.num_rb, self.rb_size, 1, embed_dim),
            initializer='zeros', trainable=True, name='rb_pos_bias')

        self.intra_blocks = [
            IntraRBBlock(self.rb_size, num_rx, embed_dim, num_heads,
                         dropout=dropout, name=f'intra_{i}')
            for i in range(num_intra_layers)
        ]
        self.inter_blocks = [
            InterRBBlock(self.num_rb, num_rx, embed_dim, num_heads,
                         dropout=dropout, name=f'inter_{i}')
            for i in range(num_inter_layers)
        ]

        # RB pooling : mean+max → Dense(2D→D)
        self.rb_pool_proj = layers.Dense(embed_dim, activation='gelu',
                                          name='rb_pool_proj')

        # Gate wideband — s_cross init 0.3 (était 0.1, mort dans V5.2)
        self.wideband_gate = layers.Dense(embed_dim, activation='gelu',
                                           name='wideband_gate')
        self.s_cross = self.add_weight(
            shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.3),
            trainable=True, name='s_cross')

        # SNR embedding (optionnel)
        if snr_aware:
            self.snr_proj = layers.Dense(embed_dim, activation='gelu',
                                          name='snr_proj')

        # Sortie joint per-SC
        self.joint_output = layers.Dense(num_rx * num_tx * 2,
                                          name='joint_output')

        print(f"TransformerPrecoderV5 | rb_size={self.rb_size} | "
              f"num_rb={self.num_rb} | embed_dim={embed_dim} | "
              f"intra={num_intra_layers}(shared) | inter={num_inter_layers} | "
              f"feat_dim={self.feat_dim} | swin_shift={use_swin_shift} | "
              f"snr_aware={snr_aware} | s_cross_init=0.3")

    # ── Feature extraction ────────────────────────────────────────────────────
    def _extract_features(self, h_sq, B, log_no=None):
        """
        h_sq   : [B, rx, tx, ofdm, fft]
        log_no : scalaire float32 ou None
        →        [Bo, fft, rx, feat_dim]

        Ordre : [feat_canal_normalisé | one-hot | log_no?]
        One-hot et log_no HORS instance norm.
        """
        Bo = B * self.num_ofdm

        h_perm = tf.reshape(
            tf.transpose(h_sq, [0, 3, 4, 1, 2]),
            [Bo, self.fft_size, self.num_rx, self.num_tx])

        # ── Features per-SC ───────────────────────────────────────────────
        h_re  = tf.math.real(h_perm)
        h_im  = tf.math.imag(h_perm)
        h_abs = tf.sqrt(h_re**2 + h_im**2 + 1e-12)
        h_cos = h_re / h_abs
        h_sin = h_im / h_abs
        feat_sc = tf.concat([h_re, h_im, h_abs, h_sin, h_cos], axis=-1)
        # [Bo, N, K, 5M]

        # ── Features per-RB broadcastées ─────────────────────────────────
        h_rb = tf.reshape(h_perm,
            [Bo, self.num_rb, self.rb_size, self.num_rx, self.num_tx])
        h_rb_mean  = tf.reduce_mean(h_rb, axis=2)
        h_rb_slope = h_rb[:, :, -1, :, :] - h_rb[:, :, 0, :, :]

        diff_re  = tf.math.real(h_rb - tf.expand_dims(h_rb_mean, 2))
        diff_im  = tf.math.imag(h_rb - tf.expand_dims(h_rb_mean, 2))
        h_rb_var = tf.math.log1p(
            tf.reduce_mean(diff_re**2 + diff_im**2, axis=2))

        slope_re_bc = tf.repeat(tf.math.real(h_rb_slope), self.rb_size, axis=1)
        slope_im_bc = tf.repeat(tf.math.imag(h_rb_slope), self.rb_size, axis=1)
        var_bc      = tf.repeat(h_rb_var, self.rb_size, axis=1)
        feat_rb     = tf.concat([slope_re_bc, slope_im_bc, var_bc], axis=-1)
        # [Bo, N, K, 3M]

        # ── Covariance spatiale broadcastée ───────────────────────────────
        cov = tf.matmul(h_rb_mean, h_rb_mean, adjoint_b=True) \
              / tf.cast(self.num_tx, h_rb_mean.dtype)
        cov_feat = tf.concat(
            [tf.math.real(cov), tf.math.imag(cov)], axis=-1)
        cov_bc = tf.repeat(cov_feat, self.rb_size, axis=1)
        # [Bo, N, K, 2K²]

        # ── Instance norm sur features canal uniquement ───────────────────
        feat_canal = tf.concat([feat_sc, feat_rb, cov_bc], axis=-1)
        f_mean = tf.reduce_mean(feat_canal, axis=-1, keepdims=True)
        f_std  = tf.math.reduce_std(feat_canal, axis=-1, keepdims=True)
        feat_norm = (feat_canal - f_mean) / (f_std + 1e-6)
        # [Bo, N, K, 8M + 2K²]

        # ── One-hot user APRÈS norm — identité user préservée ─────────────
        user_onehot = tf.tile(
            tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                       [1, 1, self.num_rx, self.num_rx]),
            [Bo, self.fft_size, 1, 1])
        # [Bo, N, K, K]

        parts = [feat_norm, user_onehot]

        if log_no is not None:
            log_no_bc = tf.fill(
                [Bo, self.fft_size, self.num_rx, 1],
                tf.cast(log_no, tf.float32))
            parts.append(log_no_bc)

        return tf.concat(parts, axis=-1)
        # [Bo, N, K, feat_dim]

    # ── Complexité ────────────────────────────────────────────────────────────
    def complexity(self, num_ofdm=14):
        M  = self.num_tx;  K  = self.num_rx
        D  = self.embed_dim;  fd = self.feat_dim
        N  = self.fft_size;  NR = self.num_rb;  RS = self.rb_size

        def df(n,i,o): return 2.0*n*i*o
        def dp(i,o):   return i*o+o
        def mf(s,d):   return 8.0*s*d**2+4.0*s**2*d
        def mp(d):     return 4*(d*d+d)
        def ff(n,d):   return df(n,d,4*d)+df(n,4*d,d)
        def fp(d):     return dp(d,4*d)+dp(4*d,d)

        f = df(N*K, fd, D);  w = dp(fd, D);  a = N*K*D
        w += NR * RS * D   # rb_pos_bias

        for _ in self.intra_blocks:
            f += (K*mf(RS,D) + RS*mf(K,D) + ff(RS*K,D)) * NR
            w += 2*mp(D) + fp(D);  a += 3*RS*K*D*NR

        f += df(NR*K, 2*D, D);  w += dp(2*D, D);  a += NR*K*D

        for _ in self.inter_blocks:
            f += K*mf(NR,D) + NR*mf(K,D) + ff(NR*K,D)
            w += 2*mp(D) + fp(D);  a += 3*NR*K*D

        f += df(NR*K, D, D);  w += dp(D, D)
        a += NR*K*D + N*K*D

        f += df(N, K*D, K*M*2);  w += dp(K*D, K*M*2);  a += N*K*M*2

        return f*num_ofdm, w, a*num_ofdm

    # ── Forward pass ──────────────────────────────────────────────────────────
    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        """
        h_freq : [B, rx, 1, 1, tx, ofdm, fft]
        no     : scalaire float32 (bruit). Si None et snr_aware → log_no=0.
        Sortie : [B, 1, ofdm, fft, tx, rx]
        """
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]
        h_sq   = tf.squeeze(h_freq, axis=[2, 3])
        Bo     = B * self.num_ofdm

        # log(no)
        if self.snr_aware:
            log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10) \
                     if no is not None else tf.constant(0.0)
        else:
            log_no = None

        # ── Features ─────────────────────────────────────────────────────
        feat = self._extract_features(h_sq, B, log_no)
        x    = self.input_norm(self.input_proj(feat))
        # [Bo, N, K, D]

        # SNR embedding injecté uniformément
        if self.snr_aware:
            snr_emb = tf.reshape(
                self.snr_proj(tf.reshape(log_no, [1, 1])),
                [1, 1, 1, self.embed_dim])
            x = x + snr_emb

        # ── Positional embeddings ─────────────────────────────────────────
        sc_pos = tf.tile(tf.range(self.rb_size), [self.num_rb])
        x = x + tf.reshape(self.pos_embed_sc(sc_pos),
                            [1, self.fft_size, 1, self.embed_dim])

        bias = tf.broadcast_to(
            tf.reshape(self.rb_pos_bias,
                       [1, self.num_rb, self.rb_size, 1, self.embed_dim]),
            [Bo, self.num_rb, self.rb_size, self.num_rx, self.embed_dim])
        x = x + tf.reshape(bias, [Bo, self.fft_size, self.num_rx, self.embed_dim])

        # ── IntraRB blocks ────────────────────────────────────────────────
        shift = self.rb_size // 2
        for i, blk in enumerate(self.intra_blocks):
            do_shift = self.use_swin_shift and (i % 2 == 1)
            if do_shift:
                x = tf.roll(x, shift=shift, axis=1)
            x_rb = tf.reshape(x, [Bo * self.num_rb, self.rb_size,
                                   self.num_rx, self.embed_dim])
            x_rb = blk(x_rb, training=training)
            x = tf.reshape(x_rb, [Bo, self.fft_size, self.num_rx, self.embed_dim])
            if do_shift:
                x = tf.roll(x, shift=-shift, axis=1)

        # ── RB pooling mean+max → inter-RB ───────────────────────────────
        x_rb2   = tf.reshape(x, [Bo * self.num_rb, self.rb_size,
                                  self.num_rx, self.embed_dim])
        rb_mean = tf.reduce_mean(x_rb2, axis=1)
        rb_max  = tf.reduce_max(x_rb2,  axis=1)
        rb_tok  = self.rb_pool_proj(tf.concat([rb_mean, rb_max], axis=-1))

        rb_tokens = tf.reshape(rb_tok,
                               [Bo, self.num_rb, self.num_rx, self.embed_dim])
        rb_tokens = rb_tokens + tf.reshape(
            self.pos_embed_rb(tf.range(self.num_rb)),
            [1, self.num_rb, 1, self.embed_dim])

        # ── InterRB blocks ────────────────────────────────────────────────
        for blk in self.inter_blocks:
            rb_tokens = blk(rb_tokens, training=training)

        # ── Gate wideband → SC ────────────────────────────────────────────
        gate    = self.wideband_gate(rb_tokens)
        gate_bc = tf.repeat(gate, self.rb_size, axis=1)
        x = x + self.s_cross * gate_bc

        # ── Joint output per-SC ───────────────────────────────────────────
        w_out = self.joint_output(
            tf.reshape(x, [Bo, self.fft_size, self.num_rx * self.embed_dim]))
        w_out = tf.transpose(
            tf.reshape(w_out, [B, self.num_ofdm, self.fft_size,
                               self.num_rx, self.num_tx, 2]),
            [0, 1, 2, 4, 3, 5])

        # ── Power norm ────────────────────────────────────────────────────
        w_re = w_out[..., 0]
        w_im = w_out[..., 1]
        s    = tf.sqrt(1.0 / (
            tf.reduce_sum(w_re**2 + w_im**2, axis=3, keepdims=True) + 1e-12))
        w_re, w_im = w_re * s, w_im * s

        if return_real_imag:
            return tf.expand_dims(w_re, 1), tf.expand_dims(w_im, 1)
        return tf.expand_dims(tf.complex(w_re, w_im), 1)