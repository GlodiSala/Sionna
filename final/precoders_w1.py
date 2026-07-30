# =============================================================================
# precoders_w.py
# =============================================================================
import tensorflow as tf
import numpy as np
from sionna.phy.mimo import rzf_precoding_matrix
from tensorflow.keras import layers, Model
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers
from tensorflow.keras.layers import RMSNormalization


# =============================================================================
# HELPER — réplique exacte de PrecodedChannel.get_desired_channels()
# (méthode d'instance non importable, donc réimplémentée ici à l'identique)
# Ref: sionna/phy/ofdm/precoding.py :: PrecodedChannel.get_desired_channels
# =============================================================================

def _get_desired_channels(h_hat, stream_management):
    """
    Réplique de PrecodedChannel.get_desired_channels() de Sionna.
    Impossible à importer directement (méthode d'instance liée à resource_grid).

    Input:  h_hat  [B, num_rx, num_rx_ant, num_tx, num_tx_ant, ofdm, fft]
    Output: h_pc   [B, num_tx, ofdm, fft, num_streams_per_tx, num_tx_ant]
    """
    # [num_tx, num_rx, num_rx_ant, num_tx_ant, ofdm, fft, B]
    h_pc = tf.transpose(h_hat, [3, 1, 2, 4, 5, 6, 0])

    # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, ofdm, fft, B]
    h_pc = tf.gather(h_pc, stream_management.precoding_ind,
                     axis=1, batch_dims=1)

    # flatten dims 1,2 → [num_tx, num_streams_per_tx, num_tx_ant, ofdm, fft, B]
    # (Sionna utilise flatten_dims(h_pc, 2, axis=1) — équivalent au reshape ci-dessous)
    s = tf.shape(h_pc)
    h_pc = tf.reshape(h_pc, [s[0], s[1] * s[2], s[3], s[4], s[5], s[6]])

    # [B, num_tx, ofdm, fft, num_streams_per_tx, num_tx_ant]
    h_pc = tf.transpose(h_pc, [5, 0, 3, 4, 1, 2])

    return h_pc


# =============================================================================
# RZF PRECODER
# =============================================================================

def rzf_precoder(h_freq, stream_management, no=None, alpha=None):
    h_pc = _get_desired_channels(h_freq, stream_management)

    if alpha is None:
        # ✅ reshape au lieu de cast direct — préserve le dynamisme
        alpha_val = tf.reshape(tf.cast(no, tf.float32), [])
    else:
        alpha_val = tf.cast(alpha, tf.float32)

    return rzf_precoding_matrix(h_pc, alpha=alpha_val)


# =============================================================================
# WMMSE PRECODER
# =============================================================================
def wmmse_precoder(h_freq, no, stream_management, num_iterations=10):

    h_pc = _get_desired_channels(h_freq, stream_management)
    H    = tf.squeeze(h_pc, axis=1)

    s        = tf.shape(H)
    B_dim    = s[0]; ofdm_dim = s[1]; fft_dim = s[2]
    K        = s[3]; M        = s[4]
    K_float  = tf.cast(K, H.dtype.real_dtype)

    # ✅ no comme tenseur scalaire réel — jamais de tf.fill
    no_scalar = tf.cast(tf.reshape(no, []), dtype=tf.float32)
    no_complex = tf.cast(no_scalar, H.dtype)

    # ✅ no_val pour denom : [1,1,1,1] broadcastable sur [B,ofdm,fft,K]
    no_val = tf.reshape(no_scalar, [1, 1, 1, 1])

    # ✅ no_mat pour régularisation : [1,1,1,1,1] broadcastable sur [B,ofdm,fft,M,M]
    no_mat = tf.reshape(no_complex, [1, 1, 1, 1, 1])

    eye_M = tf.eye(M, dtype=H.dtype)  # [M,M] — broadcast sur les dims batch

    # ✅ Init RZF avec alpha = no_scalar (dynamique, pas tracé)
    V = rzf_precoding_matrix(h_pc, alpha=no_scalar)
    V = tf.squeeze(V, axis=1)

    for _ in range(num_iterations):
        HV      = tf.matmul(H, V)
        signal  = tf.linalg.diag_part(HV)
        tot_pwr = tf.reduce_sum(tf.abs(HV)**2, axis=-1)

        # no_val [1,1,1,1] broadcast sur tot_pwr [B,ofdm,fft,K] ✅
        denom = tf.cast(tot_pwr + no_val, H.dtype)
        U     = tf.math.conj(signal) / denom

        mse = 1.0 - tf.math.real(U * signal)
        W   = 1.0 / tf.maximum(mse, 1e-7)

        weights     = tf.cast(W * tf.abs(U)**2, H.dtype)
        H_H         = tf.linalg.adjoint(H)
        weighted_HH = H_H * weights[:, :, :, None, :]
        A           = tf.matmul(weighted_HH, H)

        # no_mat [1,1,1,1,1] broadcast sur A [B,ofdm,fft,M,M] ✅
        A_reg = A + no_mat * eye_M

        target = tf.cast(W, H.dtype) * tf.math.conj(U)
        B_rhs  = H_H * target[:, :, :, None, :]
        V      = tf.linalg.solve(A_reg, B_rhs)

        pwr = tf.reduce_sum(tf.abs(V)**2, axis=[-2,-1], keepdims=True)
        V   = V * tf.cast(tf.sqrt(K_float / (pwr + 1e-12)), V.dtype)

    return tf.expand_dims(V, axis=1)   # [B, 1, ofdm, fft, M, K]

# =============================================================================

class SeparableAttentionBlock(layers.Layer):
    """
    Separable attention: frequency tokens first, then users, then FFN.
    Input/output: [B, num_tokens, num_users, embed_dim]
    """
    def __init__(self, num_tokens, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.num_users  = num_users
        self.embed_dim  = embed_dim

        self.norm_freq = RMSNormalization(epsilon=1e-6, name='norm_freq')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')

        kd = embed_dim // num_heads
        self.freq_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='freq_attn')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd, dropout=dropout, name='user_attn')

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        self.s_freq = self.add_weight(
            name='s_freq', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_user = self.add_weight(
            name='s_user', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_ffn = self.add_weight(
            name='s_ffn', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

    def call(self, x, training=False):
        B = tf.shape(x)[0]

        # Frequency attention (each user independently)
        xf = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                        [B * self.num_users, self.num_tokens, self.embed_dim])
        xf = xf + self.s_freq * self.freq_attn(
            self.norm_freq(xf), self.norm_freq(xf), training=training)
        x = tf.transpose(
            tf.reshape(xf, [B, self.num_users, self.num_tokens, self.embed_dim]),
            [0, 2, 1, 3])

        # User attention (each token independently)
        xu = tf.reshape(x, [B * self.num_tokens, self.num_users, self.embed_dim])
        xu = xu + self.s_user * self.user_attn(
            self.norm_user(xu), self.norm_user(xu), training=training)
        x = tf.reshape(xu, [B, self.num_tokens, self.num_users, self.embed_dim])

        # FFN
        xp = tf.reshape(x, [B * self.num_tokens * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(self.norm_ffn(xp), training=training)
        return tf.reshape(xp, [B, self.num_tokens, self.num_users, self.embed_dim])


class JointAttentionBlock(layers.Layer):
    """
    Joint attention over all (token × user) positions.
    Input/output: [B, num_tokens, num_users, embed_dim]
    """
    def __init__(self, num_tokens, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.TU         = num_tokens * num_users
        self.num_tokens = num_tokens
        self.embed_dim  = embed_dim

        self.norm_attn = RMSNormalization(epsilon=1e-6, name='norm_attn')
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout, name='joint_attn')

        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        self.s_attn = self.add_weight(
            name='s_attn', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_ffn = self.add_weight(
            name='s_ffn', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

    def call(self, x, training=False):
        B         = tf.shape(x)[0]
        num_users = x.shape[2]

        xf = tf.reshape(x, [B, self.TU, self.embed_dim])
        xf = xf + self.s_attn * self.attn(
            self.norm_attn(xf), self.norm_attn(xf), training=training)
        xf = xf + self.s_ffn * self.ffn(self.norm_ffn(xf), training=training)
        return tf.reshape(xf, [B, self.num_tokens, num_users, self.embed_dim])


# =============================================================================
# TransformerPrecoderV3
# =============================================================================

class TransformerPrecoderV2(Model):
    """
    Coarse-to-Fine Hybrid Transformer for Massive MIMO
    - Configurable `tokens_per_rb` for hardware/performance ablation studies.
    """
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, tokens_per_rb=1, embed_dim=128, num_heads=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_ofdm = num_ofdm
        self.fft_size = fft_size
        self.rb_size = rb_size
        self.num_rb = fft_size // rb_size
        
        # --- NEW: Token Resolution Scaling ---
        self.tokens_per_rb = tokens_per_rb
        self.sc_per_token = rb_size // tokens_per_rb
        self.total_tokens = self.num_rb * self.tokens_per_rb
        
        self.embed_dim = embed_dim
        
        # ---------------------------------------------------------------------
        # 1. Feature Extraction Projection
        # ---------------------------------------------------------------------
        self.rb_feature_projection = layers.Dense(
            embed_dim, 
            activation='gelu',
            name='token_feature_projection'
        )
        self.input_norm = RMSNormalization(epsilon=1e-6)
        
        # ---------------------------------------------------------------------
        # 2. Transformer Core (Operates on total_tokens level)
        # ---------------------------------------------------------------------
        self.pos_embedding = layers.Embedding(
            input_dim=self.total_tokens,
            output_dim=embed_dim,
            name='pos_embed'
        )
        
        self.blocks = [
            SeparableAttentionBlock(self.total_tokens, num_rx, embed_dim, num_heads)
            for _ in range(num_layers)
        ]
        
        self.coarse_projection = layers.Dense(num_tx * 2, name='coarse_proj')
        
        # ---------------------------------------------------------------------
        # 3. SC-Level High-Resolution Refinement
        # ---------------------------------------------------------------------
        self.sc_refinement = tf.keras.Sequential([
            layers.Dense(64, activation='gelu'),
            layers.Dense(2) # Outputs the Delta correction (Real + Imag)
        ], name='sc_refinement')
        
        self.alpha = self.add_weight(
            name='alpha_residual', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )

    def call(self, h_freq, training=False, return_real_imag=False, return_coarse=False):
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        
        h_sq = tf.squeeze(h_freq, axis=[2, 3]) 
        
        # =====================================================================
        # STAGE 1: COARSE PROCESSING (At Token Resolution)
        # =====================================================================
        # Group into Tokens: [B, rx, tx, ofdm, total_tokens, sc_per_token]
        h_token = tf.reshape(h_sq, [B, self.num_rx, self.num_tx, self.num_ofdm, self.total_tokens, self.sc_per_token])
        
        h_mean = tf.reduce_mean(h_token, axis=-1)   # [B, rx, tx, ofdm, total_tokens]
        h_first = h_token[..., 0]
        h_last = h_token[..., -1]
        
        # Covariance Matrix calculation
        h_mean_perm = tf.transpose(h_mean, [0, 3, 4, 1, 2]) # [B, ofdm, total_tokens, rx, tx]
        cov_matrix = tf.matmul(h_mean_perm, h_mean_perm, adjoint_b=True)
        cov_feat = tf.concat([tf.math.real(cov_matrix), tf.math.imag(cov_matrix)], axis=-1)
        
        feats_list = [h_mean, h_first, h_last]
        feats_stack = []
        for f in feats_list:
            feats_stack.append(tf.math.real(f))
            feats_stack.append(tf.math.imag(f))
        
        h_feats = tf.stack(feats_stack, axis=-1)
        h_feats = tf.transpose(h_feats, [0, 3, 4, 1, 2, 5])
        h_feats_flat = tf.reshape(h_feats, [B, self.num_ofdm, self.total_tokens, self.num_rx, self.num_tx * 6])
        
        feat_input = tf.concat([h_feats_flat, cov_feat], axis=-1)
        feat_input = tf.reshape(feat_input, [B * self.num_ofdm, self.total_tokens, self.num_rx, -1])
        
        # ---------------------------------------------------------------------
        # Transformer Core
        # ---------------------------------------------------------------------
        x = self.rb_feature_projection(feat_input)
        x = self.input_norm(x)
        
        positions = tf.range(self.total_tokens)
        pos_embed = self.pos_embedding(positions)
        pos_embed = tf.reshape(pos_embed, [1, self.total_tokens, 1, self.embed_dim])
        x = x + pos_embed
        
        for block in self.blocks:
            x = block(x, training=training)
            
        w_coarse = self.coarse_projection(x)
        w_coarse = tf.reshape(w_coarse, [B, self.num_ofdm, self.total_tokens, self.num_rx, self.num_tx, 2])
        w_coarse = tf.transpose(w_coarse, [0, 1, 2, 4, 3, 5])

        # =====================================================================
        # STAGE 2: SC-LEVEL HIGH-RESOLUTION REFINEMENT
        # =====================================================================
        # Upsample by repeating strictly sc_per_token times
        w_coarse_up = tf.repeat(w_coarse, self.sc_per_token, axis=2) 
        
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2]) 
        h_full_flat = tf.stack([tf.math.real(h_full), tf.math.imag(h_full)], axis=-1) 
        h_full_flat = tf.transpose(h_full_flat, [0, 1, 2, 4, 3, 5])
        
        combined_sc = tf.concat([w_coarse_up, h_full_flat], axis=-1)
        combined_sc_flat = tf.reshape(combined_sc, [B * self.num_ofdm * self.fft_size, self.num_tx, self.num_rx, 4])
        
        delta_flat = self.sc_refinement(combined_sc_flat)
        delta = tf.reshape(delta_flat, [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])
        
        w_final = w_coarse_up + (self.alpha * delta)
        
        w_real = w_final[..., 0]
        w_imag = w_final[..., 1]

        # =====================================================================
        # STAGE 3: STRICT POWER NORMALIZATION
        # =====================================================================
        mag_sq = w_real**2 + w_imag**2
        power_per_stream = tf.reduce_sum(mag_sq, axis=3, keepdims=True) 
        scale = tf.sqrt(1.0 / (power_per_stream + 1e-12))
        
        w_real_norm = w_real * scale
        w_imag_norm = w_imag * scale

        if return_real_imag:
            return (tf.expand_dims(w_real_norm, 1), tf.expand_dims(w_imag_norm, 1))
        
        w_complex = tf.complex(w_real_norm, w_imag_norm)
        w_complex = tf.expand_dims(w_complex, 1)

        if return_coarse:
            w_coarse_complex = tf.complex(w_coarse_up[..., 0], w_coarse_up[..., 1])
            w_coarse_complex = tf.expand_dims(w_coarse_complex, 1)
            return w_complex, w_coarse_complex

        return w_complex



# =============================================================================
# TransformerPrecoderV4  — Improved token features
# =============================================================================

class TransformerPrecoderV3(Model):
    """
    Coarse-to-Fine Hybrid Transformer with improved token DNA.

    Token features per (token, user):
      - h_mean  : complex mean over sc_per_token subcarriers  → [rx, tx, 2]
      - h_slope : h_last - h_first (frequency drift)          → [rx, tx, 2]
      - h_var   : real variance within token                  → [rx, tx, 1]
      - cov_feat: real+imag of H_mean @ H_mean^H             → [rx, rx, 2]

    Total feature dim per (token, user):
        mean:  2*num_tx
        slope: 2*num_tx
        var:   1*num_tx      ← real only, variance is non-negative real
        cov:   2*num_rx (flattened row of cov per user)
      = 5*num_tx + 2*num_rx
    """
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.num_tx    = num_tx
        self.num_rx    = num_rx
        self.num_ofdm  = num_ofdm
        self.fft_size  = fft_size
        self.rb_size   = rb_size
        self.num_rb    = fft_size // rb_size
        self.tokens_per_rb = tokens_per_rb
        self.sc_per_token  = rb_size // tokens_per_rb
        self.total_tokens  = self.num_rb * self.tokens_per_rb
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Feature dimension (per token, per user):
        #   mean  : 2 * num_tx  (real + imag)
        #   slope : 2 * num_tx  (real + imag)
        #   var   : 1 * num_tx  (real-valued variance, one per tx antenna)
        #   cov   : 2 * num_rx  (one row of cov matrix per user, re+im)
        # ------------------------------------------------------------------
        self.feat_dim = 6 * num_tx + 2 * num_rx

        # ------------------------------------------------------------------
        # Stage 1 — Coarse Transformer
        # ------------------------------------------------------------------
        self.input_projection = layers.Dense(
            embed_dim, activation='gelu', name='input_projection')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')

        self.pos_embedding = layers.Embedding(
            input_dim=self.total_tokens,
            output_dim=embed_dim,
            name='pos_embed')

        self.blocks = [
            SeparableAttentionBlock(
                self.total_tokens, num_rx, embed_dim, num_heads,
                dropout=dropout, name=f'block_{i}')
            for i in range(num_layers)
        ]

        self.coarse_projection = layers.Dense(num_tx * 2, name='coarse_proj')

        # ------------------------------------------------------------------
        # Stage 2 — SC-level refinement MLP
        # Input per (SC, tx, rx): [w_re, w_im, h_re, h_im] = 4 values
        # MLP: 4 → 64 (GELU) → 2 (delta re + im)
        # ------------------------------------------------------------------
        self.sc_refinement = tf.keras.Sequential([
            layers.Dense(64, activation='gelu', name='refine_dense1'),
            layers.Dense(2,                     name='refine_dense2'),
        ], name='sc_refinement')

        # Learnable residual scale — initialised small so training starts
        # close to the coarse solution
        self.alpha = self.add_weight(
            name='alpha_residual', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True)

    # ------------------------------------------------------------------
    # Feature extraction — the key improvement over V3
    # ------------------------------------------------------------------
    def _extract_token_features(self, h_sq, B):
        T = self.total_tokens
        S = self.sc_per_token

        h_tok = tf.reshape(
            h_sq,
            [B, self.num_rx, self.num_tx, self.num_ofdm, T, S])

        # Mean — no normalization needed, already in [-1,1] range
        h_mean = tf.reduce_mean(h_tok, axis=-1)

        # Slope — same scale as mean
        h_slope = h_tok[..., -1] - h_tok[..., 0]

        # Variance — log1p to compress dynamic range
        h_var_raw = tf.reduce_mean(
            tf.abs(h_tok - tf.expand_dims(h_mean, -1)) ** 2,
            axis=-1)
        h_var = tf.math.log1p(h_var_raw)

        # Max — log1p to compress dynamic range
        h_max_raw = tf.reduce_max(tf.abs(h_tok), axis=-1)
        h_max = tf.math.log1p(h_max_raw)

        # Spatial covariance
        h_mean_p = tf.transpose(h_mean, [0, 3, 4, 1, 2])
        cov = tf.matmul(h_mean_p, h_mean_p, adjoint_b=True)
        # ← Normalize cov by num_tx so diagonal ~ same scale as |h_mean|²
        cov = cov / tf.cast(self.num_tx, cov.dtype)
        cov_feat = tf.concat(
            [tf.math.real(cov), tf.math.imag(cov)], axis=-1)

        def cplx_to_real(h):
            p = tf.transpose(h, [0, 3, 4, 1, 2])
            return tf.concat([tf.math.real(p), tf.math.imag(p)], axis=-1)

        def real_to_perm(h):
            return tf.transpose(h, [0, 3, 4, 1, 2])

        f_mean  = cplx_to_real(h_mean)
        f_slope = cplx_to_real(h_slope)
        f_var   = real_to_perm(h_var)
        f_max   = real_to_perm(h_max)

        feat = tf.concat([f_mean, f_slope, f_var, f_max, cov_feat], axis=-1)
        # [B, ofdm, T, rx, 6*tx + 2*rx]

        # ── Instance normalization across the feature dimension ──────────
        # Normalizes each (B*ofdm, T, rx) position independently.
        # This is cheap and ensures all features enter input_projection
        # on the same scale regardless of SNR or channel conditions.
        feat_mean = tf.reduce_mean(feat, axis=-1, keepdims=True)
        feat_std  = tf.math.reduce_std(feat, axis=-1, keepdims=True)
        feat      = (feat - feat_mean) / (feat_std + 1e-6)
        # ─────────────────────────────────────────────────────────────────

        return tf.reshape(
            feat,
            [B * self.num_ofdm, T, self.num_rx, self.feat_dim])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, h_freq, training=False,
             return_real_imag=False, return_coarse=False):
        """
        h_freq : [B, num_rx, 1, 1, num_tx, num_ofdm, fft_size]  (Sionna format)
        Returns: [B, 1, num_ofdm, fft_size, num_tx, num_rx]      (complex precoder)
        """
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]

        # Remove singleton dims: [B, num_rx, num_tx, num_ofdm, fft_size]
        h_sq = tf.squeeze(h_freq, axis=[2, 3])

        # ==============================================================
        # STAGE 1: COARSE PROCESSING (token resolution)
        # ==============================================================

        # Extract improved token features
        feat_input = self._extract_token_features(h_sq, B)
        # [B*ofdm, T, rx, feat_dim]

        # Input projection + norm
        x = self.input_projection(feat_input)    # [B*ofdm, T, rx, embed_dim]
        x = self.input_norm(x)

        # Positional embedding (token position, not subcarrier position)
        positions  = tf.range(self.total_tokens)
        pos_embed  = self.pos_embedding(positions)           # [T, embed_dim]
        pos_embed  = tf.reshape(pos_embed, [1, self.total_tokens, 1, self.embed_dim])
        x = x + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Coarse precoder: [B*ofdm, T, rx, num_tx*2]
        w_coarse = self.coarse_projection(x)

        # Reshape to [B, ofdm, T, tx, rx, 2]
        w_coarse = tf.reshape(
            w_coarse,
            [B, self.num_ofdm, self.total_tokens, self.num_rx, self.num_tx, 2])
        w_coarse = tf.transpose(w_coarse, [0, 1, 2, 4, 3, 5])
        # [B, ofdm, T, tx, rx, 2]

        # ==============================================================
        # STAGE 2: SC-LEVEL REFINEMENT
        # Upsample coarse output to full subcarrier resolution, then
        # apply a lightweight per-SC MLP correction.
        # ==============================================================

        # Upsample: repeat each token sc_per_token times along SC axis
        w_coarse_up = tf.repeat(w_coarse, self.sc_per_token, axis=2)
        # [B, ofdm, fft_size, tx, rx, 2]

        # Per-SC channel (for refinement input)
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])        # [B, ofdm, fft, rx, tx]
        h_full_flat = tf.stack(
            [tf.math.real(h_full), tf.math.imag(h_full)], axis=-1)
        # [B, ofdm, fft, rx, tx, 2]
        h_full_flat = tf.transpose(h_full_flat, [0, 1, 2, 4, 3, 5])
        # [B, ofdm, fft, tx, rx, 2]

        # Concatenate coarse precoder + raw channel: 4 values per (tx,rx) position
        combined = tf.concat([w_coarse_up, h_full_flat], axis=-1)
        # [B, ofdm, fft, tx, rx, 4]

        combined_flat = tf.reshape(
            combined,
            [B * self.num_ofdm * self.fft_size, self.num_tx, self.num_rx, 4])

        # MLP correction
        delta_flat = self.sc_refinement(combined_flat)
        # [B*ofdm*fft, tx, rx, 2]

        delta = tf.reshape(
            delta_flat,
            [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])

        # Residual addition
        w_final = w_coarse_up + self.alpha * delta
        # [B, ofdm, fft, tx, rx, 2]

        w_real = w_final[..., 0]
        w_imag = w_final[..., 1]

        # ==============================================================
        # STAGE 3: POWER NORMALIZATION (per stream, per SC)
        # Normalize each precoding vector (column = stream) to unit norm.
        # axis=3 sums over tx antennas → per-stream power.
        # ==============================================================
        power = w_real**2 + w_imag**2                        # [B, ofdm, fft, tx, rx]
        power_per_stream = tf.reduce_sum(power, axis=3, keepdims=True)
        # [B, ofdm, fft, 1, rx]

        scale = tf.sqrt(1.0 / (power_per_stream + 1e-12))

        w_real = w_real * scale
        w_imag = w_imag * scale

        # ==============================================================
        # OUTPUT FORMATTING
        # ==============================================================
        if return_real_imag:
            return (tf.expand_dims(w_real, 1),
                    tf.expand_dims(w_imag, 1))

        w_complex = tf.complex(w_real, w_imag)
        w_complex = tf.expand_dims(w_complex, 1)
        # [B, 1, ofdm, fft, tx, rx]

        if return_coarse:
            # Also return the upsampled coarse precoder for diagnostic purposes
            w_c = tf.complex(w_coarse_up[..., 0], w_coarse_up[..., 1])
            w_c = tf.expand_dims(w_c, 1)
            return w_complex, w_c

        return w_complex

# =============================================================================
# TransformerPrecoderV4  —  Clean, joint multi-user output head
# =============================================================================

class TransformerPrecoderV33(Model):
    """
    RB-Aggregated Transformer Precoder — Version propre.

    Changements vs V3 :
    ─────────────────────────────────────────────────────────────────────
    1. Features simplifiées  : mean + slope + var + cov + one-hot user
       (one-hot corrige le problème de rates identiques entre users)
    2. Tête de sortie JOINT  : flatten tous les users avant la projection
       finale → le modèle décide des precoders de tous les users ensemble
    3. SC refinement JOINT   : le MLP voit [tx*rx*4] par SC
    4. tokens_per_rb         : configurable pour ablation study
    ─────────────────────────────────────────────────────────────────────
    feat_dim = 5*num_tx + 3*num_rx  (= 52 pour 8tx/4rx)
    """

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)

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

        # feat_dim par (token, user) :
        #   mean  : 2*num_tx  (re + im)
        #   slope : 2*num_tx  (re + im)
        #   var   : 1*num_tx  (réel, log1p)
        #   cov   : 2*num_rx  (ligne de H_mean @ H_mean^H, re + im)
        #   onehot: num_rx    (identifiant user explicite)
        self.feat_dim = 5 * num_tx + 3 * num_rx  # 52 pour 8tx/4rx

        # ------------------------------------------------------------------
        # 1. Projection d'entrée + norm
        # ------------------------------------------------------------------
        self.input_proj = layers.Dense(
            embed_dim, activation='gelu', name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')

        # ------------------------------------------------------------------
        # 2. Positional embedding sur les tokens
        # ------------------------------------------------------------------
        self.pos_embed = layers.Embedding(
            input_dim=self.total_tokens,
            output_dim=embed_dim,
            name='pos_embed')

        # ------------------------------------------------------------------
        # 3. Transformer blocks
        # ------------------------------------------------------------------
        self.blocks = [
            SeparableAttentionBlock(
                self.total_tokens, num_rx, embed_dim, num_heads,
                dropout=dropout, name=f'block_{i}')
            for i in range(num_layers)
        ]

        # ------------------------------------------------------------------
        # 4. Tête de sortie JOINT multi-user
        #    [B*ofdm, T, rx*embed_dim] → [B*ofdm, T, rx*tx*2]
        # ------------------------------------------------------------------
        self.joint_output_proj = layers.Dense(
            num_rx * num_tx * 2, name='joint_output_proj')

        # ------------------------------------------------------------------
        # 5. SC refinement JOINT
        #    Input  : [B*ofdm*fft, tx*rx*4]  — w_coarse + h_brut, tous users
        #    Output : [B*ofdm*fft, tx*rx*2]  — delta re + im
        # ------------------------------------------------------------------
        sc_in  = num_tx * num_rx * 4
        sc_out = num_tx * num_rx * 2
        self.sc_refine = tf.keras.Sequential([
            layers.Dense(sc_in * 2, activation='gelu', name='sc_r1'),
            layers.Dense(sc_out,                       name='sc_r2'),
        ], name='sc_refine')

        # Résidu learnable — initialisé petit pour démarrer proche du coarse
        self.alpha = self.add_weight(
            name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(self, h_sq, B):
        """
        h_sq : [B, num_rx, num_tx, num_ofdm, fft_size]
        →      [B*num_ofdm, total_tokens, num_rx, feat_dim]
        """
        T = self.total_tokens
        S = self.sc_per_token

        # Grouper en tokens : [B, rx, tx, ofdm, T, S]
        h_tok = tf.reshape(
            h_sq,
            [B, self.num_rx, self.num_tx, self.num_ofdm, T, S])

        # ── Features canal ──────────────────────────────────────────
        h_mean  = tf.reduce_mean(h_tok, axis=-1)          # [B,rx,tx,ofdm,T]
        h_slope = h_tok[..., -1] - h_tok[..., 0]          # idem
        h_var   = tf.math.log1p(
            tf.reduce_mean(
                tf.abs(h_tok - tf.expand_dims(h_mean, -1))**2,
                axis=-1))                                   # réel

        # Covariance spatiale normalisée : [B, ofdm, T, rx, rx]
        h_mean_p = tf.transpose(h_mean, [0, 3, 4, 1, 2])  # [B,ofdm,T,rx,tx]
        cov = tf.matmul(h_mean_p, h_mean_p, adjoint_b=True) \
              / tf.cast(self.num_tx, h_mean_p.dtype)
        cov_feat = tf.concat(
            [tf.math.real(cov), tf.math.imag(cov)], axis=-1)
        # [B, ofdm, T, rx, 2*rx]

        def c2r(h):
            # [B,rx,tx,ofdm,T] → [B,ofdm,T,rx, 2*tx]
            p = tf.transpose(h, [0, 3, 4, 1, 2])
            return tf.concat([tf.math.real(p), tf.math.imag(p)], axis=-1)

        def r2p(h):
            # [B,rx,tx,ofdm,T] réel → [B,ofdm,T,rx,tx]
            return tf.transpose(h, [0, 3, 4, 1, 2])

        # [B, ofdm, T, rx, 5*tx + 2*rx]
        feat = tf.concat([c2r(h_mean), c2r(h_slope), r2p(h_var), cov_feat],
                         axis=-1)

        # ── One-hot user index ───────────────────────────────────────
        # Corrige le problème SAGE-HB : tous les users ont des canaux
        # i.i.d. → features statistiquement identiques → le modèle
        # donne la même solution à tous → rates identiques.
        # Le one-hot donne un identifiant explicite par user.
        Bo = B * self.num_ofdm
        feat_flat = tf.reshape(feat, [Bo, T, self.num_rx, 5 * self.num_tx + 2 * self.num_rx])

        user_onehot = tf.eye(self.num_rx, dtype=tf.float32)       # [rx, rx]
        user_onehot = tf.reshape(user_onehot,
                                  [1, 1, self.num_rx, self.num_rx])
        user_onehot = tf.tile(user_onehot, [Bo, T, 1, 1])
        # [B*ofdm, T, rx, rx]

        feat_flat = tf.concat([feat_flat, user_onehot], axis=-1)
        # [B*ofdm, T, rx, feat_dim]  où feat_dim = 5*tx + 3*rx

        return feat_flat

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, h_freq, training=False, return_real_imag=False):
        """
        Entrée  : h_freq  [B, num_rx, 1, 1, num_tx, num_ofdm, fft]
        Sortie  : w       [B, 1, num_ofdm, fft, num_tx, num_rx]
        """
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]

        # [B, rx, tx, ofdm, fft]
        h_sq = tf.squeeze(h_freq, axis=[2, 3])

        # ── STAGE 1 : Transformer sur les tokens RB ───────────────────
        feat = self._extract_features(h_sq, B)
        # [B*ofdm, T, rx, feat_dim]

        x = self.input_proj(feat)   # [B*ofdm, T, rx, embed_dim]
        x = self.input_norm(x)

        pos = tf.range(self.total_tokens)
        x   = x + tf.reshape(self.pos_embed(pos),
                              [1, self.total_tokens, 1, self.embed_dim])

        for blk in self.blocks:
            x = blk(x, training=training)
        # x : [B*ofdm, T, rx, embed_dim]

        # ── Tête JOINT : flatten users → projeter tous ensemble ──────
        Bo = tf.shape(x)[0]
        x_joint = tf.reshape(x,
            [Bo, self.total_tokens, self.num_rx * self.embed_dim])
        # [B*ofdm, T, rx*embed_dim]

        w_tok = self.joint_output_proj(x_joint)
        # [B*ofdm, T, rx*tx*2]

        w_tok = tf.reshape(w_tok,
            [B, self.num_ofdm, self.total_tokens,
             self.num_rx, self.num_tx, 2])
        w_tok = tf.transpose(w_tok, [0, 1, 2, 4, 3, 5])
        # [B, ofdm, T, tx, rx, 2]

        # ── Upsample token → SC ───────────────────────────────────────
        w_up = tf.repeat(w_tok, self.sc_per_token, axis=2)
        # [B, ofdm, fft, tx, rx, 2]

        # ── STAGE 2 : SC refinement JOINT ────────────────────────────
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])    # [B,ofdm,fft,rx,tx]
        h_ri   = tf.stack([tf.math.real(h_full),
                           tf.math.imag(h_full)], axis=-1)
        h_ri   = tf.transpose(h_ri, [0, 1, 2, 4, 3, 5])
        # [B, ofdm, fft, tx, rx, 2]

        # Concat coarse precoder + canal brut, flatten tous les (tx,rx)
        combined = tf.concat([w_up, h_ri], axis=-1)
        # [B, ofdm, fft, tx, rx, 4]

        combined_flat = tf.reshape(
            combined,
            [B * self.num_ofdm * self.fft_size,
             self.num_tx * self.num_rx * 4])

        delta_flat = self.sc_refine(combined_flat)
        # [B*ofdm*fft, tx*rx*2]

        delta = tf.reshape(
            delta_flat,
            [B, self.num_ofdm, self.fft_size, self.num_tx, self.num_rx, 2])

        w_final = w_up + self.alpha * delta
        # [B, ofdm, fft, tx, rx, 2]

        # ── Normalisation per-stream (convention Sionna) ──────────────
        w_re = w_final[..., 0]
        w_im = w_final[..., 1]

        pwr   = tf.reduce_sum(w_re**2 + w_im**2, axis=3, keepdims=True)
        scale = tf.sqrt(1.0 / (pwr + 1e-12))
        w_re  = w_re * scale
        w_im  = w_im * scale

        # ── Format de sortie Sionna ───────────────────────────────────
        if return_real_imag:
            return (tf.expand_dims(w_re, 1),
                    tf.expand_dims(w_im, 1))

        return tf.expand_dims(tf.complex(w_re, w_im), 1)
        # [B, 1, ofdm, fft, tx, rx]
class TransformerPrecoderV4(Model):
    """
    RB-Aggregated Transformer Precoder — V4.0 / V4.1 / V4.2

    version='v4.0' — Baseline :
        - Upsampling : tf.repeat (blocs constants)
        - SC refine  : Conv1D(k=3+1), input=[w_up, h_re, h_im]
        - 4 features par (tx,rx) : w_re, w_im, h_re, h_im

    version='v4.1' — Upsampling appris :
        - Upsampling : Conv1DTranspose(kernel=stride=sc_per_token)
          → interpolation apprise, anti-checkerboard
        - SC refine  : Conv1D(k=rb_size), input=w_up seulement
        - Justification : Conv1DTranspose encode déjà le canal via features

    version='v4.2' — Info intra-token préservée :
        - Upsampling : Conv1DTranspose (identique V4.1)
        - SC refine  : Conv1D(k=rb_size), input=[w_up, h_re, h_im, |h|, ∠h]
        - 6 features par (tx,rx) : w_re, w_im, h_re, h_im, |h|, ∠h
        - Justification : |h| et ∠h sont des quantités physiques distinctes
          que le réseau n'a plus à séparer implicitement depuis h_re+h_im
        - Info intra-token récupérée via les 6 features per-SC
    """

    # Mapping version → (use_learned_upsample, sc_refine_mode)
    _VERSION_MAP = {
        'v4.0': (False, 'basic'),    # tf.repeat  + [w_up, h_ri]
        'v4.1': (True,  'minimal'),  # Conv1DTrans + [w_up]
        'v4.2': (True,  'rich'),     # Conv1DTrans + [w_up, h_ri, |h|, ∠h]
    }

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.0,
                 version='v4.0',   # ← 'v4.0', 'v4.1', ou 'v4.2'
                 **kwargs):
        super().__init__(**kwargs)

        assert version in self._VERSION_MAP, \
            f"version doit être 'v4.0', 'v4.1' ou 'v4.2', reçu: {version}"

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
        self.feat_dim      = 5 * num_tx + 3 * num_rx
        self.version       = version

        self.use_learned_upsample, self.refine_mode = self._VERSION_MAP[version]

        sc_out = num_tx * num_rx * 2

        # ── Input projection + norm ───────────────────────────────────────
        self.input_proj = layers.Dense(
            embed_dim, activation='gelu', name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')

        # ── Positional embedding ──────────────────────────────────────────
        self.pos_embed = layers.Embedding(
            input_dim=self.total_tokens,
            output_dim=embed_dim,
            name='pos_embed')

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = [
            SeparableAttentionBlock(
                self.total_tokens, num_rx, embed_dim, num_heads,
                dropout=dropout, name=f'block_{i}')
            for i in range(num_layers)
        ]

        # ── Joint output projection ───────────────────────────────────────
        self.joint_output_proj = layers.Dense(
            num_rx * num_tx * 2, name='joint_output_proj')

        # ── Upsampling ────────────────────────────────────────────────────
        if self.use_learned_upsample:
            self.upsample = layers.Conv1DTranspose(
                filters=sc_out,
                kernel_size=self.sc_per_token,
                strides=self.sc_per_token,
                padding='valid',
                activation=None,
                name='upsample_learned')
        # v4.0 : pas de layer — tf.repeat dans call()

        # ── SC refinement — taille d'input selon version ──────────────────
        #
        # v4.0 : [w_up(tx*rx*2), h_re(tx*rx), h_im(tx*rx)]
        #        = tx*rx * 4  → sc_in = M*K*4
        #
        # v4.1 : [w_up(tx*rx*2)]
        #        = tx*rx * 2  → sc_in = M*K*2
        #
        # v4.2 : [w_up(tx*rx*2), h_re(tx*rx), h_im(tx*rx),
        #         |h|(tx*rx),   ∠h(tx*rx)]
        #        = tx*rx * 6  → sc_in = M*K*6
        #
        sc_in_map = {
            'basic'  : num_tx * num_rx * 4,   # v4.0
            'minimal': num_tx * num_rx * 2,   # v4.1
            'rich'   : num_tx * num_rx * 6,   # v4.2
        }
        sc_in = sc_in_map[self.refine_mode]

        k_refine = rb_size if self.use_learned_upsample else 3

        # Remplacer partout sc_in * 2 par sc_out * 2
        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(sc_out * 2, kernel_size=k_refine,
                        padding='same', activation='gelu', name='sc_r1'),
            layers.Conv1D(sc_out, kernel_size=1,
                        padding='same', name='sc_r2'),
        ], name='sc_refine')

        # ── Alpha résiduel ────────────────────────────────────────────────
        self.alpha = self.add_weight(
            name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True)

        # ── Log de configuration ──────────────────────────────────────────
        upsample_str = (f"Conv1DTranspose(k={self.sc_per_token})"
                        if self.use_learned_upsample else "tf.repeat")
        refine_input_str = {
            'basic'  : f"[w_up, h_ri] k={k_refine}+1",
            'minimal': f"[w_up] k={k_refine}",
            'rich'   : f"[w_up, h_ri, |h|, ∠h] k={k_refine}",
        }[self.refine_mode]

        print(f"TransformerPrecoderV4 [{version.upper()}] | "
              f"tokens_per_rb={tokens_per_rb} | "
              f"sc_per_token={self.sc_per_token} | "
              f"embed_dim={embed_dim} | "
              f"upsample={upsample_str} | "
              f"refine={refine_input_str}")

    # ─────────────────────────────────────────────────────────────────────
    # Feature extraction — identique pour toutes les versions
    # ─────────────────────────────────────────────────────────────────────
    def _extract_features(self, h_sq, B):
        T = self.total_tokens
        S = self.sc_per_token

        h_tok = tf.reshape(
            h_sq,
            [B, self.num_rx, self.num_tx, self.num_ofdm, T, S])

        h_mean  = tf.reduce_mean(h_tok, axis=-1)
        h_slope = h_tok[..., -1] - h_tok[..., 0]
        h_var   = tf.math.log1p(
            tf.reduce_mean(
                tf.abs(h_tok - tf.expand_dims(h_mean, -1))**2,
                axis=-1))

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

        feat = tf.concat([c2r(h_mean), c2r(h_slope), r2p(h_var), cov_feat],
                         axis=-1)

        Bo = B * self.num_ofdm
        feat_flat = tf.reshape(
            feat, [Bo, T, self.num_rx, 5 * self.num_tx + 2 * self.num_rx])

        user_onehot = tf.tile(
            tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                       [1, 1, self.num_rx, self.num_rx]),
            [Bo, T, 1, 1])

        return tf.concat([feat_flat, user_onehot], axis=-1)

    # ─────────────────────────────────────────────────────────────────────
    # Complexity
    # ─────────────────────────────────────────────────────────────────────
    def complexity(self, num_ofdm=14):
        M  = self.num_tx;  K  = self.num_rx
        T  = self.total_tokens;  D  = self.embed_dim
        fd = self.feat_dim;      N  = self.fft_size
        S  = self.sc_per_token;  RB = self.rb_size
        sc_out = M * K * 2

        sc_in_map = {'basic': M*K*4, 'minimal': M*K*2, 'rich': M*K*6}
        sc_in = sc_in_map[self.refine_mode]
        k_r   = RB if self.use_learned_upsample else 3

        def df(n, i, o): return 2.0 * n * i * o
        def dp(i, o):    return i * o + o
        def mf(seq, d):  return 8.0*seq*d**2 + 4.0*seq**2*d
        def mp(d):       return 4 * (d*d + d)
        def ff(n, d):    return df(n, d, 4*d) + df(n, 4*d, d)
        def fp(d):       return dp(d, 4*d) + dp(4*d, d)

        # Input projection
        f = df(T*K, fd, D);  w = dp(fd, D);  a = T*K*D

        # Transformer blocks
        for _ in self.blocks:
            f += K * mf(T, D) + T * mf(K, D) + ff(T*K, D)
            w += 2*mp(D) + fp(D)
            a += 3 * T * K * D

        # Joint output proj
        f += df(T, K*D, K*M*2);  w += dp(K*D, K*M*2);  a += T*K*M*2

        if self.use_learned_upsample:
            # Conv1DTranspose
            f += 2.0 * N * S * sc_out * sc_out
            w += S * sc_out * sc_out + sc_out
            a += N * sc_out
        # v4.0 : tf.repeat = 0 FLOPs

        # SC refine : Conv1D(k_r, sc_in → sc_out*2) + Conv1D(1, sc_out*2 → sc_out)
        # Taille cachée fixe = sc_out*2 pour toutes les versions
        hidden = sc_out * 2
        f += 2.0 * N * k_r * sc_in * hidden    # k_r, in→hidden
        f += 2.0 * N * 1   * hidden  * sc_out  # k=1, hidden→out
        w += k_r * sc_in * hidden + hidden
        w += 1   * hidden * sc_out  + sc_out
        a += N * (hidden + sc_out)

        return f * num_ofdm, w, a * num_ofdm

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────
    def call(self, h_freq, training=False, return_real_imag=False):
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]
        h_sq   = tf.squeeze(h_freq, axis=[2, 3])
        Bo     = B * self.num_ofdm

        # ── Stage 1 : Transformer ─────────────────────────────────────────
        feat = self._extract_features(h_sq, B)
        x    = self.input_proj(feat)
        x    = self.input_norm(x)
        x    = x + tf.reshape(self.pos_embed(tf.range(self.total_tokens)),
                               [1, self.total_tokens, 1, self.embed_dim])
        for blk in self.blocks:
            x = blk(x, training=training)

        # ── Joint output → [Bo, T, tx*rx*2] ──────────────────────────────
        w_tok = self.joint_output_proj(
            tf.reshape(x, [Bo, self.total_tokens,
                           self.num_rx * self.embed_dim]))

        # ── Stage 2 : Upsampling → [Bo, fft, tx*rx*2] ────────────────────
        if self.use_learned_upsample:
            w_up = self.upsample(w_tok)          # Conv1DTranspose
        else:
            w_up = tf.repeat(w_tok, self.sc_per_token, axis=1)  # tf.repeat

        # ── Stage 3 : SC refinement ───────────────────────────────────────
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])
        # [B, ofdm, fft, rx, tx] — canal complet per-SC

        if self.refine_mode == 'minimal':
            # V4.1 — w_up seulement
            refine_input = w_up

        elif self.refine_mode == 'basic':
            # V4.0 — [w_up, h_re, h_im]
            h_ri = tf.reshape(
                tf.concat([tf.math.real(h_full),
                           tf.math.imag(h_full)], axis=-1),
                [Bo, self.fft_size, self.num_tx * self.num_rx * 2])
            refine_input = tf.concat([w_up, h_ri], axis=-1)

        else:  # 'rich' = V4.2
            # V4.2 — [w_up, h_re, h_im, |h|, ∠h]
            h_re    = tf.reshape(tf.math.real(h_full),
                [Bo, self.fft_size, self.num_tx * self.num_rx])
            h_im    = tf.reshape(tf.math.imag(h_full),
                [Bo, self.fft_size, self.num_tx * self.num_rx])
            h_abs   = tf.reshape(tf.abs(h_full),
                [Bo, self.fft_size, self.num_tx * self.num_rx])
            h_phase = tf.reshape(tf.math.angle(h_full),
                [Bo, self.fft_size, self.num_tx * self.num_rx])
            refine_input = tf.concat(
                [w_up, h_re, h_im, h_abs, h_phase], axis=-1)
            # [Bo, fft, tx*rx*6]

        delta        = self.sc_refine(refine_input, training=training)
        w_final_flat = w_up + self.alpha * delta

        # ── Reshape + power norm ──────────────────────────────────────────
        w_final = tf.reshape(
            w_final_flat,
            [B, self.num_ofdm, self.fft_size,
             self.num_tx, self.num_rx, 2])

        w_re = w_final[..., 0]
        w_im = w_final[..., 1]
        pwr  = tf.reduce_sum(w_re**2 + w_im**2, axis=3, keepdims=True)
        s    = tf.sqrt(1.0 / (pwr + 1e-12))
        w_re, w_im = w_re * s, w_im * s

        if return_real_imag:
            return tf.expand_dims(w_re, 1), tf.expand_dims(w_im, 1)
        return tf.expand_dims(tf.complex(w_re, w_im), 1)

class TransformerPrecoderV43(Model):
    """
    V4.3 — Trois ajouts ciblés vs V4.2 :
    1. SNR feature : log(no) broadcasté sur tous les tokens
       → permet l'adaptation amplitude/waterfilling
    2. Entraînement multi-SNR : reçoit no comme argument
       → généralisation sur [0, 20] dB au lieu de 15 dB seul
    3. SC refine avec résidu gated (sigmoid gate sur delta)
       → contrôle fin de la correction per-SC
    
    Tout le reste identique à V4.2.
    """

    _VERSION_MAP = {
        'v4.2': (True, 'rich'),
        'v4.3': (True, 'rich_snr'),   # nouveau : rich + SNR feature
    }

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 rb_size=12, tokens_per_rb=1,
                 embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.0, version='v4.3', **kwargs):
        super().__init__(**kwargs)

        assert version in self._VERSION_MAP
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

        # feat_dim = V4.2 + 1 (log_no scalaire broadcasté)
        self.feat_dim = 5 * num_tx + 3 * num_rx + 1   # +1 pour log(no)

        sc_out = num_tx * num_rx * 2

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = layers.Dense(embed_dim, activation='gelu',
                                        name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')

        # ── Positional embedding ──────────────────────────────────────────────
        self.pos_embed = layers.Embedding(self.total_tokens, embed_dim,
                                           name='pos_embed')

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = [
            SeparableAttentionBlock(
                self.total_tokens, num_rx, embed_dim, num_heads,
                dropout=dropout, name=f'block_{i}')
            for i in range(num_layers)
        ]

        # ── Joint output ──────────────────────────────────────────────────────
        self.joint_output_proj = layers.Dense(num_rx * num_tx * 2,
                                               name='joint_output_proj')

        # ── Upsampling Conv1DTranspose ────────────────────────────────────────
        self.upsample = layers.Conv1DTranspose(
            filters=sc_out,
            kernel_size=self.sc_per_token,
            strides=self.sc_per_token,
            padding='valid', activation=None,
            name='upsample_learned')

        # ── SC refine — rich + SNR ────────────────────────────────────────────
        # Input : [w_up(tx*rx*2), h_re(tx*rx), h_im(tx*rx), |h|(tx*rx), ∠h(tx*rx)]
        # = tx*rx*6 — identique V4.2
        sc_in  = num_tx * num_rx * 6
        hidden = sc_out * 2

        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(hidden, kernel_size=rb_size,
                          padding='same', activation='gelu', name='sc_r1'),
            layers.Conv1D(sc_out, kernel_size=1,
                          padding='same', name='sc_r2'),
        ], name='sc_refine')

        # ── Gate sigmoid sur le résidu SC ────────────────────────────────────
        # Apprend QUAND appliquer la correction per-SC
        # Input : w_up (contexte global) → gate [0,1] par (SC, tx, rx)
        self.sc_gate = layers.Conv1D(sc_out, kernel_size=1,
                                      padding='same', activation='sigmoid',
                                      name='sc_gate')

        # ── SNR projection : scalaire → embed_dim ─────────────────────────────
        # Injecté dans le token embedding AVANT le transformer
        self.snr_proj = layers.Dense(embed_dim, activation='gelu',
                                      name='snr_proj')

        self.alpha = self.add_weight(
            name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True)

        print(f"TransformerPrecoderV4.3 | tokens_per_rb={tokens_per_rb} | "
              f"sc_per_token={self.sc_per_token} | embed_dim={embed_dim} | "
              f"SNR-aware=True | gated-SC-refine=True")

    # ── Feature extraction — identique V4.2 sauf +log_no ─────────────────────
    def _extract_features(self, h_sq, B, log_no):
        """
        log_no : scalaire tf.float32 = log(no)
        Retourne [B*ofdm, T, rx, feat_dim] avec feat_dim = V4.2 + 1
        """
        T = self.total_tokens
        S = self.sc_per_token

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

        feat = tf.concat([c2r(h_mean), c2r(h_slope), r2p(h_var), cov_feat],
                         axis=-1)
        # [B, ofdm, T, rx, 5*tx + 2*rx]

        Bo = B * self.num_ofdm
        feat_flat = tf.reshape(
            feat, [Bo, T, self.num_rx, 5 * self.num_tx + 2 * self.num_rx])

        # One-hot user
        user_onehot = tf.tile(
            tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                       [1, 1, self.num_rx, self.num_rx]),
            [Bo, T, 1, 1])

        # log(no) broadcasté — même valeur pour tous tokens/users/SC
        # Shape: [Bo, T, rx, 1]
        log_no_cast = tf.cast(log_no, tf.float32)
        log_no_bc   = tf.fill([Bo, T, self.num_rx, 1], log_no_cast)

        return tf.concat([feat_flat, user_onehot, log_no_bc], axis=-1)
        # [Bo, T, rx, 5*tx + 3*rx + 1]

    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        """
        h_freq : [B, rx, 1, 1, tx, ofdm, fft]
        no     : bruit scalaire (float32). Si None → SNR inconnu → log(1.0)=0
        """
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]
        h_sq   = tf.squeeze(h_freq, axis=[2, 3])
        Bo     = B * self.num_ofdm

        # log(no) — normalisé pour être dans [-5, 5] sur [0,20] dB
        if no is None:
            log_no = tf.constant(0.0)
        else:
            log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10)

        # ── Features + SNR ────────────────────────────────────────────────────
        feat = self._extract_features(h_sq, B, log_no)
        x    = self.input_proj(feat)
        x    = self.input_norm(x)

        # ── SNR embedding injecté dans les tokens ────────────────────────────
        # [1, 1, 1, D] broadcasté sur [Bo, T, rx, D]
        snr_emb = self.snr_proj(
            tf.reshape(log_no, [1, 1]))          # [1, D]
        snr_emb = tf.reshape(snr_emb,
            [1, 1, 1, self.embed_dim])           # broadcastable
        x = x + snr_emb

        # ── Positional embedding ──────────────────────────────────────────────
        x = x + tf.reshape(
            self.pos_embed(tf.range(self.total_tokens)),
            [1, self.total_tokens, 1, self.embed_dim])

        # ── Transformer ───────────────────────────────────────────────────────
        for blk in self.blocks:
            x = blk(x, training=training)

        # ── Joint output → [Bo, T, tx*rx*2] ──────────────────────────────────
        w_tok = self.joint_output_proj(
            tf.reshape(x, [Bo, self.total_tokens,
                           self.num_rx * self.embed_dim]))

        # ── Upsampling → [Bo, fft, tx*rx*2] ──────────────────────────────────
        w_up = self.upsample(w_tok)

        # ── SC refine rich ────────────────────────────────────────────────────
        h_full = tf.transpose(h_sq, [0, 3, 4, 1, 2])  # [B, ofdm, fft, rx, tx]
        h_re    = tf.reshape(tf.math.real(h_full),
                             [Bo, self.fft_size, self.num_tx * self.num_rx])
        h_im    = tf.reshape(tf.math.imag(h_full),
                             [Bo, self.fft_size, self.num_tx * self.num_rx])
        h_abs   = tf.reshape(tf.abs(h_full),
                             [Bo, self.fft_size, self.num_tx * self.num_rx])
        h_phase = tf.reshape(tf.math.angle(h_full),
                             [Bo, self.fft_size, self.num_tx * self.num_rx])

        refine_input = tf.concat([w_up, h_re, h_im, h_abs, h_phase], axis=-1)
        delta        = self.sc_refine(refine_input, training=training)

        # ── Gate sigmoid : contrôle où appliquer la correction ───────────────
        gate   = self.sc_gate(w_up)                  # [Bo, fft, sc_out] ∈ [0,1]
        w_final_flat = w_up + self.alpha * gate * delta

        # ── Reshape + power norm ──────────────────────────────────────────────
        w_final = tf.reshape(w_final_flat,
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
# TransformerPrecoderV5 — Per-RB Shared + Inter-RB attention
# Architecture Swin-inspired pour MU-MIMO precoding
#
# Améliorations vs version précédente :
# 1. Cross-attention SC→RB au lieu de simple addition (broadcast intelligent)
# 2. Features enrichies : H_re/im + covariance intra-RB broadcastée
# 3. rb_size=12 fixé (standard 5G NR, justification physique)
# 4. Attention séparée SC/user dans tous les blocs
# =============================================================================

class IntraRBBlock(layers.Layer):
    """
    Bloc transformer intra-RB — appliqué à chaque RB indépendamment.
    PARTAGÉ entre tous les RBs (weight sharing).

    Justification physique du sharing :
    - Les SC à l'intérieur d'un RB obéissent aux mêmes lois de propagation
      quel que soit le RB (stationnarité intra-RB, cohérence >> 360 kHz)
    - Inductive bias correcte : translation-invariante en fréquence
      à l'échelle d'un RB

    Attention séparée SC puis user :
    - SC attention : capture la sélectivité fréquentielle intra-RB
    - User attention : capture l'interférence spatiale inter-utilisateurs
    Ces deux phénomènes sont physiquement distincts → séparation justifiée

    Input/output : [B*ofdm*num_rb, sc_per_rb, num_users, embed_dim]
    """
    def __init__(self, sc_per_rb, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.sc_per_rb  = sc_per_rb
        self.num_users  = num_users
        self.embed_dim  = embed_dim

        kd = embed_dim // num_heads

        # SC attention — sélectivité fréquentielle
        self.norm_sc   = RMSNormalization(epsilon=1e-6, name='norm_sc')
        self.sc_attn   = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            dropout=dropout, name='sc_attn')

        # User attention — interférence spatiale
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            dropout=dropout, name='user_attn')

        # FFN
        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')
        self.ffn       = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')

        # Learnable scales — initialisés petits pour stabilité
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
        # x : [B*ofdm*num_rb, sc_per_rb, num_users, embed_dim]
        B = tf.shape(x)[0]

        # ── SC attention (chaque user indépendamment) ─────────────────────
        xf = tf.reshape(
            tf.transpose(x, [0, 2, 1, 3]),
            [B * self.num_users, self.sc_per_rb, self.embed_dim])
        xf_n = self.norm_sc(xf)
        xf   = xf + self.s_sc * self.sc_attn(
            xf_n, xf_n, training=training)
        x    = tf.transpose(
            tf.reshape(xf,
                [B, self.num_users, self.sc_per_rb, self.embed_dim]),
            [0, 2, 1, 3])

        # ── User attention (chaque SC indépendamment) ─────────────────────
        xu   = tf.reshape(x,
            [B * self.sc_per_rb, self.num_users, self.embed_dim])
        xu_n = self.norm_user(xu)
        xu   = xu + self.s_user * self.user_attn(
            xu_n, xu_n, training=training)
        x    = tf.reshape(xu,
            [B, self.sc_per_rb, self.num_users, self.embed_dim])

        # ── FFN ───────────────────────────────────────────────────────────
        xp = tf.reshape(x,
            [B * self.sc_per_rb * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(
            self.norm_ffn(xp), training=training)
        return tf.reshape(xp,
            [B, self.sc_per_rb, self.num_users, self.embed_dim])


class InterRBBlock(layers.Layer):
    """
    Attention inter-RB — capture les dépendances wideband entre les 6 RBs.
    Attention séparée RB puis user.

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
        self.rb_attn   = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            dropout=dropout, name='rb_attn')

        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.user_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            dropout=dropout, name='user_attn')

        self.norm_ffn  = RMSNormalization(epsilon=1e-6, name='norm_ffn')
        self.ffn       = tf.keras.Sequential([
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
        # x : [B*ofdm, num_rb, num_users, embed_dim]
        B = tf.shape(x)[0]

        # ── RB attention (chaque user indépendamment) ─────────────────────
        xr = tf.reshape(
            tf.transpose(x, [0, 2, 1, 3]),
            [B * self.num_users, self.num_rb, self.embed_dim])
        xr_n = self.norm_rb(xr)
        xr   = xr + self.s_rb * self.rb_attn(
            xr_n, xr_n, training=training)
        x    = tf.transpose(
            tf.reshape(xr,
                [B, self.num_users, self.num_rb, self.embed_dim]),
            [0, 2, 1, 3])

        # ── User attention (chaque RB indépendamment) ─────────────────────
        xu   = tf.reshape(x,
            [B * self.num_rb, self.num_users, self.embed_dim])
        xu_n = self.norm_user(xu)
        xu   = xu + self.s_user * self.user_attn(
            xu_n, xu_n, training=training)
        x    = tf.reshape(xu,
            [B, self.num_rb, self.num_users, self.embed_dim])

        # ── FFN ───────────────────────────────────────────────────────────
        xp = tf.reshape(x,
            [B * self.num_rb * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(
            self.norm_ffn(xp), training=training)
        return tf.reshape(xp,
            [B, self.num_rb, self.num_users, self.embed_dim])

class TransformerPrecoderV5(Model):
    """
    V5.2 — features enrichies + gate additive + sin/cos phase + norm entrée
    """

    RB_SIZE = 12

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=72,
                 embed_dim=128, num_heads=4,
                 num_intra_layers=2, num_inter_layers=2,
                 dropout=0.0, use_swin_shift=True, **kwargs):
        super().__init__(**kwargs)

        assert fft_size % self.RB_SIZE == 0
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

        # feat_dim enrichi :
        #   per-SC  : h_re(M) + h_im(M) + |h|(M) + sin∠h(M) + cos∠h(M) = 5M
        #   per-RB  : slope_re(M) + slope_im(M) + var(M)  broadcastés    = 3M
        #   cov     : re(K²) + im(K²)  broadcastés                        = 2K²
        #   one-hot : K
        # total = 8M + 2K² + K = 64 + 32 + 4 = 100  (pour M=8, K=4)
        self.feat_dim = 8 * num_tx + 2 * num_rx * num_rx + num_rx

        self.input_proj   = layers.Dense(embed_dim, activation='gelu', name='input_proj')
        self.input_norm   = RMSNormalization(epsilon=1e-6, name='input_norm')
        self.pos_embed_sc = layers.Embedding(self.rb_size, embed_dim, name='pos_embed_sc')
        self.pos_embed_rb = layers.Embedding(self.num_rb,  embed_dim, name='pos_embed_rb')

        self.rb_pos_bias = self.add_weight(
            shape=(self.num_rb, self.rb_size, 1, self.embed_dim),
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

        # ── RB pooling : mean + max → Dense(2D→D) ────────────────────────
        self.rb_pool_proj = layers.Dense(
            embed_dim, activation='gelu', name='rb_pool_proj')

        # ── Gate additive : remplace cross-attention ──────────────────────
        # rb_token [D] → gate [D], broadcasté sur les 12 SC du RB
        # gradient stable : 1 seule projection au lieu de 4
        self.wideband_gate = layers.Dense(embed_dim, activation='gelu',
                                          name='wideband_gate')
        self.s_cross = self.add_weight(
            shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True, name='s_cross')

        # ── Sortie joint ──────────────────────────────────────────────────
        self.joint_output = layers.Dense(num_rx * num_tx * 2, name='joint_output')

        print(f"TransformerPrecoderV5.2 | rb_size={self.rb_size} | "
              f"num_rb={self.num_rb} | embed_dim={embed_dim} | "
              f"intra={num_intra_layers}(shared) | inter={num_inter_layers} | "
              f"feat_dim={self.feat_dim} | swin_shift={use_swin_shift} | "
              f"gate=additive")

    def _extract_features(self, h_sq, B):
        Bo = B * self.num_ofdm

        # h_perm : [Bo, N, K, M]  complex
        h_perm = tf.reshape(
            tf.transpose(h_sq, [0, 3, 4, 1, 2]),
            [Bo, self.fft_size, self.num_rx, self.num_tx])

        # ── Features per-SC (anti-NaN) ────────────────────────────────────
        h_re  = tf.math.real(h_perm)                      # [Bo, N, K, M]
        h_im  = tf.math.imag(h_perm)                      # [Bo, N, K, M]

        # Norme sécurisée — epsilon sous la racine évite grad=inf à zéro
        h_abs = tf.sqrt(h_re**2 + h_im**2 + 1e-12)        # [Bo, N, K, M]

        # sin/cos sans tf.math.angle — pas de discontinuité, pas de NaN
        h_cos = h_re / h_abs                               # ∈ [-1, 1]
        h_sin = h_im / h_abs                               # ∈ [-1, 1]

        feat_sc = tf.concat([h_re, h_im, h_abs, h_sin, h_cos], axis=-1)
        # [Bo, N, K, 5M=40]

        # ── Features per-RB broadcastées ─────────────────────────────────
        h_rb = tf.reshape(h_perm,
            [Bo, self.num_rb, self.rb_size, self.num_rx, self.num_tx])

        h_rb_mean  = tf.reduce_mean(h_rb, axis=2)         # [Bo, NR, K, M]
        h_rb_slope = h_rb[:, :, -1, :, :] - h_rb[:, :, 0, :, :]

        # Variance sécurisée — réel/imag séparés, pas de tf.abs sur complexe
        diff    = h_rb - tf.expand_dims(h_rb_mean, 2)
        diff_re = tf.math.real(diff)
        diff_im = tf.math.imag(diff)
        h_rb_var = tf.math.log1p(
            tf.reduce_mean(diff_re**2 + diff_im**2, axis=2))
        # [Bo, NR, K, M]  réel, toujours >= 0

        slope_re = tf.math.real(h_rb_slope)               # [Bo, NR, K, M]
        slope_im = tf.math.imag(h_rb_slope)

        # broadcast NR → N (repeat sur axis=1)
        slope_re_bc = tf.repeat(slope_re, self.rb_size, axis=1)  # [Bo, N, K, M]
        slope_im_bc = tf.repeat(slope_im, self.rb_size, axis=1)
        var_bc      = tf.repeat(h_rb_var,  self.rb_size, axis=1)

        feat_rb = tf.concat([slope_re_bc, slope_im_bc, var_bc], axis=-1)
        # [Bo, N, K, 3M=24]

        # ── Covariance spatiale ───────────────────────────────────────────
        cov = tf.matmul(h_rb_mean, h_rb_mean, adjoint_b=True) \
            / tf.cast(self.num_tx, h_rb_mean.dtype)     # [Bo, NR, K, K]
        cov_feat = tf.concat(
            [tf.math.real(cov), tf.math.imag(cov)], axis=-1)  # [Bo, NR, K, 2K²=32]
        cov_bc = tf.repeat(cov_feat, self.rb_size, axis=1)    # [Bo, N, K, 32]

        # ── One-hot user ──────────────────────────────────────────────────
        user_onehot = tf.tile(
            tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                    [1, 1, self.num_rx, self.num_rx]),
            [Bo, self.fft_size, 1, 1])                    # [Bo, N, K, K=4]

        # ── Concat → instance norm ────────────────────────────────────────
        feat = tf.concat([feat_sc, feat_rb, cov_bc, user_onehot], axis=-1)
        # [Bo, N, K, 5M+3M+2K²+K] = [Bo, N, K, 100]

        feat_mean = tf.reduce_mean(feat, axis=-1, keepdims=True)
        feat_std  = tf.math.reduce_std(feat, axis=-1, keepdims=True)
        feat      = (feat - feat_mean) / (feat_std + 1e-6)

        return feat   # [Bo, N, K, feat_dim=100]

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

        # Input proj
        f = df(N*K, fd, D);  w = dp(fd, D);  a = N*K*D
        w += self.num_rb * self.rb_size * self.embed_dim  # rb_pos_bias

        # IntraRB blocks (shared — compté 1× en params, NR× en FLOPs)
        for _ in self.intra_blocks:
            f += (K*mf(RS,D) + RS*mf(K,D) + ff(RS*K,D)) * NR
            w += 2*mp(D) + fp(D)
            a += 3*RS*K*D*NR

        # Activation après intra (reshape)
        f += N*K*D;  a += NR*K*D

        # RB pooling : Dense(2D→D)
        f += df(NR*K, 2*D, D);  w += dp(2*D, D);  a += NR*K*D

        # InterRB blocks
        for _ in self.inter_blocks:
            f += K*mf(NR,D) + NR*mf(K,D) + ff(NR*K,D)
            w += 2*mp(D) + fp(D)
            a += 3*NR*K*D

        # Gate additive : Dense(D→D) par RB token, broadcasté sur RS SC
        f += df(NR*K, D, D)         # gate projection
        w += dp(D, D)
        a += NR*K*D + N*K*D         # gate + broadcast add

        # Joint output Dense(K*D → K*M*2) par SC
        f += df(N, K*D, K*M*2);  w += dp(K*D, K*M*2);  a += N*K*M*2

        return f*num_ofdm, w, a*num_ofdm

    def call(self, h_freq, training=False, return_real_imag=False):
        h_freq = tf.stop_gradient(h_freq)
        B      = tf.shape(h_freq)[0]
        h_sq   = tf.squeeze(h_freq, axis=[2, 3])
        Bo     = B * self.num_ofdm

        # ── Features enrichies ────────────────────────────────────────────
        feat = self._extract_features(h_sq, B)
        # [Bo, N, K, feat_dim]

        x = self.input_norm(self.input_proj(feat))
        # [Bo, N, K, D]

        # ── Positional embeddings ─────────────────────────────────────────
        sc_pos = tf.tile(tf.range(self.rb_size), [self.num_rb])
        x = x + tf.reshape(self.pos_embed_sc(sc_pos),
                            [1, self.fft_size, 1, self.embed_dim])

        # RB pos bias
        bias = tf.broadcast_to(
            tf.reshape(self.rb_pos_bias,
                [1, self.num_rb, self.rb_size, 1, self.embed_dim]),
            [Bo, self.num_rb, self.rb_size, self.num_rx, self.embed_dim])
        x = x + tf.reshape(bias, [Bo, self.fft_size, self.num_rx, self.embed_dim])

        # ── IntraRBBlocks ─────────────────────────────────────────────────
        shift = self.rb_size // 2
        for i, blk in enumerate(self.intra_blocks):
            do_shift = self.use_swin_shift and (i % 2 == 1)
            if do_shift:
                x = tf.roll(x, shift=shift, axis=1)
            x_rb = tf.reshape(x,
                [Bo * self.num_rb, self.rb_size, self.num_rx, self.embed_dim])
            x_rb = blk(x_rb, training=training)
            x = tf.reshape(x_rb, [Bo, self.fft_size, self.num_rx, self.embed_dim])
            if do_shift:
                x = tf.roll(x, shift=-shift, axis=1)

        # ── RB pooling : mean + max → Dense ──────────────────────────────
        x_rb = tf.reshape(x,
            [Bo * self.num_rb, self.rb_size, self.num_rx, self.embed_dim])
        rb_mean = tf.reduce_mean(x_rb, axis=1)   # [Bo*NR, K, D]
        rb_max  = tf.reduce_max(x_rb,  axis=1)   # [Bo*NR, K, D]
        rb_tokens_flat = self.rb_pool_proj(
            tf.concat([rb_mean, rb_max], axis=-1))  # [Bo*NR, K, D]

        rb_tokens = tf.reshape(rb_tokens_flat,
            [Bo, self.num_rb, self.num_rx, self.embed_dim])
        rb_tokens = rb_tokens + tf.reshape(
            self.pos_embed_rb(tf.range(self.num_rb)),
            [1, self.num_rb, 1, self.embed_dim])

        # ── InterRBBlocks ─────────────────────────────────────────────────
        for blk in self.inter_blocks:
            rb_tokens = blk(rb_tokens, training=training)

        # ── Gate additive : wideband → SC ────────────────────────────────
        # rb_tokens [Bo, NR, K, D] → gate [Bo, NR, K, D]
        gate = self.wideband_gate(rb_tokens)          # [Bo, NR, K, D]
        # broadcast sur les RS SC de chaque RB
        gate_bc = tf.repeat(gate, self.rb_size, axis=1)  # [Bo, N, K, D]
        x = x + self.s_cross * gate_bc
        # [Bo, N, K, D] — chaque SC enrichi du contexte RB, gradient stable

        # ── Joint output ──────────────────────────────────────────────────
        w_out = self.joint_output(
            tf.reshape(x, [Bo, self.fft_size, self.num_rx * self.embed_dim]))
        w_out = tf.transpose(
            tf.reshape(w_out,
                [B, self.num_ofdm, self.fft_size,
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