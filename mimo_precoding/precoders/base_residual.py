"""
precoders/base_residual.py — Architectures "nettoyées" issues du diagnostic du
2026-08-07 :

TA-RB clean (TransformerPrecoderClean) :
  - Fix OFDM : traite 1 symbole représentatif (canal statique confirmé,
    corrélation temporelle = 1.0 sur les 14 symboles), tile en sortie.
    -> lève l'OOM (batch 32 -> 256+), ~14x moins de FLOPs/mémoire.
  - Décodeur v4.2 (Conv1DTranspose appris + refine features riches),
    SANS gate sigmoid v4.3 (gradient quasi nul, aucun gain mesuré
    en ablation, cf. diag_ta_rb_ablation.py).
  - Features de compression modulaires (mean/slope/var/cov) activables
    individuellement pour l'ablation -- cf. diag_feature_ablation.py.

Toutes les classes ci-dessous sont volontairement autonomes (ne
dépendent pas de precoders/classical.py) pour pouvoir itérer librement sans
perturber les scripts de diagnostic d'hier.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import RMSNormalization


class SeparableAttentionBlock(layers.Layer):
    """Identique à precoders/classical.py -- attention fréquence -> user -> FFN."""
    def __init__(self, num_tokens, num_users, embed_dim, num_heads,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.num_users = num_users
        self.embed_dim = embed_dim
        kd = embed_dim // num_heads
        self.norm_freq = RMSNormalization(epsilon=1e-6, name='norm_freq')
        self.norm_user = RMSNormalization(epsilon=1e-6, name='norm_user')
        self.norm_ffn = RMSNormalization(epsilon=1e-6, name='norm_ffn')
        self.freq_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd,
                                                     dropout=dropout, name='freq_attn')
        self.user_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd,
                                                     dropout=dropout, name='user_attn')
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ], name='ffn')
        self.s_freq = self.add_weight(name='s_freq', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_user = self.add_weight(name='s_user', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)
        self.s_ffn = self.add_weight(name='s_ffn', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        xf = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                        [B * self.num_users, self.num_tokens, self.embed_dim])
        xf = xf + self.s_freq * self.freq_attn(self.norm_freq(xf), self.norm_freq(xf), training=training)
        x = tf.transpose(tf.reshape(xf, [B, self.num_users, self.num_tokens, self.embed_dim]), [0, 2, 1, 3])
        xu = tf.reshape(x, [B * self.num_tokens, self.num_users, self.embed_dim])
        xu = xu + self.s_user * self.user_attn(self.norm_user(xu), self.norm_user(xu), training=training)
        x = tf.reshape(xu, [B, self.num_tokens, self.num_users, self.embed_dim])
        xp = tf.reshape(x, [B * self.num_tokens * self.num_users, self.embed_dim])
        xp = xp + self.s_ffn * self.ffn(self.norm_ffn(xp), training=training)
        return tf.reshape(xp, [B, self.num_tokens, self.num_users, self.embed_dim])


class TransformerPrecoderClean(Model):
    """
    TA-RB nettoyé : fix OFDM (1 symbole + tile) + décodeur v4.2 sans gate.

    feature_flags : dict optionnel {'mean':bool,'slope':bool,'var':bool,'cov':bool}
        pour l'ablation -- défaut = conclusion de l'ablation ÉTAPE 1
        (SESSION_NUIT_RESUME.md) : mean+var seulement (feat_dim 53->29 à
        M=8/K=4). slope et cov jugés dispensables (diag_v2_feature_
        ablation.py), mean indispensable, var borderline mais gardé.
        Passer {'mean':True,'slope':True,'var':True,'cov':True} pour
        restaurer l'architecture complète (comparaison/re-ablation).
        onehot user toujours présent (identité indispensable).
    """
    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4,
                 num_layers=4, dropout=0.0, feature_flags=None,
                 snr_aware=True, **kwargs):
        super().__init__(**kwargs)
        self.num_tx, self.num_rx = num_tx, num_rx
        self.num_ofdm = num_ofdm          # gardé pour l'interface Sionna (tile en sortie)
        self.fft_size, self.rb_size = fft_size, rb_size
        self.num_rb = fft_size // rb_size
        self.tokens_per_rb = tokens_per_rb
        self.sc_per_token = rb_size // tokens_per_rb
        self.total_tokens = self.num_rb * tokens_per_rb
        self.embed_dim = embed_dim
        self.snr_aware = snr_aware

        self.feature_flags = feature_flags or {'mean': True, 'slope': False, 'var': True, 'cov': False}
        ff = self.feature_flags
        # feat_dim : mean(2M) + slope(2M) + var(M) + cov(2K) + onehot(K) [+log_no]
        self.feat_dim = (2*num_tx if ff['mean'] else 0) + (2*num_tx if ff['slope'] else 0) \
                       + (num_tx if ff['var'] else 0) + (2*num_rx if ff['cov'] else 0) \
                       + num_rx + (1 if snr_aware else 0)
        assert self.feat_dim > num_rx, "au moins une feature canal doit être active"

        self.input_proj = layers.Dense(embed_dim, activation='gelu', name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')
        if snr_aware:
            self.snr_proj = layers.Dense(embed_dim, activation='gelu', name='snr_proj')
        self.pos_embed = layers.Embedding(self.total_tokens, embed_dim, name='pos_embed')
        self.blocks = [SeparableAttentionBlock(self.total_tokens, num_rx, embed_dim, num_heads,
                                                dropout=dropout, name=f'block_{i}')
                       for i in range(num_layers)]
        self.joint_output_proj = layers.Dense(embed_dim, name='joint_output_proj')
        self.upsample = layers.Conv1DTranspose(filters=embed_dim, kernel_size=self.sc_per_token,
                                                strides=self.sc_per_token, padding='valid',
                                                name='upsample_learned')
        self.final_proj = layers.Dense(2 * num_tx, name='final_proj')
        # décodeur v4.2 : refine sur [w_up, h_re, h_im, h_abs, h_phase]
        sc_in = embed_dim + num_tx * 4
        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(2 * embed_dim, kernel_size=rb_size, padding='same',
                          activation='gelu', name='sc_r1'),
            layers.Conv1D(2 * num_tx, kernel_size=1, padding='same', name='sc_r2'),
        ], name='sc_refine')
        self.alpha = self.add_weight(name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(0.1), trainable=True)

        print(f"TransformerPrecoderClean | T={tokens_per_rb} tok/RB | D={embed_dim} | "
              f"feat_dim={self.feat_dim} | features={ {k:v for k,v in ff.items()} } | "
              f"snr_aware={snr_aware} | 1-symbole-OFDM+tile (fix)")

    def _extract_features(self, h_pilot, B, log_no=None):
        """
        h_pilot : [B, rx, tx, fft]  -- UN SEUL symbole OFDM (canal statique)
        -> [B, T, rx, feat_dim]
        """
        T, S = self.total_tokens, self.sc_per_token
        ff = self.feature_flags

        h_tok = tf.reshape(h_pilot, [B, self.num_rx, self.num_tx, T, S])
        parts_canal = []
        canal_dim = 0

        def r2p(h):  # [B,rx,tx,T] -> [B,T,rx,tx]
            return tf.transpose(h, [0, 3, 1, 2])
        def c2r(h):  # complexe [B,rx,tx,T] -> réel [B,T,rx,2tx]
            p = r2p(h)
            return tf.concat([tf.math.real(p), tf.math.imag(p)], axis=-1)

        if ff['mean'] or ff['slope'] or ff['cov']:
            h_mean = tf.reduce_mean(h_tok, axis=-1)   # [B,rx,tx,T]
        if ff['mean']:
            parts_canal.append(c2r(h_mean)); canal_dim += 2*self.num_tx
        if ff['slope']:
            h_slope = h_tok[..., -1] - h_tok[..., 0]
            parts_canal.append(c2r(h_slope)); canal_dim += 2*self.num_tx
        if ff['var']:
            h_m = tf.reduce_mean(h_tok, axis=-1) if not (ff['mean'] or ff['slope'] or ff['cov']) else h_mean
            h_var = tf.math.log1p(tf.reduce_mean(tf.abs(h_tok - tf.expand_dims(h_m, -1))**2, axis=-1))
            parts_canal.append(r2p(h_var)); canal_dim += self.num_tx
        if ff['cov']:
            h_mean_p = tf.transpose(h_mean, [0, 3, 1, 2])  # [B,T,rx,tx]
            cov = tf.matmul(h_mean_p, h_mean_p, adjoint_b=True) / tf.cast(self.num_tx, h_mean_p.dtype)
            cov_feat = tf.concat([tf.math.real(cov), tf.math.imag(cov)], axis=-1)  # [B,T,rx,2rx]
            parts_canal.append(cov_feat); canal_dim += 2*self.num_rx

        feat_canal = tf.concat(parts_canal, axis=-1)  # [B,T,rx,canal_dim]
        f_mean = tf.reduce_mean(feat_canal, axis=-1, keepdims=True)
        f_std = tf.math.reduce_std(feat_canal, axis=-1, keepdims=True)
        feat_norm = (feat_canal - f_mean) / (f_std + 1e-6)

        user_onehot = tf.tile(tf.reshape(tf.eye(self.num_rx, dtype=tf.float32),
                                          [1, 1, self.num_rx, self.num_rx]), [B, T, 1, 1])
        out_parts = [feat_norm, user_onehot]
        if log_no is not None:
            out_parts.append(tf.fill([B, T, self.num_rx, 1], tf.cast(log_no, tf.float32)))
        return tf.concat(out_parts, axis=-1)

    # ── Complexité ────────────────────────────────────────────────────────────
    def complexity(self, num_ofdm=14, convention_x_ofdm=True):
        """
        FLOPs/Params/Activations.

        RÉVISION ÉTAPE 2 (SESSION_NUIT_RESUME.md) : la version précédente de
        cette méthode était un reste de l'ancien V4 (dont elle prétendait
        reprendre la convention) et ne correspondait plus à `call()` après
        le fix OFDM + décodeur v4.2 -- vérifié en comptant les
        trainable_variables réelles et en comparant terme à terme :
          - "output projection" utilisait Dense(K*D -> K*M*2) ; la couche
            réelle est `joint_output_proj` = Dense(D -> D), suivie
            séparément de `final_proj` = Dense(D -> 2M) (celle-ci n'était
            pas comptée du tout).
          - `upsample` (Conv1DTranspose) utilisait sc_out=M*K*2 comme
            largeur de canal ; la couche réelle a filters=embed_dim=D en
            entrée ET sortie (sc_out n'intervient nulle part dans
            Conv1DTranspose).
          - `sc_refine` utilisait sc_in=M*K*4 ; le vrai tenseur d'entrée
            (concat[w_up(D), h_re,h_im,h_abs,h_phase(M chacun)]) a D+4M
            canaux -- le "×K" dans l'ancien commentaire ne correspond à
            rien dans `call()`, qui traite les K users comme B*K instances
            indépendantes (pas de concat sur K).
          - poids : 3 RMSNorm (scale seul, D chacun) + 3 scalaires
            résiduels par bloc n'étaient pas comptés.
        Toutes ces couches (upsample/final_proj/sc_refine) sont appliquées
        indépendamment à chacune des K instances (B*K dans `call()`) -- FLOPs
        multipliés par K, poids NON multipliés (partagés). Formule revalidée
        en comparant le total à `sum(tf.size(v) for v in trainable_variables)`
        -- exact (308071 sur la config smoke M8K4/D64/L2/T3).

        convention_x_ofdm=True  : multiplie par num_ofdm à la fin (comme
            l'ancien V4.3, qui recalculait réellement 14x) -- comparable
            aux lignes CSV historiques Chapitre 3/4.
        convention_x_ofdm=False : coût RÉEL (calcul une seule fois par
            slot, fix OFDM) -- chiffre à utiliser pour le déploiement.
        """
        M, K = self.num_tx, self.num_rx
        T, D = self.total_tokens, self.embed_dim
        fd, N = self.feat_dim, self.fft_size
        S, RB = self.sc_per_token, self.rb_size

        def df(n, i, o): return 2.0 * n * i * o
        def dp(i, o): return i * o + o
        def mf(seq, d): return 8.0 * seq * d**2 + 4.0 * seq**2 * d
        def mp(d): return 4 * (d * d + d)
        def ff(n, d): return df(n, d, 4 * d) + df(n, 4 * d, d)
        def fp(d): return dp(d, 4 * d) + dp(4 * d, d)

        # ── input_proj + input_norm (RMSNorm, scale seul) + snr_proj + pos_embed ──
        f = df(T * K, fd, D)
        w = dp(fd, D) + D + (dp(1, D) if self.snr_aware else 0) + T * D
        a = T * K * D

        # ── L × SeparableAttentionBlock ──────────────────────────────────────
        for _ in self.blocks:
            f += K * mf(T, D) + T * mf(K, D) + ff(T * K, D)
            w += 3 * D + 3 + 2 * mp(D) + fp(D)   # 3 RMSNorm(D) + 3 scalaires + 2 MHA + FFN
            a += 3 * T * K * D

        # ── joint_output_proj : Dense(D -> D) sur T*K tokens ──────────────────
        f += df(T * K, D, D); w += dp(D, D); a += T * K * D

        # ── upsample (Conv1DTranspose, D->D, kernel=S) : K instances de N=T*S sorties ──
        f += K * 2.0 * N * S * D * D
        w += S * D * D + D
        a += K * N * D

        # ── final_proj : Dense(D -> 2M), K instances de N tokens ──────────────
        f += K * df(N, D, 2 * M); w += dp(D, 2 * M); a += K * N * 2 * M

        # ── sc_refine (décodeur v4.2, SANS gate) : K instances, sc_in=D+4M ─────
        sc_in, hidden, sc_out = D + 4 * M, 2 * D, 2 * M
        f += K * (2.0 * N * RB * sc_in * hidden + 2.0 * N * 1 * hidden * sc_out)
        w += RB * sc_in * hidden + hidden + 1 * hidden * sc_out + sc_out
        a += K * N * (hidden + sc_out)

        w += 1   # alpha (scalaire résiduel du refine)

        mult = num_ofdm if convention_x_ofdm else 1
        return f * mult, w, a * mult

    def call(self, h_freq, no=None, training=False, return_real_imag=False, pilot_idx=2):
        """
        h_freq : [B, rx, 1, 1, tx, ofdm, fft]  (interface Sionna standard)
        FIX OFDM : on ne prend qu'un symbole (canal statique confirmé),
        le reste de la pile ne voit jamais l'axe ofdm -> pas de x14 redondant.
        """
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        h_sq = tf.squeeze(h_freq, axis=[2, 3])          # [B,rx,tx,ofdm,fft]
        h_pilot = h_sq[:, :, :, pilot_idx, :]            # [B,rx,tx,fft]  <- FIX

        log_no = None
        if self.snr_aware:
            log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10) if no is not None else tf.constant(0.0)

        feat = self._extract_features(h_pilot, B, log_no)
        x = self.input_norm(self.input_proj(feat))
        if self.snr_aware:
            snr_emb = tf.reshape(self.snr_proj(tf.reshape(log_no, [1, 1])), [1, 1, 1, self.embed_dim])
            x = x + snr_emb
        x = x + tf.reshape(self.pos_embed(tf.range(self.total_tokens)), [1, self.total_tokens, 1, self.embed_dim])

        for blk in self.blocks:
            x = blk(x, training=training)

        x = self.joint_output_proj(x)  # [B,T,K,D]
        x_user = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * self.num_rx, self.total_tokens, self.embed_dim])
        w_up = self.upsample(x_user)                     # [B*K, fft, D]
        w_up_final = self.final_proj(w_up)                # [B*K, fft, 2M]

        h_user = tf.reshape(tf.transpose(h_pilot, [0, 3, 1, 2]), [B, self.fft_size, self.num_rx, self.num_tx])
        h_user = tf.reshape(tf.transpose(h_user, [0, 2, 1, 3]), [B * self.num_rx, self.fft_size, self.num_tx])
        h_re, h_im = tf.math.real(h_user), tf.math.imag(h_user)
        h_abs, h_phase = tf.abs(h_user), tf.math.angle(h_user)
        refine_input = tf.concat([w_up, h_re, h_im, h_abs, h_phase], axis=-1)
        delta = self.sc_refine(refine_input, training=training)
        w_final_flat = w_up_final + self.alpha * delta   # [B*K, fft, 2M]

        w_final = tf.reshape(w_final_flat, [B, self.num_rx, self.fft_size, self.num_tx, 2])
        w_final = tf.transpose(w_final, [0, 2, 3, 1, 4])   # [B, fft, M, K, 2]
        w_re, w_im = w_final[..., 0], w_final[..., 1]
        s = tf.sqrt(1.0 / (tf.reduce_sum(w_re**2 + w_im**2, axis=2, keepdims=True) + 1e-12))
        w_re, w_im = w_re * s, w_im * s   # [B, fft, M, K]

        def to_sionna(t):
            t = tf.expand_dims(t, axis=1)                    # [B,1,fft,M,K]
            t = tf.expand_dims(t, axis=2)                     # [B,1,1,fft,M,K]
            return tf.tile(t, [1, 1, self.num_ofdm, 1, 1, 1])  # FIX: tile en sortie

        if return_real_imag:
            return to_sionna(w_re), to_sionna(w_im)
        return to_sionna(tf.complex(w_re, w_im))


# =============================================================================
# TransformerPrecoderCleanResidual — décodeur résiduel (Partie 2,
# demande utilisateur 2026-08-07 18h00)
# =============================================================================
# Motivation (investigation "high_snr_disadvantage", même session) : le
# désavantage de TA-RB vs IntraRB grandit avec le SNR (+0.49 à 5dB ->
# +1.68 à 20dB) et est causalement attribué à la perte d'info FIXE de la
# compression T=6 -- pas à un problème de gradient (le gradient de TA-RB
# est le plus fort des 3 architectures à haut SNR, cf. même investigation).
# Le bénéfice de débruitage sous CSI imparfait (confirmé, gros effet,
# grandit avec le SNR données) vient du MOYENNAGE dans `_extract_features`
# (hérité tel quel ici, PAS touché) -- ce nouveau décodeur ne change QUE
# la reconstruction T tokens -> N sous-porteuses, pas l'extraction de
# features en amont.
#
# Ancien décodeur (TransformerPrecoderClean) : Conv1DTranspose APPRIS
# (upsample) + Dense (final_proj) reconstruisent les N sorties depuis
# zéro à partir des T tokens -- le réseau doit implicitement réapprendre
# que le canal varie lentement en fréquence (Partie 0 : rho intra-RB
# >0.8) EN PLUS d'apprendre la correction fine.
#
# Nouveau décodeur : Dense(D->2M) par token (`token_to_precoder`) donne
# une estimée de précodeur PAR TOKEN, étendue aux N SC par
# INTERPOLATION LINÉAIRE FIXE (tf.image.resize, AUCUN poids appris) --
# encode directement le bon prior physique sans le faire réapprendre.
# Le réseau (sc_refine, inchangé dans son principe) n'apprend plus QUE
# la correction résiduelle par rapport à cette base interpolée, avec les
# mêmes features per-SC riches (h_re/im/abs/phase) en entrée. Reste
# simple : une Dense en moins de paramètres que l'ancien Conv1DTranspose,
# un resize sans poids, sc_refine adapté à la nouvelle largeur d'entrée
# (6M au lieu de D+4M) -- pas de restructuration profonde.

class TransformerPrecoderCleanResidual(TransformerPrecoderClean):
    """Variante décodeur résiduel de TransformerPrecoderClean -- voir note
    ci-dessus. Hérite de `_extract_features` (feature extraction +
    moyennage cross-SC, INCHANGÉ) mais reconstruit __init__/call/
    complexity pour le nouveau décodeur (pas d'appel à
    TransformerPrecoderClean.__init__, pour éviter tout résidu de
    tracking Keras des couches upsample/final_proj qu'on ne veut plus)."""

    def __init__(self, num_tx=8, num_rx=4, num_ofdm=14, fft_size=96,
                 rb_size=12, tokens_per_rb=3, embed_dim=128, num_heads=4,
                 num_layers=4, dropout=0.0, feature_flags=None,
                 snr_aware=True, alpha_init=0.1, **kwargs):
        Model.__init__(self, **kwargs)   # bypass TransformerPrecoderClean.__init__
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

        # -- Partie partagée avec TransformerPrecoderClean (identique) --
        self.input_proj = layers.Dense(embed_dim, activation='gelu', name='input_proj')
        self.input_norm = RMSNormalization(epsilon=1e-6, name='input_norm')
        if snr_aware:
            self.snr_proj = layers.Dense(embed_dim, activation='gelu', name='snr_proj')
        self.pos_embed = layers.Embedding(self.total_tokens, embed_dim, name='pos_embed')
        self.blocks = [SeparableAttentionBlock(self.total_tokens, num_rx, embed_dim, num_heads,
                                                dropout=dropout, name=f'block_{i}')
                       for i in range(num_layers)]
        self.joint_output_proj = layers.Dense(embed_dim, name='joint_output_proj')

        # -- Nouveau décodeur résiduel --
        self.token_to_precoder = layers.Dense(2 * num_tx, name='token_to_precoder')
        sc_in  = 2 * num_tx + 4 * num_tx   # base interpolée (2M) + h_re/im/abs/phase (4M)
        hidden = 4 * num_tx                # 2x la largeur de sortie (2M), même ratio que l'original
        self.sc_refine = tf.keras.Sequential([
            layers.Conv1D(hidden, kernel_size=rb_size, padding='same',
                          activation='gelu', name='sc_r1_res'),
            layers.Conv1D(2 * num_tx, kernel_size=1, padding='same', name='sc_r2_res'),
        ], name='sc_refine_residual')
        # alpha_init : Priorité 2, piste "amplitude du résidu mal calibrée"
        # -- 0.1 (défaut, hérité de TransformerPrecoderClean) suppose que la
        # base interpolée est déjà proche de la solution ; si la base
        # interpolée est plus grossière que l'ancien décodeur appris, un
        # alpha_init plus grand pourrait laisser la correction contribuer
        # davantage dès le départ.
        self.alpha = self.add_weight(name='alpha', shape=(), dtype='float32',
            initializer=tf.keras.initializers.Constant(alpha_init), trainable=True)

        print(f"TransformerPrecoderCleanResidual | T={tokens_per_rb} tok/RB | D={embed_dim} | "
              f"feat_dim={self.feat_dim} | features={ {k:v for k,v in ff.items()} } | "
              f"snr_aware={snr_aware} | décodeur résiduel (interpolation fixe + correction apprise)")

    def call(self, h_freq, no=None, training=False, return_real_imag=False, pilot_idx=2):
        h_freq = tf.stop_gradient(h_freq)
        B = tf.shape(h_freq)[0]
        h_sq = tf.squeeze(h_freq, axis=[2, 3])
        h_pilot = h_sq[:, :, :, pilot_idx, :]

        log_no = None
        if self.snr_aware:
            log_no = tf.math.log(tf.cast(no, tf.float32) + 1e-10) if no is not None else tf.constant(0.0)

        feat = self._extract_features(h_pilot, B, log_no)   # hérité, inchangé (moyennage cross-SC)
        x = self.input_norm(self.input_proj(feat))
        if self.snr_aware:
            snr_emb = tf.reshape(self.snr_proj(tf.reshape(log_no, [1, 1])), [1, 1, 1, self.embed_dim])
            x = x + snr_emb
        x = x + tf.reshape(self.pos_embed(tf.range(self.total_tokens)), [1, self.total_tokens, 1, self.embed_dim])

        for blk in self.blocks:
            x = blk(x, training=training)

        x = self.joint_output_proj(x)   # [B,T,K,D]
        x_user = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * self.num_rx, self.total_tokens, self.embed_dim])

        # -- Base : précodeur par token, étendu par interpolation linéaire FIXE --
        token_precoder = self.token_to_precoder(x_user)          # [B*K, T, 2M]
        base = tf.image.resize(token_precoder[:, tf.newaxis, :, :],
                               size=[1, self.fft_size], method='bilinear')
        base = base[:, 0, :, :]                                   # [B*K, N, 2M]

        h_user = tf.reshape(tf.transpose(h_pilot, [0, 3, 1, 2]), [B, self.fft_size, self.num_rx, self.num_tx])
        h_user = tf.reshape(tf.transpose(h_user, [0, 2, 1, 3]), [B * self.num_rx, self.fft_size, self.num_tx])
        h_re, h_im = tf.math.real(h_user), tf.math.imag(h_user)
        h_abs, h_phase = tf.abs(h_user), tf.math.angle(h_user)
        refine_input = tf.concat([base, h_re, h_im, h_abs, h_phase], axis=-1)
        delta = self.sc_refine(refine_input, training=training)
        w_final_flat = base + self.alpha * delta                 # correction résiduelle sur la base interpolée

        w_final = tf.reshape(w_final_flat, [B, self.num_rx, self.fft_size, self.num_tx, 2])
        w_final = tf.transpose(w_final, [0, 2, 3, 1, 4])
        w_re, w_im = w_final[..., 0], w_final[..., 1]
        s = tf.sqrt(1.0 / (tf.reduce_sum(w_re**2 + w_im**2, axis=2, keepdims=True) + 1e-12))
        w_re, w_im = w_re * s, w_im * s

        def to_sionna(t):
            t = tf.expand_dims(t, axis=1)
            t = tf.expand_dims(t, axis=2)
            return tf.tile(t, [1, 1, self.num_ofdm, 1, 1, 1])

        if return_real_imag:
            return to_sionna(w_re), to_sionna(w_im)
        return to_sionna(tf.complex(w_re, w_im))

    def complexity(self, num_ofdm=14, convention_x_ofdm=True):
        """Revalidé contre trainable_variables (même méthode que ÉTAPE 2,
        SESSION_LOG_20260807.md) -- PAS juste réappliquée depuis le parent,
        le décodeur a changé."""
        M, K = self.num_tx, self.num_rx
        T, D = self.total_tokens, self.embed_dim
        fd, N = self.feat_dim, self.fft_size
        RB = self.rb_size

        def df(n, i, o): return 2.0 * n * i * o
        def dp(i, o): return i * o + o
        def mf(seq, d): return 8.0 * seq * d**2 + 4.0 * seq**2 * d
        def mp(d): return 4 * (d * d + d)
        def ff(n, d): return df(n, d, 4 * d) + df(n, 4 * d, d)
        def fp(d): return dp(d, 4 * d) + dp(4 * d, d)

        f = df(T * K, fd, D)
        w = dp(fd, D) + D + (dp(1, D) if self.snr_aware else 0) + T * D
        a = T * K * D

        for _ in self.blocks:
            f += K * mf(T, D) + T * mf(K, D) + ff(T * K, D)
            w += 3 * D + 3 + 2 * mp(D) + fp(D)
            a += 3 * T * K * D

        f += df(T * K, D, D); w += dp(D, D); a += T * K * D

        # token_to_precoder : Dense(D -> 2M), T*K tokens (remplace upsample+final_proj)
        f += df(T * K, D, 2 * M); w += dp(D, 2 * M); a += T * K * 2 * M

        # interpolation (tf.image.resize bilinéaire) : PAS de poids, FLOPs
        # négligeables (2 multiplications-additions par sortie -- pas
        # compté, cohérent avec pos_embed/interpolation jamais comptées
        # ailleurs dans ce fichier -- convention "lookup/interp = 0 FLOPs")

        # sc_refine (résiduel) : K instances, sc_in=6M
        sc_in, hidden, sc_out = 2 * M + 4 * M, 4 * M, 2 * M
        f += K * (2.0 * N * RB * sc_in * hidden + 2.0 * N * 1 * hidden * sc_out)
        w += RB * sc_in * hidden + hidden + 1 * hidden * sc_out + sc_out
        a += K * N * (hidden + sc_out)

        w += 1   # alpha

        mult = num_ofdm if convention_x_ofdm else 1
        return f * mult, w, a * mult
