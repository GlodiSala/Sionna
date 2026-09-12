# =============================================================================
# precoders/base_intra_rb.py  —  RB-Local Transformer Precoder
# Architecture : IntraRB (SC-attention + User-attention) avec poids partagés
# Fréquence-agnostique : aucun poids ne dépend de N_RB
# Compatible Sionna MU-MIMO downlink
# =============================================================================

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


# =============================================================================
# 1. BLOC DE BASE — fidèle au labo (SwiGLU + pre-norm LayerNorm)
# =============================================================================

class SwiGLU(layers.Layer):
    """SwiGLU activation : split en 2, swish sur la première moitié."""
    def call(self, x):
        a, b = tf.split(x, 2, axis=-1)
        return (a * tf.sigmoid(a)) * b


class IntraRBBlock(layers.Layer):
    """
    Bloc d'attention séparable intra-RB.

    Séquence : SC-attention → User-attention → FFN SwiGLU
    Poids partagés entre tous les RBs (appelé N_RB fois sur le même tenseur).

    Input/output : [B_rb, S, K, D]
        B_rb = B × N_RB  (tous les RBs traités en parallèle)
        S    = sc_per_rb = 12
        K    = num_users = 4
        D    = embed_dim
    """

    def __init__(self, embed_dim=128, num_heads=4, ffn_mult=4,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        kd = embed_dim // num_heads

        # --- SC attention (chaque user voit ses S SCs) ---
        self.norm_sc  = layers.LayerNormalization(epsilon=1e-6)
        self.attn_sc  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            use_bias=True, dropout=dropout)

        # --- User attention (chaque SC voit les K users) ---
        self.norm_usr = layers.LayerNormalization(epsilon=1e-6)
        self.attn_usr = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            use_bias=True, dropout=dropout)

        # --- FFN SwiGLU ---
        self.norm_ffn = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_up   = layers.Dense(embed_dim * ffn_mult * 2)  # ×2 pour SwiGLU
        self.swiglu   = SwiGLU()
        self.ffn_out  = layers.Dense(embed_dim)

        # Scalaires résidus initialisés à 0 → apprentissage progressif
        self.s_sc  = self.add_weight(shape=(), dtype='float32',
                        initializer='zeros', trainable=True, name='s_sc')
        self.s_usr = self.add_weight(shape=(), dtype='float32',
                        initializer='zeros', trainable=True, name='s_usr')
        self.s_ffn = self.add_weight(shape=(), dtype='float32',
                        initializer='zeros', trainable=True, name='s_ffn')

    def call(self, x, training=False):
        """x : [B_rb, S, K, D]"""
        B_rb = tf.shape(x)[0]
        S    = tf.shape(x)[1]
        K    = tf.shape(x)[2]
        D    = tf.shape(x)[3]

        # ── SC attention : reshape pour que K soit le batch ──────────────
        # [B_rb, S, K, D] → [B_rb×K, S, D]
        xsc = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B_rb * K, S, D])
        xsc_n = self.norm_sc(xsc)
        xsc = xsc + self.s_sc * self.attn_sc(xsc_n, xsc_n, training=training)
        # → [B_rb, S, K, D]
        x = tf.transpose(tf.reshape(xsc, [B_rb, K, S, D]), [0, 2, 1, 3])

        # ── User attention : reshape pour que S soit le batch ────────────
        # [B_rb, S, K, D] → [B_rb×S, K, D]
        xu = tf.reshape(x, [B_rb * S, K, D])
        xu_n = self.norm_usr(xu)
        xu = xu + self.s_usr * self.attn_usr(xu_n, xu_n, training=training)
        x = tf.reshape(xu, [B_rb, S, K, D])

        # ── FFN SwiGLU ───────────────────────────────────────────────────
        xf = tf.reshape(x, [B_rb * S * K, D])
        xf_n = self.norm_ffn(xf)
        xf = xf + self.s_ffn * self.ffn_out(self.swiglu(self.ffn_up(xf_n)))
        return tf.reshape(xf, [B_rb, S, K, D])


# =============================================================================
# 2. MODÈLE PRINCIPAL
# =============================================================================

class IntraRBTransformerPrecoder(Model):
    """
    Précoder transformer MU-MIMO basé sur traitement intra-RB.

    Design :
    ─────────────────────────────────────────────────────────────────
    • Token = (SC, user) — S×K tokens par RB
    • SC-attention  : chaque user voit ses 12 SCs du RB
    • User-attention: chaque SC voit les 4 users simultanément  ← clé MUI
    • Poids partagés entre tous les N_RB Resource Blocks
    • Fréquence-agnostique : N_RB absent des poids du modèle
    • SNR-aware : log(no) concaténé aux features d'entrée
    • Features réduites (§1 ÉTAPE 1, SESSION_NUIT_RESUME.md) : re+im+abs+
      log_no par défaut (25 features à M=8), cos/sin retirés — même
      conclusion d'ablation que SingleSCTransformerPrecoder (Volet 2) :
      cos/sin n'apporte que ~8.7% du gain des features dérivées contre
      82.0% pour abs seul, pour un coût HLS 9x plus grand (2 divisions vs
      1 sqrt). use_cossin=True restaure l'ancien comportement (5M+1) pour
      comparaison si besoin.

    Input Sionna  : h_freq [B, K, 1, 1, M, ofdm, fft]
    Output Sionna : g      [B, 1, ofdm, fft, M, K]

    Paramètres :
        num_tx    M=8   antennes BS
        num_rx    K=4   users (single-antenna)
        num_ofdm  14    symboles OFDM par slot
        fft_size  96    taille FFT totale
        rb_size   12    SCs par RB (fixe 5G NR)
        pilot_idx 2     indice symbole pilote utilisé (pas de moyenne)
        embed_dim 128   dimension d'embedding D
        num_heads 4     têtes d'attention
        num_layers 4    nombre de IntraRBBlocks
        snr_aware True  injecter log(no) comme feature
        use_abs   True  feature |h| (canal complexe → magnitude)
        use_cossin False feature cos/sin(angle(h)) -- retiré par défaut
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        num_tx: int   = 8,
        num_rx: int   = 4,
        num_ofdm: int = 14,
        fft_size: int = 96,
        rb_size: int  = 12,
        pilot_idx: int = 2,      # symbole OFDM pilote — pas de moyenne
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_mult: int  = 4,
        dropout: float = 0.0,
        snr_aware: bool = True,
        use_abs: bool = True,
        use_cossin: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert fft_size % rb_size == 0, \
            f"fft_size {fft_size} doit être divisible par rb_size {rb_size}"

        self.M          = num_tx
        self.K          = num_rx
        self.num_ofdm   = num_ofdm
        self.fft_size   = fft_size
        self.rb_size    = rb_size        # S = 12
        self.N_RB       = fft_size // rb_size   # 8 pour 96 SCs
        self.pilot_idx  = pilot_idx
        self.D          = embed_dim
        self.snr_aware  = snr_aware
        self.use_abs    = use_abs
        self.use_cossin = use_cossin

        # ── Feature input dim ─────────────────────────────────────────────
        # Par (SC, user) : real(M) + imag(M) [+ abs(M)] [+ cos∠(M)+sin∠(M)]
        # + log(no) si snr_aware. Défaut (use_abs=True, use_cossin=False) :
        # 3M+1 = 25 features à M=8 (était 5M+1=41).
        self.feat_dim = (2 * num_tx
                          + (num_tx if use_abs else 0)
                          + (2 * num_tx if use_cossin else 0)
                          + (1 if snr_aware else 0))

        # ── Positional encoding intra-RB ──────────────────────────────────
        # S=12 positions fixes — N_RB n'apparaît PAS ici → agnostique
        self.pos_enc_sc = layers.Embedding(rb_size, embed_dim, name='pos_sc')

        # ── Embedding d'entrée ────────────────────────────────────────────
        self.input_embed = layers.Dense(embed_dim, name='input_embed')
        self.input_norm  = layers.LayerNormalization(epsilon=1e-6,
                                                      name='input_norm')

        # ── IntraRB blocks (poids partagés entre RBs) ─────────────────────
        self.blocks = [
            IntraRBBlock(embed_dim, num_heads, ffn_mult,
                         dropout, name=f'intra_rb_{i}')
            for i in range(num_layers)
        ]

        # ── Projection de sortie ──────────────────────────────────────────
        # [B_rb, S, K, D] → [B_rb, S, K, 2M]
        # On projette depuis D, pas K×D, parce que chaque (SC,user)
        # produit son propre vecteur de précoding
        self.output_proj = layers.Dense(2 * num_tx, name='output_proj')

        print(
            f"IntraRBTransformerPrecoder | "
            f"M={num_tx} K={num_rx} | "
            f"rb_size={rb_size} N_RB={self.N_RB} | "
            f"D={embed_dim} L={num_layers} H={num_heads} | "
            f"feat_dim={self.feat_dim} (abs={use_abs}, cos/sin={use_cossin}) | "
            f"snr_aware={snr_aware} | pilot_idx={pilot_idx}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_features(self, h, no=None):
        """
        Extrait les features canal par (SC, user).

        h  : [B, K, M, fft]   canal complexe — 1 symbole OFDM, tous SCs
        no : scalaire float32  variance du bruit (optionnel)

        Retourne : [B, N_RB, S, K, feat_dim]
        """
        B  = tf.shape(h)[0]
        K  = self.K
        M  = self.M
        S  = self.rb_size
        NR = self.N_RB

        # ── Réorganiser en RBs : [B, K, M, N_RB, S] ──────────────────────
        h_rb = tf.reshape(h, [B, K, M, NR, S])

        # ── Permuter pour mettre (N_RB, S) en axes de traitement ──────────
        # [B, N_RB, S, K, M]
        h_p = tf.transpose(h_rb, [0, 3, 4, 1, 2])

        # ── Features par (SC, user, antenne) → réduire sur M ─────────────
        h_re  = tf.math.real(h_p)                          # [B,NR,S,K,M]
        h_im  = tf.math.imag(h_p)
        parts = [h_re, h_im]

        if self.use_abs or self.use_cossin:
            h_abs = tf.sqrt(h_re**2 + h_im**2 + 1e-12)
        if self.use_abs:
            parts.append(h_abs)
        if self.use_cossin:
            parts.append(h_re / h_abs)                     # cos(angle)
            parts.append(h_im / h_abs)                     # sin(angle)

        # Concaténer : [B, N_RB, S, K, feat_dim - snr]
        feats = tf.concat(parts, axis=-1)

        # ── SNR feature ────────────────────────────────────────────────────
        if self.snr_aware and no is not None:
            no_f   = tf.cast(tf.reshape(no, []), tf.float32)
            log_no = tf.math.log(no_f + 1e-10) / tf.math.log(10.0)
            snr_bc = tf.fill([B, NR, S, K, 1], log_no)
            feats  = tf.concat([feats, snr_bc], axis=-1)  # [B,NR,S,K,5M+1]

        return feats   # [B, N_RB, S, K, feat_dim]

    # ─────────────────────────────────────────────────────────────────────────
    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        """
        h_freq : [B, K, 1, 1, M, ofdm, fft]  — format Sionna MU-MIMO
        no     : variance bruit scalaire (requis si snr_aware=True)
        return_real_imag : si True, retourne (w_re, w_im) séparément
                        pour éviter tf.complex() dans le GradientTape

        Retourne g : [B, 1, ofdm, fft, M, K]  — format Sionna précoder
        """
        B   = tf.shape(h_freq)[0]
        NR  = self.N_RB
        S   = self.rb_size
        K   = self.K
        M   = self.M
        D   = self.D

        # ── 1. Nettoyage dimensions Sionna ────────────────────────────────────
        h = tf.squeeze(h_freq, axis=[2, 3])          # [B, K, M, ofdm, fft]

        # ── 2. Sélection symbole pilote (pas de moyenne) ──────────────────────
        h_pilot = h[:, :, :, self.pilot_idx, :]      # [B, K, M, fft]

        # ── 3. Features ───────────────────────────────────────────────────────
        feats = self._extract_features(h_pilot, no)  # [B, NR, S, K, feat_dim]

        # ── 4. Embedding + norm ───────────────────────────────────────────────
        x = self.input_norm(self.input_embed(feats)) # [B, NR, S, K, D]

        # ── 5. Positional encoding intra-RB ──────────────────────────────────
        sc_idx = tf.range(S)
        pos    = self.pos_enc_sc(sc_idx)             # [S, D]
        pos    = tf.reshape(pos, [1, 1, S, 1, D])
        x      = x + pos

        # ── 6. IntraRB blocks en parallèle ───────────────────────────────────
        x = tf.reshape(x, [B * NR, S, K, D])
        for blk in self.blocks:
            x = blk(x, training=training)
        # x : [B*NR, S, K, D]

        # ── 7. Projection sortie ──────────────────────────────────────────────
        out = self.output_proj(x)                    # [B*NR, S, K, 2M]

        # ── 8. Split real/imag ────────────────────────────────────────────────
        out_r, out_i = tf.split(out, 2, axis=-1)     # [B*NR, S, K, M] chacun

        # ── 9. Transpose [B*NR, S, K, M] → [B*NR, S, M, K] ──────────────────
        out_r = tf.transpose(out_r, [0, 1, 3, 2])   # [B*NR, S, M, K]
        out_i = tf.transpose(out_i, [0, 1, 3, 2])

        # ── 10. Power norm per SC per user ────────────────────────────────────
        # norm sur dim M (axis=2), keepdims → [B*NR, S, 1, K]
        norm  = tf.sqrt(out_r**2 + out_i**2)
        norm  = tf.sqrt(tf.reduce_sum(norm**2, axis=2, keepdims=True) + 1e-12)
        out_r = out_r / norm
        out_i = out_i / norm

        # ── 11. Reshape → [B, fft, M, K] ─────────────────────────────────────
        out_r = tf.reshape(out_r, [B, self.fft_size, M, K])
        out_i = tf.reshape(out_i, [B, self.fft_size, M, K])

        # ── 12. Format Sionna [B, 1, ofdm, fft, M, K] ────────────────────────
        def to_sionna(t):
            t = tf.expand_dims(t, axis=1)            # [B, 1, fft, M, K]
            t = tf.expand_dims(t, axis=2)            # [B, 1, 1, fft, M, K]
            return tf.tile(t, [1, 1, self.num_ofdm, 1, 1, 1])

        # ── 13. Retour selon return_real_imag ─────────────────────────────────
        if return_real_imag:
            # Pour le GradientTape : évite tf.complex() qui bloque les gradients
            return to_sionna(out_r), to_sionna(out_i)

        w = tf.complex(out_r, out_i)
        return to_sionna(w)
    
    def complexity(self, num_ofdm: int = 14, convention_x_ofdm: bool = True):
        """
        Calcule FLOPs, Poids, Activations pour le modèle d'énergie du labo.

        Retourne (FLOPs, Weights, Activations) — même interface que V4/V5.

        Convention :
        - 1 MAC complexe = 2 MACs réels = 4 FLOPs réels
        - On suit la convention labo : df(n, i, o) = 2*n*i*o (real FLOPs)
        - num_ofdm : pour cohérence avec V4/V5 (le précoder tile sur ofdm)
            mais le forward réel n'itère pas sur ofdm → on multiplie en fin
            comme V4/V5 le font.
        - convention_x_ofdm=True (défaut) : ×num_ofdm, comparable au CSV
            historique. False : coût réel (1 seul symbole traité, fix OFDM).

        Architecture :
        feat_dim → Dense(D)             : input embedding
        pos_enc_sc                      : Embedding(S, D) — lookup, 0 FLOPs
        L × IntraRBBlock :
            SC-attention  [B×NR×K, S, D]
            User-attention [B×NR×S, K, D]
            FFN SwiGLU    [B×NR×S×K, D]
        Dense(D → 2M)                  : output projection
        """
        M   = self.M
        K   = self.K
        D   = self.D
        S   = self.rb_size       # 12
        NR  = self.N_RB          # 6
        L   = len(self.blocks)
        fd  = self.feat_dim      # 3M+1 = 25 par défaut (was 5M+1=41 pre-fix)
        ffn = 4                  # ffn_mult

        # ── Helpers (identiques à V4/V5) ─────────────────────────────────────────
        def df(n, i, o):
            """Dense layer FLOPs : 2*n*i*o (MACs×2)"""
            return 2.0 * n * i * o

        def dp(i, o):
            """Dense layer param count : i*o + o (weights + bias)"""
            return i * o + o

        def mha_flops(seq, d):
            """
            MHA FLOPs pour séquence de longueur seq et dim d.
            Q,K,V projections : 3 × 2*seq*d*d
            Scores QK^T        : 2*seq*seq*d
            Somme pondérée scores@V : 2*seq*seq*d -- MANQUAIT (fix audit
            énergie signed_attn, demande utilisateur) : seule la moitié du
            coût d'attention (QK^T) était comptée, pas scores@V. Incohérent
            avec precoders/base_residual.py::TransformerPrecoderClean.complexity()
            (helper `mf`), qui comptait déjà les deux (coefficient 4x, pas
            2x) -- alignement des deux fichiers sur la convention complète.
            Output projection : 2*seq*d*d
            Total             : 8*seq*d² + 4*seq²*d
            """
            return 8.0 * seq * d**2 + 4.0 * seq**2 * d

        def mha_params(d):
            """MHA params : 4 matrices d×d + 4 biais d → 4*(d²+d)"""
            return 4 * (d * d + d)

        def ffn_flops(n_tokens, d, mult):
            """
            SwiGLU FFN FLOPs :
            Dense(d → 2*mult*d) : 2*n*d*(2*mult*d)
            Dense(mult*d → d)   : 2*n*(mult*d)*d
            Total               : 2*n*d²*(2*mult + mult) = 6*mult*n*d²
            """
            return 2.0 * n_tokens * d * (2 * mult * d) + \
                2.0 * n_tokens * (mult * d) * d

        def ffn_params(d, mult):
            """
            Dense(d → 2*mult*d) + Dense(mult*d → d)
            = d*(2*mult*d) + 2*mult*d + (mult*d)*d + d
            = 2*mult*d² + 2*mult*d + mult*d² + d
            = 3*mult*d² + (2*mult+1)*d
            """
            return 3 * mult * d**2 + (2 * mult + 1) * d

        # ── Tokens par sample ────────────────────────────────────────────────────
        total_tokens = NR * S * K    # 6×12×4 = 288 tokens

        # =========================================================================
        # FLOPs
        # =========================================================================
        f = 0.0

        # 1. Input embedding : Dense(feat_dim → D) sur NR×S×K tokens
        f += df(total_tokens, fd, D)

        # 2. L × IntraRBBlock
        for _ in range(L):
            # SC-attention : NR blocks, chacun traite K séquences de longueur S
            # reshape [B×NR×K, S, D] → NR×K instances de MHA(seq=S)
            f += NR * K * mha_flops(S, D)

            # User-attention : NR blocks, chacun traite S séquences de longueur K
            # reshape [B×NR×S, K, D] → NR×S instances de MHA(seq=K)
            f += NR * S * mha_flops(K, D)

            # FFN SwiGLU sur NR×S×K tokens
            f += ffn_flops(total_tokens, D, ffn)

        # 3. Output projection : Dense(D → 2M) sur NR×S×K tokens
        f += df(total_tokens, D, 2 * M)

        # =========================================================================
        # Poids
        # =========================================================================
        w = 0.0

        # Pos encoding (Embedding lookup — paramètres mais 0 FLOPs)
        w += S * D

        # Input embedding
        w += dp(fd, D)

        # ÉTAPE 2 fix : input_norm (LayerNormalization après input_embed,
        # gamma+beta = 2D) manquait -- confirmé par comptage réel des
        # trainable_variables (écart exact de 128 = 2*64 à D=64).
        w += 2 * D

        # L × IntraRBBlock
        for _ in range(L):
            w += mha_params(D)   # SC-attention
            w += mha_params(D)   # User-attention
            w += ffn_params(D, ffn)
            # LayerNorms (3 par bloc : norm_sc, norm_usr, norm_ffn)
            w += 3 * 2 * D       # scale + bias par LayerNorm
            # Scalaires résidus (s_sc, s_usr, s_ffn)
            w += 3

        # Output projection
        w += dp(D, 2 * M)

        # =========================================================================
        # Activations (mémoire intermédiaire)
        # =========================================================================
        a = 0.0

        # Input embedding
        a += total_tokens * D

        # L × IntraRBBlock
        for _ in range(L):
            a += NR * K * S * D    # après SC-attention
            a += NR * S * K * D    # après User-attention
            a += total_tokens * D  # après FFN

        # Output projection
        a += total_tokens * 2 * M

        # =========================================================================
        # Multiplicateur num_ofdm (cohérence avec V4/V5), optionnel (ÉTAPE 2)
        # Le forward réel tile après → on multiplie pour la convention historique
        # =========================================================================
        mult = num_ofdm if convention_x_ofdm else 1
        return f * mult, w, a * mult


# =============================================================================
# 2bis. ARCHITECTURE SINGLE-SUBCARRIER — traitement par SC indépendant
# =============================================================================
# Volet 3 : architecture correspondant exactement à ce qui est aujourd'hui
# réellement synthétisé/déployé en HLS -- traitement single-subcarrier, donc
# PAS de SC-attention (pas de dimension fréquentielle dans le bloc), pas de
# regroupement par RB, pas de positional encoding (rien à positionner : les
# SC ne communiquent jamais entre eux dans ce design). L'architecture
# intra-RB (IntraRBTransformerPrecoder ci-dessus, avec SC-attention) reste
# la perspective future, positionnée comme telle -- ce bloc n'y touche pas.
#
# Features par défaut = jeu réduit validé au Volet 2 (ablation poids figés,
# checkpoint IntraRB_4L_128d/best_20260615_132353) : ABS capture 82.0% du
# gain apporté par les features dérivées (5M+1 -> 3M+1 : -3.45% seulement,
# vs -19.1% en le retirant) alors que COS/SIN n'en capture que 8.7%
# (4M+1 : -17.5%) -- COS/SIN retiré par défaut (use_cossin=False), ABS gardé
# (use_abs=True). Coût HLS : abs = 1 sqrt ; cos/sin = re/abs, im/abs, donc
# 2 divisions EN PLUS du sqrt déjà nécessaire pour abs -- pour un gain de
# feature 9x plus petit que celui d'abs, ne se justifie pas.
# =============================================================================

class SingleSCBlock(layers.Layer):
    """
    Bloc single-subcarrier : User-attention → FFN SwiGLU. PAS de
    SC-attention (aucune dimension fréquentielle dans le bloc -- chaque SC
    est traité indépendamment, poids partagés entre tous les SC comme entre
    tous les RB dans IntraRBBlock).

    Input/output : [N, K, D]
        N = nombre d'instances SC indépendantes traitées en parallèle
            (typiquement B × fft_size, ou B × num_ofdm × fft_size)
        K = num_users
        D = embed_dim
    """

    def __init__(self, embed_dim=128, num_heads=4, ffn_mult=4,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        kd = embed_dim // num_heads

        self.norm_usr = layers.LayerNormalization(epsilon=1e-6)
        self.attn_usr = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=kd,
            use_bias=True, dropout=dropout)

        self.norm_ffn = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_up   = layers.Dense(embed_dim * ffn_mult * 2)
        self.swiglu   = SwiGLU()
        self.ffn_out  = layers.Dense(embed_dim)

        self.s_usr = self.add_weight(shape=(), dtype='float32',
                        initializer='zeros', trainable=True, name='s_usr')
        self.s_ffn = self.add_weight(shape=(), dtype='float32',
                        initializer='zeros', trainable=True, name='s_ffn')

    def call(self, x, training=False):
        """x : [N, K, D]"""
        xn = self.norm_usr(x)
        x = x + self.s_usr * self.attn_usr(xn, xn, training=training)

        # FFN sur tenseur 2D aplati (N*K, D) -- comme IntraRBBlock. Appeler
        # Dense directement sur un tenseur 3D [N,K,D] force un contraction
        # (tensordot/einsum) que l'optimiseur XLA/Grappler de cette version
        # de TF fusionne mal avec MultiHeadAttention juste avant : la
        # rétropropagation matérialise un tenseur intermédiaire [N,D,4D]
        # (constaté : [24576,128,1024] ≈ 12.9GB pour B=256,fft=96 -> OOM,
        # qui se manifeste en pratique comme un crash CUDA_ERROR_ILLEGAL_
        # ADDRESS plutôt qu'un OOM propre). Aplatir avant les Dense évite
        # complètement ce chemin.
        N, K, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        xf = tf.reshape(self.norm_ffn(x), [N * K, D])
        xf = self.ffn_out(self.swiglu(self.ffn_up(xf)))
        x = x + self.s_ffn * tf.reshape(xf, [N, K, D])
        return x


class SingleSCTransformerPrecoder(Model):
    """
    Précoder transformer MU-MIMO à traitement single-subcarrier.

    Design (Volet 3, priorité voulue — correspond au déploiement HLS réel) :
    ─────────────────────────────────────────────────────────────────
    • Token = user — K tokens par SC, PAS de token SC (contrairement à
      IntraRBTransformerPrecoder où le token est (SC, user))
    • User-attention seule : chaque SC voit ses K users, indépendamment de
      tous les autres SC — aucune SC-attention, aucun regroupement RB,
      aucun positional encoding (rien à positionner)
    • Poids partagés entre tous les SC (et tous les OFDM symbols si
      appliqué à plusieurs) : fréquence-agnostique par construction, encore
      plus directement que IntraRB (pas de N_RB du tout dans le modèle)
    • Features réduites (Volet 2) : re, im, abs, log(no) par défaut — 3M+1
    • SNR-aware : log(no) concaténé aux features d'entrée
    • Normalisation de sortie : norme unité par (SC, user) — même
      convention que rzf_precoding_matrix (Volet 5), pour comparaison
      RZF/IntraRB à contrainte de puissance identique

    Input Sionna  : h_freq [B, K, 1, 1, M, ofdm, fft]
    Output Sionna : g      [B, 1, ofdm, fft, M, K]
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        num_tx: int   = 8,
        num_rx: int   = 4,
        num_ofdm: int = 14,
        fft_size: int = 96,
        pilot_idx: int = 2,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_mult: int  = 4,
        dropout: float = 0.0,
        snr_aware: bool = True,
        use_abs: bool = True,
        use_cossin: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.M          = num_tx
        self.K          = num_rx
        self.num_ofdm   = num_ofdm
        self.fft_size   = fft_size
        self.pilot_idx  = pilot_idx
        self.D          = embed_dim
        self.snr_aware  = snr_aware
        self.use_abs    = use_abs
        self.use_cossin = use_cossin

        # feat_dim : re(M) + im(M) [+ abs(M)] [+ cos(M)+sin(M)] [+ log_no(1)]
        self.feat_dim = (2 * num_tx
                          + (num_tx if use_abs else 0)
                          + (2 * num_tx if use_cossin else 0)
                          + (1 if snr_aware else 0))

        self.input_embed = layers.Dense(embed_dim, name='input_embed')
        self.input_norm  = layers.LayerNormalization(epsilon=1e-6,
                                                      name='input_norm')

        self.blocks = [
            SingleSCBlock(embed_dim, num_heads, ffn_mult,
                          dropout, name=f'single_sc_{i}')
            for i in range(num_layers)
        ]

        self.output_proj = layers.Dense(2 * num_tx, name='output_proj')

        print(
            f"SingleSCTransformerPrecoder | "
            f"M={num_tx} K={num_rx} | "
            f"D={embed_dim} L={num_layers} H={num_heads} | "
            f"feat_dim={self.feat_dim} (abs={use_abs}, cos/sin={use_cossin}) | "
            f"snr_aware={snr_aware} | pilot_idx={pilot_idx}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_features(self, h, no=None):
        """
        h  : [B, K, M, fft]   canal complexe — 1 symbole OFDM, tous SCs
        no : scalaire float32  variance du bruit (optionnel)

        Retourne : [B, fft, K, feat_dim]  (pas de dimension RB/S : chaque SC
        est déjà sa propre unité de traitement)
        """
        B = tf.shape(h)[0]
        K, M, F = self.K, self.M, self.fft_size

        # [B, K, M, fft] → [B, fft, K, M]
        h_p = tf.transpose(h, [0, 3, 1, 2])

        h_re = tf.math.real(h_p)                      # [B, fft, K, M]
        h_im = tf.math.imag(h_p)
        parts = [h_re, h_im]

        if self.use_abs or self.use_cossin:
            h_abs = tf.sqrt(h_re**2 + h_im**2 + 1e-12)
        if self.use_abs:
            parts.append(h_abs)
        if self.use_cossin:
            parts.append(h_re / h_abs)                 # cos
            parts.append(h_im / h_abs)                 # sin

        feats = tf.concat(parts, axis=-1)               # [B, fft, K, 2M(+M)(+2M)]

        if self.snr_aware and no is not None:
            no_f   = tf.cast(tf.reshape(no, []), tf.float32)
            log_no = tf.math.log(no_f + 1e-10) / tf.math.log(10.0)
            snr_bc = tf.fill([B, F, K, 1], log_no)
            feats  = tf.concat([feats, snr_bc], axis=-1)

        return feats   # [B, fft, K, feat_dim]

    # ─────────────────────────────────────────────────────────────────────────
    def call(self, h_freq, no=None, training=False, return_real_imag=False):
        """
        h_freq : [B, K, 1, 1, M, ofdm, fft]  — format Sionna MU-MIMO
        Retourne g : [B, 1, ofdm, fft, M, K]  — format Sionna précoder
        """
        B = tf.shape(h_freq)[0]
        K, M, D, F = self.K, self.M, self.D, self.fft_size

        h = tf.squeeze(h_freq, axis=[2, 3])            # [B, K, M, ofdm, fft]
        h_pilot = h[:, :, :, self.pilot_idx, :]         # [B, K, M, fft]

        feats = self._extract_features(h_pilot, no)     # [B, fft, K, feat_dim]
        x = self.input_norm(self.input_embed(feats))    # [B, fft, K, D]

        # Aucun positional encoding : chaque SC est une instance indépendante,
        # rien à positionner entre elles (pas de SC-attention qui en aurait besoin)
        x = tf.reshape(x, [B * F, K, D])
        for blk in self.blocks:
            x = blk(x, training=training)
        # x : [B*F, K, D]

        out = self.output_proj(x)                       # [B*F, K, 2M]
        out_r, out_i = tf.split(out, 2, axis=-1)         # [B*F, K, M] chacun

        out_r = tf.transpose(out_r, [0, 2, 1])           # [B*F, M, K]
        out_i = tf.transpose(out_i, [0, 2, 1])

        # Puissance unité par (SC, user) -- même convention que
        # rzf_precoding_matrix (Volet 5) : norme sur M (axis=1), par user (K)
        norm = tf.sqrt(out_r**2 + out_i**2)
        norm = tf.sqrt(tf.reduce_sum(norm**2, axis=1, keepdims=True) + 1e-12)
        out_r = out_r / norm
        out_i = out_i / norm

        out_r = tf.reshape(out_r, [B, F, M, K])
        out_i = tf.reshape(out_i, [B, F, M, K])

        def to_sionna(t):
            t = tf.expand_dims(t, axis=1)               # [B, 1, fft, M, K]
            t = tf.expand_dims(t, axis=2)                # [B, 1, 1, fft, M, K]
            return tf.tile(t, [1, 1, self.num_ofdm, 1, 1, 1])

        if return_real_imag:
            return to_sionna(out_r), to_sionna(out_i)

        w = tf.complex(out_r, out_i)
        return to_sionna(w)

    def complexity(self, num_ofdm: int = 14, convention_x_ofdm: bool = True):
        """FLOPs, Poids, Activations — même convention que IntraRB.complexity().

        Comme IntraRB, le forward réel ne traite qu'1 symbole OFDM (pilote)
        et tile en sortie (`to_sionna`) -- convention_x_ofdm=True (défaut)
        reproduit le chiffre "historique" ×num_ofdm pour comparabilité avec
        le CSV existant ; False donne le coût matériel réel (calcul unique)."""
        M, K, D, L = self.M, self.K, self.D, len(self.blocks)
        fd, ffn = self.feat_dim, 4

        def df(n, i, o): return 2.0 * n * i * o
        def dp(i, o): return i * o + o
        def mha_flops(seq, d): return 8.0 * seq * d**2 + 4.0 * seq**2 * d  # fix : scores@V manquant (cf. IntraRB.complexity())
        def mha_params(d): return 4 * (d * d + d)
        def ffn_flops(n_tok, d, mult):
            return 2.0 * n_tok * d * (2 * mult * d) + 2.0 * n_tok * (mult * d) * d
        def ffn_params(d, mult): return 3 * mult * d**2 + (2 * mult + 1) * d

        total_tokens = self.fft_size * K   # pas de N_RB, pas de S : juste fft x K

        f = df(total_tokens, fd, D)
        for _ in range(L):
            f += self.fft_size * mha_flops(K, D)   # une instance MHA(seq=K) par SC
            f += ffn_flops(total_tokens, D, ffn)
        f += df(total_tokens, D, 2 * M)

        # ÉTAPE 2 fix : manquait input_norm (LayerNormalization après
        # input_embed, gamma+beta = 2D) -- confirmé par comptage réel des
        # trainable_variables (écart exact de 128 = 2*64 à D=64).
        w = dp(fd, D) + 2 * D
        for _ in range(L):
            w += mha_params(D) + ffn_params(D, ffn) + 2 * 2 * D + 2   # 2 LN + 2 scalaires (pas de norm_sc/s_sc)
        w += dp(D, 2 * M)

        a = total_tokens * D
        for _ in range(L):
            a += total_tokens * D * 2   # après attention + après FFN
        a += total_tokens * 2 * M

        mult = num_ofdm if convention_x_ofdm else 1
        return f * mult, w, a * mult


# =============================================================================
# 3. HELPERS SIONNA — RZF et WMMSE inchangés
# =============================================================================

def _get_desired_channels(h_freq, stream_management):
    h_pc = tf.transpose(h_freq, [3, 1, 2, 4, 5, 6, 0])
    h_pc = tf.gather(h_pc, stream_management.precoding_ind,
                     axis=1, batch_dims=1)
    s    = tf.shape(h_pc)
    h_pc = tf.reshape(h_pc, [s[0], s[1]*s[2], s[3], s[4], s[5], s[6]])
    return tf.transpose(h_pc, [5, 0, 3, 4, 1, 2])


def rzf_precoder(h_freq, stream_management, no=None, alpha=None):
    from sionna.phy.mimo import rzf_precoding_matrix
    h_pc = _get_desired_channels(h_freq, stream_management)
    if alpha is None:
        alpha_val = tf.cast(no / 2, tf.float32)
    else:
        alpha_val = tf.cast(alpha, tf.float32)
    return rzf_precoding_matrix(h_pc, alpha=alpha_val)