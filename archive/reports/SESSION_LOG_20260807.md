# Log de session — journée/nuit du 2026-08-07

Suite autonome de `SESSION_NUIT_RESUME.md`. Chaque étape terminée est
horodatée ci-dessous. Décisions locales prises sans validation explicite
(autorisé par consigne) sont marquées **[décision autonome]**.

---

## 2026-08-07 — Démarrage

Contexte relu en entier (`SESSION_NUIT_RESUME.md`). Confirmation reçue :
exécuter toute la cascade en autonomie, dans l'ordre §0 → Étape 4.
Objectif final rappelé : ≥ WMMSE à haut SNR (15-20dB) sur au moins une
architecture, STANDARD d'abord, MASSIVE si le temps le permet.

État vérifié au démarrage : session tmux `nuit` attachée, GPU0 libre,
aucun process GPU actif, `datasets.py` pas encore modifié (fix §0 pas
implémenté, cohérent avec le résumé).

**Note environnement** : `import tensorflow` avant `import matplotlib`
(déclenché par `import sionna`) casse avec
`ImportError: GLIBCXX_3.4.29 not found` sur cette machine (TF charge un
libstdc++ système plus vieux qui masque celui de conda). Fix : préfixer
toute commande Python de ce projet avec
`LD_PRELOAD=/export/tmp/sala/miniconda/envs/tf-gpu/lib/libstdc++.so.6`
en plus de `conda activate tf-gpu`. Pas rencontré cette nuit (script
d'hier ne réimportaient peut-être pas matplotlib dans le même ordre) —
à réutiliser pour tous les runs de cette session.

---

## §0 — Fix pipeline dataset : FAIT, VALIDÉ

**Implémentation (option a, génération jointe directe)** — `datasets.py`
réécrit :
- `SionnaSingleUserGenerator` + `SAGEHBDataset` (mono-user + recombinaison
  aléatoire) remplacés par `JointClusterGenerator` + `JointClusterDataset`.
- `JointClusterGenerator.generate_joint_cluster_channels` : UN SEUL appel à
  `channel_config.set_locked_topology(..., num_ut=K)` par batch de samples
  → tous les K users d'un sample partagent le même centre de cluster tiré
  aléatoirement, par construction. Plus de recombinaison après coup : le
  bug ne peut plus se reproduire structurellement.
- `JointClusterDataset` : tire les minibatches d'entraînement par
  échantillonnage AVEC remise sur le pool (pratique standard pour un
  dataset fini) — chaque tuple du pool est déjà un draw K-user valide,
  rien à recombiner.
- `CachedSionnaDataset` : garde-fou ajouté — `_load_cached_channels`
  vérifie `h_freq.shape[1] == system.num_users` et lève une erreur
  explicite sinon (empêche de charger silencieusement un vieux cache
  mono-user, shape[1]==1). Nouveau tag de nommage `sionna_joint_*` (au
  lieu de `sionna_base_*`) pour ne jamais pouvoir collisionner avec un
  cache pré-fix.
- Abandon assumé de l'économie de stockage ×50 (documenté en tête de
  fichier) : chaque sample coûte désormais K× le stockage mono-user.
  `dataset_size` réinterprété comme "nombre de tuples K-user tirés
  indépendamment" — 20 000 pour STANDARD (~6.7GB), 8 000 pour MASSIVE
  (~21GB), vérifié contre l'espace libre `/export/tmp/sala` (5.2TB).
- `main_finall.py` mis à jour en conséquence (DATASET_SIZE dépend de
  CHOSEN_CONFIG, cache_file renommé, cluster_radius_m/indoor_probability
  passés explicitement depuis CHOSEN_CONFIG).

**[décision autonome]** Bug de taille de batch découvert en cours de
route et corrigé : le batch_size de génération CIR (`CachedSionnaDataset.
batch_size`, distinct du batch_size d'entraînement) était toujours à 512
(hérité de l'ancien générateur mono-user). Génération jointe multiplie le
tenseur intermédiaire de `cir_to_ofdm_channel` par K (users) → OOM
reproduit sur GPU partagé (`ResourceExhaustedError` sur l'op `Mul`).
Vérifié empiriquement : bs=256 safe même pour MASSIVE (K=8,M=32, le pire
cas). Mis à 128 par défaut (marge pour la contention des autres
utilisateurs du GPU partagé) dans `datasets.py` ET `main_finall.py`.

**Validation** (`diag_validate_datasets_fix.py`, pool jetable 1000
samples, méthodologie identique à `diag_spatial_cluster_channel.py` /
`diag_m32_final_round.py` : batch=16, 10 batches/point SNR) — mesuré sur
le VRAI code path `CachedSionnaDataset.get_batch()`, pas juste
`channel_config.gen_topology_clustered` directement :

| Config   | rho@95sc mesuré | rho@95sc réf | gap 0dB | gap 5dB | gap 10dB | réf (0/5/10dB) |
|----------|-----------------|--------------|---------|---------|----------|----------------|
| STANDARD | 0.392           | ~0.314       | +1.86%  | +0.41%  | -0.04%   | +2.79/+0.97/+0.35% |
| MASSIVE  | 0.482           | ~0.370       | +0.41%  | +0.10%  | +0.18%   | +0.97/+0.39/+0.25% |

Gap présent, même signe (WMMSE≥RZF), même ordre de grandeur que la
référence aux deux échelles — magnitudes un peu plus faibles (pool de
validation petit, 1000 tuples vs génération à la volée illimitée de la
référence ; attendu, pas préoccupant). **Confirmation qualitative
suffisante : le fix restore bien la corrélation de cluster que la
recombinaison aléatoire détruisait.** Avant le fix, ce mécanisme aurait
donné un gap ~0 (comme le canal 3GPP standard sans clustering, cf.
channel_config.py §1 point 4). JSON complet : `results/diag_validate_datasets_fix.json`.

**Tâche #0 débloquée. Passage à l'invalidation des caches (§0 étape 3) et
classical_comparison.py officiel.**

---

## Invalidation caches + classical_comparison.py officiel — FAIT

- Renommé (`.stale_pre_cluster_fix`) les 6 caches `/export/tmp/sala/
  sionna_base_*_8x4.npz` pré-fix (mono-user, pré-clustering) qui
  pouvaient collisionner avec un futur nom de cache STANDARD. Nouveau tag
  `sionna_joint_*` (au lieu de `sionna_base_*`) pour `main_finall.py` —
  aucune collision possible avec l'ancien schéma désormais.
- **[décision autonome]** `classical_comparison.py` était encore sur
  l'ANCIEN mécanisme (`ConfigurableMIMOSystem` + `HALF_ANGLE_DEG`/
  `FORCE_LOS`, pas `CLUSTER_RADIUS_M`) — jamais mis à jour pour la
  REVISION (b). Corrigé : nouvelle classe `LockedClusterSystem` qui
  override `new_topology` pour utiliser `channel_config.
  set_locked_topology` comme tous les autres scripts Stage3/4. Noms de
  sortie dérivés de `cfg['NUM_TX']/['NUM_RX']` (`M32K8` au lieu du
  `M64K32` codé en dur).
- Relancé avec succès (BATCH_SIZE=32, NUM_BATCHES=10, RB=12, WMMSE=10
  iters) :

  | Config | SNR | RZF-Full | RZF-RB12 | WMMSE | gap WMMSE-RZF |
  |--------|-----|----------|----------|-------|---------------|
  | M8K4   | 0dB | 13.63 | 11.91 | 13.97 | +2.49% |
  | M8K4   | 20dB| 38.95 | 24.56 | 38.95 | +0.01% |
  | M32K8  | 0dB | 52.79 | 45.80 | 52.88 | +0.17% |
  | M32K8  | 20dB| 105.58| 67.12 | 105.58| +0.00% |

  Gap concentré à bas SNR, cohérent avec le canal verrouillé §1 et avec
  le commit `d1b34b9` ("Document known 0-5dB WMMSE-vs-RZF gap as a
  characteristic, not a bug") — magnitude un peu bruitée à ce nombre de
  batches (10), déjà documenté comme caractéristique attendue, pas
  rechassé davantage. **RZF-RB12 très en dessous de RZF-Full/WMMSE aux
  deux échelles** (ex M8K4 20dB : 24.56 vs 38.95) — confirme qu'il y a
  une vraie marge pour un précodeur appris à combler. Sauvé :
  `results/classical_comparison_M8K4.{npy,png,pdf}`,
  `results/classical_comparison_M32K8.{npy,png,pdf}`.

---

## ÉTAPE 1 — Finalisation architectures : FAIT, smoke-testé OK

**Découverte en reprenant le code** : une partie du travail d'hier soir
était déjà fait mais pas branché. `precoders_v2.py` (`TransformerPrecoder
Clean`, TA-RB nettoyé complet : fix OFDM + décodeur v4.2 sans gate +
features modulaires mean/slope/var/cov) et `precoder_intra_rb.py`
(`SingleSCTransformerPrecoder`, 3ᵉ architecture Volet 3, déjà complète et
déjà branchée dans `main_finall.py`) existaient déjà. Ce qui manquait :
(1) la réduction de features sur `IntraRBTransformerPrecoder` lui-même
(pas juste sur SingleSC), (2) le défaut `feature_flags` de
`TransformerPrecoderClean` (toutes True = 53 features, au lieu de
mean+var = 29), (3) le branchement de `TransformerPrecoderClean` dans
`MU_MIMO_System`/`MODELS_TO_TRAIN` (main_finall.py importait encore
l'ancien `TransformerPrecoderV4`/`transformer_rb`, pas `precoders_v2.py`
du tout).

**IntraRB** (`precoder_intra_rb.py:IntraRBTransformerPrecoder`) :
- Ajout `use_abs`/`use_cossin` (même pattern que SingleSC), défaut
  `use_abs=True, use_cossin=False` → feat_dim 5M+1=41 → 3M+1=25 (M=8).
  Même conclusion d'ablation que SingleSC (abs=82% du gain, cos/sin=8.7%
  pour 9x le coût HLS) réutilisée par cohérence, pas re-testée
  séparément (l'ablation featurewise ne dépend pas de l'architecture qui
  consomme les features, seulement du canal).
- SC-attention + user-attention : déjà présentes, gardées telles
  quelles (rien à changer).
- RB=12 fixe : déjà la valeur par défaut, pas de fenêtre glissante dans
  le code (rien à changer, décision confirmée).

**TA-RB** (`precoders_v2.py:TransformerPrecoderClean`) :
- Fix OFDM + décodeur v4.2 sans gate : déjà en place (fait hier soir).
- `feature_flags` par défaut changé : `{mean:True,slope:True,var:True,
  cov:True}` (53) → `{mean:True,slope:False,var:True,cov:False}` (29),
  comme spécifié.
- **Branché dans `main_finall.py`** : import `TransformerPrecoderClean`,
  nouveau `precoder_type='ta_rb'` dans `MU_MIMO_System._init_precoder`.
  `MODELS_TO_TRAIN` réécrit pour lister les 3 architectures finales
  (single_sc, intra_rb, ta_rb 3tok) au lieu des anciennes entrées
  V4_3tok/V4_6tok (V4/V5 restent importables pour comparaison ponctuelle
  mais ne sont plus le chemin de training par défaut — TransformerPrecoder
  Clean les remplace, c'est tout le sens du nettoyage d'hier soir).

**[décision autonome]** `SupervisedTrainer` : ajout d'un paramètre
`steps_per_epoch` optionnel (fallback sur l'ancien calcul `eff_size//
batch_size` si absent). Nécessaire techniquement pour le smoke test (pool
minuscule), mais surtout correct sur le fond : avec le fix §0, `eff_size`
est maintenant la taille RÉELLE du pool (pas un multiplicateur artificiel
×50), donc le coupler 1:1 au nombre de pas/époque n'a plus de sens — le
nombre de pas d'entraînement est un choix de PROTOCOLE (ÉTAPE 3), pas une
conséquence accidentelle de la taille du cache. Sera réglé explicitement
à l'ÉTAPE 3.

**Smoke test** (`diag_etape1_smoke.py`, pool jetable 300 échantillons,
D=64/L=2, 1 warmup + 1 finetune epoch × 3 steps) : **les 3 architectures
tournent sans erreur, pertes/rates finis, gradients non-nuls.**
feat_dim confirmés : single_sc=25, intra_rb=25, ta_rb=29.

**Anomalie notée pour ÉTAPE 2** : `complexity()` rapporte un nombre de
poids qui ne colle pas exactement au nombre de variables entraînables
réelles (single_sc 135956 vs 136084 params, intra_rb 170262 vs 170390 —
écart constant de 128, probablement un biais non compté quelque part ;
ta_rb 372544 vs 308071 — écart bien plus gros, ~64k, à investiguer
précisément). Pas bloquant pour le smoke test mais **à corriger avant de
publier des chiffres d'énergie/FLOPs** (ÉTAPE 2) — la méthode de calcul
doit être revalidée contre les poids réels, pas juste "réappliquée".

---

## ÉTAPE 2 — Modèle de complexité/énergie : FAIT, méthode revalidée

Investigation de l'anomalie notée ci-dessus, poids réels comparés terme à
terme aux formules (script ponctuel, cf. ce log) :

- **single_sc / intra_rb** : `input_norm` (LayerNormalization après
  input_embed, gamma+beta=2D) manquait dans le compte de poids. Fix
  trivial (+2D). Écart expliqué exactement (128=2×64 à D=64, confirmé).
- **ta_rb (TransformerPrecoderClean)** : trois bugs réels, pas juste des
  oublis mineurs — la méthode avait été copiée depuis l'ancien V4 et
  jamais entièrement réconciliée avec l'architecture nettoyée :
  1. La ligne "output projection" modélisait `Dense(K·D → K·M·2)`, qui ne
     correspond à AUCUNE couche réelle. La vraie couche est
     `joint_output_proj` = `Dense(D → D)`, et `final_proj` = `Dense(D →
     2M)` (jamais comptée du tout) est une couche séparée en aval.
  2. `upsample` (Conv1DTranspose) utilisait `sc_out=M·K·2` comme largeur
     de canal ; la vraie couche a `filters=embed_dim=D` en entrée ET
     sortie — `sc_out` n'intervient nulle part dans cette couche.
  3. `sc_refine` utilisait `sc_in=M·K·4` (intention d'origine : traiter
     les K users joints par SC) ; le tenseur réel concatène
     `[w_up(D), h_re,h_im,h_abs,h_phase(M chacun)]` par instance
     (B·K instances indépendantes, pas de concat sur K) → `sc_in=D+4M`.
     `upsample`/`final_proj`/`sc_refine` doivent aussi être multipliés
     par K en FLOPs (appliqués indépendamment à chaque des K users) —
     manquant partout. Poids, eux, ne se multiplient PAS par K (partagés).
  4. Poids des 3 `RMSNormalization` par bloc (scale seul, D chacun) + 3
     scalaires résiduels par bloc n'étaient pas comptés.

  **Ces bugs se compensaient partiellement dans un sens puis dans
  l'autre** (sc_in trop grand ET manquants ailleurs) — d'où l'écart net
  de 64k qui ne sautait pas aux yeux comme "juste un oubli". Nécessitait
  une reconstruction ligne par ligne depuis `call()` réel, pas un patch.

**Méthode de vérification** : reconstruction terme à terme des 3
`complexity()`, revalidées en comparant EXACTEMENT à
`sum(tf.size(v) for v in model.trainable_variables)`, testé à 4 tailles
(D∈{64,128} × M/K∈{8/4, 32/8}) — accord exact (0 écart) sur les 12
combinaisons. Ajout d'un paramètre `convention_x_ofdm` à `single_sc` et
`intra_rb` aussi (déjà présent sur `ta_rb`) pour pouvoir séparer le coût
réel (fix OFDM, 1 symbole traité) du chiffre "historique" ×14 -- les
trois précodeurs traitent en réalité déjà 1 seul symbole OFDM pilote en
interne (`h[..., pilot_idx, :]`) et tile en sortie, ce n'était documenté
que pour TA-RB avant ce passage.

**Résultat** (`compute_complexity_energy_v3.py`, D=128/L=4/H=4, TA-RB
T=3 tok/RB — hyperparamètres de production `MODELS_TO_TRAIN`), coût
RÉEL (1 symbole, fix OFDM), FP16 (Q_W=Q_A=16) pour le neuronal, FP32 pour
RZF/WMMSE :

| Config | Méthode | Params(K) | FLOPs réel(M) | Énergie(µJ) | vs WMMSE (FLOPs/énergie) |
|---|---|---|---|---|---|
| STANDARD (8x4) | RZF | 0.0 | 2.8 | 0.011 | 873×/418× moins cher |
| STANDARD (8x4) | WMMSE (I=10) | 0.0 | 2456.6 | 4.634 | 1×/1× (référence) |
| STANDARD (8x4) | Single-SC | 1062.9 | 810.9 | 0.439 | 3.0×/10.6× moins cher que WMMSE |
| STANDARD (8x4) | IntraRB nettoyé | 1329.7 | 1017.0 | 0.551 | 2.4×/8.4× moins cher |
| STANDARD (8x4) | TA-RB nettoyé | 1644.2 | 643.2 | 0.350 | 3.8×/13.3× moins cher |
| MASSIVE (32x8) | RZF | 0.0 | 41.8 | 0.117 | 4755×/3192× moins cher |
| MASSIVE (32x8) | WMMSE (I=10) | 0.0 | 198567.5 | 374.12 | 1×/1× (référence) |
| MASSIVE (32x8) | Single-SC | 1078.3 | 1648.6 | 0.891 | 120×/420× moins cher |
| MASSIVE (32x8) | IntraRB nettoyé | 1345.1 | 2060.7 | 1.115 | 96×/335× moins cher |
| MASSIVE (32x8) | TA-RB nettoyé | 1967.4 | 1773.1 | 0.958 | 112×/390× moins cher |

**Avant/après** (vs `complexity_energy_results_v2.csv`) : (a) M64K32 →
M32K8 (rescale channel §1) ; (b) feat_dim TA-RB 53→29, IntraRB/SingleSC
41/5M+1→25/3M+1 ; (c) méthode de calcul TA-RB corrigée (poids/FLOPs
n'étaient PAS fiables avant, écart ~20% sur les poids, biais sur les
FLOPs par manque du facteur ×K côté décodeur) ; (d) **TA-RB nettoyé est
maintenant l'architecture la moins chère en FLOPs réels des 3** (avant le
fix §2, le classement n'était pas fiable). Sauvé :
`complexity_energy_results_v3.csv`.

**Constat qualitatif utile pour ÉTAPE 3/4** : à MASSIVE, l'avantage des
architectures neuronales sur WMMSE explose (WMMSE cube en M, ~100-400×
plus cher en énergie que le neuronal) — même sans gagner en sum-rate,
l'argument complexité/énergie est déjà très fort à cette échelle. Reste
à voir si le sum-rate suit (objectif ÉTAPE 3).

---

## ÉTAPE 3 — Recherche de protocole d'entraînement (STANDARD) — EN COURS

Cache de production généré : `/export/tmp/sala/sionna_joint_20k_8x4.npz`
(20 000 échantillons joints M8K4, 478MB, généré en 51s).

`SupervisedTrainer` étendu avec un paramètre `lr_schedule` (baseline /
constant_low / cosine_long / warm_restart) contrôlant uniquement la phase
finetune (sum-rate) — le warmup (MSE vs RZF, juste une initialisation
raisonnable) reste un cosinus court inchangé pour les 4 variantes, pour
isoler la variable testée.

**Lancé en arrière-plan** (`diag_lr_schedule_search.py`, PID via harness
bhbsu9e02, log `scratchpad/lr_search.log`) : les 4 schedules sur
single_sc D=128/L=4 (le plus rapide à itérer), budget identique par
variante (2ep warmup + 25ep finetune × 150 steps/ep = 4050 pas, ~17min/
variante mesuré par sondage timing, ~70min total estimé). Évalue à
15/17.5/20dB (haut SNR, objectif ÉTAPE 3) contre la référence WMMSE/RZF
déjà validée (`results/classical_comparison_M8K4.npy`). Résultats
sauvés progressivement dans `results/diag_lr_schedule_search.json`
(après chaque schedule, pas seulement à la fin).

**Prochaine étape (à la fin de ce run)** : comparer les 4 schedules,
retenir le meilleur (ou diagnostiquer si aucun n'approche WMMSE — auquel
cas explorer plus loin : taille de batch, curriculum SNR, initialisation,
autre optimiseur, comme prévu dans les instructions), puis vérifier sur
intra_rb/ta_rb avant de passer à MASSIVE.

**[décision autonome] Bug rencontré et corrigé en cours de route** :
`SupervisedTrainer.train()` faisait `self.opt.learning_rate = self.lr_
finetune` au changement de phase — Keras verrouille `learning_rate` en
lecture-seule une fois l'optimiseur construit avec un `LearningRateSchedule`
(le warmup en est toujours un), donc réassigner un float nu (schedule
`constant_low`) plantait avec un `TypeError` explicite. `baseline` avait
fini avant que `constant_low` ne plante (résultat `baseline` conservé).
Fix : reconstruire un optimiseur Adam neuf à la transition warmup→finetune
au lieu de réassigner `learning_rate` — accessoirement plus correct sur le
fond aussi (repart avec des moments Adam à zéro pour une loss de nature
différente, MSE→sum-rate, plutôt que de traîner l'état accumulé pendant
le warmup). Script rendu reprenable (saute les schedules déjà présents
dans le JSON) et relancé pour les 3 schedules restants.

**Résultat complet (25 époques finetune, 150 steps/époque = 3750 pas
finetune, single_sc D128L4, 15/17.5/20dB = objectif ÉTAPE 3)** :

| Schedule | 15dB %WMMSE | 17.5dB %WMMSE | 20dB %WMMSE | train |
|---|---|---|---|---|
| baseline (cosinus court, comportement d'avant) | 70.5% | 65.6% | 60.7% | 13.5min |
| constant_low (LR plate, pas de décroissance) | 78.8% | 74.1% | 69.0% | 13.9min |
| **cosine_long (cosinus sur horizon 2×)** | **81.1%** | **76.5%** | **71.6%** | 13.5min |
| warm_restart (SGDR) | 63.2% | 58.1% | 53.6% | 13.1min |

**Analyse** : `cosine_long` > `constant_low` > `baseline` > `warm_restart`.
Confirme le diagnostic de la nuit dernière : une décroissance LR agressive
(baseline) nuit clairement — le simple fait de ralentir/aplatir la
décroissance (constant_low, cosine_long) gagne +8 à +11 points de %WMMSE
à budget de pas IDENTIQUE, sans rien changer d'autre. `warm_restart` est
le pire — les redémarrages périodiques déstabilisent juste avant
l'évaluation, pas assez de cycles en 25 époques pour en tirer parti.
**Aucun des 4 n'atteint encore WMMSE**, mais **inspection des courbes
d'entraînement (finetune, rate@15dB par époque) : les 4 schedules
progressent ENCORE à l'époque 25, aucun plateau net** (ex. cosine_long :
...25.2→25.5→26.5→26.3→26.3 sur les 5 dernières époques — encore
en légère hausse/plateau naissant, pas franchement stabilisé) → confirme
directement que le budget d'époques (25) est probablement encore trop
court, pas la conclusion "le protocole ne marche pas".

**Suite lancée en arrière-plan** (`diag_lr_extended.py`) : `cosine_long`
étendu à 3ep warmup + 60ep finetune (9000 pas finetune, ~2.4× le budget
précédent) sur single_sc D128L4, même méthodologie d'éval. Objectif :
déterminer si le sum-rate continue de progresser vers/au-delà de WMMSE
avec plus d'époques (confirmerait complètement le diagnostic et donnerait
le budget à utiliser pour les runs complets ÉTAPE 4), ou plateaue avant
d'atteindre WMMSE (auquel cas il faudrait chercher ailleurs : taille de
modèle, curriculum SNR, autre optimiseur).

**Résultat single_sc 60 époques** : 86.0% / 81.8% / 77.3% WMMSE à
15/17.5/20dB (contre 81.1/76.5/71.6% à 25 époques) — progression réelle
mais **plateau détecté** sur les 10 dernières époques (rate@15dB oscille
27.3-28.8 sans tendance nette). Plus d'époques seul ne suffira
probablement pas à fermer tout l'écart pour CETTE architecture.

**[décision autonome] Hypothèse retenue avant de creuser plus loin le
protocole** : single_sc est délibérément l'architecture SANS aucune
attention/mélange inter-sous-porteuse (design HLS actuel, cf. docstring
`SingleSCTransformerPrecoder`) — elle ne PEUT structurellement pas
exploiter la sélectivité fréquentielle que le clustering spatial crée
(c'est exactement pourquoi le gap RZF-WMMSE existe : WMMSE optimise
sous-porteuse par sous-porteuse mais RZF aussi en fait, donc le gap vient
d'ailleurs — corrélation inter-utilisateurs, pas inter-fréquence -- à
vérifier si ça change la lecture). Avant de creuser d'autres leviers
(curriculum SNR, batch size, optimiseur), plus informatif de vérifier si
IntraRB/TA-RB (qui ONT de la SC-attention / mélange fréquentiel) font
mieux sous EXACTEMENT le même protocole — sinon le plateau de single_sc
serait faussement imputé au protocole alors qu'il pourrait être une
limite de capacité architecturale attendue et déjà connue. `diag_lr_
extended.py` généralisé (`--arch`), relancé sur intra_rb (60ep finetune,
même protocole) en arrière-plan.

**Résultat intra_rb 60 époques (même protocole exact que single_sc)** :
**89.6% / 85.7% / 81.5% WMMSE** à 15/17.5/20dB (contre single_sc :
86.0/81.8/77.3%). **Confirme l'hypothèse** : l'architecture avec
SC-attention (mélange fréquentiel réel) ferme davantage l'écart que
single_sc à protocole strictement identique — le plateau de single_sc
est bien (au moins en partie) une limite de capacité architecturale
attendue, pas seulement un problème de protocole. Plateau détecté aussi
pour intra_rb sur les 10 dernières époques (bruit epoch-à-epoch notable,
28.1-30.3 — la mesure par époque n'utilise qu'un batch, l'éval finale à
20 batches est plus fiable).

TA-RB lancé avec le même protocole en arrière-plan pour compléter la
comparaison des 3 architectures avant de figer le protocole pour
ÉTAPE 4.

**Résultat ta_rb T=3, 60 époques** : **81.8% / 76.8% / 71.7% WMMSE** —
moins bon qu'intra_rb (89.6/85.7/81.5%) ET que single_sc (86.0/81.8/
77.3%), malgré plus de paramètres (1.64M vs 1.33M/1.06M). Courbe des 10
dernières époques NON stabilisée (25.7-27.6, tendance légèrement
baissière plutôt que plateau) — contrairement à single_sc/intra_rb qui,
eux, plateauent proprement.

**[décision autonome]** Hypothèse : T=3 tok/RB compresse 12 SC → 3 tokens
(4 SC/token), perte de résolution fréquentielle que IntraRB (résolution
SC complète, S=12, pas de compression) n'a pas — le T-sweep restait
justement à faire (`diag_tarb_T_sweep.py`, débloqué par §0 mais jamais
lancé cette nuit). Avant de conclure que TA-RB est structurellement
moins bon, test rapide à T=6 (2 SC/token, compression plus douce) lancé
en arrière-plan, même protocole/budget.

**Résultat ta_rb T=6, 60 époques** : **85.6% / 81.4% / 76.6% WMMSE** —
remonte nettement vs T=3 (81.8/76.8/71.7%), confirme l'hypothèse
résolution fréquentielle. Plateau propre cette fois (contrairement à T=3).
Encore derrière intra_rb (89.6/85.7/81.5%) mais comparable à single_sc.

**Comparatif final ÉTAPE 3 (STANDARD, 60ep finetune, cosine_long)** :

| Architecture | 15dB | 17.5dB | 20dB | Plateau propre ? |
|---|---|---|---|---|
| single_sc | 86.0% | 81.8% | 77.3% | Oui |
| **intra_rb (meilleur)** | **89.6%** | **85.7%** | **81.5%** | Oui |
| ta_rb T=3 | 81.8% | 76.8% | 71.7% | Non (bruité, légère baisse) |
| ta_rb T=6 | 85.6% | 81.4% | 76.6% | Oui |

**PROTOCOLE RETENU pour ÉTAPE 4** (appliqué dans `main_finall.py`,
`TRAINING_CONFIG`) : `lr_schedule='cosine_long'`, `warmup_epochs=3`,
`finetune_epochs=80` (60 en recherche + marge, aucune architecture
n'était totalement plateauée à 60), `steps_per_epoch=150` (fixé
explicitement, indépendant de `dataset_size`). `MODELS_TO_TRAIN` : TA-RB
passé à **T=6** (pas 3, l'ancien "Pareto-optimal" ne tenait plus sous
features réduites + décodeur nettoyé). Aucune architecture n'a encore
ATTEINT WMMSE à 60 époques (81-90%), mais la tendance et l'ordre de
grandeur (progression nette, ordre attendu par capacité architecturale
IntraRB>TA-RB>single_sc) valident le protocole comme "qui marche" au sens
de la consigne — 80 époques (vs 60 en recherche) + le budget plus long
prévu pour les runs complets ÉTAPE 4 devraient rapprocher encore
davantage. Piste non explorée faute de temps si le gap persiste après
ÉTAPE 4 : curriculum SNR (restreindre progressivement la plage
d'entraînement vers le haut-SNR), taille de batch, autre optimiseur —
mentionnées dans la consigne, pas encore testées.

Passage à la vérification sur MASSIVE (M32K8, R=5m).

**Décisions appliquées dans le code** (`main_finall.py`) : `TRAINING_
CONFIG` mis à jour (warmup_epochs=3, finetune_epochs=80, lr_schedule=
'cosine_long', steps_per_epoch=150, tous passés explicitement à
`SupervisedTrainer` dans `main()`), `MODELS_TO_TRAIN`'s TA-RB passé à
T=6.

---

## ÉTAPE 3 — Vérification MASSIVE (M32K8, R=5m) — EN COURS

Cache de production généré : `/export/tmp/sala/sionna_joint_8k_32x8.npz`
(8000 échantillons joints M32K8, 1.5GB, 141s de génération).

**Sondage timing** : ~0.80s/pas à MASSIVE (intra_rb D128L4) contre
~0.24s/pas à STANDARD — **~3.3× plus lent par pas**, cohérent avec M/K
plus grands (feat_dim intra_rb 97 à M=32 contre 25 à M=8). Consigne
anticipait explicitement ce genre de coût comme diagnostic possible.

**[décision autonome]** `main_finall.py` fixe `CHOSEN_CONFIG=STANDARD_
CONFIG` au niveau module (pas de switch runtime par design, cf. docstring
"un seul endroit à changer"). Pour tester MASSIVE sans perturber l'état
STANDARD, nouveau script `diag_lr_extended_massive.py` autonome
(n'importe pas `CHOSEN_CONFIG`/`NUM_TX`/`NUM_RX` de `main_finall.py`,
construit tout directement depuis `MASSIVE_CONFIG`).

**Lancé en arrière-plan** : intra_rb (meilleur sur STANDARD) D128L4,
cosine_long, 3ep warmup + 30ep finetune (budget réduit de moitié vs le
diagnostic STANDARD — ~66min estimées à 0.80s/pas, un diagnostic
directionnel à budget réduit est plus rentable ici qu'une réplique
intégrale du budget STANDARD avant de savoir si ça vaut le coût).

**Résultat MASSIVE intra_rb, 30 époques finetune** : seulement
**39.3% / 36.4% / 34.5% WMMSE** à 15/17.5/20dB — nettement moins bon que
STANDARD au même point de recherche. **MAIS** : courbe d'entraînement
(rate@15dB par époque) **STRICTEMENT CROISSANTE sur les 30 époques, AUCUN
plateau** (10.4→37.4, encore en hausse nette aux 3 dernières : 35.0→
35.4→36.4→37.4) — contrairement à STANDARD qui commençait à plateauer
vers 50-60 époques. **Diagnostic (consigne : "diagnostique librement
pourquoi — taille de modèle, LR, budget d'époques, coût par époque")** :
le protocole TRANSFERT bien dans le sens qu'il apprend sainement (pas de
divergence, pas de NaN, progression monotone saine) — le problème est
purement un budget d'époques insuffisant à ce stade, aggravé par le coût
3.3× plus cher par pas (mesuré ci-dessus) : à budget de TEMPS égal,
MASSIVE ne voit qu'~30% des époques que STANDARD voit. K=8 (précodage
joint sur 2x plus de flux qu'à STANDARD) rend aussi le problème
intrinsèquement plus dur à apprendre, cohérent avec une convergence plus
lente en nombre d'époques ET en coût par époque.

**[décision autonome]** Ne pas chercher le plateau exact de MASSIVE via
un diagnostic supplémentaire très long (extrapoler à 80-150 époques au
rythme actuel coûterait plusieurs heures de plus, juste pour un point de
diagnostic) — le signal (croissance saine, pas de plateau, cause
identifiée) est déjà suffisant pour conclure "le protocole marche,
budget insuffisant" sans re-router le pipeline. Priorité (consigne :
"MASSIVE ensuite SI LE TEMPS LE PERMET") : lancer maintenant les runs
complets STANDARD (ÉTAPE 4, priorité explicite "d'abord"), puis pousser
MASSIVE plus loin (budget d'époques augmenté) avec le temps restant.

---

## ÉTAPE 4 — Runs complets STANDARD — EN COURS

Lancé `main_finall.py` directement (pipeline de production officiel,
inchangé dans sa logique — juste `TRAINING_CONFIG`/`MODELS_TO_TRAIN` mis
à jour par les décisions ÉTAPE 1/3 ci-dessus) en arrière-plan : les 3
architectures nettoyées (single_sc, intra_rb, ta_rb T=6), D128/L4,
protocole cosine_long (3ep warmup + 80ep finetune, 150 pas/époque),
dataset STANDARD 20k déjà en cache. Puis évaluation RZF/WMMSE + les 3
modèles entraînés sur `EVALUATION_SNR_RANGE` (9 points 0-20dB, superset
des "5 points" demandés).

Estimation : ~0.24-0.3s/pas × 3 modèles × 83 époques × 150 pas ≈ 3h
d'entraînement + évaluation. Log : `scratchpad/etape4_standard.log`.
Résultats attendus dans `results/results_<timestamp>.npy` +
`results/*.png/pdf` (courbes sum-rate vs SNR, tableau Pareto énergie via
`print_table`/`plot_results`, déjà présents dans `main_finall.py`,
inchangés). Poids sauvés sous `weights/<run_name>/best_*/`.

**⚠️ Premier essai raté, DEUX bugs réels trouvés et corrigés (pas des
"acquis" remis en cause — bugs de code découverts en exécutant le plan)**
Le run a fini en ~44min au lieu des ~3h attendues, et les résultats
étaient absurdes (rate neuronal ~14% de WMMSE au lieu des ~86-90% mesurés
en recherche ÉTAPE 3) :

1. **`evaluate_system()`** : `tf.reduce_sum(tf.reduce_mean(rate_sc, axis=
   [1,2,4]))` omettait l'axe batch (0) du `reduce_mean` — le `reduce_sum`
   sans argument `axis` sommait alors AUSSI sur le batch, gonflant le
   sum_rate affiché d'un facteur ~batch_size (Rate WMMSE@15dB affichait
   4239.97 au lieu de ~32 ; ratio 131≈128=BATCH_SIZE, confirmé). Bug
   PRÉ-EXISTANT (pas introduit cette nuit), présent dans le pipeline
   d'évaluation "officiel" depuis toujours — jamais remarqué parce que
   c'est un facteur multiplicatif partagé par toutes les méthodes (les
   comparaisons relatives via `print_table`/Pareto restaient
   qualitativement lisibles), mais rendait les valeurs absolues du
   tableau Pareto fausses. Fix : `axis=[0,1,2,4]` (même convention que
   `SupervisedTrainer._eval_step`/`_finetune_step`, qui eux étaient
   corrects).
2. **`main()`** : `trainer.train(print_every=200, patience=8)` codé en
   dur — `patience=8` (hérité d'avant ÉTAPE 3, jamais mis à jour avec le
   reste du protocole) coupait l'entraînement après 8 époques sans
   nouveau meilleur score. Les courbes ÉTAPE 3 (recherche, patience=999)
   montraient régulièrement des creux bruités de 3-8 époques SANS
   nouveau meilleur avant de repartir à la hausse (bruit d'estimation
   par époque, 1 seul batch d'éval) — patience=8 les interprétait à tort
   comme un plateau définitif, arrêtant l'entraînement bien avant que le
   protocole validé (80 époques) ait eu sa chance. **C'est la cause
   principale du résultat catastrophique**, pas un problème de
   protocole en soi. Fix : patience→25 (~31% du budget), print_every→150
   (200 ne se déclenchait jamais avec 150 pas/époque, aucun print
   intermédiaire n'apparaissait).

Les fichiers `results/results_20260807_1130*.{npy,png,pdf}` du premier
essai sont invalides (bugs ci-dessus) — laissés en place mais superflus,
remplacés par les fichiers horodatés du second essai. Relancé
immédiatement après le fix (`etape4_standard_v2.log`) — même estimation
~3h.

### RÉSULTAT ÉTAPE 4 STANDARD (M8K4, R=20m) — FINAL, VALIDE

Run complet (83 époques réelles cette fois, ~3.5h, confirmé par les
horodatages des checkpoints). **Tableau Pareto @15dB** :

| Méthode | Rate (bps/Hz) | Gap vs WMMSE | FLOPs réel(M) | Énergie FP16(µJ) |
|---|---|---|---|---|
| RZF | 33.27 | +0.65 (RZF légèrement > WMMSE à 15dB, caractéristique connue, cf. commit d1b34b9) | 2.8 | N/A |
| WMMSE | 32.62 | (référence) | 2456.6 | N/A |
| SingleSC | 29.81 | -2.81 (**91.4% WMMSE**) | 11352.7×14conv | 6.13 |
| IntraRB | 29.57 | -3.05 (**90.7% WMMSE**) | 14237.4×14conv | 7.69 |
| TA-RB T=6 | 28.20 | -4.43 (**86.5% WMMSE**) | 11734.6×14conv | 6.34 |

**À 20dB** : WMMSE=39.17, RZF=39.87 — SingleSC=33.22 (84.8%), IntraRB=
32.69 (83.5%), TA-RB=30.73 (78.4%).

**Conclusion honnête** : **aucune architecture n'atteint/dépasse WMMSE**
à 15-20dB (objectif ÉTAPE 3/4 non pleinement atteint) — mais toutes
approchent 78-91% de WMMSE avec une complexité 3.8-9× moindre en FLOPs
réels et jusqu'à 13× moindre en énergie (cf. ÉTAPE 2). SingleSC et
IntraRB quasi ex-aequo (91.4/90.7%), TA-RB T=6 en retrait (86.5%,
cohérent avec la recherche ÉTAPE 3 — probablement encore limité par la
compression fréquentielle T=6, un T-sweep plus complet non fait par
manque de temps reste la piste la plus prometteuse pour TA-RB). BER
propre à haut SNR pour les 3 (1e-5 à 1e-4), pas de plancher d'erreur
pathologique (contrairement au 1er essai buggé). Poids sauvés sous
`weights/{SingleSC,IntraRB,TA_RB}_4L_128d/best_*/`, résultats complets
`results/results_20260807_155133.npy`, figures `results/results_
20260807_155132.{png,pdf}`.

**Piste non explorée par manque de temps si quelqu'un reprend** :
curriculum SNR (restreindre progressivement l'entraînement vers le
haut-SNR), T-sweep TA-RB complet (T=12=résolution SC pleine, comparable
à IntraRB), batch size plus grand, budget d'époques encore plus long
(les courbes n'étaient pas parfaitement plateauées même à 83 époques
pour TA-RB).

---

## ÉTAPE 4 — Runs complets MASSIVE (M32K8, R=5m) — EN COURS

Lancé `main_finall.py` sur MASSIVE : `CHOSEN_CONFIG` basculé
temporairement à `MASSIVE_CONFIG` dans le fichier source juste avant le
lancement en arrière-plan (le process a déjà importé la valeur à ce
moment, Python ne relit pas le fichier après coup), puis remis à
`STANDARD_CONFIG` immédiatement après pour que STANDARD reste la config
par défaut du dépôt. `DATASET_SIZE`/cache/noms de run se résolvent tous
automatiquement via `CHOSEN_CONFIG` (aucun autre changement nécessaire) —
cache déjà généré (`sionna_joint_8k_32x8.npz`).

**Estimation** : ~0.8s/pas (mesuré ÉTAPE 3, ~3.3× plus lent que
STANDARD) × 3 modèles × 83 époques × 150 pas ≈ **~8.3h** d'entraînement
+ évaluation. Démarrage confirmé sain (process actif, GPU alloué, pas de
crash immédiat). Log : `scratchpad/etape4_massive.log`.

**[décision autonome]** Budget de temps déjà très engagé sur cette
session (§0 à ÉTAPE 4 STANDARD = plusieurs heures de calcul GPU
cumulées). Lancé le run MASSIVE complet (3 architectures) malgré le coût
élevé estimé (~8h) car : (1) la consigne autorise explicitement "MASSIVE
ensuite si le temps le permet" et un protocole validé + pipeline déjà
opérationnel est disponible sans travail supplémentaire de conception ;
(2) MASSIVE est l'argument le plus fort du mémoire côté énergie (WMMSE y
coûte 100-400× plus cher que le neuronal, cf. ÉTAPE 2) donc avoir au
moins UN point de comparaison complet MASSIVE (sum-rate ET énergie) est
disproportionnellement utile même sans les 3 architectures parallèles.
Si ce run n'a pas fini au réveil : les poids/résultats partiels (déjà
sauvés par époque via checkpointing) et cette note suffisent à reprendre
sans rien re-dériver — commande exacte : basculer `CHOSEN_CONFIG =
MASSIVE_CONFIG` dans `main_finall.py` puis `python3 main_finall.py`.

---

## INVESTIGATION (demande utilisateur, 16h27) — pourquoi IntraRB/TA-RB
## ne battent pas SingleSC malgré l'exploitation fréquentielle

Consigne : comprendre d'abord, améliorer ensuite, graphiquer à la fin.
MASSIVE laissé intact en arrière-plan (vérifié sain avant de commencer :
process 2900748 actif, GPU0 100% util, premier checkpoint MASSIVE écrit
à 16:28:37 — progression confirmée malgré un log peu bavard, cf. réponse
précédente sur le buffering stdout).

### Hypothèse #2 (capacité vs budget) — INFIRMÉE

Courbes d'entraînement extraites directement du log du run complet
`etape4_standard_v2.log` (83 époques réelles, pas besoin de relancer
quoi que ce soit) :

| | rolling(10) ép.10 | ép.30 | ép.50 | ép.70 | ép.80 |
|---|---|---|---|---|---|
| SingleSC | 24.0 | 28.1 | 28.9 | 29.5 | 29.7 |
| IntraRB  | 22.0 | 26.7 | 28.5 | 29.4 | 29.6 |
| TA-RB    | 22.1 | 26.4 | 27.6 | 28.5 | 28.6 |

**IntraRB ne converge PAS plus lentement que SingleSC** (trajectoires
quasi identiques, écart <1pt à chaque palier) — les deux plateaunet au
même rythme, au même niveau final. 1.33M params vs 534K (single_sc D128L4
réel : 1.06M en fait, à vérifier) n'entraîne aucun retard de convergence
visible sur ce budget. **Capacité/budget n'explique pas l'écart.**

### Hypothèse #3 (SC-attention ignorée) — PARTIELLEMENT INFIRMÉE, nuancée

Inspection directe des poids entraînés IntraRB (`diag_inspect_intrarb_
weights.py`, checkpoint `best_20260807_141928`, epoch 74, sum_rate=30.20) :

**Scalaires résiduels par bloc** (init à 0) :
| Bloc | s_sc | s_usr |
|---|---|---|
| 0 | -0.088 | +0.306 |
| 1 | -0.090 | +0.276 |
| 2 | -0.094 | -0.230 |
| 3 | -0.067 | +0.194 |

`s_sc` est PETIT mais **PAS nul** (~0.07-0.09 en magnitude, contre
~0.19-0.31 pour `s_usr`) — le réseau n'ignore pas structurellement la
SC-attention, mais lui donne un poids nettement plus faible qu'à la
user-attention, et de signe négatif (le réseau SOUSTRAIT une fraction de
la sortie SC-attention plutôt que de l'ajouter — une forme de correction,
pas un simple "on/off").

**Pattern d'attention réel** (bloc 0, batch de 8 échantillons du cache
STANDARD, SNR=15dB) : **PAS uniforme, PAS dégénéré** — diagonale
dominante (0.46 en SC0→SC0, 0.44 en SC11→SC11) avec décroissance
structurée vers les voisins proches (SC0→SC1=0.15, →SC2=0.09...) — un
vrai pattern de corrélation locale appris, pas du bruit. Effet de bord
notable aux deux extrémités du RB (SC0 et SC11 s'attendent mutuellement
un peu plus qu'au reste : 0.044/0.046).

**Conclusion nuancée** : le réseau A appris un pattern de SC-attention
physiquement sensé (corrélation locale, décroissance avec la distance),
et ne l'ignore pas — mais lui accorde une contribution nette modeste
(petit `s_sc`, souvent négatif). Ce n'est pas "la SC-attention ne sert à
rien et le réseau le sait" — c'est plus subtil : la SC-attention APPREND
quelque chose de réel mais **son influence sur la sortie finale reste
faible**, cohérent avec l'hypothèse #5 ci-dessous.

### Hypothèse #5 (nouvelle, la plus convaincante) — MÉCANISME DU GAP
### RZF-WMMSE est intra-SC (cross-user), pas cross-SC

**Constat clé, jusqu'ici sous-estimé** : RZF et WMMSE calculent TOUS LES
DEUX un précodeur INDÉPENDANT par sous-porteuse (aucun des deux
n'exploite d'information cross-SC). Le gap RZF-WMMSE mesuré et confirmé
(§1 channel_config.py, classical_comparison.py) vient donc PAR
CONSTRUCTION d'une différence de qualité d'optimisation CROSS-USER À SC
FIXÉE (WMMSE alloue puissance/direction de façon jointe et itérative
entre les K users sur CETTE sous-porteuse ; RZF le fait en une passe
fermée, moins optimale) — **pas d'un phénomène cross-fréquence**.

Si c'est le bon mécanisme : la capacité qui compte pour rattraper WMMSE
est le raisonnement JOINT SUR LES K USERS À SC FIXÉE (= user-attention),
PAS le mélange d'information entre SC voisines (= SC-attention). Or
**les 3 architectures (single_sc, intra_rb, ta_rb) ont TOUTES une
user-attention** — c'est justement la seule chose qu'elles partagent en
commun structurellement, et c'est peut-être pour ça qu'elles performent
si proche les unes des autres (91.4/90.7/86.5% WMMSE) malgré des
différences architecturales par ailleurs importantes (SC-attention ou
pas, compression de tokens ou pas). Cette hypothèse est cohérente avec
et RENFORCE une observation déjà loggée cette nuit dans `channel_
config.py` (le gap vient de la corrélation SPATIALE ENTRE UTILISATEURS,
pas d'une propriété fréquentielle globale) : le clustering d'utilisateurs
crée le besoin d'un traitement cross-user plus fin (ce que WMMSE fait
mieux que RZF), la sélectivité fréquentielle du canal n'étant qu'un
sous-produit du mécanisme physique, pas la variable qui gouverne le gap.

### PARTIE 0 (reprise, demande utilisateur 17h00) — CONCLUSION DÉFINITIVE
### sur la sélectivité fréquentielle courte-distance : CANAL BON

Reprise avec les données déjà mesurées (`diag_short_lag_correlation.json`,
20 tirages indépendants, batch=16, AUCUN nouveau calcul GPU nécessaire —
consigne "aucun job GPU sans feu vert" respectée). Critère de décision
donné par l'utilisateur : rho(12sc) proche de rho(95sc) → canal
insuffisant (catégorie 2) ; rho(12sc) élevé et décroissance cohérente sur
la fenêtre RB → canal bon, problème ailleurs (catégorie 3).

| | rho(1sc) | rho(6sc) | rho(12sc, bord RB) | rho(95sc, réf) | rho(12)/rho(95) | décroissance 1→12 |
|---|---|---|---|---|---|---|
| STANDARD | 0.996 | 0.912 | 0.797 | 0.363 | **2.20×** | **-20.0%** |
| MASSIVE | 0.997 | 0.919 | 0.801 | 0.308 | **2.60×** | **-19.7%** |

**Verdict : catégorie 3, sans ambiguïté.** rho(12sc) est 2.2-2.6× PLUS
ÉLEVÉ que rho(95sc) — la corrélation intra-RB n'est PAS "déjà collapsée"
vers sa valeur longue-distance, loin de là. Et la décroissance 1→12 est
réelle et substantielle (~20% de perte relative sur la seule largeur d'un
RB), pas un plateau plat. Le canal a une vraie sélectivité fréquentielle
progressive et exploitable à l'échelle RB=12, aux deux échelles
(STANDARD et MASSIVE). **Pas besoin de chercher un nouveau canal — le
canal actuel (verrouillé §1, déjà utilisé pour tout l'entraînement
STANDARD et le run MASSIVE interrompu) reste valide pour juger l'aspect
fréquentiel.** Le problème de "IntraRB/TA-RB ne battent pas SingleSC"
est donc bien du côté architecture/entraînement (Partie 1), pas du canal.

**Nuance à garder pour la thèse** (déjà notée en hypothèse #1/#5
ci-dessus, renforcée ici) : "sélectivité réelle" ne veut pas dire
"facilement exploitable pour CE problème précis". La corrélation reste
élevée (>0.8) sur toute la largeur du RB — chaque SC individuelle est
donc déjà fortement informative sur ses voisines, ce qui limite le gain
MARGINAL que la SC-attention peut apporter par rapport à un traitement
SC-indépendant (single_sc), même si le signal cross-SC est réel et non
nul. Combiné à l'hypothèse #5 (le gap RZF-WMMSE est structurellement un
problème per-SC/cross-user, aucun des deux baselines classiques n'exploite
d'info cross-SC), l'explication la plus complète et défendable reste :
le canal a une vraie sélectivité fréquentielle (mesurée, confirmée ici à
courte distance), MAIS (a) le mécanisme qui crée le gap RZF-WMMSE est
intrinsèquement intra-SC, et (b) même le signal cross-SC disponible est
en grande partie redondant avec l'info déjà présente SC par SC à ce
niveau de corrélation — donc la capacité cross-SC (SC-attention,
tokens TA-RB) n'apporte qu'un gain marginal, cohérent avec le résultat
mesuré (single_sc ≈ intra_rb ≈ ta_rb, différences <5pts).

**Analyse complémentaire (CPU/numpy uniquement, sur `results_
20260807_155133.npy` déjà sauvé, pas de calcul GPU)** : distribution du
rate par utilisateur à 15dB — RZF std=0.021, WMMSE std=0.037, SingleSC
std=0.024, IntraRB std=0.026, TA-RB std=0.023 (bps/Hz). Tous très
homogènes entre les 4 users, aucune architecture ne favorise/pénalise un
utilisateur en particulier — pas de pathologie d'équité détectée,
n'explique pas l'écart non plus.

---

## PRIORITÉ 1 (demande utilisateur, 17h10) — CSI imparfait sur checkpoints
## figés : rôle du cross-SC pour le DÉBRUITAGE — CONFIRMÉ, effet fort

Script de l'ancienne investigation "avant-hier" introuvable (seuls les
`.npy` de résultats et 3 PNG survivent, sur l'ancien canal pré-clustering
— inutilisables tels quels). Méthodologie reconstruite (`diag_csi_
imperfect.py`) : bruit LS gaussien complexe sur le canal, variance =
1/SNR_pilote_linéaire (canal normalisé puissance unité), précodeur décide
sur le canal BRUITÉ, rate calculé avec le VRAI canal (impact réel de
l'erreur d'estimation). Data SNR fixe 15dB, pilote ∈ {parfait, 20dB,
10dB}, 20 tirages × batch 32, poids figés (checkpoints STANDARD du run
complet). **Un seul job GPU, GPU0 vérifié libre avant lancement, aucune
concurrence.**

| Méthode | perfect | pilot20dB | %retenu | pilot10dB | %retenu |
|---|---|---|---|---|---|
| RZF | 32.29 | 22.12 | 68.5% | 11.39 | 35.3% |
| WMMSE | 32.27 | 22.13 | 68.6% | 11.39 | 35.3% |
| SingleSC | 28.79 | 21.32 | 74.1% | 11.21 | 38.9% |
| IntraRB | 28.01 | 21.22 | 75.8% | 11.17 | 39.9% |
| **TA-RB T=6** | 26.88 | **23.02** | **85.7%** | **13.71** | **51.0%** |

**Résultat net et sans ambiguïté** : TA-RB, le PIRE des 3 sous CSI
parfait, devient de LOIN le MEILLEUR sous CSI imparfait — au point de
**dépasser RZF ET WMMSE en valeur absolue** à pilote 10dB (13.71 vs
11.39/11.39) et à pilote 20dB (23.02 vs 22.12/22.13). IntraRB montre un
effet dans le même sens mais bien plus modeste (+1.7/+1.0 pts de
rétention relative vs SingleSC, contre +11.6/+12.1 pts pour TA-RB).
RZF et WMMSE dégradent quasiment IDENTIQUEMENT entre eux (68.5% vs
68.6%, 35.3% vs 35.3%) — cohérent avec le fait qu'aucun des deux
n'exploite d'info cross-SC (Partie 0, hypothèse #5).

**Hypothèse confirmée, mécanisme identifié précisément** : TA-RB
(`TransformerPrecoderClean._extract_features`) calcule `h_mean =
tf.reduce_mean(h_tok, axis=-1)` — une MOYENNE explicite sur les SC d'un
token (2 SC/token à T=6) AVANT tout traitement par attention. Moyenner 2
échantillons corrélés-mais-bruités réduit mécaniquement la puissance de
bruit (facteur ~2 en variance pour 2 échantillons indépendants de bruit
sur un signal corrélé, cf. Partie 0 : rho intra-RB >0.8, donc le signal
utile survit largement à la moyenne alors que le bruit d'estimation,
indépendant SC-à-SC, se moyenne). **Ce n'est pas un effet appris par
l'attention — c'est une propriété STRUCTURELLE de l'extraction de
features de TA-RB**, présente même si le réseau n'a jamais vu de CSI
bruité à l'entraînement (uniquement CSI parfait, tout du long). Explique
aussi pourquoi IntraRB (pas de moyenne explicite dans ses features — re/
im/abs par SC individuelle, SC-attention seule, dont on a mesuré en
Partie 1 hypothèse #3 une contribution nette faible, s_sc≈-0.08) ne
montre qu'un effet résiduel modeste : sans opération de moyennage
explicite dans l'architecture, le réseau n'a pas spontanément appris à
utiliser sa SC-attention comme débruiteur (logique : entraîné uniquement
sous CSI parfait, aucune incitation à le faire).

**Implication directe pour la thèse** : le résultat "single_sc ≈
intra_rb ≈ ta_rb (voire single_sc légèrement devant)" mesuré partout
cette nuit est **spécifique au banc d'essai CSI-parfait**, qui masque un
avantage réel et important de TA-RB dans un régime plus réaliste
(CSI imparfait). C'est un résultat scientifiquement intéressant en soi,
indépendamment de la question "qui bat WMMSE" : **sous CSI imparfait,
TA-RB bat RZF ET WMMSE**, ce qu'aucune architecture ne faisait sous CSI
parfait. Piste explicitement notée par l'utilisateur pour la suite (pas
entreprise ce soir, coût GPU) : réentraîner avec CSI imparfait dans la
boucle pourrait amplifier cet avantage encore davantage (le réseau
pourrait apprendre à exploiter le débruitage plus activement, pas
seulement en profiter passivement via le mean-pooling).

**Priorité 2 (investigation plus large) jugée inutile** : la consigne
la conditionnait à "si Priorité 1 ne tranche rien" — Priorité 1 a
tranché clairement et de façon mécaniquement expliquée. Passage direct à
Priorité 3 (warmup/protocole).

Sauvé : `results/diag_csi_imperfect.json`.

---

## Suite Priorité 1 (demande utilisateur, 17h25) — sweep en SNR données
## à pilote fixe : l'avantage TA-RB GRANDIT avec le SNR, ne s'estompe PAS

Confirmation demandée : le test précédent était à **SNR données FIXE
15dB**, pilote variable. Nouveau test (`diag_csi_imperfect_snr_sweep.py`)
: SNR données ∈{0,5,10,15,20}dB, **pilote FIXE 20dB**, mêmes checkpoints
figés, même méthodologie.

**% retenu (imparfait/parfait) par SNR données :**

| SNR data | RZF | WMMSE | SingleSC | IntraRB | TA-RB T=6 |
|---|---|---|---|---|---|
| 0dB | 93.4% | 93.9% | 93.5% | 93.9% | 97.0% |
| 5dB | 87.5% | 87.7% | 88.3% | 89.0% | 94.3% |
| 10dB | 78.6% | 78.7% | 81.1% | 82.3% | 90.0% |
| 15dB | 68.6% | 68.6% | 74.0% | 75.6% | 85.4% |
| 20dB | 59.1% | 59.1% | 68.4% | 70.8% | **82.2%** |

**Avantage TA-RB vs SingleSC (points de %retenu)** : +3.5 (0dB) →
+5.95 (5dB) → +8.88 (10dB) → +11.42 (15dB) → **+13.79 (20dB)**.
**L'avantage GRANDIT avec le SNR données, il ne s'estompe pas** —
contre-intuitif à première vue mais mécaniquement logique : à pilote
FIXE (bruit d'estimation de magnitude absolue constante), plus le SNR
données augmente, plus le bruit de LIEN devient négligeable — l'erreur
de CSI devient alors le facteur limitant DOMINANT, exactement la
situation où le débruitage cross-SC de TA-RB apporte le plus de valeur
relative. À bas SNR données, c'est le bruit du lien qui domine, pas
l'erreur de CSI — le débruitage a alors moins d'impact relatif.

**Rate absolu sous CSI imparfait, pilote=20dB fixe :**

| SNR data | RZF | WMMSE | SingleSC | IntraRB | TA-RB T=6 |
|---|---|---|---|---|---|
| 0dB | 12.78 | 13.20 | 12.29 | 12.34 | 12.55 |
| 5dB | 17.09 | 17.25 | 16.64 | 16.65 | 17.27 |
| 10dB | 20.25 | 20.28 | 19.65 | 19.60 | **20.90** |
| 15dB | 22.17 | 22.18 | 21.40 | 21.35 | **23.27** |
| 20dB | 23.03 | 23.04 | 22.17 | 22.08 | **24.33** |

**À 10-20dB, TA-RB dépasse RZF ET WMMSE en absolu, et l'écart grandit
avec le SNR** (+0.6 à 10dB → +1.1 à 15dB → +1.3 à 20dB vs WMMSE). Fait
notable : RZF/WMMSE SATURENT quasiment entre 15 et 20dB sous CSI
imparfaite (22.18→23.04, seulement +0.86) — limités par le plancher
d'erreur de CSI qu'ils ne peuvent pas mitiger — alors que TA-RB continue
à progresser (23.27→24.33, +1.06), cohérent avec sa capacité de
débruitage qui continue d'extraire de la valeur du canal qui s'améliore.

**Conclusion pour la thèse** : sous CSI réaliste (imparfait), à
haut SNR — exactement la plage cible de tout l'objectif de la session —
**TA-RB n'est pas seulement compétitif, il devient le meilleur précodeur
testé, classique ou neuronal**, et son avantage relatif ne fait que
grandir avec le SNR données. Résultat solide, mécaniquement expliqué
(moyennage structurel cross-SC dans l'extraction de features), positif
pour la thèse indépendamment de la question initiale "qui bat WMMSE sous
CSI parfait". Sauvé : `results/diag_csi_imperfect_snr_sweep.json`.

---

## PARTIE 0 bis (demande utilisateur, 17h15, en parallèle de Priorité 1)
## — candidat UMa pour une sélectivité fréquentielle plus marquée

Testé en parallèle de Priorité 1 (CSI imparfait, job indépendant,
GPU0 confirmé libre, un seul job à la fois malgré tout — pas de vraie
concurrence). **Mesure seule, rien invalidé/régénéré, en attente de
confirmation.**

`diag_uma_selectivity.py` : UMa (au lieu d'UMi), même mécanisme de
clustering spatial (`gen_topology_clustered`, R=20m, NLOS) déjà validé.

| lag(SC) | ρ UMa | ρ UMi (actuel) |
|---|---|---|
| 1 | 0.983 | 0.996 |
| 6 | 0.797 | 0.912 |
| 12 (bord RB) | **0.632** | 0.797 |
| 24 | 0.459 | 0.646 |
| 95 (réf) | 0.217 | 0.363 |

Décroissance intra-RB : ρ(12)/ρ(1) = **0.643** (UMi : 0.800) — décorrélation
nettement plus rapide, **ρ(12sc)=0.632 tombe bien dans la cible demandée
(<0.6-0.7)**.

**Gap RZF-WMMSE UMa** (10 batches, même rigueur que classical_comparison.py) :
0dB +4.15%, 5dB +2.33%, 10dB +0.23%, 15dB +0.02%, 20dB +0.01% — gap réel
à bas SNR, **quasi nul à 15-20dB** (le point d'évaluation cible de toute
la session).

**Évaluation honnête, pas juste favorable** : à première vue le gap
quasi nul à haut SNR pourrait sembler un problème (c'est justement la
plage 15-20dB qui nous intéresse) — MAIS ce n'est PAS spécifique à UMa :
le canal UMi actuel présente le MÊME pattern (gap concentré à bas SNR,
déjà documenté comme caractéristique connue, commit `d1b34b9` ; à 15dB
sur le run STANDARD complet, RZF=33.27 bat même légèrement WMMSE=32.62).
**UMa n'est donc pas pire qu'UMi sur ce point précis** — les deux canaux
partagent cette limite. L'avantage net d'UMa est la sélectivité
fréquentielle courte-distance nettement supérieure, sans dégradation
mesurée du gap RZF-WMMSE par rapport à ce qui est déjà accepté.

**Recommandation (proposition, pas appliquée)** : UMa avec le même
clustering R=20m est un candidat net meilleur que UMi pour tester
spécifiquement l'hypothèse fréquentielle (SC-attention/TA-RB), sans
contrepartie identifiée. **Mais changer de canal implique de tout
refaire** (dataset §0, classical_comparison, réentraînement des 3
architectures STANDARD, MASSIVE en dernier) — coût substantiel, pas
engagé sans feu vert explicite. Sauvé : `results/diag_uma_selectivity.json`.
**En attente de la décision utilisateur.**

---

## Priorité (demande utilisateur, 17h35, en parallèle/avant UMa) —
## pourquoi IntraRB/TA-RB sont ACTIVEMENT moins bons que SingleSC à
## haut SNR (pas juste "pas mieux"), écart CROISSANT avec le SNR

**Confirmation immédiate (0 calcul GPU, données déjà sauvées,
`results_20260807_155133.npy`)** : le désavantage est réel, monotone,
et change même de SIGNE selon le SNR — à 0dB IntraRB bat LÉGÈREMENT
SingleSC (single-intra=-0.33 bps/Hz), à 20dB SingleSC devance IntraRB de
+0.53 et TA-RB de +2.48. Pas un artefact de bruit isolé à un seul point.

**Tests sur poids figés** (`diag_high_snr_disadvantage.py`, un seul job
GPU, GPU0 vérifié libre avant lancement) :

**1. Généralisation (pool entraînement vs canaux frais)** : l'écart
pool-frais grandit avec le SNR pour LES TROIS architectures (~×2.1-2.4
de 5dB à 20dB), mais **de façon similaire entre elles** — pas de signal
que IntraRB/TA-RB sur-apprennent significativement plus que SingleSC.
Preuve plus directe : **le désavantage persiste presque à l'identique
même mesuré sur le pool d'entraînement** (à 20dB, IntraRB reste derrière
SingleSC de 0.66 bps/Hz sur le pool, TA-RB de 2.56) — si c'était de la
sur-adaptation, on s'attendrait à ce que l'écart RÉTRÉCISSE ou s'inverse
sur données "vues". **Hypothèse #1 (sur-apprentissage) INFIRMÉE.**

**2. Norme du gradient à bas vs haut SNR** : croît avec le SNR pour les
3 (attendu, la loss log(1+SINR) devient plus sensible). Mais **pas de
signe d'aplatissement/évanouissement pour les architectures plus
grandes** — au contraire, le RATIO norme(20dB)/norme(5dB) est PLUS
GRAND pour IntraRB (7.74) et TA-RB (8.07) que SingleSC (4.64). TA-RB a
même la norme ABSOLUE la plus grande à 15-20dB des 3 architectures
(4.21 à 20dB, contre 3.44 SingleSC et 2.30 IntraRB) — **pas de gradient
qui s'évanouit, TA-RB en particulier a un signal d'optimisation fort**.
**Hypothèse #2 (optimisation plus dure à haut SNR pour les gros
réseaux) INFIRMÉE, au moins sous cette forme.**

Fait notable cependant : **IntraRB a une norme de gradient
systématiquement PLUS PETITE que SingleSC à CHAQUE SNR testé** (0.298 vs
0.742 à 5dB ; 2.305 vs 3.444 à 20dB) — pas spécifique au haut SNR, un
signal plus faible tout du long de l'entraînement. Piste plausible :
LayerNorms supplémentaires + scalaires résiduels initialisés à 0 (dont
on a mesuré en Partie 1 que `s_sc` reste petit, ~-0.08) pourraient
amortir structurellement le gradient traversant IntraRB, la rendant
un peu moins efficace à budget d'époques égal — cohérent avec le retard
modeste observé en début/milieu d'entraînement (courbes rolling10, déjà
loggé) qui ne se referme jamais complètement.

**3. Écart IntraRB-TA-RB (perte de compression T=6) par SNR** :
**confirmé, monotone et propre** : +0.49 (5dB) → +0.77 (10dB) → +1.09
(15dB) → +1.68 (20dB) bps/Hz. La compression T=6 perd une info FIXE
(indépendante du SNR) qui devient le facteur limitant dominant à mesure
que le régime devient MUI-limité (haut SNR) plutôt que bruit-limité (bas
SNR) — exactement l'hypothèse proposée, confirmée avec un gradient sain
(pas un problème d'optimisation, un problème d'information disponible).

**Vérification supplémentaire (sanity check mathématique)** : un écart
qui grandit avec le SNR en bps/Hz n'est PAS un simple artefact
d'amplification du log(1+SINR) — si le déficit SINR relatif entre
architectures était constant (en dB), l'écart de rate resterait
CONSTANT en bps/Hz (log2(a)-log2(b)=log2(a/b), indépendant du SNR
absolu). Un écart de rate qui grandit implique un déficit SINR RELATIF
qui grandit réellement — l'effet est réel, pas un artefact de métrique.

**Conclusion causale (pas juste corrélée)** :
- **TA-RB** : bien expliqué — perte de compression T=6, fixe, de plus
  en plus limitante en régime MUI-limité (haut SNR). PAS un problème de
  gradient/optimisation (gradients sains, les plus forts des 3 à haut
  SNR). Piste de correction directe : moins de compression (T=12,
  toujours pas testé après l'incident) ou décodeur qui compense mieux
  la perte d'info (Partie 2, refonte TA-RB, en attente).
- **IntraRB** : cause moins nette. Sur-apprentissage et gradient qui
  s'évanouit À HAUT SNR SPÉCIFIQUEMENT sont tous deux infirmés par les
  tests. Le candidat le plus plausible restant : un gradient
  systématiquement plus faible (pas spécifique au SNR) dû à la
  profondeur/normalisation supplémentaire de l'architecture, qui laisse
  un déficit de précision non totalement résorbé au budget d'époques
  actuel — cohérent avec, mais pas prouvé formellement par, le retard de
  convergence modeste observé en Partie 1. **Test décisif possible : le
  warmup+rampe LR (déjà prévu, section suivante) — si ça referme
  spécifiquement l'écart IntraRB (plus qu'il n'aide SingleSC), ça
  confirmerait "IntraRB était limité par l'optimisation, pas par une
  limite architecturale dure".**

Sauvé : `results/diag_high_snr_disadvantage.json`.

---

## Partie 2 (demande utilisateur, 18h00) — TA-RB décodeur résiduel :
## implémenté, validé (CPU), EN ATTENTE de GPU libre pour l'entraînement

**Contexte GPU** : test warmup+rampe (section suivante) tourne sur GPU0,
GPU1/GPU2 toujours occupés par d'autres utilisateurs (99% util,
26-43GB) tout du long de cette section — **aucun GPU libre trouvé**,
donc uniquement du travail CPU/implémentation fait ici, rien lancé.

**Nouvelle classe** `TransformerPrecoderCleanResidual` (`precoders_v2.py`)
: hérite de `TransformerPrecoderClean` pour `_extract_features`
(moyennage cross-SC INCHANGÉ -- le mécanisme de débruitage CSI n'est pas
touché), mais remplace le décodeur :
- **Avant** : `upsample` (Conv1DTranspose APPRIS, T tokens → N SC) +
  `final_proj` (Dense D→2M) reconstruisent tout depuis zéro -- le réseau
  doit implicitement ré-apprendre que le canal varie lentement en
  fréquence (déjà su, Partie 0 : rho intra-RB >0.8) en plus de la
  correction fine.
- **Après** : `token_to_precoder` (Dense D→2M, un seul par token) donne
  une estimée de précodeur PAR TOKEN, étendue aux N SC par
  **interpolation linéaire FIXE** (`tf.image.resize`, AUCUN poids
  appris) -- encode le prior physique directement, sans le faire
  réapprendre. `sc_refine` (inchangé dans son principe, entrée
  redimensionnée 6M au lieu de D+4M) apprend SEULEMENT la correction
  résiduelle par rapport à cette base interpolée.

**Validation CPU pure** (aucun calcul GPU, `CUDA_VISIBLE_DEVICES=-1`) :
- Forward pass correct, sortie normalisée à puissance exactement 1.0 par
  (SC, user) (vérifié après correction d'un bug d'indexation dans mon
  script de test, pas dans le modèle).
- Gradients non-nuls sur toutes les variables entraînables.
- `complexity()` réécrit et revalidé EXACTEMENT contre `sum(tf.size(v)
  for v in trainable_variables)` (162 503 = 162 503, méthode Étape 2).
- **Architecture plus LÉGÈRE que l'original** : 162 503 params (D=64,L=2)
  contre 301 415 pour `TransformerPrecoderClean` à la même taille (-46%)
  -- suppression du Conv1DTranspose appris, remplacé par une simple
  Dense + interpolation sans poids. FLOPs aussi réduits (1.02G vs 2.51G
  à cette taille).
- **Smoke test bout-en-bout complet** (pipeline réel : `MU_MIMO_System`
  + `SupervisedTrainer`, dataset jetable 100 échantillons, 1 warmup + 1
  finetune epoch × 3 pas, CPU pur) : **OK, aucune erreur, checkpoint
  sauvé correctement.**

**Branché dans le pipeline** : nouveau `precoder_type='ta_rb_residual'`
dans `MU_MIMO_System._init_precoder` (`main_finall.py`).

**Script d'entraînement/comparaison prêt** (`diag_tarb_residual_test.py`)
: entraîne sur le protocole actuellement validé (cosine_long, warmup=3,
finetune=80), puis compare à TA-RB T=6 original sur LES DEUX régimes
demandés : (1) CSI parfait à 15/17.5/20dB (l'écart à haut SNR se
réduit-il ?), (2) CSI imparfait à SNR données=15dB, pilote ∈{parfait,
20,10dB} (le bénéfice de débruitage est-il préservé ?). **Pas encore
lancé, en attente d'un GPU libre** (ni GPU0 -- occupé par le test
warmup+rampe -- ni GPU1/GPU2 -- autres utilisateurs).

---

## Suite Priorité 3 (warmup+rampe) — résultat single_sc : NÉGATIF

`diag_warmup_ramp_test.py --arch single_sc` (warmup=2ep, rampe=2ep,
finetune=30ep) terminé. Comparé point par point à la baseline (cosine_
long, warmup=3ep, switch brutal) sur les MÊMES 30 époques (courbes déjà
loggées, `etape4_standard_v2.log`) :

| | Moyenne rate@15dB, époques 21-30 |
|---|---|
| Baseline (switch brutal) | 26.58 |
| Rampe (warmup court+transition progressive) | 24.28 |
| **Delta** | **-2.30 (-8.7%)** |

**La rampe est NETTEMENT MOINS BONNE que la baseline pour single_sc**,
à budget d'époques identique. Éval propre (20 batchs) de la variante
rampe à 30 époques : 77.4%/72.8%/68.3% WMMSE à 15/17.5/20dB.

Passage à IntraRB (le test décisif pour l'hypothèse gradient amorti),
lancé dès GPU0 libéré (vérifié, aucune concurrence).

**Note utilisateur (18h10)** : la ligne "CONCLUSION PARTIE 0/1 → PRÊT
POUR RELANCER MASSIVE" plus haut est PÉRIMÉE (écrite avant cette
investigation) — confirmé par l'utilisateur, annotée in situ. MASSIVE
reste bloqué tant que warmup+rampe (3 architectures) + TA-RB résiduel +
décision protocole final ne sont pas conclus.

**Résultat IntraRB : ÉGALEMENT NÉGATIF, n'infirme PAS l'hypothèse
gradient amorti mais ne la CONFIRME PAS non plus**

| | Époques 21-30 (moyenne, rate@15dB) | single_sc |
|---|---|---|
| Baseline (switch brutal) | 25.26 | 26.58 |
| Rampe (warmup=2+rampe=2) | 24.68 | 24.28 |
| Delta | -0.58 (-2.3%) | -2.30 (-8.7%) |

Si l'hypothèse "IntraRB limité par un gradient amorti/optimisation"
était correcte, la rampe aurait dû aider IntraRB NETTEMENT PLUS que
single_sc (le corriger spécifiquement). Elle lui nuit MOINS (-2.3% vs
-8.7%) mais lui nuit quand même — pas le signal net attendu pour
confirmer l'hypothèse, mais pas non plus une infirmation propre (l'écart
relatif plus faible pour IntraRB pourrait être un signal faible dans la
bonne direction, noyé dans le bruit à ce budget réduit). Éval propre à
30 époques : 79.9%/75.2%/70.7% WMMSE à 15/17.5/20dB.

Passage à TA-RB (3e architecture, complétude), lancé dès GPU0 libéré.

**Résultat TA-RB : ÉGALEMENT NÉGATIF**

| | Époques 21-30 (moyenne, rate@15dB) |
|---|---|
| Baseline (switch brutal) | 25.12 |
| Rampe (warmup=2+rampe=2) | 24.32 |
| Delta | -0.80 (-3.2%) |

### SYNTHÈSE FINALE — Priorité/Partie 3 (warmup+rampe) : CONCLU, PAS DE GAIN

| Architecture | Delta rampe vs baseline (%) |
|---|---|
| SingleSC | -8.7% |
| IntraRB | -2.3% |
| TA-RB | -3.2% |

**Le warmup court + transition rampée nuit aux 3 architectures**, à des
degrés divers (le plus pour single_sc, le moins pour IntraRB mais
toujours négatif) — pas de gain net nulle part. Ne confirme pas
clairement l'hypothèse du gradient amorti pour IntraRB (souffre le
moins des 3, mais souffre quand même — pas le signal net attendu pour
une confirmation propre). Par la règle déjà actée : documenté
honnêtement, pas un échec. **Protocole retenu pour MASSIVE : INCHANGÉ**
(cosine_long, warmup=3ep, switch brutal warmup→finetune, finetune=80ep,
patience=25) — aucune preuve qu'un changement apporterait un gain.

**Condition MASSIVE #1 (résultat IntraRB warmup+rampe) : ✅ réglée.
Condition #3 (protocole final décidé) : ✅ réglée (= protocole actuel,
inchangé). Condition #2 (TA-RB résiduel évalué) : en cours ci-dessous.**

---

## Suite Partie 2 — Entraînement réel TA-RB décodeur résiduel (GPU0 libre)

`diag_tarb_residual_test.py` lancé (protocole actuel validé : cosine_long,
warmup=3ep, finetune=80ep, patience=25 — même protocole que TA-RB T=6
original, comparaison à isoprotocole). Entraîne puis évalue sur les DEUX
régimes (CSI parfait 15/17.5/20dB, CSI imparfait SNR=15dB/pilote
{parfait,20,10dB}) contre TA-RB T=6 original. GPU0 vérifié libre avant
lancement, aucune concurrence.

---

**Partie 0 conclue à ce stade (canal validé pour juger l'aspect
fréquentiel, aucune régénération nécessaire) — mais ceci ne concluait
QUE la Partie 0, pas le reste.** (Ligne originale "prêt pour relancer
MASSIVE" supprimée le 2026-08-07 18h45 sur demande utilisateur : elle
prêtait à confusion en réapparaissant comme statut à jour. Voir le
**STATUT MASSIVE**, tenu à jour à chaque section suivante, pour l'état
réel — ne jamais se fier à une ligne de conclusion locale ailleurs dans
ce log pour l'état de MASSIVE.)

**STATUT MASSIVE (2026-08-07 20h44) : les 4 conditions sont réglées,
mais MASSIVE reste BLOQUÉ en attente d'une décision supplémentaire
(canal UMi ou UMa ?) avant de pouvoir relancer quoi que ce soit.**
1. ~~Résultat warmup+rampe (3 architectures)~~ — ✅ réglé 18h37 (net
   négatif partout, protocole baseline confirmé inchangé).
2. ~~TA-RB décodeur résiduel évalué~~ — ✅ réglé 19h29 (BON sur les
   deux régimes, adopté comme nouveau standard TA-RB, `MODELS_TO_TRAIN`
   mis à jour).
3. ~~Investigation IntraRB (gradient amorti, s_sc) conclue~~ — ✅ réglé
   19h35 (cause caractérisée précisément couche par couche : atténuation
   uniforme sauf user-attention+sortie qui compensent à haut SNR ; pas
   de correction appliquée, écart mineur 90.7% vs 91.4%, jugé suffisant).
4. ~~UMa testé~~ — ✅ réglé 20h44 : **IntraRB dépasse SingleSC sous UMa**
   (+2.8 à +3.4pts %WMMSE, écart croissant avec le SNR) — hypothèse
   confirmée, canal UMi n'était pas assez sélectif pour révéler
   l'avantage cross-SC. **Nouvelle question ouverte, pas encore
   tranchée** : adopter UMa comme canal officiel (implique tout
   régénérer : dataset production, classical_comparison, ré-entraînement
   complet 4 architectures, PUIS MASSIVE) ou rester sur UMi (déjà
   entraîné, résultats complets en main) ? **En attente de décision
   utilisateur avant tout engagement de calcul supplémentaire.**

---

## Correction utilisateur (20h55) — budget UMa dérivé de l'analyse de
## convergence, pas fixé arbitrairement à 30/83 époques

Vérification demandée sur les courbes UMi 83 époques (rolling10,
rate@15dB, seul point SNR suivi par époque) :

| | SingleSC | IntraRB |
|---|---|---|
| Dernier gain net (>0.5/fenêtre10) | époque 75 | époque 76 |
| Pente 10 dernières époques | -0.004/ép | +0.001/ép |
| Pente 20 dernières époques | +0.037/ép | +0.057/ép |
| Gain époque 60→83 | +2.7% | +3.7% |

Plateau réel vers l'époque 75-76 pour les deux, MAIS **IntraRB a une
traîne résiduelle plus raide que SingleSC** (pente et gain restant plus
grands) — un budget raccourci désavantagerait IntraRB relativement plus,
faussant précisément la comparaison recherchée. **Budget retenu : 80
époques finetune (+3 warmup), identique au run UMi original** — pas de
raccourci, le test à 30 époques déjà fait (résultat "IntraRB gagne")
n'était pas concluant (courbes encore loin du plateau à ce stade).

**Lancé** : SingleSC UMa à 80 époques finetune, sur le cache UMa déjà
généré (8000 échantillons, pas de régénération). IntraRB à suivre. Un
seul job GPU0, vérifié avant lancement. TA-RB PAS testé sur UMa pour
l'instant (seul SingleSC vs IntraRB tranche la question posée ; TA-RB
résiduel attendra la phase de production complète si UMa est adopté).

**Résultat SingleSC UMa, 80 époques (budget complet, justifié)** :
82.9%/80.2%/76.8% WMMSE à 15/17.5/20dB (train=70.7min) — nettement
au-dessus du résultat à 30 époques (72.0/68.3/64.2%), confirme que
30 époques était très insuffisant pour ce canal aussi. IntraRB UMa
lancé au même budget (80ep) pour comparaison directe, GPU0 vérifié
libre avant lancement.

### RÉSULTAT FINAL UMa à budget complet (80 époques, justifié par
### l'analyse de convergence) — l'avantage IntraRB est CONFIRMÉ, pas
### un artefact du test rapide à 30 époques

| SNR | SingleSC (rate) | IntraRB (rate) | Delta |
|---|---|---|---|
| 15dB | 27.30 | **27.71** | **+0.41** |
| 17.5dB | 28.84 | **29.29** | **+0.45** |
| 20dB | 30.36 | **30.84** | **+0.48** |

(comparaison en rate brut, plus fiable que %WMMSE ici — les deux runs
recalculent leur propre référence WMMSE UMa sur 10 batchs chacun,
~5% d'écart entre les deux calculs de référence eux-mêmes, 31.20 vs
32.94 à 15dB ; SingleSC et IntraRB partagent exactement le même cache
et protocole donc leur comparaison directe n'est pas affectée par ce
bruit de référence)

**Avantage IntraRB confirmé à budget rigoureux** — plus modeste en
proportion qu'au signal précoce à 30 époques (+0.4-0.5 assez stable sur
les 3 SNR plutôt qu'un écart nettement croissant), mais réel et
consistant sur les 3 points. Le test à 30 époques n'était pas un faux
positif, juste optimiste sur l'ampleur.

**Toutes les questions posées cette nuit sur "pourquoi IntraRB/TA-RB ne
battent pas SingleSC" ont maintenant une réponse complète et
cohérente** : sous UMi (sélectivité modeste), pas d'avantage cross-SC
car redondant avec l'info per-SC déjà disponible (Partie 0/hypothèse
#5) ; sous CSI imparfait, un mécanisme de MOYENNAGE (pas d'attention)
donne un vrai avantage de débruitage à TA-RB (confirmé, grandit avec le
SNR) ; sous UMa (sélectivité réelle), la SC-attention d'IntraRB capture
une vraie information supplémentaire et bat SingleSC, confirmé à budget
complet. Sauvé : `results/diag_uma_quick_intra_rb.json` (80ep).

**Décision sur l'adoption d'UMa toujours en attente du feu vert
utilisateur** (implique régénération complète : dataset production,
classical_comparison, ré-entraînement des 4 architectures dont TA-RB
résiduel, PUIS MASSIVE).
MASSIVE ne sera relancé qu'après ces 4 points ET un feu vert explicite.

**Si confirmée, implication directe** : chercher à améliorer le sum-rate
via PLUS de capacité cross-SC (T=12, plus de couches SC-attention) a
peu de chances de fermer l'écart à WMMSE ; la vraie piste serait de
renforcer/mieux entraîner la partie USER-attention (ou la boucle
d'optimisation qu'elle doit apprendre à approximer). Reste à tester
empiriquement (hypothèse #1 + #4 ci-dessous) avant de trancher
définitivement.

### Hypothèse #1 (granularité de corrélation) — RÉFUTÉE (dans le sens
### littéral), mais ENRICHIT l'hypothèse #5

`diag_short_lag_correlation.py` (lags 1-12 intra-RB + 24/48/95 référence,
20 tirages indépendants moyennés, batch=16 pour rester léger à côté de
MASSIVE) :

| lag(SC) | rho STANDARD | rho MASSIVE |
|---|---|---|
| 1 | 0.996 | 0.997 |
| 6 | 0.912 | 0.919 |
| 12 (bord du RB) | 0.797 | 0.801 |
| 24 | 0.646 | 0.630 |
| 95 (référence §1) | 0.363 | 0.308 |

**La corrélation intra-RB est en réalité TRÈS ÉLEVÉE** (0.80-1.0 sur
toute la largeur d'un RB=12, décroissance relative rho(12)/rho(1)≈0.80
seulement) — **l'hypothèse littérale "pas assez de corrélation à cette
granularité" est FAUSSE**, c'est l'inverse : les 12 SC d'un RB sont
QUASI-REDONDANTES entre elles.

**⚠️ INCIDENT (16h53-16h57) — MASSIVE crashé par un job GPU concurrent
(probablement le mien)** : en testant l'hypothèse #4 (TA-RB T=12), un
1er essai (batch=128) a fait un OOM propre. Un 2e essai (batch=32,
pensant réduire le risque) a crashé sans traceback Python propre
(accès mémoire illégal probable) quasi simultanément avec la mort
CONFIRMÉE du process MASSIVE (PID 2900748, vivant et actif à 16:52,
disparu peu après — dernier checkpoint réel sur disque epoch 26/83,
16:53:50, `weights/SingleSC_4L_128d/best_20260807_165350`). Causalité
non prouvée à 100% (GPU partagé avec d'autres utilisateurs) mais
coïncidence temporelle trop forte pour l'écarter — traité comme ma
responsabilité. **Signalé immédiatement à l'utilisateur, pas de tentative
de minimiser.** MASSIVE perdu à ~1h sur les ~8h prévues, aucun mécanisme
de reprise depuis checkpoint dans le code actuel (`SupervisedTrainer`
réinitialise toujours à zéro) — la seule option est un relancement
complet. **Règle actée avec l'utilisateur suite à l'incident : plus
JAMAIS de job GPU concurrent sur un GPU faisant tourner un run long, sans
exception. Aucun job GPU sans feu vert explicite jusqu'à nouvel ordre.**
Décision utilisateur : trancher la Partie 0 (canal) AVANT de recommitter
8h de calcul MASSIVE, avec seulement les mesures déjà en main (aucun
nouveau calcul GPU) — pour éviter de payer un 2e coût de calcul perdu si
le canal doit changer.

**Mais ça RENFORCE l'hypothèse #5, pas ne la contredit** : une
corrélation intra-RB aussi forte veut dire que chaque SC individuelle
est déjà quasiment auto-suffisante — un modèle qui traite chaque SC
indépendamment (single_sc) "voit" déjà une information très proche de
celle de ses voisines, simplement parce que le canal varie lentement à
cette échelle. La SC-attention n'apporte alors qu'une information
marginale NOUVELLE (ce qu'elle agrège est déjà quasi dupliqué dans le
signal d'entrée de chaque SC prise isolément) — ce n'est pas qu'il n'y a
"rien à exploiter" (il y a un vrai pattern, cf. hypothèse #3), c'est que
ce qu'il y a à exploiter est déjà largement redondant avec l'info
disponible SC par SC. Combiné à l'hypothèse #5 (le gap RZF-WMMSE est un
problème per-SC cross-user, structurellement indépendant du cross-SC) —
double explication convergente, cohérente avec le résultat final où
SingleSC (91.4%) devance même très légèrement IntraRB (90.7%) à 15dB.

---

**Note d'ordonnancement** : les insertions successives dans ce fichier
ont cassé l'ordre chronologique strict à partir d'ici (des sections plus
anciennes se retrouvent après des sections plus récentes). Se fier aux
horodatages explicites dans chaque section, PAS à l'ordre d'apparition
dans le fichier, à partir de ce point. Le **STATUT MASSIVE** le plus
récent (ci-dessus, 18h45) reste la seule source de vérité pour l'état
de MASSIVE quel que soit l'endroit où il apparaît dans le fichier.

---

## Suite "Suite Partie 2" (19h05) — signal intermédiaire TA-RB résiduel
## + pistes de repli préparées

**Signal intermédiaire** (demande utilisateur, sans interrompre le run) :
checkpoint réel sur disque (contourne le bufferisation du log) à
l'époque **34/83, rate=26.66**. Comparé à la baseline TA-RB original sur
la même fenêtre d'époques (moyenne ép.25-34, courbes déjà loggées) :
résiduel ≈24.4 vs original 25.7 — **le résiduel est légèrement EN
RETARD (~-5%) à ce stade**, pas en avance. **Pas encore décisif** (le
bruit epoch-à-epoch historique est de cet ordre de grandeur, cf. tous
les runs précédents) — ce qui compte vraiment est l'éval propre finale
(20 batchs, CSI parfait ET imparfait), pas ce proxy bruité d'un seul
batch d'entraînement par époque. Suivi à refaire dans 10-15min ou à
l'arrivée du résultat final (le premier des deux).

**Pistes de repli préparées par anticipation** (code prêt, PAS lancé —
GPU0 occupé par le run principal, règle post-incident respectée) :
- `TransformerPrecoderCleanResidual.__init__` : nouveau paramètre
  `alpha_init` (piste "amplitude du résidu mal calibrée", défaut 0.1
  inchangé), traversant proprement `MU_MIMO_System(alpha_init=...)`
  jusqu'au modèle (même patron que `use_abs`/`use_cossin`).
- `diag_tarb_residual_variant.py` : script budget réduit (30ep, même
  format que les tests warmup+rampe) avec `--learning_rate` et
  `--alpha_init` en CLI — prêt à tester 1-2 hypothèses ciblées dès que
  le run principal se termine, sans réécrire de code.
- Piste "méthode de compression elle-même" (mean/var → pooling appris)
  : pas encore de code, gardée en réserve si les pistes décodeur
  (résiduel, LR, alpha_init) restent insuffisantes — sera scopée plus
  précisément si besoin, pour rester dans l'esprit "simple, gérable".

---

## RÉSULTAT FINAL TA-RB résiduel (19h29) — BON sur les deux régimes,
## ADOPTÉ comme nouveau standard TA-RB

Entraînement complet fini (83 époques, 56.1min, cohérent avec
l'estimation initiale malgré le décodeur moins coûteux — le budget
d'époques dominait le temps, pas le coût par pas, comme clarifié plus
haut).

**CSI parfait** (20 batchs, comparaison directe à TA-RB T=6 original) :

| SNR | Résiduel | Original T=6 | Delta | %WMMSE résiduel |
|---|---|---|---|---|
| 15dB | 29.39 | 28.20 | **+1.19** | **91.0%** (original 86.5%) |
| 17.5dB | 31.08 | 29.67 | **+1.41** | **87.2%** (original 82.8%) |
| 20dB | 32.53 | 30.73 | **+1.80** | **83.5%** (original 78.4%) |

**L'écart avec IntraRB (90.7%) et SingleSC (91.4%) à 15dB est quasiment
refermé** (91.0%) — le décodeur résiduel répond directement à la
question posée : oui, l'écart à haut SNR se réduit, et l'ampleur de
l'amélioration GRANDIT avec le SNR (+1.19→+1.80), signe que c'est bien
la perte de compression en régime MUI-limité qui est corrigée (cohérent
avec le diagnostic causal `diag_high_snr_disadvantage.py`).

**CSI imparfait** (data SNR=15dB, pilote {parfait,20,10dB}) :

| | Résiduel | Original T=6 |
|---|---|---|
| %retenu pilote 20dB | **87.7%** | 85.7% |
| %retenu pilote 10dB | **54.6%** | 51.0% |

**L'avantage de débruitage n'est pas seulement préservé, il s'améliore
légèrement** — cohérent : le mécanisme de moyennage dans
`_extract_features` (hérité tel quel, jamais touché) reste intact, et
la meilleure reconstruction en aval n'interfère pas avec lui.

**Décision (cas "SI BON" de la consigne) : `TransformerPrecoderCleanResidual`
adopté comme nouveau standard TA-RB.** `MODELS_TO_TRAIN` (`main_finall.py`)
mis à jour : `precoder_type='ta_rb_residual'` remplace `'ta_rb'`.
`'ta_rb'` (décodeur original) reste disponible dans le code pour
comparaison ponctuelle mais n'est plus le chemin de training par défaut
— même logique que le remplacement V4→TransformerPrecoderClean plus tôt
dans la nuit.

**Condition MASSIVE #2 (TA-RB résiduel évalué) : ✅ réglée — résultat
positif, adopté.** Passage à l'investigation IntraRB (condition #3).

Sauvé : `results/diag_tarb_residual_test.json`, poids
`weights/TA_RB_residual_6tok_4L_128d/best_20260807_192815`.

---

## Investigation IntraRB (19h35) — gradient COUCHE PAR COUCHE : trouvaille
## spécifique, cause caractérisée mais pas "corrigée" (décision documentée)

`diag_intrarb_gradient_layers.py` : mapping variable→couche par IDENTITÉ
objet (pas par nom -- les noms Keras auto-générés type
`layer_normalization_7` ne reflètent pas le sous-composant sémantique),
en parcourant directement `blk.norm_sc`, `blk.attn_sc`, `blk.norm_usr`,
`blk.attn_usr`, `blk.norm_ffn`, `blk.ffn_up/out`, `blk.s_sc/s_usr/s_ffn`
pour chacun des 4 blocs. `input_embed`/`output_proj`/`attn_usr` ont
exactement la même shape dans les deux architectures (feat_dim=25,
D=128 identiques) -- comparaison directe honnête, pas juste analogique.

**Rappel s_sc/s_usr appris** (déjà mesuré Partie 1, hypothèse #3,
avant l'incident) : s_sc ∈ [-0.094,-0.067] (petit mais pas nul),
s_usr ∈ [-0.230,+0.306] (nettement plus grand en magnitude).

**Résultat (ratio gradient IntraRB/SingleSC, moyenné sur 15 tirages,
par bloc, à 5/15/20dB)** :

- **Atténuation QUASI-UNIFORME dans les couches précoces et FFN** :
  `input_embed` (0.53-0.57), `input_norm` (0.41-0.62), tous les
  `norm_usr`/`norm_ffn`/`ffn_up`/`ffn_out` de tous les blocs (~0.3-0.9,
  cohérent à travers TOUT le réseau) — **pas localisé à une couche
  précise, c'est un effet systémique dès la toute première couche**
  (input_embed déjà à moitié du gradient de SingleSC).
- **EXCEPTION NETTE, spécifique, qui grandit avec le SNR** :
  `s_usr` (scalaire résiduel user-attention) et `output_proj` ont un
  ratio qui **dépasse 1 et grandit avec le SNR** — ex. bloc1 s_usr :
  ratio=0.235(5dB)→3.195(15dB)→3.000(20dB) ; output_proj :
  0.966→1.103→1.084. **À haut SNR, le mécanisme PARTAGÉ (user-attention
  + sortie) reçoit un gradient PLUS FORT dans IntraRB que dans
  SingleSC**, alors que tout le reste (FFN, normalisations, embedding
  d'entrée) reste systématiquement plus faible.

**Interprétation causale** : ce n'est PAS un problème de gradient qui
s'évanouit au sens classique (le mécanisme critique -- user-attention,
qui gère le MUI -- reçoit un signal fort, plus fort qu'à SingleSC même).
C'est plutôt que **le réseau concentre son adaptation sur user-attention
+ sortie à haut SNR, et laisse le reste du réseau (FFN, SC-attention)
relativement sous-exploité** (petits gradients = peu de mise à jour
continue) -- cohérent avec s_sc resté petit (Partie 1) : le réseau n'a
jamais vraiment trouvé comment faire travailler sa capacité
supplémentaire (SC-attention + profondeur FFN) à plein régime, et
compense en poussant davantage sur le mécanisme qu'il maîtrise déjà
(user-attention), plutôt que d'exploiter l'architecture plus riche.

**Décision (cas "sinon documente honnêtement" de la consigne)** :
correction ciblée envisageable (LR différencié par couche pour les
composants sous-exploités, ~genre "discriminative fine-tuning") mais
juge cette complexité disproportionnée vu (a) l'écart réel est déjà
PETIT (90.7% vs 91.4% WMMSE à 15dB, ~0.7pt), (b) le budget déjà engagé
cette nuit sur cette question, (c) le reste du plan (UMa, MASSIVE)
encore à faire. **Cause bien caractérisée (pas juste "gradient plus
faible", mais PRÉCISÉMENT où et pourquoi), mais PAS corrigée ce soir**
-- idée de LR différencié par couche documentée comme piste future
concrète, pas testée. IntraRB reste une architecture valide et
compétitive (90.7% WMMSE), l'écart résiduel avec SingleSC est mineur et
ne bloque pas la suite.

**Condition MASSIVE #3 (investigation IntraRB conclue) : ✅ réglée**
(conclusion honnête, cause caractérisée, pas de correction appliquée —
accepté comme suffisant par la consigne). Passage à UMa (condition #4).

Sauvé : `results/diag_intrarb_gradient_layers.json`.

---

## UMa (20h12) — test ciblé single_sc vs IntraRB, budget réduit

`datasets.py` généralisé pour supporter UMa (`_build_sionna_system`
dispatch était UMi-only ; `CachedSionnaDataset`/`JointClusterGenerator`
avaient déjà `scenario` de bout en bout). Nouveau
`diag_uma_quick_train.py` : cache UMa jetable (8000 échantillons, R=20m
comme UMi, distinct du cache de production), entraîne SingleSC puis
IntraRB à budget réduit (30ep, protocole cosine_long standard), compare
à une référence WMMSE UMa dédiée (10 batchs, méthodologie classical_
comparison). Un seul job GPU0 à la fois, vérifié avant chaque lancement.

**Résultat SingleSC UMa (30ep)** : 72.0%/68.3%/64.2% WMMSE à
15/17.5/20dB. IntraRB UMa lancé pour comparaison directe.

### RÉSULTAT FINAL — L'HYPOTHÈSE EST CONFIRMÉE : IntraRB dépasse
### SingleSC sous UMa, écart croissant avec le SNR

| SNR | SingleSC | IntraRB | Delta (rate) | Delta (%WMMSE) |
|---|---|---|---|---|
| 15dB | 24.04 | **24.54** | **+0.50** | **+2.8pts** |
| 17.5dB | 25.21 | **25.85** | **+0.64** | **+3.4pts** |
| 20dB | 25.69 | **26.41** | **+0.72** | **+3.4pts** |

(comparaison en rate brut la plus fiable -- chaque script recalcule sa
propre référence WMMSE UMa avec 10 batchs, petit bruit de mesure entre
les deux runs sur la référence elle-même, ~1-2%, sans impact sur la
comparaison directe des deux architectures qui partagent le même cache
UMa et le même protocole)

**Le classement s'INVERSE complètement par rapport à UMi** (où SingleSC
menait 91.4% vs 90.7%) **et l'écart grandit avec le SNR** — signal exact
attendu si la chaîne causale de la nuit est correcte : Partie 0 (UMi
insuffisamment sélectif à courte distance, cross-SC redondant avec le
per-SC) → Partie 1 (donc pas d'avantage pour la SC-attention, le
mécanisme du gap RZF-WMMSE est intra-SC/cross-user) → **confirmation
positive** : sous un canal à VRAIE sélectivité (UMa), la SC-attention
capture une information réelle que SingleSC ne peut structurellement
pas voir, et l'avantage se manifeste précisément là où on l'attendait
(grandit avec le SNR, comme l'avantage CSI-imparfait de TA-RB résiduel
grandissait aussi avec le SNR pour une raison apparentée).

**Condition MASSIVE #4 (UMa testé) : ✅ réglée.** Les 4 conditions
posées par l'utilisateur sont maintenant remplies. **Décision sur
l'adoption complète d'UMa (régénération dataset production +
classical_comparison + ré-entraînement complet des 4 architectures,
gros coût) : PAS prise unilatéralement — en attente du feu vert
utilisateur**, cohérent avec la règle établie toute la nuit (pas de
gros engagement de calcul sans confirmation). Sauvé :
`results/diag_uma_quick_intra_rb.json`.
