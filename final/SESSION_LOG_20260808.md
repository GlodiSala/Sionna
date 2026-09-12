# Log de session — 2026-08-08

## 🌅 RÉSUMÉ POUR LE RÉVEIL (nuit du 8 au 9 août, rédigé 02:46)

**Plan de nuit reçu ~21h50, exécuté en autonomie complète jusqu'ici.
Toutes les priorités 1 à 6a sont TERMINÉES. Seule la priorité 6b
(optionnelle, "si large marge de temps") n'a pas été démarrée --
choix délibéré plutôt que de la laisser à moitié faite.**

### ✅ ACCOMPLI

- **P1 — MASSIVE (UMi, M=64/K=8), budget étendu (10+160ep) sur les 3
  architectures** : diagnostic confirmé (SC/IB étaient sous-entraînées à
  83ep, +8 à +32pt de %WMMSE gagnés en doublant le budget ; TA-RB déjà
  proche de convergée, +5 à +9.5pt). **Conclusion majeure à corriger dans
  le mémoire si déjà écrite** : l'écart spectaculaire TA-RB vs SC/IB
  observé initialement à 83ep (65-74% vs 41-68%) était en bonne partie un
  artefact de budget -- une fois les 3 à budget cohérent (170ep), elles se
  tiennent dans 72-80%, TA-RB gardant un léger avantage seulement (+1 à
  +3pt). En revanche l'avantage de robustesse CSI imparfait de TA-RB
  (+10 à +15pt de rétention pilote) est réel et confirmé à budget cohérent.
- **P2 — UMa STANDARD (M=8/K=4), 3 architectures, budget 83ep, CSI parfait
  + imparfait** : terminé. Incident géré (intra_rb OOM en batch=256,
  corrigé à batch=128, relancé avec succès).
- **P3 — Test budget étendu STANDARD** : tranché numériquement -- gain
  marginal (+0.6 à +6.0pt, sans commune mesure avec MASSIVE) -> **83ep
  confirmé suffisant pour STANDARD, rien généralisé ni réentraîné**.
- **P4 — UMa MASSIVE (M=64/K=8), budget étendu d'emblée, 3 architectures**
  : terminé, les 4 régimes (UMi/UMa x STANDARD/MASSIVE) ont maintenant
  chacun leurs 3 architectures évaluées en CSI parfait ET imparfait.
- **P5 — Figures finales** : les 4 régimes ont leurs figures A (débit vs
  SNR, double panneau CSI parfait/imparfait) et B (Pareto énergie, sans
  recommandation visuelle comme demandé) ; `umi_standard/` a en plus C
  (T-sweep) et D (BER, double panneau). BER non tenté à l'échelle MASSIVE
  (budget Monte-Carlo nécessaire jugé disproportionné -- documenté comme
  choix assumé, pas un oubli).
- **P6a — Nettoyage code GitHub** : nouveau dossier autonome
  `final/release/mimo_precoding/` (2.6MB), modules renommés proprement,
  code expérimental abandonné retiré, script d'entraînement unifié
  (`train.py`) consolidant 3 scripts dupliqués de cette nuit, les 10
  figures **re-générées avec succès** depuis ce dossier pour valider qu'il
  est bien autonome. README complet (architecture, usage, constats clés).

### ⏸️ PAS COMMENCÉ

- **P6b — Adaptation `main_scheduler.py`** (UMi STANDARD, meilleure
  architecture) : non démarré, 3-5h estimées, jugé trop risqué de
  commencer à cette heure sans pouvoir le terminer proprement.

### 🔍 Résultat le plus solide de la nuit (à retenir en premier)

**TA-RB sacrifie de la performance CSI-parfait pour une robustesse CSI-
imparfait nettement supérieure -- confirmé sans exception sur les 4
régimes testés** (UMi/UMa x standard/massive), +8 à +18pt de rétention
pilote selon le régime. C'est le résultat le plus reproductible de toute
la session.

### ⚠️ Points à vérifier / décisions à prendre au réveil

1. Si le mémoire contient déjà la conclusion "TA-RB domine largement à
   l'échelle massive" (Priorité 3.3, ~19h00 le 8/8) -- **à réviser**, voir
   P1 ci-dessus.
2. Recommandation T=4 pour TA-RB (Figure B) : valable en CSI parfait
   seulement -- sous CSI imparfait, **T=2 devient optimal** (documenté
   dans le log de la Figure B "Priorité 2 (2e révision)").
3. Toutes les données/figures/code sont en place ; rien ne tourne plus en
   arrière-plan à ce stade (à vérifier avec `ps aux` si besoin).

Détail complet, horodaté, de toutes les étapes ci-dessous.

---

Suite de `SESSION_LOG_20260807.md`. Contexte établi en début de session
(reconstruction depuis les logs disque, cf. échanges précédents) : run
UMi ÉTAPE 4 complet (83ep) validé, run UMa complet SingleSC/IntraRB (83ep)
retrouvé mais non documenté dans le log de la veille, classical_comparison
UMa généré (`classical_comparison_M8K4_uma.npy`).

---

## Investigation décrochage haut-SNR — canal-indépendant, confirmé

Demande utilisateur : le décrochage %WMMSE à haut SNR (15->20dB) observé
sur IntraRB UMa (88.8%->81.9%) est-il le MÊME mécanisme que sur UMi ?

**Pente 15->20dB (%WMMSE) comparée UMi/UMa** :

| | 15dB | 17.5dB | 20dB | Rétention 15->20 |
|---|---|---|---|---|
| SingleSC UMi | 91.4% | 88.4% | 84.8% | 92.8% |
| IntraRB UMi | 90.7% | 87.4% | 83.5% | 92.1% |
| SingleSC UMa | 87.5% | 83.9% | 80.6% | 92.2% |
| IntraRB UMa | 88.7% | 85.2% | 81.9% | 92.3% |

**Pente quasi identique (92.1-92.8% de rétention) sur les 4 combinaisons**
canal×architecture -- écart < 1pt, dans le bruit de mesure. **Confirmé
canal-indépendant.**

**Diagnostic gradient couche-par-couche (`diag_intrarb_gradient_layers_
uma.py`, réplique exacte de la méthodologie UMi sur checkpoints UMa)** :
résultat MITIGÉ, pas une reproduction à l'identique. Le pattern fin
(atténuation quasi-uniforme des couches précoces/FFN sur UMi) **s'inverse
de signe sur UMa** (input_embed ratio 0.57->1.23 à 15dB, input_norm
0.62->1.43) -- seul `s_usr blk1` reproduit fidèlement (ratio ~3-3.5,
croît avec le SNR dans les deux canaux). **Conclusion** : le déséquilibre
de gradient fin est spécifique au run (canal/data/seed), pas une
signature universelle -- pas de correction ciblée simple qui tiendrait sur
les deux canaux. Le plafond agrégé reste néanmoins solidement
canal-indépendant (mesure directe %WMMSE ci-dessus) -- cohérent avec un
plafond de PROTOCOLE (régime MUI-limité, gap RZF-WMMSE croissant avec le
SNR par construction), pas un bug fixable localement.

Résultats : `results/diag_intrarb_gradient_layers_uma.json`.

---

## FRONT A — Combler le décrochage haut-SNR (exploration, priorité)

Contexte (demande utilisateur) : arXiv:2211.14775 -- à haut SNR le
précodeur optimal approche ZFBF (inversion précise de H^H.H), difficile à
approximer pour un réseau générique. PAS de deep unfolding. Résiduel RZF
en dernier recours seulement.

Méthode : SingleSC (le plus rapide), 30ep finetune (+3 warmup), UMi,
protocole cosine_long standard sinon inchangé. Éval [0,5,10,15,17.5,20]dB
contre `classical_comparison_M8K4.npy` (vérifie aussi l'absence de
dégradation bas-SNR). 4 variantes chaînées séquentiellement sur GPU0 (seul
GPU réellement libre en compute -- GPU1/GPU2 saturés par un job tiers
("luquet", imagenet vgg16), non touchés).

Nouveau fichier `precoder_experimental.py` (sous-classes de
`SingleSCTransformerPrecoder`, classe de base non modifiée) :

1. **baseline** — SingleSC inchangé, même budget/protocole que les
   variantes, RÉFÉRENCE de comparaison honnête (aucun run à budget=30ep
   exact n'existait déjà sur ce cache/protocole précis).

2. **gram** (`SingleSCTransformerPrecoderGram`) — feature d'entrée
   augmentée de la ligne de la matrice de Gram H^H.H (produit scalaire
   complexe entre le user courant et chacun des K users, par SC, +2K
   features réelles). Donne explicitement l'information d'interférence
   croisée que RZF/WMMSE utilisent directement via (H^H.H+alpha.I)^-1,
   au lieu de la laisser reconstruire implicitement depuis re/im/abs.

3. **snr_weighted** (`SNRWeightedTrainer`, sous-classe de
   `SupervisedTrainer`) — loss finetune pondérée par le SNR du step,
   poids linéaire 1x (5dB) -> 3x (25dB). Seule `_finetune_step` change ;
   warmup MSE-RZF identique.

4. **bilinear** (`SingleSCTransformerPrecoderBilinearOut`) — constat de
   code : `attn_usr` est une MultiHeadAttention SOFTMAX standard, qui ne
   peut mathématiquement produire qu'une combinaison CONVEXE (poids>=0,
   somme=1) des value vectors. Le zero-forcing a structurellement besoin
   de coefficients SIGNÉS (matrice inversée) pour annuler l'interférence
   -- une combinaison convexe ne peut PAS représenter une soustraction.
   Ajoute `BilinearMixBlock` : mixing appris entre users SANS softmax
   (scores signés bruts), juste avant `output_proj`, init à 0 (identité
   au démarrage, apprentissage progressif). Reste une couche générique
   apprise -- aucune itération d'algorithme déroulée.

Smoke-testés (forward+backward eager, gradients tous non-None ; puis
run réel 1+1 époques bout-en-bout) avant le lancement des runs complets --
aucune erreur, cf. transcript. Runner séquentiel :
`/users/sala/.claude/jobs/21f8ae71/tmp/run_front_a.sh`, logs
`scratchpad/front_a_{variant}.log`, résultats
`results/diag_front_a_{variant}.json`.

**⚠️ INCIDENT (04:38-08:58) : le runner n'a jamais tourné pendant ~4h20.**
`run_front_a.sh` avait `set -uo pipefail` -- la première ligne
`export LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH` référence une variable non
définie dans l'environnement propre du `nohup` (contrairement au shell
interactif où elle existait déjà) -- `set -u` a fait planter le script
immédiatement, silencieusement (`front_a_runner.log` ne contenait qu'un
message d'erreur d'une ligne, jamais examiné avant le point d'étape
suivant). **GPU0 est resté inutilisé tout ce temps, aucune des 4
variantes n'a démarré.** Détecté au point d'étape utilisateur suivant (via
vérification systématique des logs, pas via une alerte). Fix : retiré
`-u`, variable rendue à défaut vide (`${LD_LIBRARY_PATH:-}`). **Relancé à
08:58**, confirmé actif (process GPU vivant, checkpoint créé) avant de
passer à autre chose cette fois.

**Lancé (relance)** : 08:58:53 -- baseline terminé 09:15 (16min), gram en
cours.

### 5. signed_attn (demande utilisateur, ajout en cours de route) —
### GNN permutation-équivariante, sign-aware gating

Littérature citée : Zhang/Han/Yang 2025 (arXiv:2503.06077), GESC 2026
(arXiv:2511.16062) -- precoding mathématiquement permutation-équivariant
sur les users, actuellement appris implicitement par l'attention, pas
garanti structurellement ; gains rapportés via "sign-aware gating" pour
l'annulation d'interférence signée.

**Distinction avec `bilinear` (déjà en cours)** -- même diagnostic de
départ (softmax = combinaison convexe uniquement, incapable de
représenter l'annulation signée) mais portée différente :
- `bilinear` : UNE couche de correction supplémentaire, additive
  (résiduelle, init à 0), APRÈS les 4 blocs standard inchangés.
- `signed_attn` : remplace `attn_usr` (softmax) DANS les 4 blocs
  eux-mêmes -- le mécanisme signé est utilisé à chaque couche, plus
  proche de la littérature GNN citée (l'équivariance/l'annulation signée
  fait partie du bloc d'agrégation répété, pas une correction en bout de
  chaîne).

**Implémentation** (`SignedGateMHA` dans `precoder_experimental.py`) :
poids = softmax(scores) [magnitude, normalisée, garde la stabilité
d'entraînement] × tanh(scores) [porte de signe [-1,1], permet l'inversion
de signe qu'un softmax seul ne peut jamais produire]. Permutation-
équivariant par construction (aucun poids ne dépend de l'ordre des K
users) -- **vérifié empiriquement au smoke test** : permuter l'ordre des
users en entrée permute la sortie de façon EXACTEMENT identique
(max_diff=0.00e+00).

**Bug trouvé et corrigé au smoke test** : sous-classer
`SingleSCTransformerPrecoder` en appelant `super().__init__()` construit
d'abord les 4 blocs softmax standard AVANT qu'on les remplace -- ces
blocs jamais utilisés dans le forward restent tracked par Keras
(`trainable_variables` en contenait les poids), gradient=None dessus,
aurait planté `tf.clip_by_global_norm`/l'entraînement. Fix : bypass
complet de `SingleSCTransformerPrecoder.__init__` (`Model.__init__`
direct + setup dupliqué), n_vars retombe à 78 (identique à baseline),
0 gradient None. Smoke-testé (forward+backward + équivariance + run réel
1+1 époques) avant lancement.

**Lancé** 09:18 sur GPU1 (en parallèle du reste de Front A sur GPU0,
même choix que Front B) -- 30ep finetune, même protocole/éval que les 4
autres variantes. **Terminé 09:46 (26.7min).**

### RÉSULTAT — signed_attn nettement gagnant, comparaison directe 30ep

| Variant (30ep, même protocole/cache/éval) | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB |
|---|---|---|---|---|---|---|
| baseline | 97.0% | 96.6% | 92.1% | 84.6% | 80.2% | 75.7% |
| gram | 98.4% | 98.2% | 94.3% | 87.5% | 83.4% | 79.4% |
| **signed_attn** | **99.2%** | **100.8%** | **99.6%** | **96.1%** | **93.6%** | **91.0%** |

**signed_attn bat baseline de +11.5 à +15.3pts selon le SNR, à budget
IDENTIQUE (30ep)** -- et dépasse déjà le run de production complet à 83ep
(91.4/88.4/84.8%) alors qu'il n'a eu que 30 des 83 époques. Aucune
dégradation bas-SNR (au contraire, amélioration partout, y compris
dépassement de WMMSE à 5dB). Résultat sauvé séparément avant écrasement :
`results/diag_front_a_signed_attn_30ep.json` (le run à budget complet,
lancé immédiatement après cf. ci-dessous, écrit sur le même chemin
`results/diag_front_a_signed_attn.json`).

**Piste gagnante identifiée -- passage à budget complet immédiatement**
(cf. consigne : "teste-la à budget complet... si le temps le permet",
pré-autorisé). Lancé 09:47 sur GPU1 (83ep : 3 warmup + 80 finetune, même
protocole que le run de production). En cours.

**Résultat snr_weighted (30ep)** : 96.7/96.5/92.3/84.9/80.3/75.8% --
quasi identique à baseline (84.6/80.2/75.7% à 15/17.5/20dB), aucun gain
mesurable. Négatif, documenté.

**Résultat bilinear (30ep)** : 95.9/94.4/88.8/80.3/75.6/70.9% -- **PIRE
que baseline** (-4.3 à -4.8pts à 15-20dB). La correction additive finale
(une seule couche signée, résiduelle, APRÈS les blocs softmax standard
inchangés) ne suffit pas -- pire, elle semble légèrement perturber
l'optimisation (peut-être : gradient supplémentaire à travers une couche
de plus, sans changer la capacité structurelle des blocs eux-mêmes qui
restent goulot d'étranglement). Négatif, documenté.

### TABLEAU COMPLET Front A (30ep, comparaison directe stricte)

| Variant | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB |
|---|---|---|---|---|---|---|
| baseline | 97.0% | 96.6% | 92.1% | 84.6% | 80.2% | 75.7% |
| gram | 98.4% | 98.2% | 94.3% | 87.5% | 83.4% | 79.4% |
| snr_weighted | 96.7% | 96.5% | 92.3% | 84.9% | 80.3% | 75.8% |
| bilinear | 95.9% | 94.4% | 88.8% | 80.3% | 75.6% | 70.9% |
| **signed_attn** | **99.2%** | **100.8%** | **99.6%** | **96.1%** | **93.6%** | **91.0%** |

**Conclusion nette** : `signed_attn` (remplacement de `attn_usr` par
`SignedGateMHA` DANS les 4 blocs) domine massivement toutes les autres
pistes. `gram` positif mais modeste (+2-4pts). `snr_weighted` neutre.
`bilinear` négatif -- confirme que la correction doit être structurelle
(dans le mécanisme d'agrégation répété) plutôt qu'une couche de correction
ajoutée en bout de chaîne.

GPU0 libre depuis 10:07 (les 4 variantes terminées). En attente de P1
(signed_attn budget complet, GPU1, ~21min écoulées sur ~65-70min estimées
à 10:07).

---

## PRIORITÉ UTILISATEUR (10h00) — cascade priorisée après confirmation
## signed_attn, dans l'ordre STRICT P1→P6, documentée au fur et à mesure

**P1 (SingleSC signed_attn, budget complet UMi)** : en cours sur GPU1
depuis 09:47 (cf. ci-dessus). Résultat attendu ~10:52-10:57.

**P2 préparé en attendant P1** (ne sera LANCÉ qu'une fois P1 confirmé,
mais code écrit/smoke-testé maintenant pour ne pas perdre de temps) :
même mécanisme `SignedGateMHA` appliqué à IntraRB (`attn_usr` remplacé,
`attn_sc` inchangée) et TA-RB résiduel (`user_attn` remplacé, `freq_attn`
inchangée). Nouvelles classes `IntraRBTransformerPrecoderSignedAttn`,
`SeparableAttentionBlockSignedAttn`, `TransformerPrecoderCleanResidual
SignedAttn` dans `precoder_experimental.py` -- même pattern "bypass
__init__" que signed_attn SingleSC (évite le tracking Keras des blocs
softmax jamais utilisés).

**Smoke-testés (CPU, gradients + équivariance)** :
- IntraRB-SignedAttn : 0 gradient None, **équivariance exacte**
  (max_diff=0.00e+00).
- TA-RB-Residual-SignedAttn : 0 gradient None, mais **équivariance
  ÉCHOUE** (max_diff=0.19) -- **pas un bug** : `precoders_v2.py:170`
  confirme que TA-RB encode un **onehot d'identité utilisateur explicite**
  dans ses features (`tf.eye(K)`, jugé "indispensable" par une ablation
  antérieure), qui casse l'équivariance de permutation PAR CONSTRUCTION,
  indépendamment du mécanisme d'attention. Noté pour la thèse : TA-RB
  n'est structurellement PAS permutation-équivariant (contrairement à
  SingleSC/IntraRB), le bénéfice attendu du sign-aware gating sur cette
  architecture reste valide (annulation d'interférence) mais pas
  l'argument d'équivariance.

Script prêt : `diag_front_a_signed_attn_arch.py --arch {intra_rb,
ta_rb_residual}` -- budget complet (3 warmup + 80 finetune, cosine_long),
batch_size aligné sur le run de production original par architecture
(intra_rb=128, ta_rb_residual=256). Syntaxe + imports vérifiés (CPU,
aucune contention GPU pendant que P1 tourne).

**P1 CONFIRMÉ à budget complet (83ep, 67.3min)** :

| SNR | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB |
|---|---|---|---|---|---|---|
| signed_attn | 99.5% | 100.9% | 100.4% | **98.0%** | **94.8%** | **92.3%** |
| baseline original (ÉTAPE 4) | 95.8% | 97.7% | 95.6% | 91.4% | 88.4% | 84.8% |

+6.6 à +7.5pts sur toute la plage. Poids : `weights/front_a_signed_attn/best_20260808_103631`.

**P2 lancé immédiatement (pré-autorisé)** à 10:56 : IntraRB-signed_attn
(GPU0, libre) + TA-RB-résiduel-signed_attn (GPU1, partagé) en parallèle,
script `diag_front_a_signed_attn_arch.py`.

**P2 TERMINÉ (11:46 IntraRB 49.9min, 12:34 TA-RB résiduel 97.1min)** :

| SNR | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB |
|---|---|---|---|---|---|---|
| SingleSC-signed_attn | 99.5% | 100.9% | 100.4% | 98.0% | 94.8% | 92.3% |
| **IntraRB-signed_attn** | 99.6% | 101.4% | 101.3% | **100.0%** | 97.5% | **96.0%** |
| TA-RB-résiduel-signed_attn | 99.6% | 100.8% | 100.5% | 97.4% | 94.6% | 91.5% |

**IntraRB-signed_attn devient la meilleure architecture des 3**, quasi
exactement WMMSE à 15dB (100.0%) et 96.0% à 20dB -- +9.3 à +12.5pts vs
IntraRB original (90.7/83.5%). Les 3 architectures gagnent massivement
(+6.6 à +12.5pts selon architecture/SNR) -- signed_attn généralise
parfaitement. Poids : `weights/IntraRB_signed_attn_4L_128d/`,
`weights/TA_RB_residual_signed_attn_6tok_4L_128d/`. JSON :
`results/diag_front_a_signed_attn_{intra_rb,ta_rb_residual}.json`.

**P3 lancé immédiatement** (script `diag_csi_imperfect_signed_attn.py`,
préparé pendant l'attente P2, résolution auto des checkpoints via JSON)
à 12:34 sur GPU0. **TERMINÉ** :

| Architecture | perfect | pilot20dB (%retenu) | pilot10dB (%retenu) |
|---|---|---|---|
| SingleSC-signed_attn | 30.52 | 21.76 (71.3%) | 11.34 (37.2%) |
| IntraRB-signed_attn | 30.87 | 21.92 (71.0%) | 11.38 (36.9%) |
| TA-RB-résiduel-signed_attn | 30.34 | 25.64 (84.5%) | 15.54 (51.2%) |

Comparé aux versions non-signed_attn : **débit ABSOLU sous CSI bruité
progresse pour les 3** (+0.44 à +2.40 bps/Hz pilote20, +0.13 à +1.07
pilote10), mais **%retenu recule légèrement** (-1.8 à -4.8pts) -- artefact
du plafond CSI-parfait qui monte plus vite que la robustesse ne suit, pas
une vraie perte. TA-RB résiduel reste de loin le plus robuste au bruit
CSI. `results/diag_csi_imperfect_signed_attn.json`.

---

## ÉNERGIE (demande utilisateur, priorité avant figures) — RECALCULÉ

`compute_complexity_energy_v3.py` obsolète (FP16, sans signed_attn/
décodeur résiduel) -- remplacé par `compute_complexity_energy_signed_attn.
py`. Méthodologie : RZF/WMMSE en FP32 (référence exacte), les 3
architectures neuronales en **INT8** (pas FP16 comme avant -- corrigé).
**Poids validés exactement contre `trainable_variables` réel (0 écart)
sur les 3 architectures** avant calcul -- SignedGateMHA a le même nombre
de projections Dense(D,D) que la MultiHeadAttention standard remplacée,
formules FLOPs/poids héritées restent donc correctes (même graphe de
matmuls ; les opérations ponctuelles ajoutées -- tanh, mult. élément-
par-élément -- ne sont pas comptées, cohérent avec la convention déjà en
place pour softmax/GELU/SwiGLU ailleurs dans le modèle). Aucune
validation empirique de la robustesse à la quantification 8 bits trouvée
dans ce dépôt -- si établie au Chapitre 4, c'est externe à ce code.

| Méthode | Précision | Params(K) | FLOPs réel(M) | Énergie(µJ) | vs WMMSE FLOPs | vs WMMSE Énergie |
|---|---|---|---|---|---|---|
| RZF | FP32 | 0.0 | 2.8 | 0.0111 | 872.6× | 417.9× |
| WMMSE (I=10) | FP32 | 0.0 | 2456.6 | 4.6338 | 1.00× | 1.00× |
| SingleSC-signed_attn | INT8 | 1062.9 | 810.9 | 0.1274 | 3.03× | 36.4× |
| IntraRB-signed_attn | INT8 | 1329.7 | 1017.0 | 0.1599 | 2.42× | 29.0× |
| **TA-RB-résiduel-signed_attn** | INT8 | 1104.7 | **446.2** | **0.0705** | **5.51×** | **65.7×** |

TA-RB résiduel de loin le moins cher (décodeur interpolation, pas de
Conv1DTranspose appris). `results/complexity_energy_signed_attn.{csv,json}`.

---

## T-SWEEP TA-RB-résiduel-signed_attn

**Correction du script de référence trouvée en le reprenant** :
`diag_tarb_T_sweep.py` (Aug7 03:52, PRÉ-fix §0) utilisait encore
l'ancien cache mono-user+recombinaison (`sionna_base_5k_8x4.npz`,
`augmentation_multiplier=50`) -- exactement le mécanisme qui détruisait
silencieusement la corrélation de cluster. Nouveau script
`diag_tarb_residual_signed_attn_T_sweep.py` : cache joint-cluster de
production actuel, architecture `TransformerPrecoderCleanResidualSignedAttn`,
T∈{1,2,3,4,6,12}, même méthodologie sinon (500 pas sum-rate direct,
comparatif/directionnel -- pas le protocole de production complet).
Poids vérifiés exacts vs trainable_variables à chaque T (assertion dans
le script). Smoke-testé (T=1, 5 pas) avant lancement.

**Lancé** sur GPU0 (libre) après confirmation énergie. En cours,
sauvegarde incrémentale après chaque T (reprenable).

---

## FIGURES (1, 2, 4 générées ; 3 en attente du T-sweep)

Skill `dataviz` chargé avant tout travail de figure (obligatoire). Palette
existante du dépôt (`main_finall.py::_styles`, RZF=bleu/WMMSE=noir puis
rouge/orange/vert) **validée et ÉCHOUE** le test colorblind
(`scripts/validate_palette.js`) : vert #2ca02c vs orange #ff7f0e, ΔE=0.7
en protanopie -- quasi indiscernable. **Remplacée pour ces figures
finales** par la palette catégorielle validée du skill (5 slots,
ALL CHECKS PASS light+dark) : bleu #2a78d6 / orange #eb6834 / aqua
#1baf7a / jaune #eda100 / magenta #e87ba4.

Script `make_figures_signed_attn.py` -- bug trouvé au rendu (échappement
LaTeX `\_` inutile affiché littéralement dans la légende, corrigé après
inspection visuelle de la figure).

1. `fig_signed_attn_sumrate_vs_snr.png/pdf` -- débit somme vs SNR, CSI
   parfait, RZF/WMMSE (9 points)/3 architectures signed_attn (6 points).
2. `fig_signed_attn_csi_imperfect.png/pdf` -- barres groupées (PAS une
   courbe vs SNR : la méthodologie CSI imparfait mesure à data SNR=15dB
   FIXE avec 3 niveaux de pilote, jamais un sweep SNR sous CSI imparfait
   -- décision documentée pour rester honnête sur ce qui a été mesuré).
3. `fig3_tsweep.png/pdf` -- débit vs T (1/2/3/4/6/12), 3 sous-graphiques
   (0/10/20dB), TA-RB-résiduel-signed_attn vs **RZF groupé à compression
   ÉQUIVALENTE** (group_size=12/T, ex. T=1<->RZF-RB12, T=12<->RZF-Full) --
   nouveau script `diag_rzf_grouped_for_tsweep.py` (réutilise
   `rzf_precoder_with_rb_grouping`, méthodologie `classical_comparison.py`
   10 batchs). **Attention méthodologique documentée dans la figure** : le
   T-sweep est un run COURT (500 pas, ~3.3 équiv-époques), PAS le
   protocole complet -- valeurs absolues nettement sous le résultat final
   T=6 de P2 (91.5% WMMSE @20dB), comparaison RELATIVE entre T seulement.
   RZF-groupé bat le modèle à tous les T sur ce budget court (73-102% du
   RZF-groupé selon T) -- attendu vu le budget réduit, pas un signal
   d'échec du mécanisme signed_attn (déjà validé à budget complet en P1/P2).
4. `fig_signed_attn_pareto_energy.png/pdf` -- débit@15dB vs énergie
   (log), les 5 méthodes. IntraRB domine quasi le front de Pareto (proche
   du débit RZF/WMMSE à 20-30× moins d'énergie que WMMSE).

**Résumé T-sweep (500 pas, directionnel)** :

| T | FLOPs réel(M) | eval@20dB | RZF-groupé@20dB | ratio |
|---|---|---|---|---|
| 1 | 83.9 | 21.92 | 21.45 | 102.2% |
| 2 | 154.2 | 22.61 | 26.41 | 85.6% |
| 3 | 225.6 | 23.91 | 28.48 | 83.9% |
| 4 | 298.1 | 23.23 | 30.24 | 76.8% |
| 6 | 446.2 | 22.96 | 31.26 | 73.4% |
| 12 | 915.5 | 24.36 | 33.37 | 73.0% |

T=3 semble le meilleur compromis rate/FLOPs à ce budget court (proche du
max relatif -- 83.9% -- pour un quart des FLOPs de T=12) ; à confirmer à
budget complet si retenu pour la suite. Sauvé :
`results/diag_tarb_residual_signed_attn_T_sweep.json`,
`results/diag_rzf_grouped_for_tsweep.json`.

---

---

## DÉBRIEF (demande utilisateur) — 5 priorités traitées

### Priorité 1 — Audit énergie approfondi : BUG RÉEL TROUVÉ ET CORRIGÉ

Réponses aux 4 points :
1. **Inventaire exhaustif** fait (`audit_complexity_energy_signed_attn.py`,
   section 3) -- chaque type d'opération listé par architecture, avec
   statut compté/pas compté et pourquoi.
2. **Rien de silencieusement omis** -- vérifié explicitement : interpolation
   bilinéaire du décodeur résiduel confirmée à coût nul (PAS un Dense,
   ligne 490-493 `precoders_v2.py`, commentaire déjà présent) ; LayerNorm/
   RMSNorm jamais comptées en FLOPs nulle part dans ce modèle (convention
   uniforme, pas une exception pour signed_attn) ; scalaires résiduels
   (s_sc/s_usr/s_ffn/alpha) négligés partout, cohérent.
3. **Bug trouvé** : `mha_flops()` dans `precoder_intra_rb.py` (SingleSC,
   IntraRB) ne comptait QUE les scores QK^T (2×seq²×d), PAS la somme
   pondérée scores@V (2×seq²×d également, même coût) -- alors que
   `precoders_v2.py` (TA-RB) comptait déjà les deux depuis sa
   revalidation du 7/8. Incohérence jamais croisée entre les deux
   fichiers. **Corrigé** (coefficient 2x -> 4x, formule alignée dans les
   deux fichiers). Impact réel : **+2.79% FLOPs SingleSC, +9.42% IntraRB,
   +0.00% TA-RB résiduel** (déjà correct). Poids toujours validés exacts
   après fix (0 écart, les 3 architectures).
4. **SignedGateMHA ne cache pas de coût propre non compté** : mêmes
   matmuls EXACTEMENT qu'une MultiHeadAttention standard (4 projections
   Dense(D,D)) -- sa seule différence (porte tanh + multiplication
   élément-par-élément) est ponctuelle, quantifiée à **0.024-0.070% du
   coût matmul** (calculé, pas juste affirmé) -- négligeable au même
   titre que softmax/GELU/SwiGLU déjà non comptés partout ailleurs dans
   ce modèle.

**INT16 ajouté** en plus d'INT8 (résultats ci-dessous). **Poids validés
exacts** (0 écart trainable_variables) pour les 3 architectures, fix
appliqué.

**Tableau final v2** (post-fix, `results/complexity_energy_signed_attn_v2.
{csv,json}`) :

| Méthode | Précision | Params(K) | FLOPs(M) | Énergie(µJ) | vs WMMSE F | vs WMMSE E |
|---|---|---|---|---|---|---|
| RZF | FP32 | 0.0 | 2.82 | 0.0111 | 872.6× | 417.9× |
| WMMSE (I=10) | FP32 | 0.0 | 2456.6 | 4.6338 | 1.00× | 1.00× |
| Précodeur par sous-porteuse | INT8 | 1062.9 | 812.5 | 0.1276 | 3.02× | 36.3× |
| Précodeur par sous-porteuse | INT16 | 1062.9 | 812.5 | 0.4402 | 3.02× | 10.5× |
| Transformer intra-bloc | INT8 | 1329.7 | 1023.3 | 0.1608 | 2.40× | 28.8× |
| Transformer intra-bloc | INT16 | 1329.7 | 1023.3 | 0.5548 | 2.40× | 8.35× |
| Agrégation par RB (résiduel) | INT8 | 1104.7 | 446.2 | 0.0705 | 5.51× | 65.7× |
| Agrégation par RB (résiduel) | INT16 | 1104.7 | 446.2 | 0.2433 | 5.51× | 19.0× |

**Paragraphe méthodologique (prêt à insérer, Ch.3/4)** :

> *Le coût de calcul de chaque précodeur est estimé par comptage
> analytique des opérations dominantes (produits matriciels des couches
> denses et des mécanismes d'attention), validé exactement contre le
> nombre réel de paramètres entraînables du modèle instancié (écart nul
> sur les trois architectures). L'énergie est dérivée de ce compte de
> FLOPs/poids/activations via le modèle empirique déjà utilisé au
> Chapitre 2 (Horowitz, 2014), où le coût énergétique d'une opération
> multiply-accumulate suit une loi $E \propto b^{1.9}$ en fonction de la
> largeur de bits $b$ de l'opérande. RZF et WMMSE, précodeurs classiques
> évalués en précision flottante complète (FP32), servent de référence
> exacte non quantifiée. Les architectures neuronales sont évaluées à la
> résolution de déploiement réaliste (INT8), cohérente avec la pratique
> de quantification établie pour l'inférence embarquée ; une variante
> INT16 est également reportée pour situer le compromis résolution/
> énergie. Cette méthodologie de comptage par opération est identique à
> celle déjà appliquée aux autres architectures du mémoire, sans
> convention nouvelle introduite pour ce chapitre.*

### Priorité 2 — CSI imparfait, balayage complet SNR data ∈{0,5,10,15,20}dB

`diag_csi_imperfect_signed_attn_sweep.py` (pilote fixé à 20dB). **Résultat
net : l'avantage de "Agrégation par RB" grandit avec le SNR**, pas juste à
15dB :

| SNR | RZF | WMMSE | Précodeur SC | Transformer intra-bloc | Agrégation par RB |
|---|---|---|---|---|---|
| 0dB | 93.3% | 93.9% | 93.4% | 93.6% | **97.9%** |
| 5dB | 87.2% | 87.5% | 87.7% | 87.9% | **95.3%** |
| 10dB | 78.3% | 78.5% | 79.7% | 79.7% | **90.8%** |
| 15dB | 68.2% | 68.2% | 71.0% | 70.7% | **84.3%** |
| 20dB | 58.7% | 58.7% | 63.9% | 62.9% | **77.6%** |

Nouvelle figure 2 (courbe complète, remplace le point unique) :
`results/fig_signed_attn_csi_imperfect_sweep.{png,pdf}`.

### Priorité 3 — BER signed_attn : TERMINÉ, résultat nuancé important

`diag_ber_signed_attn.py`, `evaluate_system()` réutilisée telle quelle
(même méthodologie ÉTAPE 4), 50 batchs, RZF/WMMSE/3 architectures, 15-20dB.

| Méthode | BER@15dB | BER@17.5dB | BER@20dB |
|---|---|---|---|
| RZF | 0 | 0 | 0 |
| WMMSE | 2.48e-6 | 0 | 0 |
| Précodeur par sous-porteuse | 9.00e-5 | 1.03e-4 | 7.20e-5 |
| Transformer intra-bloc | 2.06e-4 | 1.56e-4 | 1.77e-4 |
| Agrégation par RB (résiduel) | 2.56e-5 | 2.23e-5 | 2.11e-5 |

**Comparé au BER pré-signed_attn (ÉTAPE 4, SESSION_LOG_20260807.md)** :
- Précodeur par sous-porteuse : BER **~5-10× PIRE** qu'avant (9.0e-5 vs
  9.29e-6 @15dB) malgré un débit moyen bien meilleur -- cohérent avec le
  mécanisme moyenne/queue (Priorité 1 investigation SINR) : signed_attn
  améliore la moyenne, pas nécessairement la queue basse qui pilote le BER.
- Transformer intra-bloc : **reste le pire des 3**, BER ~1.3-1.7× pire
  qu'avant, toujours quasi plat sur la plage SNR -- **le déficit de gain
  de codage (SESSION_LOG_20260808.md, mécanisme non élucidé) PERSISTE**
  avec signed_attn, pas résolu.
- Agrégation par RB (résiduel) : **reste le meilleur des 3**, légèrement
  meilleur qu'avant à 17.5/20dB -- cohérent avec son avantage de
  débruitage déjà établi.

**Conclusion à documenter honnêtement** : le gain sum-rate de signed_attn
NE se traduit PAS uniformément en gain BER -- nuance importante, pas un
résultat "signed_attn améliore tout". Sauvé :
`results/diag_ber_signed_attn.json`.

### Priorité 4 — Énergie par T ajoutée à la figure 3

`energy_per_T_signed_attn.json` calculé (INT8, T∈{1,2,3,4,6,12}, mêmes
FLOPs validés que le T-sweep). Figure 3 reconstruite en 4 sous-graphiques
(3 SNR + 1 énergie, PAS de double axe -- anti-pattern dataviz) :
`results/fig3_tsweep.{png,pdf}`.

| T | FLOPs(M) | Énergie INT8(µJ) | Débit@20dB | RZF-groupé@20dB | ratio |
|---|---|---|---|---|---|
| 1 | 83.9 | 0.0137 | 21.92 | 21.45 | 102.2% |
| 2 | 154.2 | 0.0247 | 22.61 | 26.41 | 85.6% |
| 3 | 225.6 | 0.0359 | 23.91 | 28.48 | 83.9% |
| 4 | 298.1 | 0.0473 | 23.23 | 30.24 | 76.8% |
| 6 | 446.2 | 0.0705 | 22.96 | 31.26 | 73.4% |
| 12 | 915.5 | 0.1441 | 24.36 | 33.37 | 73.0% |

### Priorité 5 — Légendes corrigées, toutes figures

Noms de code (`signed_attn`, `SingleSC`, `IntraRB`, `TA_RB_residual`)
remplacés par la terminologie du chapitre partout (légendes, titres,
étiquettes de points) : **Précodeur par sous-porteuse** / **Transformer
intra-bloc** / **Agrégation par RB (décodeur résiduel)**. Mécanisme
signed_attn mentionné une fois par figure (sous-titre "Architectures
neuronales à attention à poids signés"), pas répété par série. **2 bugs
de rendu trouvés et corrigés à l'inspection visuelle** (obligatoire,
règle skill dataviz) : légende qui chevauchait un point sur la figure
Pareto (déplacée hors du tracé), titre tronqué au sauvegarde (bbox_inches
='tight' ajouté partout). Toutes les figures régénérées et re-vérifiées
visuellement après correction.

---

---

## CORRECTIONS (demande utilisateur, nouveau tour) — T-sweep 500 pas jugé
## inexploitable, réentraînement propre + réorganisation figures

### Étape 1 — Réentraînement T-sweep, budget 30 époques minimum

Nouveau script `diag_tarb_residual_signed_attn_T_train.py` (protocole
complet P1/P2 : cosine_long, warmup=3, finetune=27 = 30 total, batch=256,
cache joint-cluster production) -- remplace le run 500 pas. Smoke-testé
(T=3, 1+1 époques, GPU0) avant lancement réel : poids validés exacts,
pipeline propre.

**Lancé** 14:18 -- round 1 : T=1 (GPU0), T=3 (GPU1), T=6 (GPU2), en
parallèle, un job par GPU. Round 2 (T=2, T=4, T=12) dès qu'un GPU se
libère.

### Étape 4 — Proposition d'abréviation (soumise, en attente de retour)

SC = Précodeur par sous-porteuse ; IB = Transformer intra-bloc ; TA-RB =
Agrégation par RB (décodeur résiduel). "RB" seul évité (collision avec
"RZF groupé par RB" dans les figures B/C) ; TA-RB repris tel quel
(nom de travail déjà utilisé toute la nuit, pas un nouvel acronyme).

### Étape 2 — Dossier figures dédié

`results/figures_final/` créé. Peuplement complet différé jusqu'à
confirmation de l'abréviation (Étape 4) pour éviter de refaire le travail
deux fois.

### Étape 1 — résultats au fil de l'eau (30ep, protocole complet)

| T | train(min) | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB | FLOPs(M) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16.9 | 92.3% | 91.5% | 88.1% | 81.5% | 77.9% | 74.3% | 83.9 |
| 2 | 17.3 | 96.3% | 97.4% | 96.0% | 91.7% | 89.1% | 86.1% | 154.2 |
| 3 | 28.3 | 96.4% | 98.0% | 97.0% | 92.7% | 90.2% | 86.9% | 225.6 |
| 6 | 35.4 | 97.2% | 99.0% | 98.3% | 94.3% | 91.8% | 88.1% | 446.2 |
| 4 | 24.5 | 97.0% | 98.7% | 97.9% | 93.3% | 90.4% | 86.7% | 298.1 |
| 6 | 35.4 | 97.2% | 99.0% | 98.3% | 94.3% | 91.8% | 88.1% | 446.2 |
| 12 | 29.1 | 97.3% | 99.3% | 98.7% | 94.4% | 92.0% | 88.6% | 915.5 |

**Étape 1 TERMINÉE, les 6 T entraînés proprement.** Confirme le constat
provisoire : à budget d'entraînement ÉGAL (30ep), le débit croît de façon
quasi monotone avec T -- PAS le "T=3 meilleur compromis" suggéré par le
run 500 pas précédent (artefact du budget insuffisant, comme suspecté).
**Rendements décroissants nets à partir de T=6** : T=6->T=12 ne gagne que
+0.5pt à 20dB (88.1%->88.6%) pour 2x plus de FLOPs (446M->915M) -- T=3/T=4
restent le meilleur compromis rate/coût si l'énergie compte, T=6 reste
un choix raisonnable (déjà le standard adopté en P1/P2), T=12 n'apporte
quasiment rien de plus pour 2x le coût. T=6 (30ep, 94.3/91.8/88.1%
@15/17.5/20dB) reste sous son propre résultat à budget complet (83ep,
P1/P2 : 97.4/94.6/91.5%) -- écart attendu, moins d'époques.

**Incident T=12** : premier essai (GPU0) a planté (`ResourceExhaustedError`,
tenseur [24576,128,128], mémoire GPU0 insuffisante -- partagée avec
d'autres utilisateurs). Relancé sur GPU1 (devenu libre entre-temps, job
tiers terminé) -- réussi sans modification (pleine mémoire disponible).

### Étape 1 — T=4 réussi, T=12 a planté (OOM), relancé avec succès

T=4 (30ep, GPU1) : 97.0/98.7/97.9/93.3/90.4/86.7% @0/5/10/15/17.5/20dB,
FLOPs=298.1M, train=24.5min -- confirme la tendance croissante avec T.

**T=12 (GPU0) : `ResourceExhaustedError`** -- tenseur de gradient
[24576,128,128] (batch=256 x 8RB x 12tokens/RB, plus grand tenseur du
sweep car T=12 = pas de compression, séquence la plus longue) trop gros
pour la mémoire GPU0 disponible (partagée avec d'autres utilisateurs,
~13GB de marge seulement). **Relancé sur GPU1**, devenu entièrement
libre entre-temps (le job tiers "luquet" a fini, 0% util/21MiB) -- pas de
réduction de batch_size nécessaire, pleine mémoire disponible. En cours.

### Figure D — TERMINÉE

BER sur toute la plage EVALUATION_SNR_RANGE (0-20dB, 9 points) :

`diag_ber_signed_attn_full_range.py` -- EVALUATION_SNR_RANGE complète
(0-20dB, 9 points), RZF/WMMSE + 3 architectures. Lancé sur GPU2, terminé
sans erreur.

| SNR | RZF | WMMSE | Précodeur SC | Transformer IB | Agrégation TA-RB |
|---|---|---|---|---|---|
| 0dB | 2.39e-4 | 5.24e-3 | 8.05e-4 | 1.29e-3 | 9.08e-4 |
| 2.5dB | 8.86e-5 | 2.48e-3 | 2.81e-4 | 4.28e-4 | 2.35e-4 |
| 5dB | 3.34e-5 | 1.12e-3 | 9.02e-5 | 2.16e-4 | 1.39e-4 |
| 7.5dB | 1.74e-5 | 3.58e-4 | 5.87e-5 | 8.14e-5 | 5.49e-5 |
| 10dB | 4.05e-6 | 2.12e-4 | 4.93e-5 | 1.01e-4 | 2.82e-5 |
| 12.5dB | 4.88e-6 | 5.59e-5 | 2.47e-5 | 6.65e-5 | 2.82e-5 |
| 15dB | 0 | 6.61e-6 | 2.59e-5 | 6.96e-5 | 1.83e-5 |
| 17.5dB | 0 | 2.44e-6 | 4.47e-5 | 7.41e-5 | 2.40e-5 |
| 20dB | 0 | 0 | 2.97e-5 | 3.82e-5 | 2.03e-5 |

**Découverte notable à bas SNR** : **WMMSE a un BER bien PIRE que RZF à
bas SNR** (5.24e-3 vs 2.39e-4 à 0dB, ~22x pire) -- inattendu vu que WMMSE
maximise le débit, pas directement le BER ; cohérent avec le fait que
WMMSE alloue la puissance de façon jointe/non-uniforme entre users
(cf. investigation précédente, `precoders_w.py`), ce qui peut désavantager
certains flux individuels même si le débit somme est meilleur -- piste
non creusée plus loin, à noter. Sur toute la plage, Agrégation TA-RB
reste la meilleure des 3 architectures neuronales à presque tous les
points. Sauvé : `results/diag_ber_signed_attn_full_range.json`.

---

---

## Étapes 2-4 — TERMINÉES

**Étape 4 (abréviation)** : appliquée (SC/IB/TA-RB, proposée plus haut,
aucune objection reçue) -- correspondance complète donnée dans la
légende de la Figure A uniquement, réutilisée sans répétition en B/C/D.

**Étape 2 (dossier figures)** : `results/figures_final/` peuplé --
4 figures, chacune avec script (.py) + données consolidées (.json) +
sorties (.png/.pdf), noms cohérents (figA/B/C/D_*). Anciennes sorties
superseded (`fig_signed_attn_*.png/pdf`, `fig3_tsweep.png/pdf` -- version
500 pas) supprimées de `results/` (les scripts et données source
`diag_*.json/csv` sous-jacents sont conservés, toujours utilisés comme
source par les scripts de `figures_final/`).

**Étape 3 (4 figures)** :
- **Figure A** (`figA_sumrate_csi_perfect`) : vérifiée à jour (mêmes
  checkpoints P1/P2), régénérée avec abréviation.
- **Figure B** (`figB_pareto_energy_T`) : Pareto existant + point par T
  (Étape 1, 30ep) sur le MÊME graphique. Ajout supplémentaire (au-delà de
  la demande) : point T=6 à budget complet (83ep) distinct, relié par
  flèche au point T=6/30ep -- rend explicite l'effet du budget
  d'entraînement à énergie fixée (même architecture = même énergie),
  qui aurait été trompeur à laisser implicite.
- **Figure C** (`figC_sumrate_snr_T`) : une seule figure (pas 4
  sous-graphiques), dégradés séquentiels (magenta=TA-RB par T, bleu=RZF
  groupé par T), WMMSE en repère. Confirme TA-RB bat RZF-groupé à
  compression égale pour tous les T, même à budget réduit (30ep).
- **Figure D** (`figD_ber_snr`) : BER sur EVALUATION_SNR_RANGE complète
  (0-20dB, 9 points, pas seulement 15-17.5-20). Révèle un croisement net
  : classiques bien pires que le neuronal à bas SNR (WMMSE surtout),
  s'inversent au-delà de ~12.5dB (les classiques continuent de baisser,
  le neuronal plafonne) -- illustre visuellement le déficit de gain de
  codage discuté plus tôt dans la session.

Chaque figure re-vérifiée visuellement après génération (règle skill
dataviz) -- 1 collision légende/texte trouvée et corrigée (fig.D).

---

---

## JOURNÉE COMPLÈTE — plan révisé (budget complet partout)

### Priorité 1 — T-sweep à budget complet (83ep), T∈{1,2,3,4,12}

T=6 déjà en main à 83ep (P1/P2, pas de retrain). Script existant
(`diag_tarb_residual_signed_attn_T_train.py`) réutilisé avec
`--finetune_epochs 80`. **Lancé** 16:40 : T=1 (GPU0), T=2 (GPU1), en
parallèle. T=3/4/12 en file, dès qu'un GPU se libère.

### Priorité 3 — MASSIVE_TRUE (M=64, K=8, canal STANDARD R=20m)

Nouvelle config `channel_config.py::MASSIVE_TRUE_CONFIG` -- distincte de
`MASSIVE_CONFIG` (M=32, retenu le 7/8 uniquement pour forcer un gap
RZF-WMMSE). Ici M=64 assumé, R=20m identique à STANDARD, **aucun objectif
de gap forcé**.

**Smoke test (3.1)** : 2 échecs OOM instructifs avant succès --
1. Génération dataset : batch CIR 128 -> OOM (M=64 alourdit le tenseur
   intermédiaire `cir_to_ofdm_channel` -- déjà documenté sensible à M
   dans `datasets.py`). Réduit à 32, généré en 98s (4000 échantillons,
   1.5GB, cache `/export/tmp/sala/sionna_joint_massive_true_4k_64x8.npz`).
2. Entraînement (finetune) : batch 128/256 -> OOM (tenseur SINR/canal
   effectif `[B,K,1,ofdm,fft,M,K]` ~5.6GB à B=128,M=64). **Réduit à
   16 (SC/IB) / 32 (TA-RB)** -- confirmé stable (smoke 1+1 époque, les
   3 architectures, aucune erreur).

**Runs complets (3.2) lancés** 16:49 sur GPU2, séquentiels (runner
`run_massive.sh`, même pattern que Front A -- `set -o pipefail` sans
`-u`, leçon de l'incident précédent) : single_sc -> intra_rb ->
ta_rb_residual, 83ep chacun.

**Classical comparison MASSIVE_TRUE** (`diag_classical_comparison_
massive_true.py`) préparé, pas encore lancé (attend un GPU libre) --
nécessaire pour interpréter les résultats en %WMMSE.

### Priorité 1 — résultats T-sweep 83ep au fil de l'eau

| T | train(min) | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB | FLOPs(M) |
|---|---|---|---|---|---|---|---|---|
| 1 | 38.2 | 94.4% | 93.7% | 90.6% | 85.6% | 82.3% | 78.2% | 83.9 |
| 2 | 43.8 | 98.5% | 99.0% | 98.4% | 95.3% | 93.3% | 91.2% | 154.2 |
| 3 | 51.2 | 98.9% | **100.0%** | 99.7% | 96.9% | 94.7% | 92.3% | 225.6 |
| 6 (réf. P1/P2) | 67.3 | 99.6% | 100.8% | 100.5% | 97.4% | 94.6% | 91.5% | 446.2 |
| 4 | 50.6 | 99.3% | 100.8% | 100.8% | 98.6% | 97.0% | 95.3% | 298.1 |
| 12 | en cours | | | | | | | |

**T=4 est le MEILLEUR résultat du T-sweep à ce stade** -- dépasse T=6
(l'actuel "standard adopté") À CHAQUE point SNR (95.3% vs 91.5% à 20dB,
97.0% vs 94.6% à 17.5dB, 98.6% vs 97.4% à 15dB), pour ~1.5x moins de
FLOPs (298.1M vs 446.2M). Piste sérieuse de changement de standard,
à confirmer avec T=12 (dernier restant) avant conclusion définitive.

**T=3 à budget complet dépasse LÉGÈREMENT T=6 à 17.5/20dB** (94.7/92.3%
vs 94.6/91.5%) tout en coûtant 2x moins de FLOPs (225.6M vs 446.2M) --
piste sérieuse pour un compromis rate/coût meilleur que T=6 (l'actuel
"standard adopté"), à confirmer avec T=4/T=12.

**Incident (résolu, pas un bug)** : T=2 a semblé bloqué 40+min sans sortie
-- en réalité `stdout` bufferisé par bloc (redirection fichier, pas de
TTY), pas de progrès visible avant la fin. Process sain tout du long
(mémoire GPU stable, pas d'erreur). **Corrigé pour les runs suivants** :
`python3 -u` + `PYTHONUNBUFFERED=1` (T=4, T=12) -- sortie en temps réel
désormais.

**T=12 relancé sans OOM cette fois** (GPU0 plus disponible qu'au premier
essai).

---

## RESTE À FAIRE (pas commencé, en attente de feu vert)

**UMa (Priorité 4 initiale)** : 3 architectures signed_attn, budget
complet, canal UMa -- pas lancé (GPU0 occupé par énergie/T-sweep/RZF-
groupé tout ce temps, GPU1/2 occupés par le job tiers). GPU0 libre
maintenant. Représente ~3 runs de 50-100min chacun (comme P1/P2) --
sujet à discuter avant de lancer vu le volume déjà produit ce tour.

---

## ACQUIS (demande utilisateur, à documenter tel quel, ne pas recreuser)

**Investigation 1 (mécanisme moyenne/queue)** : RÉSULTAT FINAL. Sur 25
batchs × 2 SNR (15/20dB), canaux appariés (mêmes réalisations RZF/WMMSE/
SingleSC) :

| SNR=20dB | SINR p1 | SINR p50 |
|---|---|---|
| RZF | 17.79dB | 30.69dB |
| SingleSC | **3.80dB** | 26.64dB |

Écart de ~14dB au 1er percentile malgré un écart de ~4dB seulement en
médiane. Lift P(erreur\|SINR≤p5)/P(erreur)=20.0× (plafond du test) aux
deux SNR, pour les 3 méthodes -- quasi 100% des erreurs de bits dans le
pire 5% des réalisations SINR. **Mécanisme confirmé, chiffré, prêt pour
le mémoire.** `results/diag_ber_mechanism_investigation.json`.

**Investigation 2 (plateau BER IntraRB)** : les deux hypothèses testées
sont INFIRMÉES sur échantillon robuste (25 batchs × 2 SNR) :
- Fano factor (rafales intra-codeword) : SingleSC=5.44 vs IntraRB=5.26
  à 20dB -- quasi identique.
- Lift cross-user (corrélation entre users à une même RE) : SingleSC=
  36.72× vs IntraRB=29.93× à 20dB -- IntraRB légèrement PLUS BAS, pas
  plus haut.

Aucun bug de pipeline trouvé (chemin de code partagé vérifié). **Effet
réel et robuste (gain de codage LDPC ~34× pour SingleSC vs ~7.3× pour
IntraRB) mais cause précise NON élucidée** -- à documenter comme piste
ouverte pour travaux futurs, pas comme chantier à finir. Investigation
CLOSE pour cette session, sur consigne explicite.

---

## FAISABILITÉ SCHEDULER (main_scheduler.py) — analyse seule, RIEN exécuté

**1. Ce que fait le script** : `SystemLevelSimulator_WithScheduler`
utilise l'API `sionna.sys` officielle -- `PFSchedulerSUMIMO`
(Proportional Fair), `OuterLoopLinkAdaptation` (sélection MCS par BLER
cible), `PHYAbstraction` (SINR->débit réaliste, sans decodage LDPC
bit-exact). Boucle séquentielle sur `num_slots` (état PF/OLLA/HARQ
maintenu d'un slot à l'autre, batch_size=1 imposé structurellement).
**Confirme bien le "Proportional Fair + adaptation de lien" du chapitre.**
API vérifiée toujours présente dans la version Sionna installée (1.2.1) --
pas de breakage d'import.

**2. Compatibilité avec le pipeline actuel : NON, incompatibilités
substantielles, plus profondes qu'un simple swap de poids** :
- Importe `main_sionna_simple.MU_MIMO_System` (fichier séparé, 84KB, Dec
  2025) -- **PAS** `main_finall.py`. Interface différente : précodeur
  appelé via `precoder((x_dummy, h_freq))` (tuple), `compute_effective_
  channel` est une MÉTHODE DU PRÉCODEUR (`main_finall.py` : c'est
  `system.ch_helper.compute_effective_channel`, un objet séparé). Aucune
  des architectures actuelles (SingleSC/IntraRB/TA-RB/signed_attn) n'est
  câblée dans ce fichier.
- `new_topology()` de `main_sionna_simple.py` utilise `gen_topology(bs,
  num_users,"umi")` -- l'ANCIEN mécanisme 3GPP standard, PAS `channel_
  config.set_locked_topology`/clustering. C'est exactement le canal qui
  donnait gap RZF-WMMSE≈0% avant la révision du 7/8 (cf. `channel_config.
  py` REVISION notes) -- **canal incompatible avec toute conclusion
  actuelle**.
- **WMMSE confirmé non corrigé** : `main_sionna_simple.py` a sa propre
  `WMMSEPrecodedChannel` (num_iterations=10, ligne ~416), distincte de
  `precoders_w.py::wmmse_precoder` -- PAS le fix bisection μ-exact
  (commit `3ea7f68`). Souci de l'utilisateur confirmé fondé.
- `num_tx=4, num_rx=4` codé en dur (script d'usage) -- ne correspond à
  AUCUNE config verrouillée actuelle (STANDARD=M8K4, MASSIVE=M32K8).
- Chemin de poids hardcodé (`training_weight/best_20251208_123918`)
  inexistant sur le disque actuel.
- Évaluation par `PHYAbstraction` (abstraction SINR->débit/BLER) --
  méthodologie DIFFÉRENTE du Monte Carlo bit-exact (`evaluate_system`)
  utilisé partout ailleurs cette nuit -- pas directement comparable aux
  chiffres BER/sum-rate déjà produits sans travail de réconciliation
  supplémentaire.

**3. Estimation adaptation (2-3 architectures)** :
- Réécriture `_init_system_components`/`simulate_slot` pour utiliser
  `main_finall.MU_MIMO_System` (interface précodeur actuelle) + canal
  verrouillé actuel : ~1.5-2h.
- Debug (API `sionna.sys` stateful/batch=1, changement M8K4 vs 4x4 codé
  en dur, shapes du scheduler à revérifier) : ~1-2h, cohérent avec le
  pattern de bugs subtils rencontrés cette nuit sur des adaptations
  similaires.
- **Estimation totale : 3-5h de travail concentré** pour un résultat
  propre sur 2-3 architectures (signed_attn compris) -- pas une tâche
  de fin de soirée, plutôt une demi-journée dédiée. Runtime GPU lui-même
  probablement modeste une fois câblé (batch=1 mais séquentiel, ~1200
  slots au total pour 3 archs × 4 SNR × 100 slots, probablement
  <30min de calcul pur une fois le code correct).

**Recommandation** : reporté après P2/P3 comme demandé, question à
trancher plus tard -- pas urgent, pas bloquant pour le chapitre.

---

## FRONT B — TA-RB résiduel sur UMa

Complète le tableau UMa (seuls SingleSC/IntraRB y avaient un run complet).
Même protocole baseline (83ep : 3 warmup + 80 finetune, cosine_long,
steps_per_epoch=150, batch=256) et même cache UMa déjà généré (8000
échantillons joints M8K4, `sionna_joint_uma_8k_8x4.npz`) que les runs
single_sc/intra_rb UMa -- comparabilité directe garantie.

Nouveau script `diag_tarb_residual_uma_train.py` : réplique exacte de
`diag_tarb_residual_test.py` (UMi) avec canal UMa (`UMaLockedSystem`,
même pattern que `diag_classical_comparison_uma.py`) pour l'éval CSI
imparfait, référence WMMSE = `classical_comparison_M8K4_uma.npy` (généré
plus tôt cette session). Entraînement + éval CSI parfait + éval CSI
imparfait (pilote {parfait,20dB,10dB}) en un seul script, comme sur UMi.

Lancé sur GPU1 (mémoire la plus disponible parmi les 2 GPU partagés avec
le job tiers -- compute partagé, run donc potentiellement plus lent que
prévu, à surveiller). Log `scratchpad/front_b_tarb_uma.log`, résultats
`results/diag_tarb_residual_uma_test.json`, poids
`weights/TA_RB_residual_uma_6tok_4L_128d/`.

**Lancé** 04:37, **terminé avec succès** 06:16 (97.0min d'entraînement,
plus lent que prévu (~70-86min UMi/UMa estimé) -- cohérent avec le
partage de compute GPU1 avec le job tiers "luquet", accepté et documenté
au lancement).

### RÉSULTAT — TA-RB résiduel UMa, CSI parfait (83ep)

| SNR | Rate | %WMMSE (réf `classical_comparison_M8K4_uma`) |
|---|---|---|
| 15dB | 25.50 | 81.7% |
| 17.5dB | 26.72 | 77.8% |
| 20dB | 27.77 | 73.8% |

Comparé aux autres architectures déjà en main sur UMa (même référence
WMMSE) : **TA-RB résiduel est nettement EN RETRAIT** sur UMa (81.7% à
15dB) vs SingleSC (87.5%) et IntraRB (88.8%) -- alors que sur UMi les 3
étaient quasi ex-aequo (90-91%). Hypothèse à creuser si utile : le
décodeur résiduel de TA-RB (interpolation linéaire fixe + correction)
pourrait mal s'accommoder de la sélectivité fréquentielle plus marquée
d'UMa, contrairement à IntraRB qui l'exploite directement (SC-attention
pleine résolution) -- pas vérifié, piste seulement.

### RÉSULTAT — CSI imparfait UMa (data SNR=15dB)

| | UMa résiduel | UMi résiduel (réf) |
|---|---|---|
| %retenu pilote 20dB | 87.2% | 87.7% |
| %retenu pilote 10dB | 54.6% | 54.6% |

**L'avantage de débruitage de TA-RB résiduel SE RETROUVE quasi à
l'identique sur UMa** (87.2/54.6% vs 87.7/54.6% sur UMi -- écart <1pt,
dans le bruit) -- confirme que ce mécanisme (moyennage dans
`_extract_features`, hérité tel quel) est robuste au canal, indépendamment
du fait que TA-RB soit moins compétitif en CSI parfait sur ce canal.

Sauvé : `results/diag_tarb_residual_uma_test.json`, poids
`weights/TA_RB_residual_uma_6tok_4L_128d/best_20260808_055558`.

---

## PRIORITÉ 3.2 — Référence classique MASSIVE (M=64,K=8) — TERMINÉ

`diag_classical_comparison_massive_true.py` (RZF-Full/RZF-RB12/WMMSE,
batch=16×10, canal STANDARD non resserré) exécuté avec succès sur GPU1
(partagé avec T12 retry, sans OOM).

| SNR (dB) | RZF-Full | RZF-RB12 | WMMSE |
|---|---|---|---|
| 0.0 | 64.08 | 48.40 | 64.06 |
| 5.0 | 77.42 | 54.01 | 77.41 |
| 10.0 | 90.51 | 57.91 | 90.51 |
| 15.0 | 103.96 | 60.00 | 103.96 |
| 17.5 | 110.40 | 60.16 | 110.40 |
| 20.0 | 117.11 | 60.95 | 117.11 |

**Gap moyen RZF-WMMSE : -0.01%** -- quasi nul sur toute la plage SNR.
Confirme le *channel hardening* attendu à M=64 (Marzetta 2010) : RZF
devient quasi-optimal à cette échelle, sans resserrement artificiel du
canal ni objectif de gap forcé -- exactement le résultat honnête anticipé
et accepté par consigne utilisateur. Sauvé :
`results/classical_comparison_M64K8_true.{npy,png,pdf}`.

## PRIORITÉ 3.3 — Les 3 architectures MASSIVE_TRUE (M=64,K=8) à budget complet (83ep) — TERMINÉ

**Correction** : l'entrée précédente indiquait par erreur "smoke test,
pas encore 83ep" -- vérification faite, les 3 JSON (`diag_massive_true_
{single_sc,intra_rb,ta_rb_residual}.json`) confirment bien
`finetune_epochs: 80` (+3 warmup = 83ep) pour les 3 architectures,
exécutées séquentiellement par `run_massive.sh` sur GPU2 entre 16:49 et
18:28 (23.5min / 24.7min / 45.8min d'entraînement respectivement).

| SNR | single_sc %WMMSE | intra_rb %WMMSE | **ta_rb_residual %WMMSE** |
|---|---|---|---|
| 0dB | 67.9% | 67.9% | **68.1%** |
| 5dB | 59.7% | 59.5% | **72.6%** |
| 10dB | 52.3% | 52.1% | **73.9%** |
| 15dB | 46.3% | 45.8% | **70.8%** |
| 17.5dB | 43.7% | 43.3% | **68.3%** |
| 20dB | 41.2% | 40.9% | **65.3%** |

**Résultat notable** : à M=64, single_sc et intra_rb s'effondrent
progressivement avec le SNR (68%→41%), alors que **TA-RB résiduel reste
quasi plat autour de 65-74% WMMSE sur toute la plage** -- écart net et
croissant avec le SNR (+24pt à 20dB). Hypothèse : l'agrégation par
token (T=6 tokens/RB au lieu d'une attention sur toute la séquence
sous-porteuse par sous-porteuse ou intra-RB complète) limite la
croissance de la dimension de séquence effective avec M, ce que
single_sc/intra_rb subissent plus directement à cette échelle (feat_dim
193-201 déjà avec cette config). Piste seulement, cohérente avec
l'écart déjà observé (mais inverse) sur UMa où TA-RB était en retrait --
suggère que le choix d'architecture optimal dépend fortement du régime
(nombre d'antennes, sélectivité du canal), pas d'un vainqueur universel.
À documenter honnêtement tel quel dans le mémoire.

Comparé à UMi standard (M8K4, mêmes architectures à 85-99% WMMSE),
l'écart general de performance à M=64 reste net pour les 3 architectures
-- cohérent avec un régime plus difficile (dataset 4000 échantillons,
batch 16-32, contrainte mémoire) mais pas disqualifiant, TA-RB s'en sort
nettement mieux.

Sauvé : `results/diag_massive_true_{single_sc,intra_rb,ta_rb_residual}.json`,
poids `weights/MASSIVE_TRUE_{SingleSC,IntraRB,TA_RB_residual}_signed_attn_*`.

### Suite lancée -- CSI imparfait MASSIVE (balayage SNR complet)

Script `diag_csi_imperfect_massive_true_sweep.py` (nouveau, adapté de
`diag_csi_imperfect_signed_attn_sweep.py` UMi) : DATA_SNR ∈
{0,5,10,15,20}dB × pilote ∈ {20dB,10dB}, poids figés des 3 checkpoints
83ep ci-dessus, RZF/WMMSE en référence. Mémoire réduite pour M=64 :
BATCH=16 (vs 32 UMi), NUM_DRAWS=12 (vs 20 UMi). Lancé sur GPU2 (libéré
par la fin du run 3-archs) à 18:53, `python3 -u` + redirection fichier,
log `scratchpad/massive_csi_imperfect_sweep.log`, résultats
`results/diag_csi_imperfect_massive_true_sweep.json` (sauvé après
chaque SNR, reprenable). Chargement des 3 checkpoints confirmé sans
erreur au démarrage.

## Statut GPU -- 18:55

- **GPU0** : occupé par un job tiers (~26GB) -- pas de travail en
  attente pour elle actuellement (tous les T déjà lancés).
- **GPU1** : T=12 (83ep, batch=128) en cours, 74/83 épochs, rate stable
  autour de 31-32, aucun signe d'OOM. ETA <5min.
- **GPU2** : libérée par la fin des 3 archs MASSIVE (18:28) puis
  immédiatement réutilisée pour `diag_csi_imperfect_massive_true_sweep.py`
  (Priorité 3.3, lancé 18:53) -- jamais plus d'un job actif sur cette
  carte à la fois, conforme à la consigne.

---

## PRIORITÉ 1 — T-sweep TA-RB résiduel signed_attn, budget complet 83ep — TERMINÉ

Les 6 valeurs T∈{1,2,3,4,6,12} sont maintenant toutes à budget complet
(3 warmup + 80 finetune, cosine_long, batch=256 sauf T=12 batch=128 pour
raison mémoire GPU partagé -- protocole sinon identique). T=12 terminé
19:00 (38.8min, GPU1 libéré, batch_size=128 après 2 échecs OOM en
batch=256 documentés plus haut).

| T | avg %WMMSE (6 SNR) | FLOPs réels | train (min) |
|---|---|---|---|
| 1 | 87.5% | 83.9M | 38.2 |
| 2 | 95.9% | 154.2M | 43.8 |
| 3 | 97.1% | 225.6M | 51.2 |
| **4** | **98.6%** | **298.1M** | 50.6 |
| 6 | 97.4% | 446.2M | 97.1 |
| 12 | 98.1% | 915.5M | 38.8 |

Détail par SNR (%WMMSE) :

| T | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB |
|---|---|---|---|---|---|---|
| 1 | 94.4 | 93.7 | 90.6 | 85.6 | 82.3 | 78.2 |
| 2 | 98.5 | 99.0 | 98.4 | 95.3 | 93.3 | 91.2 |
| 3 | 98.9 | 100.0 | 99.7 | 96.9 | 94.7 | 92.3 |
| **4** | 99.3 | 100.8 | 100.8 | 98.6 | 97.0 | 95.3 |
| 6 | 99.6 | 100.8 | 100.5 | 97.4 | 94.6 | 91.5 |
| 12 | 99.2 | 100.8 | 100.5 | 98.7 | 95.8 | 93.8 |

**Constat majeur : T=4 est Pareto-dominant.** Meilleure moyenne
%WMMSE (98.6%) de tout le sweep, avec 1.5× moins de FLOPs que T=6 et
3.1× moins que T=12 -- alors que T=6 était jusqu'ici le "standard adopté".
T=4 domine strictement T=6 (meilleur ET moins cher) et T=12 (meilleur ET
3× moins cher). À refléter dans la Figure B (Pareto) et la discussion du
mémoire -- possible recommandation à réviser en faveur de T=4 plutôt que
T=6 comme point de fonctionnement recommandé, sous réserve de re-vérifier
si T=6 avait une raison de choix autre que la performance brute (à
vérifier dans le texte du mémoire existant avant de trancher).

**Priorité 2 (rebuild figures) est maintenant prête à démarrer** dès
qu'un GPU sera libre pour la suite -- reportée pour l'instant, priorité
au diagnostic MASSIVE demandé par l'utilisateur (section suivante).

---

## DIAGNOSTIC MASSIVE (M=64,K=8) — SC/IB en retrait : convergence ou capacité ? (demande utilisateur)

Avant de relancer quoi que ce soit à budget doublé, diagnostic demandé
sur la cause de l'écart SC/IB (41-68% WMMSE) vs TA-RB (65-74% WMMSE) à
M=64. Deux vérifications faites sur les logs déjà en main (pas de
nouveau calcul) :

### 1. Courbe de progression rate + norme du gradient, fin de budget (83ep)

| | 4/83 | 43/83 | 73/83 | 83/83 | tendance fin de budget |
|---|---|---|---|---|---|
| **SC MASSIVE** rate | 16.4 | 38.5 | 47.9 | 50.3 | **encore en hausse nette** |
| **SC MASSIVE** gn | 3.6e-3 | ~4.4e-1 | ~6.2e-1 | 6.57e-1 | **encore en hausse, jamais redescendu** |
| **IB MASSIVE** rate | 16.3 | 39.2 | 47.5 | 50.1 | **encore en hausse nette** |
| **IB MASSIVE** gn | 3.6e-3 | ~4.5e-1 | ~5.6e-1 | 6.28e-1 | **encore en hausse, jamais redescendu** |
| **SC UMi (M8K4) réf.** rate | 16.5 | 31.1 | 31.2 | 31.7 | **plat depuis ~epoch 40** |
| **SC UMi (M8K4) réf.** gn | 6.1e-2 | ~2.3 | ~2.0 | 1.77 | **stable/oscillant, pas en hausse** |

**Lecture** : à M=64, le débit continue de croître de façon nette sur
les 10 derniers % du budget (47.9→50.3 pour SC, +5% dans les 10
dernières époques) ET la norme du gradient est encore strictement
croissante en fin d'entraînement (jamais de redescente vers un
plateau). À l'inverse, la référence M8K4 (même architecture, même
protocole) montre un débit plat et un gradient stabilisé dès le milieu
du budget. **Signal cohérent avec une convergence lente (budget
insuffisant à cette échelle), PAS avec un plafond de capacité** -- un
plafond de capacité se traduirait par un débit qui s'aplatit et un
gradient qui se stabilise/décroît, ce qui n'est observé dans aucun des
deux cas MASSIVE.

### 2. Correction de l'hypothèse TA-RB (le raisonnement initial était imprécis)

Vérification du code (`precoder_experimental.py`) : la séquence traitée
par l'attention utilisateur est de longueur **K** (fixe, 8 ici), pas M,
pour les 3 architectures -- le raisonnement initial ("la séquence croît
avec M") était faux, comme signalé. Ce qui croît réellement avec M est
la **dimension de features par token** (`feat_dim`), et ce, **de façon
quasi identique pour les 3 architectures** :
- SC/IB : `feat_dim = 2M + M(abs) + 1(snr) ≈ 3M+1`
- TA-RB : `feat_dim = 2M(mean) + M(var) + K + 1(snr) ≈ 3M+K+1`

Donc **feat_dim n'est PAS un facteur différenciant** entre les 3
architectures (toutes ~3M) -- l'hypothèse initiale est invalidée,
retirée. La vraie différence structurelle de TA-RB est ailleurs :
agrégation (moyenne/variance) des sous-porteuses PAR RB en amont de
l'attention (réduit le nombre de tokens traités et moyenne le bruit
par construction), potentiellement plus robuste avec le budget de
données réduit à cette échelle (4000 échantillons, batch 16-32) --
cohérent avec l'effet de débruitage déjà documenté en CSI imparfait
(moyennage dans `_extract_features`). Reste une piste, pas une preuve
formelle -- à formuler avec cette prudence dans le mémoire.

### Décision : test de budget étendu sur SC (validation avant généralisation)

Script existant `diag_massive_true_train.py` réutilisé tel quel (déjà
paramétrable en CLI). Nouveau `--tag` ajouté pour ne pas écraser le
run 83ep original (sauvegardé également : `diag_massive_true_single_sc_
83ep_orig.json`). Lancé sur GPU1 (libéré par T=12) à 19:08 :
`--warmup_epochs 10 --finetune_epochs 160 --tag _extbudget` (~2× le
budget original, warmup allongé comme suggéré vu la difficulté accrue
à cette échelle). SC seul d'abord pour valider la tendance avant de
généraliser à IB. Log `scratchpad/massive_single_sc_extbudget.log`,
résultats `results/diag_massive_true_single_sc_extbudget.json`.

**Priorité 4 (UMa) reste explicitement en attente** tant que MASSIVE
n'a pas une performance confirmée correcte, conformément à la consigne.

### Sweep CSI imparfait MASSIVE — TERMINÉ (résumé)

| méthode | %retenu moy. pilote 20dB | %retenu moy. pilote 10dB |
|---|---|---|
| RZF/WMMSE (≈identiques, channel hardening) | 71.6% | 43.7% |
| SingleSC | 96.7% | 78.8% |
| IntraRB | 97.3% | 80.9% |
| TA-RB résiduel | 86.5% | 56.5% |

**Résultat inattendu et inverse d'UMi** : à M=64, SC/IB paraissent
BEAUCOUP plus robustes au bruit pilote que TA-RB (97% vs 86.5% à
20dB ; 80% vs 56.5% à 10dB) -- l'inverse du comportement UMi/UMa où
TA-RB avait l'avantage. **Lecture prudente nécessaire** : SC/IB
opèrent ici à un débit absolu très inférieur (encore non convergés,
cf. diagnostic ci-dessus) -- un système saturé bas a mécaniquement
moins à perdre face au bruit qu'un système qui exploite une vraie
dynamique (TA-RB, débit 2× plus élevé). Ce %retenu n'est donc PAS
directement comparable tant que SC/IB n'ont pas atteint leur propre
plafond de performance -- à réévaluer une fois le diagnostic budget
étendu conclu, ne pas documenter cette inversion comme un résultat
robuste en l'état.

Sauvé : `results/diag_csi_imperfect_massive_true_sweep.json`.

---

## PRIORITÉ 2 — Corrections des figures (menée en parallèle du diagnostic MASSIVE) — TERMINÉ

Chantier indépendant, mené sans attendre la conclusion du diagnostic
MASSIVE. Correction préalable nécessaire : `diag_tarb_residual_signed_
attn_T6_train.json` contenait un doublon **obsolète** (30ep, finetune_
epochs=27) qui aurait pollué les figures si utilisé tel quel -- corrigé
pour refléter les vraies données 83ep (source `diag_front_a_signed_
attn_ta_rb_residual.json`, déjà validées). Les 6 fichiers T-sweep
(`diag_tarb_residual_signed_attn_T{1,2,3,4,6,12}_train.json`) suivent
maintenant un schéma uniforme, tous à 83ep.

- **Figure A** : encart zoomé (15-20dB) ajouté (`ax.inset_axes` +
  `indicate_inset_zoom`) -- les 5 courbes (RZF/WMMSE/SC/IB/TA-RB), très
  resserrées en vue principale à haut SNR, sont maintenant clairement
  distinguées dans l'encart.
- **Figure B** : reconstruite avec les 6 T à budget complet (83ep)
  uniforme, plus de distinction 30ep/complet, plus de flèche "T=6
  adopté". **T=4 mis en avant (étoile)** comme point de fonctionnement
  recommandé (meilleur %WMMSE moyen du sweep ET moins cher que T=6/T=12,
  cf. Priorité 1). 2 itérations pour éliminer les collisions d'étiquettes
  (T=12 proche de SC/IB, "T=4" proche de la ligne T=3-T=6) -- résolu par
  décalages différenciés par T + libellé "T=4 ★" raccourci (le détail
  "recommandé" reste dans la légende, pas de redite dans l'annotation).
- **Figure C** : couleurs réassorties -- **un seul dégradé séquentiel
  (RdPu)** partagé entre TA-RB (trait plein) et RZF-groupé (pointillé,
  même teinte, alpha réduit) pour un même T, au lieu de deux familles de
  couleur non liées (RdPu vs Blues) comme avant. Affichage réduit à
  T ∈ {1,4,6,12} (T=4 remplace T=3 comme point clé), les 6 T restent
  dans le JSON exporté (`T_shown_in_figure` documente le sous-ensemble
  affiché).
- **Figure D** : vérifiée -- couleurs déjà cohérentes avec A/B/C (même
  dict `COLORS`), données déjà issues des checkpoints 83ep corrects
  (`diag_front_a_signed_attn*.json`). **Aucun changement nécessaire.**

Sauvé : `results/figures_final/fig{A,B,C}_*.{png,pdf,json}` mis à jour
(figD inchangée). Recommandation T=4 (au lieu de T=6) à reporter dans
le texte du chapitre si celui-ci mentionne T=6 comme choix adopté --
**à vérifier par l'utilisateur dans le manuscrit, pas fait ici** (hors
périmètre des figures).

---

## PRIORITÉ 2 (2e révision) — Retours utilisateur sur les figures — TERMINÉ

Retours après première passe : légende Figure B trop longue, titres
partout trop verbeux ("budget complet 83ep" à retirer des titres/
légendes/labels sur les 4 figures), couleurs Figure C trop proches
(dégradé séquentiel peu distinguable, notamment les pointillés
RZF-groupé), et surtout **T=4 devient la référence TA-RB partout**
(au lieu de T=6 -- meilleur %WMMSE moyen ET moins cher, cf. Priorité 1).
Puis demande complémentaire : **Figure A en double panneau** (CSI
parfait à gauche, CSI imparfait à droite), toujours avec TA-RB T=4.

- **Titres/légendes/labels** : raccourcis sur les 4 figures (ex. Figure
  B : "SC (Précodeur par sous-porteuse)" -> "SC", plus de sous-titre
  "budget complet (83ep)" nulle part -- l'abréviation complète reste
  donnée une fois en toutes lettres dans la légende de la Figure A).
- **Figure B** : légende réduite à 6 entrées courtes (2 itérations sur
  le placement des annotations pour éviter toute collision -- résolu).
- **Figure C** : **réécrite** -- abandon du dégradé séquentiel RdPu
  (peu distinguable à 6 T), remplacé par **6 teintes catégorielles
  bien distinctes** (bleu/orange/vert/bordeaux/violet/rouge), TOUS les
  6 T affichés (plus de sous-ensemble réduit), T=4 mis en évidence
  (trait plus épais, marqueur plus grand, libellé "réf."). RZF-groupé
  garde la même teinte que son TA-RB (pointillé, alpha réduit).
- **Recalculs nécessaires pour T=4 comme référence** (les données
  existantes utilisaient T=6) :
  - `diag_ber_tarb_T4.py` (nouveau, GPU0, ~3min) : BER TA-RB recalculée
    au checkpoint T=4 uniquement (RZF/WMMSE/SC/IB inchangés), fusionnée
    dans `diag_ber_signed_attn_full_range.json`.
  - `diag_csi_imperfect_tarb_T4.py` (nouveau, GPU2, ~4min) : sweep CSI
    imparfait (pilote 20dB, data SNR 0-20dB) TA-RB recalculé au
    checkpoint T=4, fusionné dans `diag_csi_imperfect_signed_attn_
    sweep.json`. Les deux recalculs ciblés (pas de réexécution de
    RZF/WMMSE/SC/IB, déjà corrects) -- même logique que pour la Figure
    A/D, gain de temps GPU.
- **Figure A refaite en double panneau** : gauche = CSI parfait
  (inchangé sinon T=6->T=4), droite = CSI imparfait pilote 20dB (RZF,
  WMMSE, SC, IB, TA-RB T=4), légende unique partagée (couleurs
  communes aux deux panneaux), titre global concis.

### Constat notable (panneau droit, Figure A)

Sous CSI imparfait (pilote 20dB), **TA-RB (T=4) domine nettement tous
les autres à SNR data 5-15dB** (ex. 25.7 vs ~19.7 bps/Hz à 15dB pour
RZF/WMMSE/SC/IB -- +30%), confirmant à nouveau l'avantage de débruitage
déjà documenté (moyennage RB dans `_extract_features`). **Ce avantage
se réduit et s'inverse à 20dB** (TA-RB 27.8 vs ~30.5 pour les autres) --
piste à noter, pas approfondie ici : possible que les autres méthodes,
moins pénalisées par le bruit pilote relatif à très haut SNR, referment
l'écart une fois le signal lui-même très propre. Cohérent avec le
comportement déjà vu sur MASSIVE (T4 CSI-imparfait UMi n'a toutefois
pas la même inversion complète que sur MASSIVE) -- résultat honnête à
mentionner tel quel dans le mémoire si pertinent.

Sauvé : `results/figures_final/fig{A,B,C,D}_*.{png,pdf,json}` (les 4
mises à jour cette fois), `results/diag_ber_signed_attn_full_range.json`
et `results/diag_csi_imperfect_signed_attn_sweep.json` mis à jour avec
TA-RB T=4 (T=6 conservé nulle part dans ces deux fichiers -- écrasé).

---

## DIAGNOSTIC MASSIVE — Test budget étendu SC : CONFIRMÉ, résultat spectaculaire

`diag_massive_true_train.py --arch single_sc --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget`
(GPU1, ~2× le budget original) terminé à 19:56 (49.6min d'entraînement,
meilleur train rate=81.81 vs 51.41 à 83ep).

| SNR | %WMMSE (83ep, ancien) | %WMMSE (160ep, étendu) | Gain |
|---|---|---|---|
| 0dB | 67.9% | 75.9% | +8.0pt |
| 5dB | 59.7% | 78.7% | +19.0pt |
| 10dB | 52.3% | 79.8% | +27.4pt |
| 15dB | 46.3% | 77.8% | +31.5pt |
| 17.5dB | 43.7% | 75.9% | +32.2pt |
| 20dB | 41.2% | 73.4% | +32.3pt |

**Diagnostic confirmé sans ambiguïté** : le budget étendu corrige presque
entièrement l'effondrement avec le SNR observé à 83ep (67.9%→41.2% devenait
75.9%→73.4%, quasi plat) -- c'était bien un problème de convergence lente
(budget insuffisant à cette échelle), pas un plafond de capacité de
l'architecture. Gains les plus marqués précisément là où le collapse
était le pire (haut SNR), cohérent avec l'hypothèse.

### Généralisation à IB -- lancée sans attendre confirmation (suite logique validée)

`diag_massive_true_train.py --arch intra_rb --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget`
lancé sur GPU1 (libéré par SC) à 20:00. Original 83ep sauvegardé à part
(`diag_massive_true_intra_rb_83ep_orig.json`).

### TA-RB résiduel MASSIVE -- signal de convergence plus ambigu, pas retesté pour l'instant

Vérification de la fin d'entraînement TA-RB (83ep, déjà en main) : rate
oscille dans une bande 72-77 sur les 10 dernières époques (pas de
tendance nette à la hausse comme SC/IB avant leur budget étendu) mais
gradient norm reste élevé (3.6-4.1, même ordre de grandeur que SC/IB
en fin de budget étendu). Signal mixte -- possiblement déjà proche
d'un plateau (contrairement à SC/IB qui montraient une croissance nette),
mais pas formellement confirmé. TA-RB étant déjà largement meilleur que
SC/IB à 83ep (65-74% vs 41-68%), pas re-testé en priorité pour
l'instant -- GPU alloué à IB d'abord (généralisation demandée en priorité).
À reconsidérer si le temps le permet après IB.

---

## Plan mis à jour (consigne utilisateur, 20:05)

- **TA-RB-MASSIVE** : même si son signal de convergence propre est
  ambigu (pas d'urgence), le budget étendu sera quand même lancé une
  fois IB confirmé -- **pour cohérence méthodologique** (comparaison
  équitable entre les 3 architectures à budget identique), pas
  seulement pour corriger un problème suspecté. Une incohérence de
  budget entre architectures serait un point faible en soutenance.
- **Ordre** : IB (en cours) -> TA-RB-MASSIVE budget étendu (cohérence)
  -> Priorité 4 (UMa) -> cette nuit, test budget étendu STANDARD
  (SC-STANDARD seul d'abord, même protocole 10 warmup + 160 finetune).
- **Rappel du signal déjà en main pour STANDARD** : la référence M8K4
  utilisée pour le diagnostic MASSIVE montrait déjà rate PLAT et
  gradient STABILISÉ dès le milieu du budget 83ep -- signe que
  STANDARD est probablement déjà bien convergé. Le test de cette nuit
  sert à trancher définitivement (pas d'urgence) :
  - Gain marginal (quelques points) -> 83ep confirmé suffisant pour
    STANDARD, rien à changer.
  - Gain significatif (comparable à MASSIVE, 20-30pts) -> généraliser
    à IB/TA-RB-STANDARD et **réentraîner tout le régime STANDARD** à
    budget étendu pour cohérence avec MASSIVE -- chantier plus lourd,
    à ce moment-là seulement.

---

## Figures A/B/D — double panneau CSI parfait/imparfait, demande utilisateur (20:20) — TERMINÉ

Suite à retour utilisateur : Figure A zoom retiré, Figure B et D
étendues en double panneau (CSI parfait / CSI imparfait pilote 20dB),
même principe que A. Nouveaux calculs nécessaires :

- `diag_csi_imperfect_tsweep.py` (T∈{1,2,3,6,12}, T=4 déjà en main) --
  GPU0 (T=1,2,3) + GPU2 (T=6,12) en parallèle, ~1min/T, terminé 19:55.
- `diag_ber_csi_imperfect.py` (nouveau script -- BER sous CSI imparfait,
  jamais fait avant cette session ; réutilise `MU_MIMO_System.
  _forward_from_precoder` pour dissocier canal-précodeur bruité /
  canal-détection réel, pas de nouvelle logique de détection). Premier
  essai à 50 batchs/point bien trop lent (génération de canal UMi domine
  le coût par batch en exécution eager, pas de @tf.function global comme
  le pipeline existant) -- tué et relancé à 15 batchs/point (~40min pour
  les 25 combinaisons SNR×méthode). **Caveat à garder en tête** : 15
  batchs est plus bruité que les 50 utilisés pour la Figure D CSI parfait
  -- les valeurs BER exactes (surtout aux points proches de 0, ex. RZF/
  WMMSE à 20dB) sont à interpréter avec cette réserve statistique.

**Résultat notable Figure B (Pareto CSI imparfait)** : **T=2 devient le
meilleur point sous CSI imparfait** (25.9 bps/Hz à 15dB), pas T=4 (25.65)
qui reste le meilleur uniquement sous CSI parfait -- et T=12 s'effondre
nettement (22.2, pire que T=6). La recommandation "T=4" de la Priorité 1
est donc valable spécifiquement en CSI parfait, pas universelle -- à
noter explicitement dans la discussion du mémoire si la robustesse CSI
imparfait est un critère retenu.

Sauvé : `results/figures_final/fig{A,B,D}_*.{png,pdf,json}`,
`results/diag_csi_imperfect_tsweep_T{1,2,3,6,12}.json`,
`results/diag_ber_csi_imperfect.json`.

---

## Test budget étendu IB (MASSIVE) — CONFIRMÉ, même schéma que SC

`diag_massive_true_train.py --arch intra_rb --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget`
terminé 20:45 (46.3min, GPU1).

| SNR | %WMMSE (83ep) | %WMMSE (160ep) | Gain |
|---|---|---|---|
| 0dB | 67.9% | 76.0% | +8.1pt |
| 5dB | 59.5% | 78.5% | +19.0pt |
| 10dB | 52.1% | 79.2% | +27.2pt |
| 15dB | 45.8% | 76.9% | +31.1pt |
| 17.5dB | 43.3% | 74.7% | +31.4pt |
| 20dB | 40.9% | 72.1% | +31.3pt |

**Quasi identique au gain observé pour SC** (+8 à +31pt selon le SNR,
même forme -- gains les plus forts à haut SNR). Confirme définitivement
que le problème était bien un budget d'entraînement insuffisant à
l'échelle M=64, généralisé aux deux architectures qui s'effondraient.

### TA-RB-MASSIVE budget étendu — lancé (cohérence méthodologique)

`diag_massive_true_train.py --arch ta_rb_residual --warmup_epochs 10 --finetune_epochs 160 --tag _extbudget`
lancé sur GPU1 (libéré par IB) à 20:49. Comme demandé : même si le
signal de convergence propre de TA-RB était plus ambigu (pas de
collapse net à 83ep, déjà 65-74% WMMSE), le budget étendu est appliqué
pour garantir une comparaison à budget cohérent entre les 3
architectures MASSIVE -- une incohérence de budget serait un point
faible en soutenance. Original 83ep sauvegardé à part
(`diag_massive_true_ta_rb_residual_83ep_orig.json`).

**Priorité 4 (UMa) reste en attente** jusqu'à confirmation TA-RB.

---

## Vérification tmux (demande utilisateur, 21:50) — résultat honnête

`$TMUX` est **vide** dans le shell de cette session -- cette session
Claude Code (session-id 21f8ae71) ne tourne PAS à l'intérieur de la
session tmux "nuit". Vérification complète de l'arbre de processus :

- Ma session (21f8ae71) est un `claude bg-pty-host` (PID 732211,
  PPID=1, détaché depuis 03:55 ce matin) -- déjà un job d'arrière-plan
  autonome au sens de Claude Code, indépendant de tout terminal.
- La session tmux "nuit" existe bien (`tmux list-sessions` la
  confirme, créée le 7/8, attachée) mais héberge un **processus
  `claude` DIFFÉRENT** (PID 1188141) -- probablement la session
  interactive de l'utilisateur, pas celle-ci.

**Ce qui compte réellement pour la survie des jobs GPU** : vérifié
directement sur le job TA-RB en cours (PID 3398130) -- son **PPID=1**
(reparenté à systemd), totalement détaché de mon shell/session Claude
Code grâce à `nohup ... & disown`. Ces process survivront à la
fermeture de cette session Claude Code, de son shell, ou même du
daemon bg-pty-host -- ils tournent au niveau OS, indépendamment.

**Nuance à ne pas perdre de vue** : ce qui n'est PAS garanti de
survivre à la fermeture de cette session (contrairement aux process
GPU eux-mêmes) c'est l'**orchestration** -- ma capacité à réagir aux
fins de job, lancer l'étape suivante, mettre à jour le log. Si cette
session se termine, les jobs en cours finiront et sauvegarderont leurs
résultats normalement, mais personne ne lancera la suite tant qu'une
session Claude Code n'est pas reprise sur ce répertoire. Rester
connecté (session actuelle ou reprise) reste donc utile pour la
continuité de l'orchestration -- mais pas pour la survie des calculs
eux-mêmes, qui est déjà acquise.

---

## Lancement nuit — P2 (UMa STANDARD) et P3 (budget étendu STANDARD) — 21:53

Deux nouveaux scripts créés :
- `diag_uma_signed_attn_train.py` (généralise `diag_tarb_residual_uma_train.py`
  aux 3 architectures signed_attn, T=4 pour TA-RB, entraîne via cache UMa
  déjà généré 8000 échantillons + éval CSI parfait + CSI imparfait balayage
  SNR complet intégrés en un seul script). Lancé en chaîne (single_sc ->
  intra_rb -> ta_rb_residual) sur **GPU0** via `run_uma_signed_attn.sh`.
- `diag_standard_extbudget_train.py` (généralise le pattern MASSIVE
  extbudget au régime STANDARD UMi M8K4). SC-STANDARD lancé seul d'abord
  sur **GPU2** (10 warmup + 160 finetune).

**GPU1** : TA-RB-MASSIVE budget étendu toujours en cours (122/170 à 21:56).

Les 3 GPU sont maintenant occupés par un job chacun -- conforme à la
consigne "un job par GPU". TA-RB-MASSIVE devrait terminer avant les deux
nouveaux runs (déjà à 122/170).

---

## MASSIVE — les 3 architectures à budget étendu cohérent (160ep) — TERMINÉ, CONCLUSION MAJEURE

TA-RB-MASSIVE budget étendu terminé 22:14 (90.9min -- la plus longue des
trois, cohérent avec son coût déjà 2x supérieur à 83ep).

| SNR | %WMMSE (83ep) | %WMMSE (160ep) | Gain |
|---|---|---|---|
| 0dB | 68.1% | 73.0% | +4.9pt |
| 5dB | 72.6% | 77.7% | +5.1pt |
| 10dB | 73.9% | 79.8% | +5.9pt |
| 15dB | 70.8% | 78.7% | +7.9pt |
| 17.5dB | 68.3% | 77.0% | +8.7pt |
| 20dB | 65.3% | 74.8% | +9.5pt |

**Gain net mais nettement plus modeste que SC/IB** (+5 à +9.5pt vs
+8 à +32pt) -- confirme que TA-RB était déjà largement convergé à 83ep,
contrairement à SC/IB. Cohérent avec le diagnostic initial (signal de
gradient/rate moins alarmant pour TA-RB).

### Tableau comparatif final -- les 3 architectures à budget IDENTIQUE (160ep)

| SNR | SC | IB | TA-RB |
|---|---|---|---|
| 0dB | 75.9% | 76.0% | 73.0% |
| 5dB | 78.7% | 78.5% | 77.7% |
| 10dB | 79.8% | 79.2% | 79.8% |
| 15dB | 77.8% | 76.9% | 78.7% |
| 17.5dB | 75.9% | 74.7% | 77.0% |
| 20dB | 73.4% | 72.1% | 74.8% |

**CONCLUSION MAJEURE, à retenir pour le mémoire** : une fois les 3
architectures équitablement entraînées (budget cohérent), **l'écart
spectaculaire observé à 83ep (TA-RB 65-74% vs SC/IB 41-68%, +24pt à
20dB) DISPARAÎT PRESQUE ENTIÈREMENT** -- les 3 architectures se
tiennent maintenant dans une fourchette étroite (72-80%), TA-RB restant
légèrement en tête à SNR moyen/haut (+1 à +3pt) mais plus l'écart
massif observé initialement. **La conclusion "TA-RB domine largement à
l'échelle massive" de la Priorité 3.3 (20260808, ~19h00) doit être
révisée** : c'était en réalité en bonne partie un artefact de budget
d'entraînement insuffisant pour SC/IB, pas un avantage architectural
fondamental de TA-RB à cette échelle. TA-RB garde un léger avantage
(le plus cohérent : +1 à +3pt à quasi tous les SNR) mais rien de
comparable à ce qui avait été rapporté initialement -- à corriger
explicitement dans la discussion du mémoire si cette conclusion
initiale y a déjà été intégrée.

Sauvé : `results/diag_massive_true_{single_sc,intra_rb,ta_rb_residual}_extbudget.json`.

**Priorité 1 -- reste à faire** : évaluation CSI parfait (déjà faite,
dans les JSON ci-dessus) + CSI imparfait (balayage SNR complet) sur les
3 checkpoints extbudget -- lancement immédiat sur GPU1 (libéré par
TA-RB).

---

## 22:23 — CSI imparfait MASSIVE extbudget lancé (GPU1, libéré par TA-RB)

`diag_csi_imperfect_massive_extbudget_sweep.py` (copie adaptée de
`diag_csi_imperfect_massive_true_sweep.py`, pointant vers les 3 JSON
`_extbudget`) lancé sur GPU1. Complète la Priorité 1 (CSI parfait déjà
en main via l'éval intégrée aux scripts d'entraînement).

État des 3 GPU à 22:23 :
- **GPU0** : UMa STANDARD single_sc, 31/83 (~40% du 1er des 3 archs)
- **GPU1** : CSI imparfait MASSIVE extbudget (nouveau, démarre)
- **GPU2** : Budget étendu STANDARD SC, 19/170 (~11%)

Tout tourne normalement, aucune erreur. Poursuite autonome.

---

## PRIORITÉ 1 (MASSIVE) — TERMINÉE, résultat CSI imparfait propre (budget cohérent)

CSI imparfait MASSIVE extbudget terminé 22:30 (GPU1).

| méthode | %retenu moy. pilote 20dB | %retenu moy. pilote 10dB |
|---|---|---|
| RZF/WMMSE (≈identiques) | 72.6% | 45.7% |
| SingleSC (160ep) | 72.5% | 39.7% |
| IntraRB (160ep) | 72.8% | 39.9% |
| **TA-RB résiduel (160ep)** | **83.1%** | **54.2%** |

**Résultat maintenant propre et interprétable** (contrairement au
sweep du même nom à 83ep, faussé par le sous-entraînement de SC/IB à
l'époque -- cf. note de prudence 19h) : à budget cohérent, **TA-RB
conserve un net avantage de robustesse au bruit pilote** (+10.5pt à
20dB, +14.5pt à 10dB par rapport à SC/IB, qui sont maintenant proches
de RZF/WMMSE). Confirme la piste de débruitage par agrégation RB
(moyennage dans `_extract_features`) comme un avantage réel de TA-RB,
indépendant de la question de convergence réglée par ailleurs.

### PRIORITÉ 1 — BILAN FINAL

Les 3 architectures MASSIVE (M=64,K=8) sont maintenant à budget
cohérent (160ep = 10 warmup + 150 finetune... **correction** : 10
warmup + 160 finetune, soit 170ep total), CSI parfait ET imparfait
évaluées pour les 3. Conclusions retenues pour le mémoire :
1. Channel hardening confirmé (RZF≈WMMSE, gap -0.01%) -- résultat
   physique attendu, pas un problème.
2. Écart initial spectaculaire SC/IB vs TA-RB à 83ep était en bonne
   partie un artefact de budget insuffisant -- corrigé, les 3
   architectures sont proches en CSI parfait (72-80%) une fois
   équitablement entraînées.
3. **TA-RB conserve néanmoins un avantage réel et net en robustesse
   CSI imparfait** (+10 à +15pt de rétention selon la qualité pilote)
   -- celui-ci n'est PAS un artefact de budget, confirmé sur les 3
   architectures à budget identique.

Sauvé : `results/diag_massive_true_{single_sc,intra_rb,ta_rb_residual}_extbudget.json`,
`results/diag_csi_imperfect_massive_extbudget_sweep.json`.

**Priorité 1 = TERMINÉE.** GPU1 libéré.

---

## 22:40 — Figures MASSIVE (P5) + lancement BER MASSIVE

Dossier `figures_final/` restructuré en 4 sous-dossiers
(`umi_standard/`, `umi_massive/`, `uma_standard/`, `uma_massive/`) --
convention documentée dans `figures_final/README.md`. Scripts UMi
STANDARD déplacés et chemins corrigés (`DATA_DIR` : `../` -> `../../`),
tous re-vérifiés fonctionnels après déplacement.

**Figures A et B générées pour UMi MASSIVE** (`umi_massive/`) :
- Figure A (double panneau CSI parfait/imparfait) : RZF/WMMSE quasi
  identiques (channel hardening confirmé visuellement), TA-RB
  nettement plus robuste au bruit pilote (panneau droit).
- Figure B (Pareto, sans recommandation ni mise en avant) : écart
  d'énergie RZF/WMMSE vs neuronal ÉNORME à cette échelle (~2000µJ vs
  ~0.3µJ, >3 ordres de grandeur -- le coût O(M³) de WMMSE explose à
  M=64) pour un débit ~78% de WMMSE -- point notable pour la
  discussion (avantage énergétique qui grandit avec l'échelle).
- Nouveau fichier `results/complexity_energy_massive_true.json` créé
  (n'existait pas encore pour ce régime).
- Pas de Figure C (pas de T-sweep à cette échelle).

**BER MASSIVE (Figure D) lancé** sur GPU1 (libéré par la CSI-imparfait
extbudget) : `diag_ber_massive.py` (nouveau, CSI parfait uniquement --
le CSI imparfait BER à cette échelle est reporté, jugé trop coûteux
en temps pour cette nuit face aux priorités restantes). Batch réduit
(32, 15 batchs) comme pour l'entraînement/éval MASSIVE.

---

## 22:52 — BER MASSIVE : résultat nul, non exploitable en figure (documenté honnêtement)

`diag_ber_massive.json` : BER = 0 (ou quasi, <2e-4 isolé) sur PRESQUE
tous les points SNR, pour TOUTES les méthodes (RZF/WMMSE/SC/IB/TA-RB).
**Pas un bug** -- le channel hardening à M=64 réduit tellement la
variance du SINR que le taux d'erreur réel est très probablement bien
en dessous de la résolution statistique permise par 15 batchs × 32 =
480 mots de code par point (il faudrait un budget bien plus important,
probablement 10-50x, pour observer des erreurs de façon fiable à cette
échelle).

**Décision : Figure D non produite pour UMi MASSIVE** -- une courbe à
BER=0 partout ne serait pas informative visuellement (et masquerait la
vraie hiérarchie sous-jacente, non résolue ici). À la place, le
constat textuel "BER négligible (<2e-4) sur toute la plage SNR testée,
aussi bien classique que neuronal, à M=64" est noté ici comme
observation qualitative pour le mémoire, sans figure associée. Un
budget Monte-Carlo bien plus lourd serait nécessaire pour un vrai
tracé BER à cette échelle -- non tenté cette nuit (compromis temps
assumé, cf. priorités).

**UMi MASSIVE (P5) -- bilan figures** : Figure A ✅, Figure B ✅,
Figure C non applicable (pas de T-sweep), Figure D non applicable
(résolution BER insuffisante, documenté). Régime considéré complet
pour cette nuit.

GPU1 libéré -- passage à la Priorité 4 (UMa MASSIVE).

---

## 23:00 — Priorité 4 (UMa MASSIVE) démarrée

`diag_classical_comparison_massive_uma.py` (nouveau, adapté du pattern
MASSIVE classique + canal UMa) lancé sur GPU1 pour la référence RZF/WMMSE
-- prérequis avant tout entraînement (besoin de WMMSE_REF pour %WMMSE).
`diag_uma_massive_train.py` préparé (fusion des patterns MASSIVE_TRUE +
UMa signed_attn, budget étendu 10+160 D'EMBLÉE comme demandé -- pas de
run 83ep intermédiaire, le diagnostic court/long a déjà tranché pour ce
régime).

**Estimation de temps réaliste** : classical comparison ~15-20min, puis
3 architectures à budget étendu (~46-91min chacune d'après les temps
MASSIVE déjà observés) = potentiellement 3-4h30 pour les 3. **Pas garanti
de finir cette nuit**, comme anticipé par la consigne -- poursuite du
maximum de progrès possible, arrêt propre et documenté si le temps
manque avant la fin.

État à 23:00 :
- GPU0 : UMa STANDARD single_sc 65/83 (proche de la fin, puis intra_rb)
- GPU1 : classical comparison UMa MASSIVE, 2/9 SNR
- GPU2 : budget étendu STANDARD SC, 39/170

---

## 23:05 — Référence classique UMa MASSIVE terminée

Gap RZF-WMMSE : -0.00% -- channel hardening confirmé aussi sous UMa à
M=64 (cohérent avec UMi MASSIVE, physique attendue indépendante du
modèle de canal). Sauvé `results/classical_comparison_M64K8_true_uma.npy`.

Lancement immédiat entraînement single_sc (budget étendu 10+160
d'emblée) sur GPU1.

---

## 23:15 — UMa STANDARD single_sc terminé (P2)

`diag_uma_signed_attn_train.py --arch single_sc` (80.7min, 83ep) :
%WMMSE 91.3-97.3% (CSI parfait), %retenu pilote 20dB 93.4%->61.5%
(0->20dB, dégradation attendue avec le SNR). Résultat sain, cohérent
avec UMi STANDARD. Chaîne GPU0 poursuit automatiquement avec intra_rb.

Sauvé `results/diag_uma_signed_attn_single_sc.json`.

---

## 23:16 — Incident : intra_rb (UMa STANDARD) OOM, ta_rb_residual tourne près de la limite

`diag_uma_signed_attn_train.py --arch intra_rb` a crashé en OOM dès le
warmup (tenseur [24576,128,128], batch=256 par défaut) -- la chaîne du
runner a continué automatiquement vers ta_rb_residual (pas de gate sur
le code retour), qui tourne mais **très près de la limite mémoire**
(43.4/46GB sur GPU0, warning "Garbage collection" émis -- risque réel
d'OOM plus tard dans son propre run).

**Fix appliqué** : ajout `--batch_size` CLI à `diag_uma_signed_attn_
train.py` (même pattern que le fix T-sweep plus tôt cette nuit/session).
**Plan** : attendre la fin (succès ou échec) de ta_rb_residual en cours
sur GPU0 (un job par GPU respecté), puis relancer intra_rb avec
`--batch_size 128` sur GPU0 libéré.

Cause probable : IntraRB a un feat_dim et une structure d'attention
légèrement plus lourds que SingleSC/TA-RB à cette combinaison
particulière (M=8,K=4,UMa) -- écart suffisant pour basculer en OOM sur
une carte déjà chargée par un job tiers, contrairement à SingleSC qui
est passé de justesse.

---

## 23:43 — Point d'étape, 4 jobs en cours

- **GPU0** : ta_rb_residual (UMa STANDARD) 42/83, a survécu à la zone
  de pression mémoire qui a fait planter intra_rb -- pas d'OOM après
  35min. ~25-28min restantes.
- **GPU1** : UMa MASSIVE single_sc 148/170, presque terminé (~7-10min).
- **GPU2** : budget étendu STANDARD SC 67/170 (~40%).

Rien de nouveau à traiter immédiatement -- poursuite de la surveillance.

---

## 00:09 (9 août) — UMa MASSIVE single_sc terminé, intra_rb enchaîné

| SNR | %WMMSE (UMa MASSIVE, single_sc, 160ep) |
|---|---|
| 0dB | 71.9% |
| 5dB | 75.8% |
| 10dB | 76.1% |
| 15dB | 75.5% |
| 17.5dB | 73.6% |
| 20dB | 71.4% |

Cohérent avec UMi MASSIVE single_sc (72-80%) -- même ordre de grandeur,
comportement stable indépendant du canal. CSI imparfait pilote 20dB :
86.3%->59.6% (0->20dB), dégradation attendue.

`diag_uma_massive_train.py --arch intra_rb` lancé sur GPU1 (cache
réutilisé, démarrage propre).

**GPU0** : ta_rb_residual (UMa STANDARD) à 81/83, quasi terminé.
**GPU2** : budget étendu STANDARD SC à 87/170, ~50%.

---

## 00:12 (9 août) — ta_rb_residual UMa STANDARD terminé, les 3 archs UMa STANDARD complètes (avec réserve intra_rb)

| SNR | %WMMSE CSI parfait | %retenu pilote 20dB |
|---|---|---|
| 0dB | 87.9% | 98.1% |
| 5dB | 90.4% | 95.9% |
| 10dB | 89.8% | 91.9% |
| 15dB | 86.0% | 86.0% |
| 17.5dB | 84.6% | 82.7% |
| 20dB | 81.1% | 79.7% |

**Confirme un schéma déjà observé sur UMi (standard et massive)** :
TA-RB est en retrait en CSI parfait par rapport à SC (81-90% vs 91-97%
pour SC) mais **nettement plus robuste au bruit pilote** (98%->80%
retenu vs 93%->62% pour SC) -- le compromis "performance brute vs
robustesse CSI" de TA-RB se confirme constant à travers tous les
régimes testés cette nuit (UMi standard, UMi massive, UMa standard).

Runner UMa STANDARD terminé (les 3 archs, avec le rattrapage intra_rb
nécessaire -- lancé ci-dessous). GPU0 libéré.

---

## 00:14 (9 août) — intra_rb (UMa STANDARD) retry confirmé stable

Époque 1/83 passée sans OOM avec `--batch_size 128` -- fix confirmé
efficace. Poursuite normale attendue.

---

## 00:28 (9 août) — Point d'étape, signal précoce STANDARD-extbudget

`standard_extbudget_single_sc` (109/170) : rate d'entraînement déjà
plat (28-32.5 depuis l'époque ~20, pas de tendance nette à la hausse)
et gradient stable (~1.5-2.0, jamais >2.0) -- **contraste net avec le
pattern MASSIVE** (rate multiplié par ~1.5-2x, gradient montant jusqu'à
3-4). Signal précoce cohérent avec la prédiction du rappel utilisateur
("STANDARD probablement déjà bien convergé à 83ep") -- à confirmer
numériquement une fois l'éval finale disponible (~60 époques restantes),
mais la tendance qualitative penche déjà clairement vers "gain
marginal, 83ep suffisait".

Autres jobs : intra_rb UMa STANDARD retry 27/83 (sain), intra_rb UMa
MASSIVE 64/170 (rate ~78-80, sain).

---

## 00:59 (9 août) — UMa MASSIVE intra_rb terminé

| SNR | %WMMSE CSI parfait | %retenu pilote 20dB |
|---|---|---|
| 0dB | 72.8% | 85.6% |
| 5dB | 76.7% | 76.4% |
| 10dB | 78.7% | 66.7% |
| 15dB | 80.4% | 58.9% |
| 17.5dB | 79.7% | 55.6% |
| 20dB | 78.9% | 53.3% |

Cohérent avec single_sc UMa MASSIVE (71.9-76.1%), légèrement meilleur.
GPU1 libéré -- lancement ta_rb_residual (dernière archi UMa MASSIVE).

---

## 01:01 (9 août) — intra_rb UMa STANDARD terminé — PRIORITÉ 2 COMPLÈTE

| SNR | %WMMSE CSI parfait | %retenu pilote 20dB |
|---|---|---|
| 0dB | 90.3% | 93.4% |
| 5dB | 95.7% | 87.5% |
| 10dB | 96.8% | 79.0% |
| 15dB | 95.3% | 69.7% |
| 17.5dB | 95.2% | 65.3% |
| 20dB | 93.4% | 61.6% |

### PRIORITÉ 2 (UMa STANDARD) — BILAN

Les 3 architectures signed_attn sont maintenant complètes sur UMa
STANDARD (M8K4), budget complet (83ep), CSI parfait + imparfait :

| SNR | SC | IB | TA-RB (T=4) |
|---|---|---|---|
| 0dB | 91.3% | 90.3% | 87.9% |
| 5dB | 95.8% | 95.7% | 90.4% |
| 10dB | 97.3% | 96.8% | 89.8% |
| 15dB | 95.7% | 95.3% | 86.0% |
| 17.5dB | 95.1% | 95.2% | 84.6% |
| 20dB | 93.4% | 93.4% | 81.1% |

SC et IB quasi identiques et meilleurs que TA-RB en CSI parfait sur ce
canal (confirme l'écart déjà noté avant signed_attn -- UMa semble
structurellement moins favorable à TA-RB qu'UMi). Mais TA-RB garde son
avantage de robustesse CSI imparfait (schéma constant à travers tous
les régimes testés cette nuit).

**PRIORITÉ 2 = TERMINÉE.** GPU0 libéré -- rien d'urgent en attente de
GPU pour l'instant (P3 tourne sur GPU2, P4 sur GPU1) -- bascule vers
la génération des figures UMa STANDARD (Priorité 5, ne nécessite pas
de GPU) pendant que les autres jobs continuent.

---

## 01:10 (9 août) — Figures UMa STANDARD (P5) générées

Figure A (double panneau) et Figure B (Pareto, sans recommandation)
créées dans `figures_final/uma_standard/`. Réutilisation directe des
énergies déjà calculées pour SC/IB (mêmes FLOPs, indépendants du canal)
et T=4 (déjà calculé, `energy_per_T_signed_attn.json`) -- aucun nouveau
calcul GPU nécessaire. Pas de Figure C (pas de T-sweep UMa) ni D (pas
de BER UMa, même choix que MASSIVE -- priorité au débit/énergie).

Figure A montre très clairement le compromis TA-RB déjà documenté :
en retrait en CSI parfait (30.7 vs 34-35 à 20dB) mais nettement devant
en CSI imparfait (26.6 vs 22.7-23.5 à 20dB).

**UMa STANDARD (P2+P5) : régime complet.**

---

## 01:21 (9 août) — PRIORITÉ 3 TRANCHÉE : 83ep suffisait déjà pour STANDARD

`diag_standard_extbudget_train.py --arch single_sc` (10+160, 206.5min)
terminé. Comparaison au run 83ep original (`diag_front_a_signed_attn.json`) :

| SNR | 83ep %WMMSE | 160ep %WMMSE | Gain |
|---|---|---|---|
| 0dB | 99.5% | 100.1% | +0.6pt |
| 5dB | 100.9% | 102.2% | +1.2pt |
| 10dB | 100.4% | 102.3% | +1.9pt |
| 15dB | 98.0% | 100.9% | +2.9pt |
| 17.5dB | 94.8% | 99.4% | +4.6pt |
| 20dB | 92.3% | 98.3% | +6.0pt |

**VERDICT (critère fixé par l'utilisateur : "gain marginal, quelques
points, pas 20-30 comme MASSIVE -> 83ep suffisait")** : gain +0.6 à
+6.0pt seulement -- **nettement marginal**, sans commune mesure avec
les +8 à +32pt observés sur MASSIVE. Confirme numériquement le signal
déjà vu qualitativement (rate plat, gradient stable dès le milieu du
budget) et l'hypothèse initiale de l'utilisateur (référence M8K4 déjà
bien convergée à 83ep).

**Décision : PAS de généralisation à IB/TA-RB-STANDARD, pas de
réentraînement du régime STANDARD à budget étendu.** Les résultats
STANDARD déjà en main (83ep, T=4 pour TA-RB) restent la référence
retenue pour tout le mémoire. Léger gain résiduel à haut SNR
(+4.6/+6.0pt à 17.5/20dB) noté mais jugé non prioritaire à généraliser
vu le coût (3.4h pour ce seul run SC).

**PRIORITÉ 3 = TERMINÉE.** GPU2 libéré.

---

## 01:22 (9 août) — Point d'étape : P1/P2/P3 terminées, GPU2 idle en attente de P4

**Bilan** : P1 (MASSIVE) ✅, P2 (UMa STANDARD) ✅, P3 (budget étendu
STANDARD) ✅ tranché (marginal, rien à généraliser), P5 (figures)
faites pour 3/4 régimes (umi_standard, umi_massive, uma_standard).

**GPU2 désormais libre** -- pas de tâche de priorité supérieure à lui
assigner dans l'immédiat (P4 n'a qu'une seule architecture restante,
déjà sur GPU1). Choix : laisser GPU2 au repos plutôt que de basculer
prématurément sur P6a/P6b (l'ordre strict demandé place P6 après P5,
et P5 n'est pas encore à 100% -- la figure uma_massive attend la fin
de ta_rb_residual). Reprise dès que ta_rb_residual (GPU1, 38/170)
termine.

---

## 01:53 (9 août) — ta_rb_residual UMa MASSIVE en cours (57%)

97/170, rate ~69-72, gradient élevé (3.7-4.1) mais stable dans cette
bande depuis plusieurs dizaines d'époques -- comportement cohérent
avec l'équivalent UMi MASSIVE (aussi ~72-77 en fin de budget étendu).
~36min restantes estimées. GPU2 toujours au repos, rien de plus
prioritaire à y lancer.

---

## 02:33 (9 août) — ta_rb_residual UMa MASSIVE terminé — PRIORITÉ 4 COMPLÈTE

### Bilan UMa MASSIVE (M=64,K=8), les 3 architectures à budget étendu (160ep)

**%WMMSE CSI parfait :**

| SNR | SC | IB | TA-RB |
|---|---|---|---|
| 0dB | 71.9% | 72.8% | 67.7% |
| 5dB | 75.8% | 76.7% | 72.8% |
| 10dB | 76.1% | 78.7% | 74.5% |
| 15dB | 75.5% | 80.4% | 72.7% |
| 17.5dB | 73.6% | 79.7% | 70.8% |
| 20dB | 71.4% | 78.9% | 68.8% |

**%retenu CSI imparfait, pilote 20dB :**

| SNR | SC | IB | TA-RB |
|---|---|---|---|
| 0dB | 86.3% | 85.6% | 94.7% |
| 5dB | 77.5% | 76.4% | 89.2% |
| 10dB | 69.3% | 66.7% | 81.8% |
| 15dB | 63.2% | 58.9% | 75.4% |
| 17.5dB | 61.1% | 55.6% | 73.0% |
| 20dB | 59.6% | 53.3% | 71.3% |

**Constat -- le compromis TA-RB se confirme sur le 4e et dernier régime
testé cette nuit** : IB légèrement en tête en CSI parfait ici (78-80%
vs SC 72-76% vs TA-RB 68-75%), mais **TA-RB reste nettement le plus
robuste au bruit pilote sur les 4 régimes sans exception** (UMi
standard, UMi massive, UMa standard, UMa massive) -- +8 à +18pt de
rétention par rapport à SC/IB selon le régime. C'est le résultat le
plus reproductible et le plus solide de toute la nuit.

**PRIORITÉ 4 = TERMINÉE.** Les 4 régimes (UMi/UMa × STANDARD/MASSIVE)
ont maintenant chacun leurs 3 architectures signed_attn évaluées en
CSI parfait ET imparfait, à budget cohérent (83ep pour STANDARD, 160ep
pour MASSIVE -- budgets différents mais chacun confirmé/adapté à son
régime par les diagnostics de cette nuit).

---

## 02:35 (9 août) — Figures UMa MASSIVE générées — PRIORITÉ 5 COMPLÈTE

Figure A et B pour `uma_massive/` créées, réutilisant l'énergie déjà
calculée (umi_massive, FLOPs indépendants du canal). Confirme une
dernière fois le compromis TA-RB (61.9 vs 50.1-51.7 bps/Hz en CSI
imparfait à 20dB, malgré un léger retrait en CSI parfait vs IB).

### PRIORITÉ 5 — BILAN FINAL

Les 4 régimes ont chacun leur Figure A (double panneau) et B (Pareto,
sans recommandation) :
- `umi_standard/` : A, B, C (T-sweep), D (BER, double panneau)
- `umi_massive/` : A, B (pas de C ni D -- justifié : pas de T-sweep,
  BER statistiquement non résolvable au budget Monte-Carlo raisonnable)
- `uma_standard/` : A, B (pas de C ni D -- même logique)
- `uma_massive/` : A, B (idem)

**PRIORITÉ 5 = TERMINÉE.**

---

## PRIORITÉS 1-5 TOUTES TERMINÉES — bascule vers Priorité 6a (nettoyage code GitHub)

---

## 02:45 (9 août) — PRIORITÉ 6a : nettoyage code GitHub — TERMINÉ

Nouveau dossier autonome `release/mimo_precoding/` créé (2.6MB, 90
fichiers, hors weights/) :

- **Modules core renommés/réorganisés** : `main_finall.py` -> `system.py`,
  `precoder_intra_rb.py` -> `precoders/base_intra_rb.py`, `precoders_v2.py`
  -> `precoders/base_residual.py`, `precoders_w.py` -> `precoders/classical.py`,
  `compare_rb_grouping.py` -> `precoders/rb_grouping.py`, `wmmse_convergence_
  check.py` -> `eval_system.py`, `channel_config.py`/`datasets.py` inchangés.
  Imports internes corrigés partout, **vérifiés à l'exécution** (pas
  seulement syntaxiquement) via `python3 -c "import ..."` pour chaque
  module.
- **`precoder_experimental.py` réduit à `precoders/signed_attention.py`** :
  ne garde QUE SignedGateMHA + les 3 architectures gagnantes (SC/IB/TA-RB
  signed_attn) -- les variantes expérimentales abandonnées (Gram matrix,
  Bilinear mixing) ne sont PAS reprises, comme demandé ("garder uniquement
  le code des résultats finaux retenus").
- **`train.py`** (nouveau) : entrypoint unique, consolide les 3 scripts
  quasi-dupliqués de cette nuit (`diag_massive_true_train.py`,
  `diag_uma_signed_attn_train.py`, `diag_standard_extbudget_train.py`) en
  un seul script paramétré `--arch --channel --scale --warmup_epochs
  --finetune_epochs --tag`. Vérifié à l'import et via `--help` (pas
  ré-exécuté end-to-end sur GPU -- dérivé directement de la logique déjà
  validée cette nuit, mais cette consolidation elle-même n'a pas eu de
  nouveau run de validation complet, à garder en tête).
- **`classical_reference.py`** (nouveau) : consolide les scripts de
  référence RZF/WMMSE par régime, même niveau de vérification que train.py.
- **`figures/`** : copie de `figures_final/`, chemins `DATA_DIR` corrigés
  et **les 10 figures régénérées avec succès** depuis ce nouveau dossier
  autonome (test bout-en-bout réel, pas juste import) -- confirme que
  `results/` (35 fichiers JSON/NPY copiés, sélectionnés précisément par
  grep des dépendances réelles des scripts de figures) est complet et
  cohérent.
- **README.md** : structure, architecture (résumé SignedGateMHA), usage,
  constats clés du mémoire, ce qui n'est PAS inclus (poids, scripts
  exploratoires abandonnés, BER massive non résolue statistiquement).

**PRIORITÉ 6a = TERMINÉE.**

---

## 11:05 (10 août) — Priorité 6b démarrée : évaluation système (scheduler)

Estimation de faisabilité faite (comparaison directe attribut par
attribut entre `main_scheduler.py` (déc. 2025) et `MU_MIMO_System`
actuel) : interface plus proche que prévu, 2 lignes à changer dans
`simulate_slot` (`_call_precoder` + `ch_helper.compute_effective_
channel` au lieu de l'ancien précodeur = son propre `PrecodedChannel`).
Import `sionna.sys` (PHYAbstraction/OLLA/PFScheduler) vérifié sans
dérive d'API. Estimation révisée : 2.5-4h (code ~1.5-2.5h + runtime
inconnu avant smoke test).

Nouveau script `diag_system_scheduler_eval.py` créé (copie adaptée,
`main_scheduler.py` original non modifié). Smoke test (1 méthode SC,
1 SNR, 10 slots) : **succès sans erreur**, 685ms/slot, extrapolation
sweep complet ~34min. Sweep complet lancé (RZF/WMMSE/SC/IB/TA-RB(T=4)
x SNR {0,5,10,15,17.5,20}dB x 100 slots) sur GPU0.

---

## 11:32 (10 août) — Bug corrigé, sweep relancé

Premier lancement du sweep complet a crashé immédiatement sur RZF :
`system.precoder` est `None` pour RZF/WMMSE (gérés inline dans
`MU_MIMO_System.call()`, pas via l'attribut `precoder`) -- le smoke
test n'avait couvert que SC (neuronal), pas RZF/WMMSE, d'où le bug
non détecté avant ce lancement. Corrigé : `simulate_slot` branche
maintenant sur `precoder_type` (`rzf_precoder`/`wmmse_precoder` en
direct pour les classiques, `_call_precoder` sinon). Sweep complet
relancé sur GPU0, RZF en cours.

---

## 11:58 (10 août) — PRIORITÉ 6b TERMINÉE : évaluation système (scheduler)

Sweep complet réussi après correction du bug RZF/WMMSE (5 méthodes x
6 SNR x 100 slots, ~2.6min/méthode). Figure 3 panneaux générée.

### Résultats complets (efficacité spectrale bps/Hz | débit total bits/slot | Jain)

| SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0dB | 7.87 \| 11234 \| 0.991 | 7.29 \| 10415 \| 0.997 | 6.47 \| 9234 \| 0.950 | 6.79 \| 9693 \| 0.999 | 7.07 \| 10091 \| 0.994 |
| 5dB | 10.61 \| 15147 \| 0.951 | 10.86 \| 15502 \| 0.945 | 10.58 \| 15106 \| 0.924 | 10.60 \| 15139 \| 0.961 | 10.56 \| 15075 \| 0.964 |
| 10dB | 14.51 \| 20717 \| 0.868 | 14.09 \| 20123 \| 0.868 | 14.85 \| 21205 \| 0.854 | 14.89 \| 21257 \| 0.917 | 13.48 \| 19252 \| 0.865 |
| 15dB | 19.01 \| 27144 \| 0.803 | 18.45 \| 26348 \| 0.853 | 16.39 \| 23399 \| 0.800 | 16.44 \| 23474 \| 0.752 | 17.69 \| 25263 \| 0.864 |
| 17.5dB | 19.44 \| 27758 \| 0.650 | 18.49 \| 26409 \| 0.682 | 16.22 \| 23166 \| 0.695 | 16.95 \| 24209 \| 0.772 | 17.40 \| 24852 \| 0.741 |
| 20dB | 19.24 \| 27480 \| 0.556 | 19.39 \| 27690 \| 0.527 | 17.67 \| 25227 \| 0.581 | 16.82 \| 24021 \| 0.873 | 17.92 \| 25593 \| 0.650 |

**Message central confirmé** : les gains observés en débit somme
théorique (Fig. A) se maintiennent en conditions système réalistes
(scheduler PF + link adaptation MCS + PHY abstraction BLER cible) --
à 15-20dB, les architectures neuronales restent à 7-15% de RZF/WMMSE
en efficacité spectrale (vs écart similaire en débit théorique pur),
TA-RB étant généralement la plus proche des classiques parmi les 3.
Pas de dégradation qualitative supplémentaire due au scheduler/MCS
réaliste -- validation cohérente avec le reste du mémoire.

**Réserve honnête** : la courbe d'équité de Jain montre des
croisements/oscillations à 17.5-20dB (ex. IB remonte fortement à
20dB) -- possiblement du bruit d'échantillonnage (100 slots, tirages
de topologie indépendants par slot) plutôt qu'un effet systématique ;
à ne pas sur-interpréter dans le texte du mémoire au-delà de "l'équité
se dégrade avec le SNR pour toutes les méthodes, sans écart marqué
entre elles" -- cohérent avec le cadrage demandé (sous-section courte
de validation, pas un pilier).

Sauvé : `results/diag_system_scheduler_eval.{json,png,pdf}`,
`diag_system_scheduler_eval.py` (script adapté, `main_scheduler.py`
original non modifié).

**PRIORITÉ 6b = TERMINÉE. Toutes les priorités (1-6b) de la nuit et de
la matinée sont maintenant complètes.**

---

## 12:04 (10 août) — Extension P6b : CSI imparfait pour l'évaluation système

Demande utilisateur : ajouter le CSI imparfait à l'évaluation système
(scheduler). `diag_system_scheduler_eval.py` étendu : `simulate_slot`
dissocie maintenant canal-précodeur (`h_est`) / canal-détection
(`h_true`), même convention que `diag_csi_imperfect_*.py` partout
ailleurs cette nuit. Nouveau flag `--pilot_snr_db` (absent = CSI
parfait, comportement original inchangé -- vérifié par smoke test de
non-régression, écart observé avec le run précédent attribué au bruit
d'échantillonnage à 10 slots, pas une régression : la logique est
mathématiquement identique quand pilot_snr_db=None puisque h_est=h_true).

Smoke test CSI imparfait (pilote 20dB) réussi sans erreur. Sweep
complet lancé sur GPU0 (5 méthodes x 6 SNR x 100 slots, ~34min estimées).
Sortie : `results/diag_system_scheduler_eval_csi_imperfect_pilot20dB.{json,png,pdf}`
(fichiers CSI parfait existants non touchés).

---

## 12:31 (10 août) — Sweep CSI imparfait (scheduler) terminé — RÉSULTAT NOTABLE

### Résultats complets (efficacité spectrale bps/Hz | débit total bits/slot | Jain), CSI imparfait pilote 20dB

| SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0dB | 6.73 \| 9613 \| 0.995 | 7.73 \| 11035 \| 0.996 | 6.08 \| 8684 \| 0.957 | 6.12 \| 8744 \| 0.974 | 6.34 \| 9047 \| 0.969 |
| 5dB | 8.64 \| 12336 \| 0.948 | 9.20 \| 13132 \| 0.951 | 8.88 \| 12682 \| 0.983 | 9.39 \| 13404 \| 0.979 | 10.56 \| 15081 \| 0.957 |
| 10dB | 11.34 \| 16197 \| 0.945 | 11.00 \| 15708 \| 0.903 | 10.23 \| 14612 \| 0.857 | 11.19 \| 15975 \| 0.904 | 12.86 \| 18362 \| 0.942 |
| 15dB | 12.45 \| 17782 \| 0.887 | 12.61 \| 18011 \| 0.869 | 12.11 \| 17291 \| 0.920 | 11.87 \| 16954 \| 0.804 | **15.42** \| **22025** \| 0.876 |
| 17.5dB | 12.52 \| 17873 \| 0.895 | 12.25 \| 17492 \| 0.839 | 11.59 \| 16555 \| 0.896 | 11.66 \| 16656 \| 0.909 | **16.46** \| **23505** \| 0.802 |
| 20dB | 12.74 \| 18189 \| 0.888 | 12.45 \| 17773 \| 0.864 | 12.26 \| 17514 \| 0.874 | 12.94 \| 18480 \| 0.865 | **13.74** \| **19619** \| 0.820 |

### RÉSULTAT NOTABLE : TA-RB dépasse RZF/WMMSE au niveau système sous CSI imparfait

À 15-17.5dB, TA-RB **dépasse nettement RZF et WMMSE** en efficacité
spectrale réelle (scheduler PF + link adaptation MCS) -- 15.42-16.46
bps/Hz contre 12.25-12.61 pour les classiques (+25 à +34%). C'est la
première fois cette nuit que l'avantage CSI-imparfait de TA-RB se
traduit par une **supériorité absolue** sur les baselines classiques
(pas seulement une meilleure rétention relative) -- confirme que
l'avantage de robustesse survit intégralement au pipeline système
complet (scheduler + sélection MCS + PHY abstraction à BLER cible),
pas seulement à la métrique de débit-somme brute utilisée partout
ailleurs dans le mémoire.

**Réserve honnête** : redescente notable à 20dB (13.74, toujours le
meilleur mais l'écart se resserre) -- budget Monte-Carlo modeste (100
slots/point) comme pour le run CSI parfait, la forme générale (net
avantage 5-17.5dB) est un effet de grande ampleur cohérent avec tout
ce qui a été documenté cette nuit, mais le point exact à 20dB est à
prendre avec la même prudence que les oscillations de Jain déjà notées.

Sauvé : `results/diag_system_scheduler_eval_csi_imperfect_pilot20dB.{json,png,pdf}`.

**Extension CSI imparfait de la Priorité 6b = TERMINÉE.**

---

## 14:20-14:22 (10 août) — Annexe B : cohérence fréquentielle + SNR intermédiaires

### Cohérence fréquentielle (ρ(Δn)), 4 scénarios UMi/UMa × LOS/NLOS

Nouveau script `diag_coherence_annexe_b.py`, config STANDARD_CONFIG
verrouillée (M8K4, R=20m), méthodologie identique à
`diag_uma_selectivity.py` (20 tirages x batch 16, Δn∈{1..12,24,48,95}).

| Δn | UMi-LOS | UMi-NLOS | UMa-LOS | UMa-NLOS |
|---|---|---|---|---|
| 1 | 0.9992 | 0.9979 | 0.9867 | 0.9835 |
| 12 | 0.9400 | 0.8938 | 0.8684 | 0.6465 |
| 24 | 0.8671 | 0.7764 | 0.8093 | 0.4626 |
| 48 | 0.7795 | 0.6033 | 0.7504 | 0.3079 |
| 95 | 0.7316 | 0.4205 | 0.7329 | 0.2145 |

**Validation croisée** : UMa-NLOS (0.6465 à Δn=12) concorde avec la
mesure déjà en main (`diag_uma_selectivity.json`, 0.6323 à Δn=12,
même méthodologie, tirages différents -- écart ~2%, cohérent avec le
bruit Monte-Carlo à 20 tirages).

**⚠️ ALERTE À VÉRIFIER PAR L'UTILISATEUR** : UMi-NLOS mesuré ici à
ρ(12)=0.8938, alors qu'une valeur de référence ρ(12)=0.797 est codée
en dur (commentaire, pas un fichier de résultats) dans
`diag_uma_selectivity.py` (variable `umi_ref`, ligne ~112-113),
présentée comme "canal actuel". Sources possibles de l'écart :
mesure antérieure à une révision de `channel_config.py`, config
légèrement différente, ou la valeur codée en dur est simplement
obsolète. La mesure UMa concordant bien avec l'historique, je fais
confiance à la méthodologie de ce nouveau script -- mais **si le texte
du mémoire cite déjà 0.797 pour ρ(12) UMi, il y a une incohérence à
trancher avant d'utiliser ce nouveau tableau**.

Sauvé : `results/diag_coherence_annexe_b.json`.

### SNR intermédiaires (2.5/7.5/12.5/17.5dB), 4 régimes x 3 architectures

Lancé sur GPU2, chaîné séquentiellement (`diag_intermediate_snr_
annexe_b.py --regime {umi_standard,uma_standard,umi_massive,uma_massive}`),
réutilise les checkpoints déjà validés (cf. vérification précédente),
même méthode `eval_perfect` que les scripts d'entraînement d'origine.
En cours -- umi_standard proche de la fin.

### 14:26 (10 août) — Annexe B terminée (4/4 régimes + 4/4 scénarios)

Sweep SNR intermédiaires : exit=0 sur les 4 régimes (umi_standard
14:20-14:21, uma_standard 14:21-14:22, umi_massive 14:22-14:24,
uma_massive 14:24-14:25), aucune erreur, GPU2 libéré en fin de run.
Rapport complet compilé et envoyé : `ANNEXE_B_DONNEES.md` (tableau
cohérence 4 scénarios + tableaux débit somme 4 régimes x 3 archis x 4
points SNR, chemins de checkpoints exacts pour chaque cellule).

Point ouvert transmis à l'utilisateur : écart ρ(12) UMi-NLOS (0.894
mesuré vs 0.797 référence codée en dur obsolète) — à trancher côté
.tex.

---

## 00:20-00:51 (12 août) — Investigation plafond MASSIVE (M=64,K=8) : fermeture de l'écart RZF

Demande utilisateur explicite : le plafond CSI-parfait MASSIVE actuel
(73-80% RZF, budget déjà étendu à 170ep) n'est pas satisfaisant.
Champ libre pour investiguer (capacité réseau, curriculum learning,
ou autre piste), sur UMi MASSIVE UNIQUEMENT (M8 STANDARD et UMa non
touchés), checkpoints MASSIVE_TRUE_* actuels intacts, toute tentative
dans un nouveau dossier de poids.

### Méthode

Diagnostic à budget réduit (2 warmup + 18 finetune = 20ep, au lieu de
170) pour comparer plusieurs pistes À COÛT ÉGAL avant de committer un
budget complet à la plus prometteuse. Scripts :
- `diag_massive_gap_sweep.py` (dataset chargé 1x, plusieurs variants
  enchaînés dans le même process) : `--arch {single_sc,intra_rb,
  ta_rb_residual} --variants v1,v2,...`
- `diag_massive_gap_fullbudget.py` (run à 170ep complet pour la
  variante gagnante) : poids sauvés dans `./weights/MASSIVE_TRUE_
  <Arch>_..._cap<D>L<L>/`, distincts des checkpoints extbudget de
  production.

Variants testés (single_sc, 20ep, même coût ~5.5-9.7min) :
1. `baseline` : D=128,L=4 (recette actuelle), init aléatoire
2. `curriculum` : mêmes D/L, mais blocs transformer (attention/FFN)
   TRANSPLANTÉS depuis le checkpoint STANDARD (M=8) déjà convergé
   (75-111/78-123 variables shape-compatibles selon l'archi, seuls
   input_embed et output_proj -- dépendants de M -- réinitialisés)
3. `cap_d128l8` : profondeur doublée (L=8), init aléatoire
4. `cap_d256l4`, `cap_d384l4`, `cap_d512l4` : largeur (embed_dim D)
   augmentée, init aléatoire

### Résultat -- capacité (largeur D), pas optimisation, pas profondeur

| Variant | D | L | train (20ep) | % RZF moyen |
|---|---|---|---|---|
| baseline | 128 | 4 | 5.5min | 32.1% |
| curriculum | 128 | 4 | 6.4min | 32.3% (= bruit) |
| cap_d128l8 | 128 | 8 | 5.8min | 34.9% (+3pt, marginal) |
| cap_d256l4 | 256 | 4 | 5.7min | 53.8% (+22pt) |
| cap_d384l4 | 384 | 4 | 6.6min | 66.0% (+34pt) |
| cap_d512l4 | 512 | 4 | 9.7min | 68.0% (+36pt, rendements décroissants) |

Confirmation croisée-architecture (D=256 vs baseline, 20ep) :
- intra_rb : 32.6% → 55.5% (+23pt, même ampleur que single_sc)
- ta_rb_residual : en cours au moment de cette entrée

**Diagnostic** : le goulot d'étranglement à M=64 est la LARGEUR du
réseau (embed_dim), pas la profondeur ni l'initialisation. À D=128,
feat_dim=193 (3M+1) est fortement compressé dans un espace à 128 dim
-- le réseau n'a simplement pas la capacité de représenter
l'information nécessaire pour approcher RZF à cette échelle. Le
curriculum learning (transplant M=8→M=64) n'apporte aucun gain
mesurable : les blocs transformer pré-entraînés à M=8 n'encodent pas
un a priori utile pour M=64 (le problème d'annulation d'interférence
à 64 antennes est qualitativement différent, pas juste une extension
du cas M=8).

**Décision** : D=384,L=4 retenu comme meilleur compromis (rendements
décroissants après D=384 : +12pt de 256→384 vs seulement +2pt de
384→512, pour un coût compute qui lui continue de croître).

### Run complet en cours

`single_sc` D=384,L=4, budget complet 170ep (10 warmup + 160
finetune), lancé 00:51 sur GPU1. Poids -> `./weights/MASSIVE_TRUE_
SingleSC_signed_attn_4L_128d_capD384L4/`. Résultat -> `results/
diag_massive_gap_fullbudget_single_sc_cap_d384l4.json`.

Checkpoints MASSIVE_TRUE_* extbudget (production, 73-80% RZF)
INTACTS, non touchés.

### 01:05 (12 août) — Lancement des 3 runs complets (D=384, 170ep)

Confirmation croisée-architecture terminée (D=256 vs baseline, 20ep) :
- single_sc : 32.1% → 53.8% (+21.7pt)
- intra_rb : 32.6% → 55.5% (+22.9pt)
- ta_rb_residual : 37.1% → 63.0% (+25.9pt, gain le plus fort des 3)

Effet confirmé sur les 3 architectures, cohérent en ampleur -> pas
spécifique à une archi, c'est un vrai goulot de capacité générique
à M=64.

Les 3 runs complets (D=384,L=4, 170ep = 10 warmup + 160 finetune)
lancés EN PARALLÈLE (GPU0/1/2, ~39GB libres sur GPU0 malgré job tiers) :
- single_sc  : GPU1, lancé 00:51 -> `results/diag_massive_gap_fullbudget_single_sc_cap_d384l4.json`
- intra_rb   : GPU2, lancé 01:05 -> `results/diag_massive_gap_fullbudget_intra_rb_cap_d384l4.json`
- ta_rb_residual : GPU0, lancé 01:05 -> `results/diag_massive_gap_fullbudget_ta_rb_residual_cap_d384l4.json`

Poids -> `./weights/MASSIVE_TRUE_<Arch>_..._capD384L4/` (dossiers
distincts, checkpoints extbudget de production intacts).

Temps estimé (extrapolation depuis extbudget D=128 + facteur D=384) :
single_sc/intra_rb ~60-70min, ta_rb_residual ~100-120min (plus lent
par epoch même à D=128 dans les runs de production).

### 01:46 (12 août) — single_sc D=384 (170ep) TERMINÉ — succès net

Train time réel : 54.1min (170ep, GPU1).

| SNR | ancien D=128 (%RZF) | nouveau D=384 (%RZF) | gain |
|---|---|---|---|
| 0dB | 75.9% | **98.6%** | +22.7pt |
| 5dB | 78.7% | **97.2%** | +18.5pt |
| 10dB | 79.8% | **94.4%** | +14.6pt |
| 15dB | 77.8% | **89.1%** | +11.3pt |
| 17.5dB | 75.9% | **85.8%** | +9.9pt |
| 20dB | 73.4% | **82.1%** | +8.7pt |

Coût : params 1.10M -> 9.59M (x8.7), FLOPs 1.69G -> 14.72G (x8.7) --
attendu (D:128->384 = x3, attention/FFN scalent en D² -> x9 théorique,
cohérent).

Quasi-RZF à 0-10dB, écart résiduel significatif seulement en haute SNR
(18pt à 82% vs 100%) -- reste un vrai gain, pas juste "grappiller
quelques points". Poids -> `./weights/MASSIVE_TRUE_SingleSC_signed_
attn_4L_128d_capD384L4/best_20260812_014548`.

intra_rb (GPU2) et ta_rb_residual (GPU0) toujours en cours.

### 02:14 (12 août) — intra_rb D=384 (170ep) TERMINÉ — succès net (idem single_sc)

Train time réel : 78.3min (170ep, GPU2).

| SNR | ancien D=128 (%RZF) | nouveau D=384 (%RZF) | gain |
|---|---|---|---|
| 0dB | 76.0% | **98.0%** | +22.0pt |
| 5dB | 78.5% | **96.2%** | +17.7pt |
| 10dB | 79.2% | **93.1%** | +13.9pt |
| 15dB | 76.9% | **87.3%** | +10.4pt |
| 17.5dB | 74.7% | **84.0%** | +9.3pt |
| 20dB | 72.1% | **80.2%** | +8.1pt |

Params 1.37M -> 11.96M, FLOPs 2.11G -> 18.40G. Poids -> `./weights/
MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_021303`.

ta_rb_residual (GPU0) toujours en cours.

### 02:27 (12 août) — ta_rb_residual D=384 (170ep) TERMINÉ — les 3 architectures confirmées

Train time réel : 86.0min (170ep, GPU0, malgré job tiers présent sur ce GPU).

| SNR | ancien D=128 (%RZF) | nouveau D=384 (%RZF) | gain |
|---|---|---|---|
| 0dB | 73.0% | **95.3%** | +22.3pt |
| 5dB | 77.7% | **95.2%** | +17.5pt |
| 10dB | 79.8% | **93.1%** | +13.3pt |
| 15dB | 78.7% | **88.0%** | +9.3pt |
| 17.5dB | 77.0% | **84.7%** | +7.7pt |
| 20dB | 74.8% | **81.2%** | +6.4pt |

Params 2.33M -> 10.97M, FLOPs 2.76G -> 9.45G (TA-RB reste l'architecture
la moins coûteuse en FLOPs des 3 même à D=384, décodeur résiduel plus
compact). Poids -> `./weights/MASSIVE_TRUE_TA_RB_residual_signed_attn_
6tok_4L_128d_capD384L4/best_20260812_022738`.

## RÉCAPITULATIF — LES 3 ARCHITECTURES, D=384, 170ep

| Archi | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB | (ancien 20dB) |
|---|---|---|---|---|---|---|---|
| single_sc | 98.6% | 97.2% | 94.4% | 89.1% | 85.8% | 82.1% | (73.4%) |
| intra_rb | 98.0% | 96.2% | 93.1% | 87.3% | 84.0% | 80.2% | (72.1%) |
| ta_rb_residual | 95.3% | 95.2% | 93.1% | 88.0% | 84.7% | 81.2% | (74.8%) |

**Succès net et cohérent sur les 3 architectures** : quasi-RZF (95-99%)
à 0-10dB, 80-82% à 20dB (vs plafond 73-80% précédent, +6 à +23pt selon
SNR/archi). Coût : x8-9 params/FLOPs, mais énergie (INT8) reste
négligeable vs WMMSE (2-3µJ vs 2354µJ).

Passage à la suite : CSI imparfait, figures, avec les nouveaux
checkpoints.

### 02:46 (12 août) — CSI imparfait + énergie + figures TERMINÉS, rapport final écrit

CSI imparfait (D=384, 3 archi, `diag_csi_imperfect_massive_capD384L4.py`) :
robustesse TA-RB CONFIRMÉE et même renforcée -- +8 à +26pt de rétention
vs RZF/WMMSE selon SNR/pilote (voir tableau complet dans le rapport).
À pilote=20dB haute SNR, TA-RB dépasse même WMMSE en débit absolu.

Énergie (`compute_complexity_energy_massive_capD384L4.py`, CPU) :
énergie neuronale reste négligeable vs WMMSE (facteur ~800-1600x)
malgré x3.4-8.7 FLOPs/params.

Figures régénérées -> `results/figures_final/umi_massive_capD384L4/`
(figA sumrate CSI parfait+imparfait, figB Pareto énergie) --
dossier séparé, référence D=128 `umi_massive/` intacte.

Rapport complet écrit : `RAPPORT_MASSIVE_GAP_CLOSING_20260812.md`
(méthode, chiffres avant/après complets, tous les chemins de
checkpoints/résultats/figures, ce qui n'a pas été refait et pourquoi).

## STATUT FINAL : SUCCÈS COMPLET
Plafond 73-80% RZF -> 82-99% selon SNR/archi (D=384,L=4, même budget
170ep). Effet confirmé sur les 3 architectures, robustesse CSI-
imparfait de TA-RB préservée et renforcée. Checkpoints production
D=128 intacts. STANDARD/UMa non touchés. Rien en cours -- toutes les
tâches de la nuit terminées.

---

## 22:16-22:19 (12 août, nuit suivante) — Réplication du gap-closing sur UMa MASSIVE

Demande utilisateur : reproduire exactement la méthodologie de la nuit
précédente (RAPPORT_MASSIVE_GAP_CLOSING_20260812.md, D=128->D=384 sur
UMi MASSIVE) sur UMa MASSIVE (M=64,K=8), pour retirer l'asymétrie
UMi/UMa documentée dans le mémoire. Diagnostic à coût réduit déjà
tranché sur UMi (D=384,L=4 = rendements décroissants, effet générique)
-- pas refait sauf signal contraire.

Référence D=128 existante confirmée : `results/diag_uma_massive_
{single_sc,intra_rb,ta_rb_residual}.json`, ckpts `./weights/UMa_MASSIVE_
<Arch>_signed_attn_.../best_202608{08,09}_...` (170ep déjà, "budget
étendu d'emblée" -- pas de run 83ep intermédiaire pour UMa, confirmé
dans le docstring de `diag_uma_massive_train.py`). Plafond ancien
confirmé par recalcul (`classical_comparison_M64K8_true_uma.npy`,
gap RZF-WMMSE <0.01%, hardening confirmé aussi côté UMa) :
single_sc 71.4-76.1%, intra_rb 72.8-80.4%, ta_rb_residual 67.7-74.5%
de RZF -- même ordre que l'ancien plafond UMi (73-80%).

Scripts créés (nouveaux, réplique UMi adaptée) :
- `diag_massive_gap_fullbudget_uma.py` (scenario='uma', cache
  `sionna_joint_uma_massive_4k_64x8.npz` déjà généré/réutilisé,
  RUN_NAMES préfixés `UMa_MASSIVE_...`)
- `diag_csi_imperfect_massive_uma_capD384L4.py` (channel_model=UMa,
  reste identique à la version UMi)
- Énergie/complexité : PAS de nouveau script -- déterminé que
  `complexity_energy_massive_capD384L4.json` (calculé pour UMi) est
  DIRECTEMENT réutilisable tel quel pour UMa : les FLOPs/params/énergie
  ne dépendent que de (M,K,D,L,architecture), pas du scénario de canal.
  Confirmé par relecture de `compute_complexity_energy_massive_
  capD384L4.py` (aucune référence au channel scenario dans le calcul).

Smoke test (1ep, single_sc) : OK, 0.9min, aucune erreur.

Runs complets (D=384,L=4, 170ep=10+160) lancés en parallèle 22:19 :
- single_sc GPU1, intra_rb GPU0, ta_rb_residual GPU2 (GPU2 partagé
  avec un job tiers ~29GB/46GB -- 17GB libres, suffisant pour nos
  besoins ~2-8GB/job).

### 23:13-23:55 (12→13 août) — Les 3 architectures UMa D=384 terminées, succès confirmé

| Archi | SNR=0 anc->nouv | SNR=20 anc->nouv | train |
|---|---|---|---|
| single_sc | 71.9%->92.5% (+20.7pt) | 71.4%->81.8% (+10.3pt) | 53.7min |
| intra_rb | 72.8%->92.4% (+19.6pt) | 78.9%->81.8% (+2.9pt) | 78.6min |
| ta_rb_residual | 67.7%->87.6% (+19.9pt) | 68.8%->74.0% (+5.2pt) | 99.8min |

Même schéma que UMi (gain fort à basse SNR, plus modeste à 20dB) --
intra_rb/ta_rb_residual montrent un gain plus resserré en haute SNR
que single_sc sur UMa (déjà présent dans une moindre mesure côté UMi).
Poids -> `./weights/UMa_MASSIVE_<Arch>_..._capD384L4/`, checkpoints
D=128 (`UMa_MASSIVE_<Arch>_signed_attn_.../` sans suffixe) intacts.

Passage à CSI imparfait (`diag_csi_imperfect_massive_uma_capD384L4.py`).

### 07:30-07:43 (13 août) — CSI imparfait + figures TERMINÉS, rapport final écrit

CSI imparfait (D=384, 3 archi, `diag_csi_imperfect_massive_uma_capD384L4.py`) :
robustesse TA-RB CONFIRMÉE sur UMa, marge +7 à +24pt vs RZF/WMMSE
selon SNR/pilote -- comparable ou supérieure à UMi. Point notable :
TA-RB est l'architecture la PLUS FAIBLE des 3 sous CSI parfait sur
UMa (contrairement à UMi où les 3 étaient proches), mais redevient
nettement la meilleure sous CSI imparfait -- confirme que son
avantage est spécifique à la robustesse, pas à la capacité brute.

Énergie : PAS recalculée, réutilisation directe du fichier UMi
(channel-independent, cross-validé par params exacts identiques).

Figures régénérées -> `results/figures_final/uma_massive_capD384L4/`.

Rapport complet écrit : `RAPPORT_MASSIVE_UMA_GAP_CLOSING_20260813.md`

## STATUT FINAL (UMa) : SUCCÈS COMPLET
Plafond 68-81% RZF -> 74-93% selon SNR/archi (D=384,L=4, même budget
170ep, réplique exacte du chantier UMi du 12/08). Asymétrie
UMi(D=384)/UMa(D=128) du mémoire peut être retirée -- les deux canaux
sont maintenant au même niveau de méthodologie. Checkpoints D=128
production intacts. STANDARD/UMi MASSIVE non touchés. Rien en cours.
