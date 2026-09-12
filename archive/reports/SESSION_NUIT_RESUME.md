# Résumé de session — nuit du 2026-08-06 au 2026-08-07

Document de reprise de contexte, MIS À JOUR une deuxième fois (le canal a été verrouillé dans le code depuis la première version de ce résumé, et un problème critique de pipeline a été trouvé — voir §0). Si vous reprenez cette session à froid (nouvelle instance, tmux) : **lisez §0 en entier avant de lancer quoi que ce soit d'entraînement.**

---

## 0. ⚠️ BLOQUANT — pipeline de génération de dataset incompatible avec le nouveau canal

**À corriger AVANT tout run d'entraînement ce soir (T-sweep TA-RB, refenêtrage IntraRB, schedules LR).** Les scripts de *mesure* de canal (recherche de canal, `classical_comparison.py`) ne sont PAS affectés — ils génèrent déjà les utilisateurs correctement (voir pourquoi ci-dessous). Seul le pipeline d'entraînement (`datasets.py`) l'est.

### Le problème
`channel_config.py` implémente désormais le clustering spatial (§1) : les K utilisateurs d'un même échantillon doivent être tirés **ensemble**, autour d'un **même** centre de cluster, pour être corrélés entre eux — c'est cette corrélation conjointe qui crée le gap RZF-WMMSE mesuré cette nuit.

`datasets.py` (`CachedSionnaDataset` → `SionnaSingleUserGenerator.generate_single_user_channels` → `SAGEHBDataset`) fait l'inverse : il génère 5000 canaux **mono-utilisateur indépendants** (chacun son propre tirage, donc son propre centre de cluster aléatoire), puis pour former un batch K-utilisateurs, pioche K index **au hasard parmi les 5000** (`np.random.choice(..., replace=False)` dans `SAGEHBDataset._infinite_generator`, fichier `datasets.py` ~ligne 227). C'est le mécanisme "SAGE-HB" qui donne 250 000 échantillons effectifs à partir de 5000 tirages (économie de stockage ~98%).

Ce recombinage aléatoire était **sans conséquence** avec l'ancien canal (fenêtre d'angle étroite) : chaque utilisateur respectait la contrainte indépendamment, recombiner des utilisateurs indépendants entre eux préservait la propriété recherchée.

Avec le nouveau canal (clustering), c'est **incompatible** : 4 utilisateurs piochés au hasard parmi 5000 tirages indépendants viennent presque certainement de 4 clusters différents → la corrélation spatiale qui crée le gap disparaît silencieusement à l'entraînement, même si `channel_config.py` est correct. Résultat : on s'entraînerait sur un canal qui redevient statistiquement "sans gap" (comme le test 3GPP standard qui a donné 0.00% cette nuit), sans avertissement.

### Pourquoi les scripts de mesure de cette nuit restent valides
Tous les `diag_*channel*.py`, `diag_massive_*.py`, `diag_m32_final_round.py`, ainsi que `classical_comparison.py`, utilisent `ConfigurableMIMOSystem`/`ClusteredSystem` (basé sur `wmmse_convergence_check.py`) qui tire les K utilisateurs **conjointement** par batch, à chaque appel de `new_topology()` — pas de mise en cache ni de recombinaison. Les résultats R=20m (STANDARD) et R=5m (MASSIVE) sont donc fiables tels quels.

### Fix à faire (pas encore implémenté)
Modifier `datasets.py` pour générer les échantillons multi-utilisateurs **conjointement** par cluster, au lieu de mono-utilisateur-puis-recombinaison. Deux approches possibles (à choisir/juger en reprenant) :
- **(a) Simple** : abandonner le multiplicateur d'augmentation ×50, générer directement N tuples K-utilisateurs joints (chaque tuple = un appel à `gen_topology_clustered` avec `num_ut=K` directement, pas `num_ut=1`). Perd l'économie de stockage mais élimine le problème structurellement. Probablement le choix le plus sûr vu l'urgence.
- **(b) Plus économe** : taguer chaque tirage mono-utilisateur avec l'ID de son cluster (générer plusieurs utilisateurs par cluster dès la génération de base, pas juste 1), et modifier `SAGEHBDataset._infinite_generator` pour ne recombiner qu'à l'intérieur d'un même cluster. Garde une partie de l'économie mais demande de repenser la structure de cache (`h_freq_all` doit porter les IDs de cluster).

Recommandation : commencer par (a) pour débloquer les runs de ce soir (le fix est mécaniquement simple — remplacer l'appel `set_locked_topology(..., num_ut=1, ...)` en boucle par un seul appel `num_ut=K` par échantillon), quitte à optimiser vers (b) plus tard si la génération devient trop lente. **Tester le fix avec `diag_grad_vs_snr.py` ou un script similaire léger avant de lancer un smoke test complet**, pour vérifier que le cache régénéré contient bien la corrélation attendue (mesurer ρ et le gap RZF-WMMSE sur un batch tiré du nouveau cache, comparer aux chiffres de §1).

---

## 1. Canal verrouillé — décision finale, ÉCRITE DANS LE CODE

**`channel_config.py` a été mis à jour** (REVISION 2026-08-07 (b) dans le docstring) :

**STANDARD_CONFIG : M=8, K=4, CLUSTER_RADIUS_M=20.0**
**MASSIVE_CONFIG : M=32, K=8, CLUSTER_RADIUS_M=5.0** (rescale depuis M64K32 — voir pourquoi ci-dessous)

Nouvelle fonction canonique `gen_topology_clustered()` dans `channel_config.py` (tire un centre de cluster dans le secteur 3GPP standard 120°, puis chaque utilisateur dans un disque de rayon `CLUSTER_RADIUS_M` autour de ce centre). `gen_topology_locked()` et `set_locked_topology()` mis à jour pour l'utiliser. **Signature changée** : `set_locked_topology(..., cluster_radius_m=..., force_los=..., ...)` — l'ancien paramètre positionnel `half_angle_deg` n'existe plus à cette position (attention si du code externe l'appelait positionnellement). `main_finall.py` (`MU_MIMO_System.new_topology`) déjà corrigé pour utiliser `CHOSEN_CONFIG['CLUSTER_RADIUS_M']`.

### Pourquoi ce canal (raisonnement physique, condensé — voir docstring complet dans `channel_config.py` pour le détail chronologique)

**Point de départ** : le canal verrouillé en début de nuit (LOS forcé + fenêtre ±7.5°) était quasi mono-trajet en fréquence (mesuré : >99% de la puissance dans un seul tap retard, |ρ(lag)|>0.80 sur toute la bande FFT=96) — ça annule la prémisse même des architectures fréquentielles de la thèse (rien à exploiter entre sous-porteuses).

**Recherche (tout mesuré, rien supposé)** :
1. NLOS + divers angles (7.5°/15°/30°/60°, LOS naturel) : restaure la sélectivité (ρ@95sc→0.41-0.55) mais le gap RZF-WMMSE s'effondre à quasi rien une fois mesuré avec 10 batches (bruit, pas signal réel).
2. K=M=64 (100% loading) : gap énorme (+35.8%) mais **rejeté** — contredit M≫K (définition MIMO massif de la thèse).
3. WMMSE re-vérifié correct sous NLOS (0/42 violations de non-décroissance du WSR interne, conforme à Shi et al. 2011 formule par formule — récepteur MMSE scalaire, poids 1/mse, mise à jour V avec contrainte de puissance totale via μ exact par bisection). Le gap qui se ferme n'est pas un problème WMMSE.
4. Canal 3GPP standard (secteur 120° stock, pas de réglage d'angle) : gap **exactement 0.00%** aux 3 échelles testées. Mesure géométrique : écart angulaire moyen entre PAIRES d'utilisateurs = 39.6° sous ce mécanisme — les utilisateurs sont naturellement bien séparés, RZF déjà quasi-optimal.
5. **Insight retenu** : ce qui compte n'est pas l'angle du secteur global (vu du BS), mais la proximité **entre utilisateurs eux-mêmes** — sous NLOS, chaque utilisateur a son propre tirage de multipath indépendant (spatial consistency 3GPP ne corrèle que 7 paramètres scalaires, pas la géométrie cluster/rayon). Mécanisme : tirer un centre de cluster aléatoire dans le secteur standard, puis chaque utilisateur dans un disque de rayon R autour.
6. Sweep R sur STANDARD (M8K4) : R=20m optimal (3/5 points SNR, 0dB:+2.79%/5dB:+0.97%/10dB:+0.35%, meilleure sélectivité que R=30/60m aussi).
7. MASSIVE à M=64 : **aucune combinaison K∈{4,8,16}×R∈{5,10,20}m n'a rouvert de gap** (0-0.27%, jamais >0.3%). Channel hardening massif (Marzetta 2010) confirmé par recherche exhaustive, pas supposé.
8. M=32 testé en dernier recours (4× STANDARD, mentionné Chapitre 3) : K=8,R=5m donne un vrai gap (2/5 points : 0dB+0.97%/5dB+0.39%), sélectivité cohérente (ρ@95sc=0.370), M/K=4 pas dégénéré. **Retenu pour MASSIVE.**

Les deux rayons diffèrent (20m vs 5m) parce qu'un réseau à 32 antennes résout angulairement beaucoup plus finement qu'à 8 — il faut resserrer davantage pour produire un effet de corrélation comparable. Même mécanisme physique aux deux échelles, rayon adapté à la résolution du réseau.

### Décision de fond sur le gap (rappel important)
Le gap RZF-WMMSE n'est qu'un **indicateur secondaire** de l'espace disponible pour un précodeur appris — pas l'objectif. L'objectif principal : voir si le Transformer (surtout la version fréquentielle) bat RZF/WMMSE. Un MASSIVE avec gap modeste est acceptable ; même un MASSIVE sans gap aurait été un résultat scientifique valide à documenter (cohérent avec la théorie).

---

## 2. État de chaque volet

### Volet 1 — Refonte architecturale

**TA-RB** :
- Fix OFDM (1 symbole pilote + tile, canal statique confirmé bit-à-bit identique sur les 14 symboles) intégré dans **`precoders_v2.py` → `TransformerPrecoderClean`**, avec décodeur style v4.2 SANS gate sigmoid (ablation : gradient quasi nul, aucun gain mesuré).
- Ablation features (mean/slope/var/cov) faite (`diag_v2_smoke.py`, résultats `results/diag_v2_feature_ablation.json`) : slope et covariance dispensables, mean indispensable, var borderline. Pas encore confirmé sur run plus long.
- **T-sweep (1/2/3/4/6 tok/RB) : PAS FAIT.** Script prêt (`diag_tarb_T_sweep.py`) mais **bloqué par le problème §0** (utilise `CachedSionnaDataset`/`datasets.py`) — ne PAS lancer tel quel avant le fix.

**IntraRB** :
- Sweep RB=4/6/12/24/48 fait sur l'**ancien** canal plat (`diag_intrarb_window.py`) — non discriminant, à refaire. **Aussi bloqué par §0** si on relance ce script tel quel (même dépendance à `datasets.py`/`CachedSionnaDataset`).
- Fenêtre glissante Swin : pas explorée.

*Note* : mes scripts de mesure de canal (qui n'utilisent PAS `datasets.py`) restent valides ; mais tout ce qui utilise `CachedSionnaDataset` pour de l'entraînement (T-sweep, refenêtrage IntraRB, schedules LR) doit attendre le fix §0.

### Volet 2 — Protocole d'entraînement
- Diagnostic (session précédente) : plafonnement du sum-rate direct pas dû au gradient (croît avec le SNR, mesuré) ni à la contrainte de puissance (RZF a la même contrainte et égale WMMSE). Cause probable : schedule LR trop agressif pour le budget d'époques.
- **Schedules LR (constant_bas/cosinus_long/warm_restart/baseline) : INTERROMPU, pas de résultat**, et de toute façon **bloqué par §0** (utilise `CachedSionnaDataset` via IntraRB).
- Curriculum SNR : pas testé.

### Volet 3 — Complexité/énergie
- Premier passage fait pour TA-RB nettoyé (`complexity_energy_results_v2.csv`) : fix OFDM = ~14× moins de FLOPs réels, chiffres conventionnels (×num_ofdm, pour comparabilité historique) et réels tous deux documentés.
- Pas fait : IntraRB avec fenêtre finale, features réduites si confirmées, configs finales du nouveau canal (M32K8 remplace M64K32 partout — tout FLOPs/énergie calculé pour M64K32 cette nuit est maintenant obsolète).

### Nettoyage disque (fait, validé)
`/users/sala` 94%→67%, rien supprimé, déplacé vers `/export/tmp/sala/` avec liens symboliques : `hls_projects/results/` (3.8G, projet sans rapport), checkpoints obsolètes pré-30/07 (`weights_archive/`), `training_weights_transformer_rb_unsupervised/` (`training_weights_tr/`). Checkpoints actifs conservés sur NFS : `SingleSC_light_run`, `IntraRB_fullres`, `TA_RB_T3`.

---

## 3. Rien ne tourne actuellement en arrière-plan

Vérifié (deux fois, `ps -ef` + `nvidia-smi --query-compute-apps`) : **aucun process GPU à moi actif**. GPU0 libre. GPU1/GPU2 occupés toute la nuit par d'autres utilisateurs (16GB/28GB, 99-100% util) — ne jamais y lancer de travail, seul GPU0 utilisé cette nuit.

**Note opérationnelle sur `run_in_background`** : les jobs lancés via l'outil background du harness survivent à la fermeture d'un terminal (détachés comme avec `disown`), mais PAS à la fermeture complète de la session/app Claude Code (ils restent rattachés au runtime qui héberge la session). L'utilisateur passe en session tmux séparée pour ce soir précisément pour éviter ce risque sur les runs longs à venir.

### Prochaines étapes, DANS CET ORDRE :
1. **Corriger le pipeline `datasets.py` (§0)** — bloquant absolu avant tout entraînement. Recommandation : approche (a) (génération jointe directe, abandon temporaire du multiplicateur ×50).
2. Valider le fix avec un script léger (mesurer ρ et gap RZF-WMMSE sur un batch du nouveau cache régénéré, comparer aux chiffres de §1 point 6/8).
3. Invalider les caches dataset existants (`/export/tmp/sala/sionna_base_5k_8x4*.npz` — déjà invalidé une fois pour la révision NLOS précédente, à refaire pour cette révision clustering ; pas encore de cache 32x8 pour MASSIVE, sera généré au premier usage).
4. Régénérer `classical_comparison.py` officiel sur STANDARD (M8K4) et MASSIVE (M32K8) avec le nouveau canal — **celui-ci n'est PAS bloqué par §0** (n'utilise pas `datasets.py`), peut être relancé immédiatement pour produire le tableau de référence final.
5. Cascade (après le fix §0 uniquement) : T-sweep TA-RB (`diag_tarb_T_sweep.py`), refenêtrage IntraRB (`diag_intrarb_window.py`, relancer sur le nouveau canal), reprise schedules LR (`diag_lr_schedules.py`).
6. Mettre à jour le modèle de complexité/énergie (Volet 3) au fil de l'eau à mesure que les architectures se stabilisent, avec les configs finales (M32K8, pas M64K32).

---

## 4. Décisions tranchées — NE PAS remettre en question sans raison nouvelle

- **RB=12 fixe** comme référence pour IntraRB (le refenêtrage teste des alternatives, RB=12 reste le point de comparaison).
- **Fix OFDM sur TA-RB** (1 symbole pilote + tile) : preuve physique (canal identique bit-à-bit sur 14 symboles), acquis.
- **Gate sigmoid v4.3 retiré** de TA-RB : gradient quasi nul, aucun gain mesuré. Acquis.
- **WMMSE (bisection sur μ, commit `3ea7f68`)** : validé sous LOS ET sous NLOS, conforme à Shi et al. 2011. Ne plus remettre en doute.
- **K=M=64 rejeté** pour MASSIVE : contredit M≫K, même si gap énorme obtenu.
- **Le gap RZF-WMMSE est secondaire** à l'objectif (Transformer bat-il les baselines) — ne pas chercher à le maximiser au-delà de ce qui est déjà trouvé.
- **Ancien pipeline PyTorch/TF pré-Sionna** : confirmé sans warmup caché.
- **Hypothèses réfutées, ne pas retester** : gradient qui s'aplatit à haut SINR (il croît, mesuré) ; contrainte de puissance par (SC,user) comme cause du plafonnement (RZF a la même contrainte et égale WMMSE).
- **Mécanisme de canal** : clustering spatial (§1), PAS l'ancien mécanisme d'angle étroit. Ne pas revenir en arrière sans raison nouvelle et mesurée.

---

## 5. Fichiers créés/modifiés cette nuit (tous dans `final/` sauf mention contraire)

**Modifiés** : `channel_config.py` (révision clustering, voir §1), `main_finall.py` (call site `set_locked_topology` corrigé), `precoder_intra_rb.py`, `precoders_w.py`.

**Créés — scripts de diagnostic** (fonctionnels, réutilisables ; ceux marqués ⚠️ dépendent de `datasets.py` et sont bloqués par §0) :
`diag_ta_rb_memory.py`, `diag_ta_rb_gradient.py`, `diag_ta_rb_ablation.py`, `diag_grad_vs_snr.py` ⚠️, `diag_lr_continuation.py` ⚠️, `diag_v2_smoke.py` ⚠️ (+ `precoders_v2.py`, pas affecté lui-même), `diag_intrarb_window.py` ⚠️, `diag_lr_schedules.py` ⚠️, `diag_tarb_T_sweep.py` ⚠️ (prêt, pas lancé), `diag_new_channel_search.py`/`search2.py`, `diag_new_channel_massive.py`/`massive2.py`, `diag_massive_K_sweep.py`, `diag_wmmse_nlos_check.py`, `diag_standard_3gpp_channel.py`, `diag_spatial_cluster_channel.py` (contient l'implémentation originale de `gen_topology_clustered`, migrée dans `channel_config.py` — celle de `channel_config.py` fait foi désormais), `diag_spatial_cluster_r20_k8.py`, `diag_massive_cluster_sweep.py`, `diag_massive_tighter_r.py`, `diag_m32_final_round.py`.

**Créé — autre** : `complexity_energy_results_v2.csv` (Volet 3, partiel, valeurs M64K32 à refaire pour M32K8).

---

*Fin du résumé. Reprendre à l'étape 1 de la section 3 ci-dessus (corriger `datasets.py` en premier).*
