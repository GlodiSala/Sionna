# Archive

Rien ici n'est nécessaire pour reproduire les résultats du mémoire : tout ce
qui porte un chiffre cité vit dans `mimo_precoding/`. Ce dossier garde la
trace du chemin parcouru — en particulier les impasses, qui sont la seule
raison pour laquelle on sait que les choix retenus sont les bons.

**Avertissement** : beaucoup de chiffres présents ici sont périmés. Voir
§6.6 du `README.md` racine pour la liste. En cas de contradiction avec
`mimo_precoding/results/`, c'est `mimo_precoding/` qui fait foi.

## `exploration/`

~110 scripts de diagnostic et leurs 190 fichiers de résultats. Les familles :

| Préfixe | Ce qui s'y trouve | Pourquoi c'est ici |
|---|---|---|
| `diag_new_channel_*`, `diag_spatial_cluster_*`, `diag_standard_3gpp_channel`, `reconcile_discrepancy` | recherche du canal (février → 7 août) | canal LOS forcé / fenêtre angulaire étroite abandonnés : mesurés quasi plats en fréquence, donc non discriminants pour des architectures fréquentielles |
| `diag_front_a_*` (baseline, bilinear, gram, snr_weighted) | variantes du front-end d'attention | battues par SignedGateMHA |
| `diag_massive_gap_{diagnostic,sweep,tighter_r}`, `diag_massive_K_sweep`, `diag_massive_cluster_sweep` | diagnostic du plafond 73-80 % RZF à M=64 | a conclu à D=384/L=4 + budget 170 ep, conclusion reportée dans `training/` |
| `diag_lr_*`, `diag_warmup_ramp_*`, `diag_grad_vs_snr` | ordonnancement du pas d'apprentissage | a produit le protocole `cosine_long` / la rampe de warmup, maintenant dans `system.py` |
| `diag_csi_imperfect_*` (non-seedfix), `classical_comparison_*` (non-seedfix), `tsweep_*` (non-2560) | campagnes d'évaluation antérieures | remplacées par la campagne seedfix du 20 août (tirages non appariés, graine Sionna non verrouillée) |
| `compute_complexity_energy_v3`, `complexity_energy_results_v{2,3}.csv` | énergie/complexité anciennes | « MASSIVE » = M32×K8 (ancienne config), TA-RB T=3 sans attention signée |
| `main_finall.py.bak_energy_fix` | `system.py` avant le correctif énergie du 3 septembre | sert de référence pour le diff du correctif (cf. §6.1 du README racine) |

## `reports/`

Journaux et rapports chronologiques, par ordre d'écriture :

| Fichier | Date | Statut |
|---|---|---|
| `SESSION_NUIT_RESUME.md` | 7 août | historique |
| `SESSION_LOG_20260807.md` | 7 août | historique |
| `SYNTHESE_RESULTATS.md` | 8 août | **chiffres périmés** (pré-D=384, pré-seedfix) |
| `CHIFFRES_FINAUX_CHAPITRE3.md` | 9 août | **chiffres périmés** malgré le nom |
| `ANNEXE_B_DONNEES.md` | 10 août | données d'annexe, toujours valide (cohérence/SNR intermédiaires) |
| `SESSION_LOG_20260808.md` | 8 → 13 août | journal principal du gap-closing D=384, UMi puis UMa |
| `RAPPORT_MASSIVE_GAP_CLOSING_20260812.md` | 12 août | gap-closing UMi massive, méthode + chiffres avant/après |
| `RAPPORT_INVESTIGATION_FRANCOIS_ROUND2_20260812.md` | 12 août | question du directeur : NLOS forcé ou probabiliste ? |
| `RAPPORT_LOS_NLOS_FORCE_20260812.md` | 12 août | réponse : **forcé** explicitement (`los=False`), preuve par le code et la docstring Sionna |
| `RAPPORT_MASSIVE_UMA_GAP_CLOSING_20260813.md` | 13 août | réplication du gap-closing sur UMa massive |

Aucun rapport ne couvre la campagne seedfix du 20 août — c'est le `README.md`
racine qui en tient lieu.

## `release_20260809/`

Bundle « release » assemblé le 9 août. Sa structure a servi de modèle à
`mimo_precoding/` (même découpe, mêmes noms de modules), mais son contenu est
antérieur au gap-closing D=384 et à la campagne seedfix. Conservé pour
référence ; ne pas y puiser de chiffres.

## `legacy/`

Tout l'avant-projet, de 2025 à la mi-2026 : 176 PNG, 20 scripts racine
(`main_sionna_*`, `transformer_5d_sionna*`, `wmmse_precoder.py`…), et les
anciennes études `frequency_study_sionna/`, `presentation_figures/`,
`results_ee/`, `transformer_frequency_analysis/`. Aucun lien avec le code
actuel.

## `weights_massive_gap_diag/`

228 Mo de checkpoints produits par le diagnostic du plafond massive
(`diag_massive_gap_*`). Ignorés par git. Supprimables : les checkpoints de
production sont dans `mimo_precoding/weights/`.
