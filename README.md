# Précodeurs MU-MIMO OFDM à attention signée

Code de simulation et d'entraînement (Sionna / TensorFlow) pour trois
précodeurs linéaires appris, équivariants par permutation des utilisateurs,
évalués contre RZF et WMMSE sur quatre régimes : canal {UMi, UMa} × échelle
{standard M=8/K=4, massive M=64/K=8}.

Mémoire de maîtrise, Polytechnique Montréal — Glodi Sala, dir. François
Leduc-Primeau. Le texte du mémoire (`7-Theme1.tex`, chapitre 3) vit dans un
dépôt Overleaf séparé ; **ici c'est le code et les données sources des
tableaux et figures**.

---

## 1. La contribution, en une page

Le point de départ est un coût mesuré : en MU-MIMO OFDM, calculer un
précodeur par sous-porteuse est cher, et la parade classique — grouper les
sous-porteuses par bloc de ressources (RB) et n'envoyer qu'un précodeur par
RB — coûte beaucoup de débit. Sur le canal UMi standard, RZF appliqué par
RB de 12 sous-porteuses ne rend plus que **70 %** du débit de RZF par
sous-porteuse à 20 dB (**50 %** sur UMa, canal plus sélectif). Le mécanisme
est chiffré dans `experiments/mechanism_*.py` : un plancher d'interférence
résiduelle `Δ_sc · pinv(H_avg)` qui ne dépend que du canal, donc qui domine
à haut SNR.

Les trois architectures apprennent à compenser cette perte :

| | Ce qu'elle fait | Structure fréquentielle |
|---|---|---|
| **SC** (`SingleSCTransformerPrecoderSignedAttn`) | attention entre utilisateurs, une passe par sous-porteuse | aucune (référence de capacité) |
| **IB** (`IntraRBTransformerPrecoderSignedAttn`) | attention sous-porteuses puis utilisateurs, à l'intérieur de chaque RB | intra-RB |
| **TA-RB** (`TransformerPrecoderCleanResidualSignedAttn`) | chaque RB est résumé en `T` tokens (moyenne/variance) avant attention utilisateurs, décodeur résiduel vers les poids par sous-porteuse | compression par RB, réglable par `T` |

Le mécanisme commun est **SignedGateMHA** (`precoders/signed_attention.py`) :
une attention multi-têtes dont la magnitude softmax est multipliée par une
porte de signe `tanh`. L'attention softmax standard ne produit qu'une
*combinaison convexe* des valeurs (poids ≥ 0, somme 1), alors que le
précodeur optimal à haut SNR tend vers le forçage à zéro, qui exige des
coefficients **signés** entre utilisateurs pour annuler l'interférence.
SignedGateMHA donne ce degré de liberté en gardant la stabilité du softmax,
et reste exactement équivariante par permutation des K utilisateurs.

Le canal n'est pas un détail : la configuration retenue (`channel_config.py`)
est NLOS **forcé** (`los=False`, ce qui désactive le tirage probabiliste
3GPP §7.4.2) sur le secteur standard 120°, avec **regroupement spatial** des
utilisateurs — un centre de cluster tiré dans le secteur, puis K
utilisateurs dans un disque de rayon R autour. Les révisions successives et
les raisons de chaque abandon (LOS forcé = canal quasi plat en fréquence,
fenêtre angulaire étroite = utilisateurs toujours trop séparés entre eux)
sont documentées en tête de `channel_config.py`.

---

### Les six chiffres à retenir

1. **Le coût à battre.** RZF par RB de 12 sous-porteuses ne rend que
   **70,4 %** du débit de RZF par sous-porteuse à 20 dB sur UMi standard, et
   **50,1 %** sur UMa. La dégradation est continue en fonction de la taille
   du groupe (SINR moyen 28,05 dB à T=1 → 23,37 dB à T=12,
   `mechanism_rzf_grouping_degradation`).
2. **Sous CSI parfait, les réseaux récupèrent presque tout** : 93 à 99 % de
   RZF-SC sur UMi standard (IB 95,4 %, TA-RB 95,0 %, SC 93,3 % à 20 dB),
   74 à 93 % à l'échelle massive (D=384).
3. **Le résultat principal est la robustesse au CSI imparfait.** À pilote
   20 dB, TA-RB tient **27,33 bps/Hz contre 23,13 pour RZF et WMMSE** à
   20 dB (standard) — **+18 %**, avec une rétention de **70,7 % contre
   56,9 %**. À l'échelle massive (D=384) : **75,27 contre 67,83 bps/Hz**
   (+11 %), rétention **86,2 % contre 58,1 %**. TA-RB ne fait pas que
   résister : il **dépasse** WMMSE dès que l'estimation de canal est bruitée.
4. **L'énergie reste négligeable** devant WMMSE : TA-RB T=6 consomme
   **0,087 µJ contre 4,31 µJ** (50×) à l'échelle standard, et **1,82 µJ
   contre 2 190 µJ** (1 204×) à l'échelle massive, pour respectivement 5,5×
   et 132× moins de FLOPs (modèle analytique INT8 vs FP32, §4.3).
5. **Le compromis sur T est double.** T=4 est optimal en CSI parfait
   (97,2 % de WMMSE en moyenne) mais la rétention sous CSI imparfait décroît
   de façon monotone avec T (97,6 % à T=1 → 79,2 % à T=12) : l'optimum
   descend à T=2-3 si le CSI est bruité.
6. **Le durcissement du canal à M=64 est vérifié** : WMMSE ≡ RZF à moins de
   0,01 % près, sur les deux canaux. La cible légitime à cette échelle est
   donc RZF, et ce n'est pas un artefact de réglage du canal — aucun
   regroupement artificiel d'utilisateurs n'est ajouté à l'échelle massive.

---

## 2. Structure

```
mimo_precoding/
  channel_config.py          Configs de canal verrouillées (STANDARD_CONFIG M8K4,
                             MASSIVE_TRUE_CONFIG M64K8) + topologie à cluster spatial.
  datasets.py                JointClusterGenerator / JointClusterDataset : tirage
                             conjoint des K utilisateurs, cache disque .npz.
  system.py                  MU_MIMO_System (chaîne Sionna complète : LDPC → précodage
                             → canal OFDM → égalisation LMMSE → décodage), SupervisedTrainer
                             (warmup MSE vs RZF puis finetune débit-somme),
                             compute_energy_uJ / compute_classical_complexity.
  eval_system.py             ConfigurableMIMOSystem : système allégé (débit seulement,
                             sans LDPC) pour les balayages CSI et les références classiques.
  classical_reference.py     LockedClusterSystem + référence RZF/WMMSE par régime.
  precoders/
    classical.py             RZF, WMMSE (allocation de puissance eigh + bissection sur µ
                             en float64/complex128). Contient aussi les précodeurs
                             transformer V4/V5 historiques, importés par system.py.
    rb_grouping.py           RZF avec regroupement par RB + compute_{rzf,wmmse}_flops.
    base_intra_rb.py         Classes de base SC / IB (attention softmax standard).
    base_residual.py         Classe de base TA-RB résiduel.
    signed_attention.py      SignedGateMHA + les trois architectures retenues.
  training/                  Un script par régime (voir §5).
  experiments/               Les évaluations canoniques (voir §4).
  figures/<régime>/          Un dossier par régime, chaque fig*.py régénère ses
                             .png/.pdf/.json depuis ../../results/.
  results/                   Les 88 fichiers JSON/NPY qui portent les chiffres du mémoire.
  weights/                   Checkpoints (657 Mo, hors dépôt — régénérables par training/).
archive/
  exploration/               ~110 scripts de diagnostic et leurs 190 résultats : canaux
                             abandonnés, variantes d'attention non retenues, balayages
                             à budget insuffisant. Valeur documentaire (les impasses).
  reports/                   Logs de session et rapports chronologiques. Attention :
                             plusieurs contiennent des chiffres périmés, cf. archive/README.md.
  release_20260809/          Ancien bundle « release » du 9 août : bonne structure,
                             contenu pré-D=384 et pré-seedfix. Remplacé par mimo_precoding/.
  legacy/                    Tout l'avant-projet (2025 → mi-2026) : 176 PNG, 20 scripts.
```

---

## 3. Environnement et exécution

L'environnement conda contient TensorFlow 2.20, Sionna 1.2.1 et matplotlib 3.10.6 :

```bash
export LD_LIBRARY_PATH=/export/tmp/sala/miniconda/envs/tf-gpu/lib:$LD_LIBRARY_PATH
PY=/export/tmp/sala/miniconda/envs/tf-gpu/bin/python
```

Le `LD_LIBRARY_PATH` n'est pas optionnel : sans lui, matplotlib échoue sur
`GLIBCXX_3.4.29 not found` (le `libstdc++` système est plus ancien que celui
attendu par les roues conda).

Tous les scripts de `experiments/` et `training/` se placent eux-mêmes à la
racine de `mimo_precoding/` au démarrage : ils peuvent être lancés depuis
n'importe quel répertoire, et écrivent toujours dans `results/` et
`weights/` de ce paquet.

```bash
$PY experiments/complexity_energy_standard.py        # CPU, quelques secondes
$PY experiments/eval_umi_standard.py --num_batches 10 # GPU
```

Les caches de canaux (`.npz`, plusieurs Go) vivent sous `/export/tmp/sala/`
et sont régénérés automatiquement s'ils manquent (coûteux : compter ~10-30
min par régime).

---
## 4. Résultats canoniques — campagne « seedfix » du 20 août 2026

**Un seul jeu de résultats fait foi** : celui du 20 août. C'est le plus
récent et le seul produit sous les trois garanties méthodologiques
suivantes, absentes des campagnes antérieures :

1. `sionna_config.seed = 42` en plus de `tf.random.set_seed` — le vrai
   verrou de `config.tf_rng`, que `tf.random.set_seed` seul ne fixe pas ;
2. **un seul tirage de canal par lot, partagé par les 6 méthodes** —
   avant, les méthodes neuronales évaluaient sur un tirage distinct de
   celui des classiques, ce qui mélangeait écart de méthode et bruit
   d'échantillonnage ;
3. **double métrique**, calculée sur les mêmes canaux : la métrique Sionna
   (`_sum_rate` / `LMMSEPostEqualizationSINR`, celle des tableaux actuels
   du mémoire) et le SINR manuel direct
   `SINR_k = |h_k^H w_k|² / (Σ_{j≠k} |h_k^H w_j|² + n₀)`, c'est-à-dire le
   critère interne de WMMSE appliqué uniformément aux 6 méthodes.

Chaque cellule des tableaux ci-dessous est donc **`Sionna (% RZF-SC) /
SINR manuel (% RZF-SC)`**. L'écart entre les deux métriques n'est pas du
bruit : il est systématique et il est informatif — la post-égalisation
LMMSE de la métrique Sionna pénalise plus durement les précodeurs à
représentation fréquentielle compressée (RZF-RB12 passe de 70,4 % à
77,1 % de RZF-SC à 20 dB selon la métrique).

### 4.1 Où vit chaque chiffre

| Tableau / figure du mémoire | Script | Fichiers de résultats |
|---|---|---|
| `tab:results_umi` (T3.9), UMi std | `experiments/eval_umi_standard.py` | `classical_comparison_M8K4_seedfix_{sionna,manual_sinr}_20260820_023135.json` |
| T3.10, UMa std | `experiments/eval_uma_standard.py` | `classical_comparison_M8K4_uma_seedfix_*_20260820_125429.json` |
| `tab:results_massive`, UMi M64K8 | `experiments/eval_massive_d128_d384.py` | `…_D384_*_20260820_032242.json` (D=384), `…_D128_*_20260820_090729.json` |
| CSI imparfait, UMi std | `experiments/eval_csi_imperfect_umi_standard.py` | `diag_csi_imperfect_seedfix_*_20260820_030418.json` |
| CSI imparfait, UMi massive D=384 | `experiments/eval_csi_imperfect_umi_massive_d384.py` | `…_massive_capD384L4_seedfix_*_20260820_031306.json` |
| CSI imparfait, UMa massive D=384 + UMi massive D=128 | `experiments/eval_csi_imperfect_massive_2regimes.py` | `…_uma_massive_D384_*_144320.json`, `…_umi_massive_D128_*_144542.json` |
| `tab:ablation_T` | `experiments/ablation_tokens_T.py` + `ablation_tokens_T_rzf_ref.py` | `tsweep_seedfix_*_2560_20260820_153545.json`, `tab_ablation_T_seedfix_2560_recompute.json` |
| Mécanisme (plancher d'interférence, plafond SINR, dégradation RZF selon T) | `experiments/mechanism_*.py` | `verify_{interference_floor,sinr_ceiling,rzf_grouping_degradation}_result.json` |
| Complexité / énergie | `experiments/complexity_energy_{standard,massive_d384,per_T}.py` | `complexity_energy_*.json`, `energy_per_T_*.json` |
| Niveau système (ordonnanceur, MCS) | `experiments/eval_system_scheduler.py` | `diag_system_scheduler_eval*.json` |
| BER, FER, mécanisme du plancher (§4.6) | `experiments/eval_ber_{umi_standard,csi_imperfect}.py`, `experiments/mechanism_ber_error_floor.py` | `diag_ber_umi_standard_seedfix_*.json`, `diag_ber_csi_imperfect_seedfix_*.json`, `mechanism_ber_error_floor_*.json` |
| `tab:coherence`, Annexe B | `experiments/channel_coherence*.py`, `intermediate_snr_annex_b.py` | `diag_coherence_*.json`, `diag_intermediate_snr_*.json` |

#### Couverture : ce qui existe, régime par régime

| Régime | Débit CSI parfait | Débit CSI imparfait | BER | Niveau système |
|---|---|---|---|---|
| **UMi standard** M8K4 | ✔ | ✔ (TA-RB T=6) | ✔ | ✔ |
| **UMa standard** M8K4 | ✔ | ✔ (TA-RB T=4) | — | — |
| UMi massive M64K8 D=384 | ✔ | ✔ | — | — |
| UMi massive M64K8 D=128 | ✔ | ✔ | — | — |
| UMa massive M64K8 D=384 | ✔ | ✔ | — | — |

Le BER et l'évaluation système n'existent **qu'en UMi standard** : c'est le
régime de référence (seul avec le balayage T complet), `eval_system_scheduler.py`
est câblé sur `STANDARD_CONFIG`, et à l'échelle massive le durcissement du canal
pousse le BER sous la résolution du budget Monte-Carlo utilisé ici. Aucune
affirmation sur le BER ou le débit livré ne peut donc être étendue aux autres
régimes.

### 4.2 Débit somme, CSI parfait

Format des cellules : `Sionna (% RZF-SC) / SINR manuel (% RZF-SC)`, en bps/Hz.

#### UMi standard (M=8, K=4) — D=128, 10 lots × 32 canaux

| SNR | RZF-SC | RZF-RB12 | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 0 dB | **14.85** / 14.96 | 13.57 (91.4%) / 13.92 (93.1%) | 14.95 (100.7%) / 15.21 (101.7%) | 14.61 (98.4%) / 14.64 (97.9%) | 14.60 (98.3%) / 14.63 (97.8%) | 14.48 (97.5%) / 14.59 (97.5%) |
| 5 dB | **21.01** / 21.05 | 18.70 (89.0%) / 19.43 (92.3%) | 20.99 (99.9%) / 21.13 (100.4%) | 20.77 (98.9%) / 20.86 (99.1%) | 20.82 (99.1%) / 20.88 (99.2%) | 20.66 (98.3%) / 20.82 (98.9%) |
| 10 dB | **27.39** / 27.41 | 23.01 (84.0%) / 24.32 (88.7%) | 27.37 (99.9%) / 27.42 (100.1%) | 26.90 (98.2%) / 27.12 (98.9%) | 27.02 (98.6%) / 27.17 (99.1%) | 26.80 (97.8%) / 27.05 (98.7%) |
| 15 dB | **33.89** / 33.89 | 26.28 (77.5%) / 28.33 (83.6%) | 33.88 (100.0%) / 33.90 (100.0%) | 32.71 (96.5%) / 33.21 (98.0%) | 33.05 (97.5%) / 33.40 (98.5%) | 32.85 (96.9%) / 33.25 (98.1%) |
| 20 dB | **40.66** / 40.66 | 28.61 (70.4%) / 31.35 (77.1%) | 40.65 (100.0%) / 40.66 (100.0%) | 37.95 (93.3%) / 39.01 (95.9%) | 38.80 (95.4%) / 39.52 (97.2%) | 38.64 (95.0%) / 39.37 (96.8%) |

#### UMa standard (M=8, K=4) — D=128, 20 lots × 32 canaux

| SNR | RZF-SC | RZF-RB12 | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 0 dB | **13.60** / 13.76 | 10.60 (77.9%) / 11.34 (82.4%) | 13.91 (102.3%) / 14.22 (103.3%) | 13.27 (97.6%) / 13.30 (96.7%) | 13.24 (97.4%) / 13.27 (96.4%) | 13.07 (96.1%) / 13.26 (96.3%) |
| 5 dB | **19.36** / 19.44 | 14.21 (73.4%) / 15.56 (80.0%) | 19.42 (100.3%) / 19.63 (101.0%) | 19.14 (98.9%) / 19.21 (98.8%) | 19.08 (98.6%) / 19.17 (98.6%) | 18.66 (96.4%) / 18.97 (97.6%) |
| 10 dB | **25.65** / 25.69 | 16.85 (65.7%) / 18.90 (73.6%) | 25.65 (100.0%) / 25.76 (100.3%) | 25.29 (98.6%) / 25.46 (99.1%) | 25.16 (98.1%) / 25.37 (98.8%) | 24.29 (94.7%) / 24.82 (96.6%) |
| 15 dB | **32.10** / 32.12 | 18.67 (58.1%) / 21.35 (66.5%) | 32.10 (100.0%) / 32.14 (100.1%) | 31.25 (97.3%) / 31.60 (98.4%) | 30.95 (96.4%) / 31.41 (97.8%) | 29.61 (92.2%) / 30.47 (94.9%) |
| 20 dB | **38.67** / 38.67 | 19.39 (50.1%) / 22.55 (58.3%) | 38.66 (100.0%) / 38.68 (100.0%) | 36.70 (94.9%) / 37.45 (96.8%) | 35.99 (93.1%) / 36.98 (95.6%) | 33.81 (87.5%) / 35.32 (91.3%) |

#### UMi massive (M=64, K=8) — D=384, L=4, 5 lots × 16 canaux

| SNR | RZF-SC | RZF-RB12 | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 0 dB | **63.81** / 63.85 | 50.76 (79.5%) / 54.08 (84.7%) | 63.94 (100.2%) / 64.00 (100.2%) | 59.45 (93.2%) / 60.21 (94.3%) | 59.06 (92.6%) / 59.88 (93.8%) | 56.81 (89.0%) / 58.56 (91.7%) |
| 5 dB | **76.97** / 76.99 | 56.05 (72.8%) / 60.56 (78.7%) | 77.00 (100.0%) / 77.04 (100.1%) | 69.70 (90.5%) / 71.28 (92.6%) | 69.41 (90.2%) / 71.08 (92.3%) | 67.83 (88.1%) / 69.98 (90.9%) |
| 10 dB | **90.31** / 90.32 | 60.15 (66.6%) / 65.73 (72.8%) | 90.31 (100.0%) / 90.32 (100.0%) | 78.55 (87.0%) / 81.52 (90.3%) | 78.27 (86.7%) / 81.37 (90.1%) | 77.62 (86.0%) / 80.81 (89.5%) |
| 15 dB | **103.52** / 103.52 | 62.30 (60.2%) / 68.64 (66.3%) | 103.51 (100.0%) / 103.52 (100.0%) | 83.68 (80.8%) / 88.52 (85.5%) | 82.88 (80.1%) / 87.91 (84.9%) | 83.88 (81.0%) / 88.64 (85.6%) |
| 20 dB | **116.75** / 116.75 | 63.92 (54.7%) / 70.79 (60.6%) | 116.75 (100.0%) / 116.75 (100.0%) | 86.98 (74.5%) / 93.36 (80.0%) | 85.78 (73.5%) / 92.31 (79.1%) | 87.33 (74.8%) / 93.62 (80.2%) |

#### UMi massive (M=64, K=8) — D=128 budget étendu (référence avant gap-closing)

| SNR | RZF-SC | RZF-RB12 | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 0 dB | **66.15** / 66.15 | 54.36 (82.2%) / 57.52 (86.9%) | 66.14 (100.0%) / 66.16 (100.0%) | 46.78 (70.7%) / 47.24 (71.4%) | 46.84 (70.8%) / 47.28 (71.5%) | 46.01 (69.6%) / 47.32 (71.5%) |
| 5 dB | **79.43** / 79.43 | 60.58 (76.3%) / 65.02 (81.9%) | 79.43 (100.0%) / 79.43 (100.0%) | 58.41 (73.5%) / 59.47 (74.9%) | 58.59 (73.8%) / 59.61 (75.0%) | 59.16 (74.5%) / 60.44 (76.1%) |
| 10 dB | **92.75** / 92.75 | 64.23 (69.2%) / 69.77 (75.2%) | 92.75 (100.0%) / 92.75 (100.0%) | 67.44 (72.7%) / 69.84 (75.3%) | 67.94 (73.3%) / 70.22 (75.7%) | 70.97 (76.5%) / 72.70 (78.4%) |
| 15 dB | **105.84** / 105.84 | 67.39 (63.7%) / 73.74 (69.7%) | 105.84 (100.0%) / 105.84 (100.0%) | 74.21 (70.1%) / 78.56 (74.2%) | 74.71 (70.6%) / 78.84 (74.5%) | 80.70 (76.2%) / 83.63 (79.0%) |
| 20 dB | **119.49** / 119.49 | 70.68 (59.2%) / 77.69 (65.0%) | 119.49 (100.0%) / 119.49 (100.0%) | 79.44 (66.5%) / 85.43 (71.5%) | 80.62 (67.5%) / 86.45 (72.4%) | 87.56 (73.3%) / 92.35 (77.3%) |

### 4.3 CSI imparfait (pilote 20 dB) — débit et rétention vs CSI parfait


**UMi standard M8K4 (TA-RB T=6)**

| SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0 dB | 12.79 (86.1%) | 13.21 (88.4%) | 12.47 (85.3%) | 12.48 (85.5%) | 12.98 (89.6%) |
| 5 dB | 17.12 (81.5%) | 17.22 (82.1%) | 16.91 (81.4%) | 16.98 (81.5%) | 18.17 (88.0%) |
| 10 dB | 20.39 (74.4%) | 20.42 (74.6%) | 20.14 (74.8%) | 20.27 (75.0%) | 22.60 (84.3%) |
| 15 dB | 22.26 (65.7%) | 22.27 (65.7%) | 21.94 (67.1%) | 22.10 (66.8%) | 25.69 (78.2%) |
| 20 dB | 23.13 (56.9%) | 23.14 (56.9%) | 22.77 (60.0%) | 22.96 (59.2%) | 27.33 (70.7%) |

**UMa standard M8K4 (TA-RB T=4)**

| SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0 dB | 12.04 (88.5%) | 12.82 (92.1%) | 11.67 (87.9%) | 11.66 (88.1%) | 12.18 (93.2%) |
| 5 dB | 15.97 (82.5%) | 16.22 (83.5%) | 15.80 (82.5%) | 15.80 (82.8%) | 17.02 (91.2%) |
| 10 dB | 19.09 (74.4%) | 19.14 (74.6%) | 18.94 (74.9%) | 18.95 (75.3%) | 21.43 (88.2%) |
| 15 dB | 21.00 (65.4%) | 21.01 (65.5%) | 20.82 (66.6%) | 20.80 (67.2%) | 24.68 (83.3%) |
| 20 dB | 21.85 (56.5%) | 21.85 (56.5%) | 21.65 (59.0%) | 21.63 (60.1%) | 26.49 (78.3%) |

**UMi massive M64K8 D=384**

| SNR | RZF | WMMSE | SingleSC | IntraRB | TA_RB_residual |
|---|---|---|---|---|---|
| 0 dB | 58.77 (92.1%) | 58.75 (91.9%) | 56.08 (94.3%) | 55.82 (94.5%) | 57.90 (101.9%) |
| 5 dB | 64.21 (83.4%) | 64.20 (83.4%) | 60.99 (87.5%) | 60.65 (87.4%) | 66.62 (98.2%) |
| 10 dB | 66.79 (74.0%) | 66.78 (74.0%) | 63.51 (80.8%) | 63.17 (80.7%) | 72.48 (93.4%) |
| 15 dB | 67.56 (65.3%) | 67.56 (65.3%) | 64.00 (76.5%) | 63.71 (76.9%) | 74.40 (88.7%) |
| 20 dB | 67.83 (58.1%) | 67.83 (58.1%) | 64.27 (73.9%) | 63.95 (74.6%) | 75.27 (86.2%) |

### 4.4 Ablation T (TA-RB, UMi standard, 2560 échantillons/point)

| T | % WMMSE (CSI parfait, moy. 0-20 dB) | % retenu (CSI imparfait 15 dB) | FLOPs (M) | Énergie INT8 (µJ) — pré / post-correctif |
|---|---|---|---|---|
| 1 | 88.8% | 97.6% | 83.9 | 0.0137 / **0.0171** |
| 2 | 95.5% | 94.8% | 154.2 | 0.0247 / **0.0307** |
| 3 | 96.1% | 93.1% | 225.6 | 0.0359 / **0.0444** |
| 4 | 97.2% | 91.3% | 298.1 | 0.0473 / **0.0584** |
| 6 | 97.0% | 89.7% | 446.2 | 0.0705 / **0.0869** |
| 12 | 97.2% | 79.2% | 915.5 | 0.1441 / **0.1773** |

### 4.5 Complexité et énergie

Modèle analytique (pas de mesure matérielle) : RZF/WMMSE en FP32 comme
référence exacte, les trois architectures en INT8. **Ces chiffres sont ceux
du code actuel, après le correctif énergie du 3 septembre** (voir §6.1) —
ils diffèrent de ceux stockés dans `results/complexity_energy_*.json`, qui
sont antérieurs au correctif et conservés pour traçabilité.

| Régime | Méthode | FLOPs (M) | Énergie (µJ) | Gain énergie vs WMMSE |
|---|---|---|---|---|
| STANDARD M8K4 | RZF | 2,8 | 0,0092 | 469× |
| | WMMSE (I=10) | 2 456,6 | 4,3096 | 1× |
| | SC (INT8) | 812,5 | 0,1570 | 27× |
| | IB (INT8) | 1 023,2 | 0,1979 | 22× |
| | TA-RB T=6 (INT8) | 446,2 | 0,0869 | **50×** |
| | TA-RB T=4 (INT8) | 298,1 | 0,0584 | **74×** |
| MASSIVE M64K8, D=384 | RZF | 80,3 | 0,1917 | 11 423× |
| | WMMSE (I=10) | 1 249 485,8 | 2 189,85 | 1× |
| | SC (INT8) | 14 722,6 | 2,8234 | 776× |
| | IB (INT8) | 18 403,1 | 3,5304 | 620× |
| | TA-RB T=6 (INT8) | 9 452,4 | 1,8184 | **1 204×** |

Les FLOPs et les paramètres sont inchangés par le correctif (vérifié : les
six valeurs de `energy_per_T` recalculées reproduisent exactement les FLOPs
et les paramètres historiques, seule l'énergie bouge). Les paramètres sont
systématiquement vérifiés par `assert` contre `trainable_variables`.

---

### 4.6 BER, FER et mécanisme du plancher d'erreur

La chaîne BER est la chaîne Sionna complète (`system.py` : LDPC → précodage →
canal OFDM → égalisation LMMSE → décodage LDPC), à **modulation et codage
fixes** : QPSK, LDPC rate 1/2, et l'axe SNR est un **Eb/N0**. Elle transmet
donc 1 bit/symbole/utilisateur alors que le SINR post-égalisation à 20 dB en
autorise ~9 : il y a un facteur 9 de marge **sur la moyenne**. Un plancher de
BER ne peut donc pas venir du bruit — il vient de la **queue** de distribution.

À MCS fixe, une trame est perdue si le SINR de **cet utilisateur-là** passe
sous le seuil de décodage (~1,2 bps/Hz d'efficacité spectrale ; la limite de
Shannon pour QPSK r=1/2 est 1,0). Le BER ne mesure donc pas la qualité
moyenne du précodeur : il compte la fréquence des utilisateurs passés sous le
seuil. C'est une statistique du **pire**, pas de la **moyenne**.

#### Le mécanisme, mesuré

`experiments/mechanism_ber_error_floor.py` mesure, mot de code par mot de code
(un mot = un couple réalisation × utilisateur, k = 1152 bits d'information),
le nombre de bits erronés et l'efficacité spectrale du mot. Sur 4096 mots par
point, UMi standard, CSI parfait :

**Efficacité spectrale moyenne** — les cinq méthodes sont interchangeables :

| bps/Hz | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| RZF | 3,55 | 5,03 | 6,62 | 8,25 | 9,90 |
| WMMSE | 3,60 | 5,04 | 6,62 | 8,25 | 9,90 |
| SC | 3,47 | 4,96 | 6,47 | 7,90 | 9,12 |
| IB | 3,47 | 4,97 | 6,51 | 7,99 | 9,33 |
| TA-RB | 3,46 | 4,94 | 6,45 | 7,91 | 9,21 |

**Efficacité spectrale du pire mot de code** — c'est là que tout se joue
(seuil ≈ 1,2) :

| bps/Hz | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | comportement |
|---|---|---|---|---|---|---|
| RZF | 0,85 | 1,42 | 2,29 | 3,47 | **4,93** | croît avec le SNR → limité par le bruit |
| WMMSE | **0,02** | 0,85 | 1,70 | 3,06 | **4,78** | croît, converge vers RZF |
| SC | 0,50 | 0,85 | 1,15 | 1,32 | **1,36** | **sature** → limité par l'interférence |
| IB | 0,56 | 0,92 | 1,26 | 1,51 | **1,63** | sature |
| TA-RB | 0,61 | 0,96 | 1,25 | 1,46 | **1,58** | sature |

**Trames en échec (sur 4096)** — la conséquence directe :

| | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| RZF | 5 | 0 | 0 | 0 | 0 |
| WMMSE | **71** | 8 | 2 | 0 | 0 |
| SC | 29 | 4 | 2 | **1** | **1** |
| IB | 27 | 6 | 3 | **2** | **2** |
| TA-RB | 20 | 5 | 0 | 0 | 0 |

Trois faits en découlent, tous mesurés :

1. **Les erreurs sont concentrées, pas étalées.** 100 % des bits erronés
   tiennent dans au plus 3 mots de code sur 4096, et ces mots ont une
   efficacité spectrale de 1,3 à 2,0 bps/Hz contre 6,5 à 9,9 pour tous les
   autres. C'est de l'outage, pas une dégradation systématique — un défaut
   d'échelle de LLR, de normalisation de puissance ou de pilotes donnerait
   des erreurs réparties sur tous les mots.
2. **WMMSE s'effondre à bas SNR, pas à haut SNR.** À 0 dB son pire
   utilisateur est à 0,02 bps/Hz — éteint — alors que sa moyenne (3,60) est
   la meilleure du tableau. C'est le comportement attendu d'un algorithme qui
   maximise la **somme** des débits : servir un utilisateur mal placé coûte
   plus qu'il ne rapporte, l'optimum de la somme est donc de l'affamer. À MCS
   fixe ce sacrifié tombe sous le seuil. L'effet disparaît à haut SNR parce
   que WMMSE tend vers le forçage à zéro quand le bruit tend vers zéro : son
   pire mot rejoint celui de RZF (4,78 vs 4,93).
3. **Les précodeurs appris plafonnent à haut SNR.** Leur pire mot de code
   **sature** (SC : 1,15 → 1,32 → 1,36 de 10 à 20 dB, soit +0,04 entre 15 et
   20 dB quand RZF gagne +1,46). Ce qui limite ce mot n'est pas le bruit mais
   l'**interférence résiduelle** : c'est le signal destiné aux autres
   utilisateurs qui fuit, et cette fuite croît avec la puissance d'émission
   exactement comme le signal utile — le rapport reste constant, monter le
   SNR ne sert à rien. RZF annule l'interférence par inversion explicite du
   canal, sous-porteuse par sous-porteuse ; un réseau en produit une
   approximation, excellente en moyenne (92-94 % de RZF) mais qui laisse,
   sur environ 1 utilisateur sur 2 000, une fuite au niveau du seuil.

C'est exactement le mécanisme déjà établi pour le groupement RB dans ce dépôt
(`experiments/mechanism_interference_floor.py` : le terme `Δ_sc · pinv(H_avg)`
ne dépend pas du bruit, donc il domine à haut SNR). Les précodeurs appris en
héritent une version atténuée. Un précodeur appris **sans** plancher serait le
résultat surprenant, pas l'inverse.

Résumé en une ligne : **les précodeurs appris perdent ~7 % sur la moyenne
(9,12-9,33 vs 9,90 bps/Hz, ce qui recoupe les 93-95 % de RZF du §4.2) et
~70 % sur la queue (1,36-1,63 vs 4,93).**

#### Le même mécanisme sous CSI imparfait — le plancher devient la règle

`experiments/mechanism_ber_error_floor.py --csi pilot20` rejoue la mesure
quand le précodeur ne voit qu'une estimée bruitée du canal (pilote 20 dB) :

**SE moyenne (bps/Hz)**

| | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| RZF | 3.32 | 5.23 | 5.90 |
| WMMSE | 3.39 | 5.23 | 5.90 |
| SC | 3.25 | 5.17 | 5.82 |
| IB | 3.25 | 5.21 | 5.87 |
| TA-RB | 3.38 | 5.87 | 7.11 |

**SE du pire mot de code (seuil ≈ 1,2)**

| | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| RZF | 0.56 | 0.75 | 0.93 |
| WMMSE | 0.08 | 0.83 | 0.93 |
| SC | 0.29 | 0.51 | 0.53 |
| IB | 0.34 | 0.66 | 0.73 |
| TA-RB | 0.33 | 0.73 | 0.94 |

**Trames en échec sur 4096**

| | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| RZF | 15 | 2 | 1 |
| WMMSE | 68 | 2 | 1 |
| SC | 28 | 4 | 3 |
| IB | 31 | 2 | 2 |
| TA-RB | 25 | 2 | 1 |

Trois conséquences, et elles renversent la lecture du paragraphe précédent :

1. **RZF perd son annulation exacte.** Son pire mot de code plafonne à
   **0,93 bps/Hz** — *sous* le seuil de décodage — au lieu de monter à 4,93,
   et sa SE moyenne sature à 5,90 au lieu d'atteindre 9,90. RZF n'était exact
   que parce quon lui donnait le canal exact ; sur une estimée bruitée il
   inverse une matrice fausse, et l'erreur d'estimation produit exactement le
   même type de fuite proportionnelle au signal.
2. **L'écart de queue se referme.** À 20 dB : 1 trame en échec pour RZF,
   WMMSE et TA-RB, 2 pour IB, 3 pour SC — contre 0 vs 1-2 sous CSI parfait.
   Le plancher cesse d'être une faiblesse propre aux précodeurs appris.
3. **TA-RB est le seul à garder une moyenne nettement supérieure** : 7,11
   contre 5,90 bps/Hz pour RZF à 20 dB (+20,5 %, cohérent avec le +18 à
   +21 % de débit-somme du §4.3), avec une queue au niveau de celle de RZF
   (0,94 contre 0,93).

Autrement dit : sous CSI parfait, les précodeurs appris paient leur débit par
une queue plus lourde. Sous CSI imparfait — le cas réaliste — tout le monde a
une queue lourde, et TA-RB est le seul à ne pas payer ce défaut en moyenne.
Figure : `figE_se_mean_vs_tail_csi_imperfect.png`.

#### Quoi présenter, dans quel ordre

1. **`figures/umi_standard/figE_se_mean_vs_tail.png`** (et sa variante
   `_csi_imperfect`) — moyenne vs pire mot de code, et le FER qui en découle.
   C'est la figure explicative : elle rend le reste lisible. À montrer avant la
   courbe de BER.
2. **`figD_ber_snr_seedfix.png`** — la courbe BER vs SNR classique, double
   panneau CSI parfait / imparfait.
3. **Le niveau système** (§4.7) — avec adaptation de MCS, l'utilisateur en
   queue descend d'un cran de modulation et passe ; le plancher se paie en
   MCS, pas en débit.

Présenter le BER seul, sans (1) et (3), conduit le lecteur à « le BER est
moins bon donc ça ne marche pas », ce que les données ne disent pas.

#### Réserves

- **Taille d'échantillon.** 4096 mots de code ne résolvent pas un FER de
  ~10⁻⁴ : TA-RB affiche 0/4096 ci-dessus mais 8,2·10⁻⁵ de BER sur le run à
  50 tirages. Les runs BER complets (200 tirages = 204 800 mots, résolution
  4,9·10⁻⁶ en FER) tranchent ; le diagnostic ci-dessus sert au mécanisme, pas
  au classement fin des trois architectures.
- **Comptabilité des bits.** Un tirage de 256 porte 256 × 4 utilisateurs ×
  1152 bits = 1 179 648 bits d'information, donc 50 tirages résolvent
  1,7·10⁻⁸ en BER. Les « 0 » de RZF sont de vrais zéros sur 59 millions de
  bits, pas une limite d'échantillonnage.
- **Échelle massive.** Pas de BER à M=64 : le durcissement du canal pousse le
  BER sous la résolution du budget Monte-Carlo utilisé ici. BER = échelle
  standard uniquement.
- **T différent selon le panneau.** Le BER CSI parfait évalue TA-RB à T=6,
  le BER CSI imparfait à T=4 (checkpoints différents). `figD_ber_snr.py`
  annonce T=4 pour les deux, ce qui est faux pour le panneau de gauche ;
  `figD_ber_snr_seedfix.py` corrige l'étiquette.

### 4.7 Débit livré au niveau système (ordonnanceur + MCS adaptatif)

`experiments/eval_system_scheduler.py` remplace le MCS fixe du §4.6 par une
adaptation de modulation et de codage par utilisateur, avec ordonnanceur.
C'est la mesure qui dit ce que le précodeur **livre** réellement, une fois
que la queue de distribution est absorbée par un cran de MCS plutôt que par
une trame perdue.

**CSI parfait** — efficacité spectrale livrée (bps/Hz), équité de Jain entre parenthèses

| Méthode | 0.0 dB | 5.0 dB | 10.0 dB | 15.0 dB | 17.5 dB | 20.0 dB |
|---|---|---|---|---|---|---|
| RZF | 7.87 (0.99) | 10.61 (0.95) | 14.51 (0.87) | 19.01 (0.80) | 19.44 (0.65) | 19.24 (0.56) |
| WMMSE | 7.29 (1.00) | 10.86 (0.95) | 14.09 (0.87) | 18.45 (0.85) | 18.49 (0.68) | 19.39 (0.53) |
| SC | 6.47 (0.95) | 10.58 (0.92) | 14.85 (0.85) | 16.39 (0.80) | 16.22 (0.69) | 17.67 (0.58) |
| IB | 6.79 (1.00) | 10.60 (0.96) | 14.89 (0.92) | 16.44 (0.75) | 16.95 (0.77) | 16.82 (0.87) |
| TA-RB | 7.07 (0.99) | 10.56 (0.96) | 13.48 (0.87) | 17.69 (0.86) | 17.40 (0.74) | 17.92 (0.65) |

**CSI imparfait (pilote 20 dB)** — efficacité spectrale livrée (bps/Hz), équité de Jain entre parenthèses

| Méthode | 0.0 dB | 5.0 dB | 10.0 dB | 15.0 dB | 17.5 dB | 20.0 dB |
|---|---|---|---|---|---|---|
| RZF | 6.73 (1.00) | 8.64 (0.95) | 11.34 (0.94) | 12.45 (0.89) | 12.52 (0.90) | 12.74 (0.89) |
| WMMSE | 7.73 (1.00) | 9.20 (0.95) | 11.00 (0.90) | 12.61 (0.87) | 12.25 (0.84) | 12.45 (0.86) |
| SC | 6.08 (0.96) | 8.88 (0.98) | 10.23 (0.86) | 12.11 (0.92) | 11.59 (0.90) | 12.26 (0.87) |
| IB | 6.12 (0.97) | 9.39 (0.98) | 11.19 (0.90) | 11.87 (0.80) | 11.66 (0.91) | 12.94 (0.87) |
| TA-RB | 6.34 (0.97) | 10.56 (0.96) | 12.86 (0.94) | 15.42 (0.88) | 16.46 (0.80) | 13.74 (0.82) |

Deux lectures :

- **Sous CSI parfait**, les précodeurs appris livrent un peu moins que RZF et
  WMMSE (17,7-17,9 contre 19,2-19,4 bps/Hz à 20 dB) : cohérent avec leur perte
  de 7 % sur la moyenne et leur queue plus lourde. En revanche ils sont plus
  **équitables** à haut SNR (Jain 0,87 pour IB contre 0,56 pour RZF à 20 dB).
- **Sous CSI imparfait**, TA-RB domine nettement sur toute la plage utile :
  **16,46 contre 12,52 bps/Hz à 17,5 dB, soit +31 %** sur RZF. C'est le même
  avantage de robustesse que le §4.3 mesure en débit-somme, mais cette fois en
  débit effectivement livré, MCS et ordonnanceur compris.

Autrement dit : le plancher de BER du §4.6 se paie en un cran de MCS sur une
poignée d'utilisateurs, pas en perte de débit — et sous CSI imparfait le gain
de TA-RB survit intégralement au passage au niveau système.

#### Ce que le §4.7 modélise, et ce qu'il ne modélise pas

La boucle est celle de `sionna.sys` : ordonnanceur proportionnel équitable
(`PFSchedulerSUMIMO`) + adaptation de lien en boucle externe
(`OuterLoopLinkAdaptation`, cible **BLER = 10 %**) + abstraction PHY
(`PHYAbstraction`).

- **Le retour HARQ (ACK/NACK) est présent** et c'est lui qui pilote le choix
  du MCS : `PHYAbstraction` tire l'échec du bloc de transport contre son BLER
  (`harq_feedback = 0` si échec, `1` sinon), et l'OLLA ajuste le MCS slot
  après slot pour tenir la cible de 10 %.
- **Le débit compté est déjà net des erreurs** : `num_decoded_bits =
  harq_feedback × num_cb × cb_size`, donc un bloc en échec rapporte **zéro
  bit**. Les chiffres du tableau ci-dessus paient donc déjà le prix des
  trames perdues — le +31 % de TA-RB sous CSI imparfait est un gain **net**.
- **Il n'y a pas de retransmission HARQ** : un bloc en échec est perdu, pas
  renvoyé en redondance incrémentale, et l'ordonnanceur passe au slot
  suivant. Ces chiffres sont donc une **borne inférieure** : un vrai système
  récupérerait l'essentiel de ces 10 % à la deuxième transmission. C'est
  d'ailleurs la raison pour laquelle une cible de 10 % de BLER est le point
  de fonctionnement normal en LTE/NR — on l'assume parce que le HARQ la
  rattrape.
- **Ce n'est pas la même chaîne que le §4.6.** Le §4.6 fait tourner le vrai
  codec LDPC à MCS fixe (QPSK r=1/2) et compte les bits faux ; le §4.7 utilise
  l'abstraction PHY (courbes de BLER en fonction du SINR effectif) avec MCS
  adaptatif. Les deux sont complémentaires : le premier dit *pourquoi* la
  queue de distribution fait mal, le second dit *combien* elle coûte une fois
  le lien adapté.


## 5. Refaire les résultats

Les scripts sont indépendants : rien n'oblige à tout relancer. Ordre de
dépendance : `training/` produit un checkpoint + un JSON de manifeste dans
`results/`, que les scripts `experiments/` relisent pour retrouver le
checkpoint (clé `best_ckpt`).

### 5.1 Sans réentraîner (les checkpoints sont dans `weights/`)

```bash
# Tableaux principaux, les 4 régimes
$PY experiments/eval_umi_standard.py                 # ~10 min GPU
$PY experiments/eval_uma_standard.py --num_batches 20
$PY experiments/eval_massive_d128_d384.py            # D=128 et D=384

# CSI imparfait (pilote 20 dB)
$PY experiments/eval_csi_imperfect_umi_standard.py
$PY experiments/eval_csi_imperfect_umi_massive_d384.py
$PY experiments/eval_csi_imperfect_massive_2regimes.py

# Ablation T (2560 échantillons/point : ~1 h)
$PY experiments/ablation_tokens_T_rzf_ref.py
$PY experiments/ablation_tokens_T.py

# BER : courbes complètes (~1 h à 200 tirages) et mécanisme du plancher (~2 min)
$PY experiments/eval_ber_umi_standard.py  --num_batches 200
$PY experiments/eval_ber_csi_imperfect.py --num_batches 30
$PY experiments/mechanism_ber_error_floor.py --csi perfect  --snrs 0 5 10 15 20
$PY experiments/mechanism_ber_error_floor.py --csi pilot20  --snrs 0 10 20

# Mécanisme, complexité, système — CPU ou GPU léger
$PY experiments/mechanism_rzf_grouping_degradation.py
$PY experiments/mechanism_interference_floor.py
$PY experiments/mechanism_sinr_ceiling.py
$PY experiments/complexity_energy_standard.py
$PY experiments/complexity_energy_massive_d384.py
$PY experiments/complexity_energy_per_T.py

# Figures (depuis leur propre dossier)
cd figures/umi_standard && $PY figA_sumrate_seedfix.py
cd figures/umi_standard && $PY figE_se_mean_vs_tail.py    # figure explicative du BER
cd figures/umi_standard && $PY figD_ber_snr_seedfix.py    # courbes BER appariées
```

### 5.2 En réentraînant

| Régime | Script | Budget | Checkpoints produits |
|---|---|---|---|
| UMi std, SC + variantes d'attention | `training/train_umi_standard_sc_variants.py --variant signed_attn` | 83 ep (3+80) | `front_a_signed_attn` |
| UMi std, IB et TA-RB T=6 | `training/train_umi_standard_ib_tarb.py` | 83 ep | `IntraRB_signed_attn_4L_128d`, `TA_RB_residual_signed_attn_6tok_4L_128d` |
| UMi std, TA-RB balayage T | `training/train_umi_standard_tarb_Tsweep.py --T 4` | 83 ep | `TA_RB_residual_signed_attn_T4_4L_128d` |
| UMa std, les 3 | `training/train_uma_standard.py --arch …` | 83 ep | `UMa_*` |
| UMi massive D=128 | `training/train_umi_massive_d128.py --arch … --tag _extbudget` | **170 ep (10+160)** | `MASSIVE_TRUE_*_extbudget` |
| UMi massive D=384 | `training/train_umi_massive_d384.py --arch … --variant cap_d384l4` | 170 ep | `MASSIVE_TRUE_*_capD384L4` |
| UMa massive D=128 | `training/train_uma_massive_d128.py --arch …` | 170 ep | `UMa_MASSIVE_*` |
| UMa massive D=384 | `training/train_uma_massive_d384.py --arch … --variant cap_d384l4` | 170 ep | `UMa_MASSIVE_*_capD384L4` |

Compter 50 à 100 min par architecture à l'échelle massive (GPU A100/H100,
parfois partagé). **Le budget n'est pas négociable à M=64** : à 83 époques,
SC et IB ne sont pas convergés (débit encore en hausse, norme du gradient
encore croissante en fin de budget) et s'effondrent — voir §6.2.

---

## 6. Pièges connus et points à corriger dans le mémoire

### 6.1 Le correctif énergie du 3 septembre n'est pas propagé

`system.py` a été corrigé (correctif F. Leduc-Primeau) : l'énergie d'accès
mémoire `E_M`/`E_L` doit être **linéaire** en nombre de bits transférés, et
non hériter de l'exposant 1,9 du multiplieur matériel.

```python
# avant            EMAC = α·(Q/16)^1.9 ;  EM = 2·EMAC ;        EL = EMAC
# après            EMAC = α·(Q/16)^1.9 ;  EM = 2·α·(Q/16) ;    EL = α·(Q/16)
```

Tous les fichiers `results/complexity_energy_*.json` et
`energy_per_T_signed_attn.json` datent d'**avant** ce correctif. Les
recalculs post-correctif sont fournis à côté (`*_postfix_20260912.json`) et
ce sont eux qui figurent au §4.3. Conséquence à porter dans le mémoire :
l'avantage énergétique se réduit, **sans changer la conclusion** —
standard 66× → **50×** (TA-RB T=6 vs WMMSE), massive 1 591× → **1 204×**.
L'énergie INT8 *augmente* avec le correctif (à Q=8, `(1/2)^1,9 = 0,27 < 0,5`)
alors que l'énergie FP32 *diminue*, d'où la compression des rapports.

Au passage, `experiments/complexity_energy_massive_d384.py` relisait
l'énergie RZF/WMMSE depuis un fichier pré-correctif au lieu de la
recalculer : le tableau massive mélangeait donc classiques pré-correctif et
neuronaux post-correctif. Corrigé dans ce dépôt (les classiques sont
recalculés avec le même modèle que les architectures).

### 6.2 Checkpoints massive à 83 époques : résultats invalides, pas un plafond

`results/diag_massive_true_{single_sc,intra_rb}.json` (sans `_extbudget`)
correspondent à des modèles sous-entraînés qui s'effondrent à ~32 bps/Hz
(contre ~80 avec le budget étendu). `experiments/audit_massive_checkpoints.py`
le démontre côte à côte. Toute valeur dérivée de ces fichiers est une
erreur de checkpoint, pas une mesure de capacité. Les deux fichiers sont
conservés **uniquement** comme sujet de cet audit.

### 6.3 Écart de reproduction à l'échelle massive

`experiments/audit_massive_repro.py` réévalue les checkpoints D=128 avec un
budget Monte-Carlo plus large (20 lots × 64 au lieu de 5 × 16) et obtient
pour SC 70,9 / 78,6 / 82,9 % à 10 / 15 / 20 dB, contre **72,2 / 80,9 / 86,0 %**
publiés. L'écart (1 à 3 points) n'est pas expliqué à ce jour : il est
cohérent avec une différence de protocole d'échantillonnage, mais ce n'est
pas démontré. À vérifier avant de citer les valeurs publiées.

### 6.4 TA-RB standard : T=4 en CSI parfait, T=6 en CSI imparfait

Le tableau CSI parfait UMi std utilise le checkpoint T=4
(`TA_RB_residual_signed_attn_T4_4L_128d`), le tableau CSI imparfait utilise
T=6 (`…_6tok_…`). Les deux colonnes « TA-RB » ne désignent donc pas le même
modèle. Ce n'est pas une erreur de mesure, mais ça doit être dit dans le
texte, sinon la rétention de TA-RB (70,7 % à 20 dB) n'est pas comparable
ligne à ligne avec son débit CSI parfait (95,0 % de RZF).

Et la recommandation « T=4 » est incomplète : T=4 est le meilleur point en
CSI parfait (97,2 % de WMMSE en moyenne), mais la rétention sous CSI
imparfait **décroît de façon monotone avec T** (97,6 % à T=1 → 79,2 % à
T=12). Si le CSI est bruité, l'optimum descend vers T=2-3.

### 6.5 Provenance des figures

Les figures dont le nom contient `seedfix` ou `2560seedfix` sont construites
sur les données canoniques du 20 août. Les autres (`figA_sumrate_csi_perfect`,
`figB_pareto_energy` des régimes massive, `figD_ber_snr`) viennent encore
des campagnes antérieures : cohérentes entre elles, mais pas alignées sur
les tableaux du §4.2. À regénérer avant dépôt final. Les scripts `figB_*`
consomment par ailleurs les fichiers d'énergie **pré-correctif** (§6.1).

### 6.6 Ce qui est périmé et pourquoi

| Fichier | Statut |
|---|---|
| `archive/reports/CHIFFRES_FINAUX_CHAPITRE3.md` (9 août) | antérieur au gap-closing D=384 (12-13 août) **et** au seedfix (20 août) — malgré son nom |
| `archive/reports/SYNTHESE_RESULTATS.md` (8 août) | idem |
| `archive/exploration/complexity_energy_results_v{2,3}.csv` | « MASSIVE » = M32×K8, ancienne config ; TA-RB T=3 sans attention signée |
| `archive/exploration/` — canaux LOS forcé, M64K32, M32K8 | canal abandonné le 7 août (quasi plat en fréquence) |
| `archive/release_20260809/` | bonne structure, contenu pré-D=384 et pré-seedfix |
| `figures/{umi,uma}_massive/` (D=128) | remplacés par `*_capD384L4` |

---

## 7. Ce que ce dépôt ne contient pas

- Les poids entraînés (657 Mo, `weights/`, ignorés par git) et les caches de
  canaux (`/export/tmp/sala/*.npz`). Régénérables, cf. §5.2.
- Les checkpoints de diagnostic du gap-closing (228 Mo,
  `archive/weights_massive_gap_diag/`, ignorés).
- Aucune validation empirique de la robustesse à la quantification INT8 :
  le modèle d'énergie est analytique, la quantification réelle des poids et
  activations n'est pas simulée ici.
- Les courbes BER à l'échelle massive : le durcissement du canal pousse le
  BER sous la résolution du budget Monte-Carlo utilisé ailleurs dans ce
  dépôt (quelques centaines de mots de code par point).
