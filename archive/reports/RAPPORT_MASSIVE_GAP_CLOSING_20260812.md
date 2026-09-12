# Rapport — Fermeture du plafond MASSIVE (M=64,K=8), nuit du 11 au 12/08

**Statut : SUCCÈS.** Le plafond de 73-80% de RZF sous CSI parfait est
fermé à 82-99% selon SNR/architecture, sur les 3 architectures, à
budget d'entraînement identique (170ep). Effet réel et stable, pas
quelques points grappillés.

Portée respectée : UMi MASSIVE (M=64,K=8) uniquement. STANDARD (M=8)
et UMa non touchés. Checkpoints `*_extbudget` (production, D=128)
intacts — voir chemins ci-dessous, aucun n'a été écrasé.

---

## 1. Méthode

### 1.1 Diagnostic à budget réduit (avant de committer un budget complet)

Comparaison à coût égal (20 épisodes : 2 warmup + 18 finetune, au lieu
de 170) de 4 pistes sur `single_sc` :

| Piste | Description |
|---|---|
| `baseline` | recette actuelle (D=128,L=4), init aléatoire |
| `curriculum` | mêmes D/L, blocs transformer transplantés depuis le checkpoint STANDARD (M=8) déjà convergé — seuls `input_embed`/`output_proj` (dépendants de M) réinitialisés |
| `cap_d128l8` | profondeur doublée (L=8) |
| `cap_d256l4`, `cap_d384l4`, `cap_d512l4` | largeur (embed_dim) augmentée |

**Résultat** (% de RZF moyen sur 6 points SNR, même coût ~5.5-9.7min) :

| Variant | D | L | %RZF | Δ vs baseline |
|---|---|---|---|---|
| baseline | 128 | 4 | 32.1% | — |
| curriculum | 128 | 4 | 32.3% | +0.2pt (= bruit, aucun effet) |
| cap_d128l8 | 128 | 8 | 34.9% | +2.8pt (marginal) |
| cap_d256l4 | 256 | 4 | 53.8% | +21.7pt |
| cap_d384l4 | 384 | 4 | 66.0% | +33.9pt |
| cap_d512l4 | 512 | 4 | 68.0% | +35.9pt (rendements décroissants, coût compute en hausse) |

**Diagnostic** : le goulot est la LARGEUR du réseau, pas la profondeur
ni l'initialisation. À D=128, `feat_dim=193` (3M+1 à M=64) est
fortement compressé dans un espace à 128 dimensions — capacité
insuffisante pour représenter l'information nécessaire à cette
échelle. Le curriculum learning (transplant M=8→M=64) n'aide pas : les
blocs pré-entraînés à M=8 n'encodent pas un a priori utile pour un
problème d'annulation d'interférence à 64 antennes, qualitativement
différent.

Confirmation croisée-architecture (D=256 vs baseline, 20ep) :
`intra_rb` +22.9pt, `ta_rb_residual` +25.9pt — effet générique, pas
spécifique à une architecture.

**D=384,L=4 retenu** : rendements décroissants nets après ce point
(+12pt de 256→384 contre seulement +2pt de 384→512, pour un coût
compute qui continue de croître).

### 1.2 Runs complets (D=384,L=4, 170ep = 10 warmup + 160 finetune)

Lancés en parallèle sur les 3 GPU disponibles.

Scripts : `diag_massive_gap_diagnostic.py` / `diag_massive_gap_sweep.py`
(diagnostic), `diag_massive_gap_fullbudget.py` (run complet).
Poids : `./weights/MASSIVE_TRUE_<Arch>_..._capD384L4/` — **dossiers
distincts** des checkpoints de production `*_extbudget` (D=128,
intacts).

---

## 2. Résultats — CSI parfait, débit somme (bps/Hz), % de RZF

Source RZF/WMMSE : `results/classical_comparison_M64K8_true.npy`
(inchangé, indépendant de D).

### single_sc
Ancien : `results/diag_massive_true_single_sc_extbudget.json`
(ckpt `./weights/MASSIVE_TRUE_SingleSC_signed_attn_4L_128d_extbudget/best_20260808_195627`)
Nouveau : `results/diag_massive_gap_fullbudget_single_sc_cap_d384l4.json`
(ckpt `./weights/MASSIVE_TRUE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_014548`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 75.9% | **98.6%** | +22.7pt |
| 5dB | 78.7% | **97.2%** | +18.5pt |
| 10dB | 79.8% | **94.4%** | +14.6pt |
| 15dB | 77.8% | **89.1%** | +11.3pt |
| 17.5dB | 75.9% | **85.8%** | +9.9pt |
| 20dB | 73.4% | **82.1%** | +8.7pt |

Train : 54.1min (170ep). Params 1.10M→9.59M (×8.7). FLOPs 1.69G→14.72G (×8.7).

### intra_rb
Ancien : `results/diag_massive_true_intra_rb_extbudget.json`
(ckpt `./weights/MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_extbudget/best_20260808_204513`)
Nouveau : `results/diag_massive_gap_fullbudget_intra_rb_cap_d384l4.json`
(ckpt `./weights/MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_021303`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 76.0% | **98.0%** | +22.0pt |
| 5dB | 78.5% | **96.2%** | +17.7pt |
| 10dB | 79.2% | **93.1%** | +13.9pt |
| 15dB | 76.9% | **87.3%** | +10.4pt |
| 17.5dB | 74.7% | **84.0%** | +9.3pt |
| 20dB | 72.1% | **80.2%** | +8.1pt |

Train : 78.3min. Params 1.37M→11.96M (×8.7). FLOPs 2.11G→18.40G (×8.7).

### ta_rb_residual (T=6)
Ancien : `results/diag_massive_true_ta_rb_residual_extbudget.json`
(ckpt `./weights/MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d_extbudget/best_20260808_221356`)
Nouveau : `results/diag_massive_gap_fullbudget_ta_rb_residual_cap_d384l4.json`
(ckpt `./weights/MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_022738`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 73.0% | **95.3%** | +22.3pt |
| 5dB | 77.7% | **95.2%** | +17.5pt |
| 10dB | 79.8% | **93.1%** | +13.3pt |
| 15dB | 78.7% | **88.0%** | +9.3pt |
| 17.5dB | 77.0% | **84.7%** | +7.7pt |
| 20dB | 74.8% | **81.2%** | +6.4pt |

Train : 86.0min. Params 2.33M→10.97M (×4.7). FLOPs 2.76G→9.45G (×3.4
— TA-RB reste l'architecture la moins coûteuse en FLOPs des 3 même à
D=384, décodeur résiduel plus compact).

**Lecture** : quasi-RZF (95-99%) à 0-10dB, écart résiduel significatif
seulement en haute SNR (18-19pt à 82% vs 100% à 20dB) — reste un vrai
gain net partout, pas juste "grappiller quelques points".

---

## 3. Résultats — CSI imparfait (relancé avec les 3 nouveaux checkpoints)

Script : `diag_csi_imperfect_massive_capD384L4.py`
Résultat : `results/diag_csi_imperfect_massive_capD384L4.json`
Méthode identique à la référence production (`diag_csi_imperfect_
massive_extbudget_sweep.py`) : bruit LS gaussien sur le canal estimé,
canal vrai pour SINR/détection, 12 tirages, pilote {10,20}dB.

**Le résultat central de la thèse (robustesse CSI-imparfait de TA-RB)
tient toujours, et de façon encore plus nette** :

| pilote | SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 20dB | 0 | 88.5% | 88.4% | 89.5% | 89.5% | **96.4%** |
| 20dB | 10 | 71.4% | 71.4% | 76.5% | 76.8% | **88.0%** |
| 20dB | 20 | 56.3% | 56.3% | 70.3% | 71.0% | **82.1%** |
| 10dB | 0 | 58.7% | 58.7% | 60.7% | 60.7% | **77.1%** |
| 10dB | 10 | 43.0% | 43.0% | 47.4% | 47.5% | **61.3%** |
| 10dB | 20 | 33.4% | 33.4% | 43.0% | 43.3% | **55.5%** |

TA-RB dépasse RZF/WMMSE de **+8 à +26pt** de rétention selon SNR/pilote
— marge similaire ou supérieure à ce qui était observé avec les
anciens checkpoints D=128. À pilote=20dB haute SNR, TA-RB dépasse même
WMMSE en valeur absolue (débit somme), pas seulement en % retenu (voir
figure A, panneau droit).

---

## 4. Énergie / complexité

Script : `compute_complexity_energy_massive_capD384L4.py` (CPU, même
méthodologie que la référence : RZF/WMMSE en FP32 exact, les 3
architectures neuronales en INT8, formule `compute_energy_uJ`
inchangée).
Résultat : `results/complexity_energy_massive_capD384L4.json`

| Méthode | params | FLOPs (M) | Énergie (µJ, INT8) |
|---|---|---|---|
| RZF | — | 80.3 | 0.221 (FP32) |
| WMMSE (I=10) | — | 1,249,486 | 2353.99 (FP32) |
| SingleSC (D=384) | 9,587,848 | 14,722.6 | 2.299 |
| IntraRB (D=384) | 11,960,972 | 18,403.1 | 2.875 |
| TA_RB_residual (D=384) | 10,968,845 | 9,452.4 | 1.479 |

Malgré ×3.4-8.7 de FLOPs/params vs D=128, l'énergie par précodage
reste **négligeable devant WMMSE** (facteur ~800-1600×) — le compromis
débit/énergie des architectures neuronales reste extrêmement
favorable (voir figure B).

---

## 5. Figures régénérées

Nouveau dossier `results/figures_final/umi_massive_capD384L4/`
(la référence `umi_massive/` en D=128 reste **intacte** pour
comparaison) :
- `figA_sumrate_csi_perfect.{png,pdf,json}` — double panneau CSI
  parfait / CSI imparfait pilote 20dB
- `figB_pareto_energy.{png,pdf,json}` — Pareto débit(15dB)/énergie

---

## 6. Ce qui n'a PAS été refait (hors périmètre explicite)

- **Validation système (scheduler PF+OLLA+PHY abstraction)** : cette
  évaluation n'a JAMAIS existé à l'échelle MASSIVE — elle a été
  scopée UMi STANDARD uniquement (P6b, `diag_system_scheduler_eval.py`,
  confirmé par lecture du code : `M, K = STANDARD_CONFIG[...]`). Il
  n'y a donc rien à "refaire" pour MASSIVE sur ce point ; ce serait une
  extension nouvelle, pas une mise à jour. Pas tentée cette nuit
  (respecte aussi la consigne de ne pas toucher au régime STANDARD).
- **BER MASSIVE** (`diag_ber_massive.py` → `results/diag_ber_massive.json`,
  figure D) : existe, mais construite sur les anciens checkpoints
  D=128 (`RESULT_JSONS` pointe vers `*_extbudget.json`) — donc
  maintenant obsolète par rapport aux nouveaux poids D=384. Pas dans
  la liste explicite (CSI imparfait / validation système / énergie) —
  pas régénérée cette nuit, mais candidate évidente si voulue.

---

## 7. Fichiers — récapitulatif des chemins exacts

**Scripts créés** (`final/`) :
`diag_massive_gap_diagnostic.py`, `diag_massive_gap_sweep.py`,
`diag_massive_gap_fullbudget.py`, `diag_csi_imperfect_massive_capD384L4.py`,
`compute_complexity_energy_massive_capD384L4.py`

**Poids nouveaux** (`final/weights/`, dossiers distincts des
`*_extbudget` de production, intacts) :
- `MASSIVE_TRUE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_014548`
- `MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_021303`
- `MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_022738`

**Résultats** (`final/results/`) :
- `diag_massive_gap_{single_sc,intra_rb,ta_rb_residual}_{baseline,curriculum,cap_d128l8,cap_d256l4,cap_d384l4,cap_d512l4}*.json` (diagnostics)
- `diag_massive_gap_fullbudget_{single_sc,intra_rb,ta_rb_residual}_cap_d384l4.json` (runs complets)
- `diag_csi_imperfect_massive_capD384L4.json`
- `complexity_energy_massive_capD384L4.json`
- `figures_final/umi_massive_capD384L4/` (nouvelles figures)

**Log détaillé avec horodatage** : `SESSION_LOG_20260808.md`
(entrées 00:20 → 02:46 du 12/08).
