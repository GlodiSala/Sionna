# Rapport — Fermeture du plafond MASSIVE sur UMa (M=64,K=8), nuit du 12 au 13/08

**Statut : SUCCÈS.** Réplique directement le chantier UMi du 12/08
(`RAPPORT_MASSIVE_GAP_CLOSING_20260812.md`) sur le canal UMa. Même
levier (D=128→D=384, L=4 inchangé, budget 170ep inchangé), même
ampleur de gain : le plafond CSI-parfait UMa passe de 68-81% de RZF à
74-93% selon SNR/architecture. L'asymétrie UMi(D=384)/UMa(D=128)
documentée dans le mémoire peut être retirée.

Portée respectée : UMa MASSIVE (M=64,K=8) uniquement. UMi MASSIVE et
STANDARD (M=8) non touchés. Checkpoints UMa `UMa_MASSIVE_<Arch>_
signed_attn_.../` (D=128, production) intacts — aucun écrasé.

---

## 1. Méthode

### 1.1 Diagnostic à coût réduit — pas refait, décision justifiée

Conformément au plan, le diagnostic 4-pistes (baseline/curriculum/
profondeur/largeur) n'a **pas été refait** pour UMa : il avait déjà
établi sans ambiguïté sur UMi que le goulot est `feat_dim=193` (3M+1
à M=64) comprimé dans un espace à 128 dimensions — un phénomène
purement dimensionnel, indépendant du canal (le nombre d'antennes/
utilisateurs et la formule feat_dim ne changent pas entre UMi et UMa).

**Aucun signal contraire trouvé en cours de run** justifiant de
reconsidérer ce choix : le run complet direct à D=384,L=4 a convergé
normalement sur les 3 architectures (voir §2), avec des temps
d'entraînement quasi identiques à ceux d'UMi (53.7/78.6/99.8min UMa
vs 54.1/78.3/86.0min UMi) — pas d'anomalie de convergence qui aurait
suggéré un comportement différent sur ce canal.

### 1.2 Référence D=128 existante (canal UMa)

Déjà en place, entraînée en amont (nuit du 8-9/08) avec budget étendu
d'emblée (170ep, pas de run 83ep intermédiaire — confirmé par le
docstring de `diag_uma_massive_train.py` : *"budget étendu D'EMBLÉE
[...] pas de run 83ep intermédiaire, cf. diagnostic budget MASSIVE
déjà tranché"*). Fichiers : `results/diag_uma_massive_{single_sc,
intra_rb,ta_rb_residual}.json`, ckpts `./weights/UMa_MASSIVE_<Arch>_
signed_attn_4L_128d[_6tok]/best_202608{08,09}_...`.

Recalcul du plafond ancien (%RZF, source `results/classical_
comparison_M64K8_true_uma.npy`, gap RZF-WMMSE <0.01% sur toute la
plage SNR — durcissement du canal confirmé aussi côté UMa) :
single_sc 71.4-76.1%, intra_rb 72.8-80.4%, ta_rb_residual 67.7-74.5%
— même ordre de grandeur que l'ancien plafond UMi (73-80%).

### 1.3 Runs complets (D=384,L=4, 170ep = 10 warmup + 160 finetune)

Script neuf : `diag_massive_gap_fullbudget_uma.py` (réplique de
`diag_massive_gap_fullbudget.py`, seules différences : `scenario='uma'`
passé à `CachedSionnaDataset`, cache `sionna_joint_uma_massive_
4k_64x8.npz` déjà généré et réutilisé tel quel, `RUN_NAMES` préfixés
`UMa_MASSIVE_...` pour cohérence avec le nommage D=128 existant).

Lancés en parallèle sur les 3 GPU disponibles (22:19, nuit du 12/08).
Poids : `./weights/UMa_MASSIVE_<Arch>_..._capD384L4/` — **dossiers
distincts** des checkpoints D=128 de production, intacts.

---

## 2. Résultats — CSI parfait, débit somme (bps/Hz), % de RZF

Source RZF/WMMSE : `results/classical_comparison_M64K8_true_uma.npy`
(inchangé, indépendant de D).

### single_sc
Ancien : `results/diag_uma_massive_single_sc.json`
(ckpt `./weights/UMa_MASSIVE_SingleSC_signed_attn_4L_128d/best_20260808_234158`)
Nouveau : `results/diag_massive_gap_fullbudget_uma_single_sc_cap_d384l4.json`
(ckpt `./weights/UMa_MASSIVE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_231346`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 71.9% | **92.5%** | +20.7pt |
| 5dB | 75.8% | **92.4%** | +16.6pt |
| 10dB | 76.1% | **90.5%** | +14.5pt |
| 15dB | 75.5% | **87.4%** | +11.9pt |
| 17.5dB | 73.6% | **84.0%** | +10.4pt |
| 20dB | 71.4% | **81.8%** | +10.3pt |

Train : 53.7min (170ep). Params 1.10M→9.59M (×8.7). FLOPs 1.69G→14.72G (×8.7).

### intra_rb
Ancien : `results/diag_uma_massive_intra_rb.json`
(ckpt `./weights/UMa_MASSIVE_IntraRB_signed_attn_4L_128d/best_20260809_005701`)
Nouveau : `results/diag_massive_gap_fullbudget_uma_intra_rb_cap_d384l4.json`
(ckpt `./weights/UMa_MASSIVE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_233842`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 72.8% | **92.4%** | +19.6pt |
| 5dB | 76.7% | **92.3%** | +15.5pt |
| 10dB | 78.7% | **90.5%** | +11.8pt |
| 15dB | 80.4% | **87.4%** | +7.1pt |
| 17.5dB | 79.7% | **84.0%** | +4.3pt |
| 20dB | 78.9% | **81.8%** | +2.9pt |

Train : 78.6min. Params 1.37M→11.96M (×8.7). FLOPs 2.11G→18.40G (×8.7).

**Note** : le gain d'intra_rb rétrécit nettement en haute SNR (+2.9pt
seulement à 20dB) — l'ancien checkpoint D=128 était déjà relativement
performant à haute SNR sur UMa (78.9% à 20dB, le meilleur des 3
anciens checkpoints à ce point). Toujours un gain net, mais moins
spectaculaire que single_sc/ta_rb_residual à cette extrémité de la
plage SNR.

### ta_rb_residual (T=6)
Ancien : `results/diag_uma_massive_ta_rb_residual.json`
(ckpt `./weights/UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d/best_20260809_022421`)
Nouveau : `results/diag_massive_gap_fullbudget_uma_ta_rb_residual_cap_d384l4.json`
(ckpt `./weights/UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_235531`)

| SNR | ancien (%RZF) | nouveau (%RZF) | gain |
|---|---|---|---|
| 0dB | 67.7% | **87.6%** | +19.9pt |
| 5dB | 72.8% | **88.1%** | +15.3pt |
| 10dB | 74.5% | **85.1%** | +10.6pt |
| 15dB | 72.7% | **80.6%** | +7.9pt |
| 17.5dB | 70.8% | **77.7%** | +6.9pt |
| 20dB | 68.8% | **74.0%** | +5.2pt |

Train : 99.8min. Params 2.33M→10.97M (×4.7). FLOPs 2.76G→9.45G (×3.4
— TA-RB reste l'architecture la moins coûteuse en FLOPs des 3, comme
sur UMi).

**Observation propre à UMa (à signaler, pas lissée)** : sous CSI
parfait, TA-RB est ici l'architecture la PLUS FAIBLE des 3 (74.0% à
20dB, contre 81.8%/81.8% pour single_sc/intra_rb) — un écart plus net
que sur UMi, où les 3 architectures étaient plus proches (80.2-82.1%
à 20dB). Voir figure A (panneau gauche) : TA-RB est visiblement
en-dessous de SC/IB sous UMa MASSIVE.

**Lecture d'ensemble** : gain fort et cohérent à basse/moyenne SNR
(0-10dB : +10.6 à +20.7pt sur les 3 architectures), qui se resserre en
haute SNR de façon plus marquée que sur UMi (surtout pour intra_rb).
Reste un gain net partout — jamais de régression.

---

## 3. Résultats — CSI imparfait (relancé avec les 3 nouveaux checkpoints)

Script neuf : `diag_csi_imperfect_massive_uma_capD384L4.py` (réplique
de `diag_csi_imperfect_massive_capD384L4.py`, seule différence :
`channel_model = UMa(...)` au lieu de `UMi(...)` par défaut).
Résultat : `results/diag_csi_imperfect_massive_uma_capD384L4.json`
Méthode identique : bruit LS gaussien sur canal estimé, canal vrai
pour SINR/détection, RZF/WMMSE recalculés sur les mêmes tirages
(12 tirages, pilote {10,20}dB).

**Le résultat central du mémoire (robustesse CSI-imparfait de TA-RB)
tient sur UMa aussi, avec une marge comparable voire supérieure à
UMi** :

| pilote | SNR | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|---|
| 20dB | 0 | 88.7% | 88.6% | 88.8% | 88.8% | **95.7%** |
| 20dB | 10 | 71.7% | 71.7% | 73.9% | 73.8% | **85.5%** |
| 20dB | 20 | 56.7% | 56.7% | 65.5% | 65.4% | **77.8%** |
| 10dB | 0 | 58.9% | 58.9% | 59.6% | 59.6% | **76.0%** |
| 10dB | 10 | 43.3% | 43.3% | 45.0% | 44.9% | **59.0%** |
| 10dB | 20 | 33.6% | 33.6% | 39.3% | 39.1% | **51.9%** |

TA-RB dépasse RZF/WMMSE de **+7 à +24pt** de rétention selon
SNR/pilote. Point notable : **sous CSI parfait TA-RB était
l'architecture la plus faible des 3 sur UMa (§2)** — mais sous CSI
imparfait, elle redevient nettement la meilleure, confirmant que son
avantage est spécifiquement lié à la robustesse au bruit d'estimation
de canal, pas à une capacité brute supérieure. Ce contraste est en
fait plus net sur UMa que sur UMi, et renforce l'argument central du
mémoire plutôt que de l'affaiblir.

---

## 4. Énergie / complexité

**Aucun nouveau calcul effectué — réutilisation directe du fichier
UMi** (`results/complexity_energy_massive_capD384L4.json`), décision
justifiée : les FLOPs/params/énergie d'une architecture ne dépendent
que de (M,K,D,L) — confirmé par relecture de `compute_complexity_
energy_massive_capD384L4.py`, qui ne référence à aucun moment le
scénario de canal dans son calcul de complexité (le graphe de calcul
du précodeur est identique quel que soit UMi/UMa). RZF/WMMSE FLOPs
dépendent aussi seulement de (M,K,N_SC,N_OFDM,I_wmmse), identiques
ici. Seul le débit à 15dB (axe Y de la figure Pareto) diffère,
utilisant les données UMa fraîches.

| Méthode | params | FLOPs (M) | Énergie (µJ, INT8) |
|---|---|---|---|
| RZF | — | 80.3 | 0.221 (FP32) |
| WMMSE (I=10) | — | 1,249,486 | 2353.99 (FP32) |
| SingleSC (D=384) | 9,587,848 | 14,722.6 | 2.299 |
| IntraRB (D=384) | 11,960,972 | 18,403.1 | 2.875 |
| TA_RB_residual (D=384) | 10,968,845 | 9,452.4 | 1.479 |

Cross-check numérique : les params réels du checkpoint UMa D=384
(`diag_massive_gap_fullbudget_uma_*.json`, §2) correspondent
EXACTEMENT à ces valeurs (ex. single_sc 9,587,848 des deux côtés) —
confirme que la réutilisation est valide, pas une supposition.

---

## 5. Figures régénérées

Nouveau dossier `results/figures_final/uma_massive_capD384L4/`
(la référence `uma_massive/` en D=128 reste **intacte**) :
- `figA_sumrate_csi_perfect.{png,pdf,json}` — double panneau CSI
  parfait / CSI imparfait pilote 20dB
- `figB_pareto_energy.{png,pdf,json}` — Pareto débit(15dB)/énergie

---

## 6. Ce qui n'a PAS été refait (hors périmètre explicite)

- **Diagnostic à coût réduit (4 pistes)** : pas refait, cf. §1.1 —
  décision justifiée (levier déjà établi comme indépendant du canal
  sur UMi, aucun signal contraire observé pendant le run direct).
- **Énergie/complexité** : pas recalculée séparément, cf. §4 —
  réutilisation directe justifiée (grandeur channel-independent).
- **Validation système (scheduler)** : comme pour UMi, cette
  évaluation n'a jamais existé à l'échelle MASSIVE (scopée UMi
  STANDARD uniquement) — rien à "refaire" pour UMa non plus.
- **BER UMa MASSIVE** : aucun script `diag_ber_massive_uma.py`
  trouvé dans le dépôt — contrairement à UMi, il n'existe même pas
  de version D=128 obsolète à signaler ; cette figure n'a simplement
  jamais été produite pour UMa (ni avant, ni maintenant).

---

## 7. Fichiers — récapitulatif des chemins exacts

**Scripts créés** (`final/`) :
`diag_massive_gap_fullbudget_uma.py`, `diag_csi_imperfect_massive_uma_capD384L4.py`

**Poids nouveaux** (`final/weights/`, dossiers distincts des
checkpoints D=128 de production, intacts) :
- `UMa_MASSIVE_SingleSC_signed_attn_4L_128d_capD384L4/best_20260812_231346`
- `UMa_MASSIVE_IntraRB_signed_attn_4L_128d_capD384L4/best_20260812_233842`
- `UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d_capD384L4/best_20260812_235531`

**Résultats** (`final/results/`) :
- `diag_massive_gap_fullbudget_uma_{single_sc,intra_rb,ta_rb_residual}_cap_d384l4.json` (runs complets)
- `diag_csi_imperfect_massive_uma_capD384L4.json`
- `complexity_energy_massive_capD384L4.json` (réutilisé de UMi, cf. §4)
- `figures_final/uma_massive_capD384L4/` (nouvelles figures)

**Log détaillé avec horodatage** : `SESSION_LOG_20260808.md`
(entrées 22:16 du 12/08 → 07:43 du 13/08, section "Réplication du
gap-closing sur UMa MASSIVE").
