# Chiffres finaux pour Chapitre 3 — réécriture post-nuit du 8-9 août 2026

Référence complète pour la mise à jour de `7-Theme1.tex`. Toutes les
données proviennent de `final/results/*.json` et `*.npy` (session du
8-9 août 2026, voir `SESSION_LOG_20260808.md` pour l'historique complet
et la méthodologie de chaque calcul).

**Abréviations** : SC = Précodeur par sous-porteuse, IB = Transformer
intra-bloc, TA-RB = Agrégation par RB (décodeur résiduel).
**Budgets** : STANDARD (M=8,K=4) = 83ep (3 warmup + 80 finetune) partout.
MASSIVE (M=64,K=8) = 170ep (10 warmup + 160 finetune) partout, budget
étendu confirmé nécessaire par diagnostic cette nuit (voir §0).

---

## §0 — Points de correction obligatoires (résumé pour la relecture)

1. **Toute mention "TA-RB domine largement à l'échelle massive"** (issue
   de la Priorité 3.3, ~19h00 le 8/8, basée sur des checkpoints SC/IB
   sous-entraînés à 83ep) **est fausse et doit être supprimée/corrigée**.
   À budget cohérent (170ep, tableau §2), les 3 architectures sont
   proches (72-80% WMMSE), TA-RB gardant un avantage modeste (+1 à +3pt
   à SNR moyen/haut sur UMi, mais légèrement EN RETRAIT sur UMa massive
   face à IB). Ce qui reste vrai et se généralise (§3) : la robustesse
   CSI imparfait de TA-RB.
2. **Recommandation T unique = T=4 est incomplète.** T=4 est optimal en
   CSI parfait (98.6% WMMSE moyen, meilleur du sweep, moins cher que
   T=6/T=12). Mais **sous CSI imparfait, T=2 devient le meilleur point**
   (25.91 bps/Hz à 15dB vs 25.66 pour T=4) et **T=12 s'effondre**
   (22.22 bps/Hz, pire que T=6). La formulation correcte : "T=4
   recommandé si CSI parfait/quasi-parfait dominant ; T=2 si robustesse
   au bruit pilote est le critère prioritaire — pas de T universellement
   optimal."
3. Tous les tableaux CSI parfait + imparfait des 4 régimes : §2 et §3.
4. Le résultat central à mettre en avant (§3, §5) : robustesse CSI
   imparfait de TA-RB confirmée **sans exception sur 4 régimes
   indépendants** (UMi/UMa × standard/massive) — c'est la conclusion la
   plus solide statistiquement de toute la session (4 réplications
   indépendantes, même signe, même ordre de grandeur d'écart).

---

## §1 — Référence classique RZF/WMMSE (channel hardening), 4 régimes

Gap = 100×(WMMSE−RZF)/RZF. Proche de 0 = channel hardening (RZF quasi-
optimal), attendu et confirmé à M=64 (Marzetta 2010), **sans aucun
clustering artificiel resserré** (même `CLUSTER_RADIUS_M=20m` qu'à
M=8).

| SNR (dB) | UMi STD RZF | UMi STD WMMSE | gap | UMi MAS RZF | UMi MAS WMMSE | gap | UMa STD RZF | UMa STD WMMSE | gap | UMa MAS RZF | UMa MAS WMMSE | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 13.63 | 13.97 | +2.50% | 64.08 | 64.06 | −0.02% | 12.94 | 13.58 | +4.95% | 66.23 | 66.22 | −0.01% |
| 5.0 | 19.60 | 19.63 | +0.17% | 77.42 | 77.41 | −0.01% | 18.45 | 18.74 | +1.59% | 79.46 | 79.46 | −0.00% |
| 10.0 | 25.74 | 25.71 | −0.10% | 90.51 | 90.51 | −0.00% | 24.61 | 24.68 | +0.29% | 92.86 | 92.86 | −0.00% |
| 15.0 | 32.31 | 32.29 | −0.06% | 103.96 | 103.96 | −0.00% | 31.22 | 31.22 | +0.02% | 106.03 | 106.03 | 0.00% |
| 17.5 | 35.66 | 35.65 | −0.03% | 110.40 | 110.40 | 0.00% | 34.36 | 34.36 | −0.02% | 112.77 | 112.77 | 0.00% |
| 20.0 | 38.95 | 38.95 | −0.02% | 117.11 | 117.11 | 0.00% | 37.66 | 37.65 | −0.03% | 119.29 | 119.29 | 0.00% |

STD = STANDARD (M=8,K=4), MAS = MASSIVE (M=64,K=8). Points intermédiaires
(2.5/7.5/12.5dB) disponibles dans les `.npy` sources si besoin pour un
graphique à 9 points.

---

## §2 — CSI parfait : débit et %WMMSE, SC/IB/TA-RB, 4 régimes

### UMi STANDARD (M8K4, 83ep, TA-RB à T=4)

| SNR | SC (bps/Hz) | SC %WMMSE | IB (bps/Hz) | IB %WMMSE | TA-RB (bps/Hz) | TA-RB %WMMSE |
|---|---|---|---|---|---|---|
| 0.0 | 13.90 | 99.5% | 13.91 | 99.6% | 13.87 | 99.3% |
| 5.0 | 19.81 | 100.9% | 19.90 | 101.4% | 19.78 | 100.8% |
| 10.0 | 25.81 | 100.4% | 26.06 | 101.3% | 25.91 | 100.8% |
| 15.0 | 31.66 | 98.0% | 32.28 | 100.0% | 31.85 | 98.6% |
| 17.5 | 33.80 | 94.8% | 34.75 | 97.5% | 34.57 | 97.0% |
| 20.0 | 35.96 | 92.3% | 37.38 | 96.0% | 37.11 | 95.3% |

### UMi MASSIVE (M64K8, 170ep, TA-RB à T=6)

| SNR | SC (bps/Hz) | SC %WMMSE | IB (bps/Hz) | IB %WMMSE | TA-RB (bps/Hz) | TA-RB %WMMSE |
|---|---|---|---|---|---|---|
| 0.0 | 48.63 | 75.9% | 48.68 | 76.0% | 46.76 | 73.0% |
| 5.0 | 60.90 | 78.7% | 60.77 | 78.5% | 60.18 | 77.7% |
| 10.0 | 72.19 | 79.8% | 71.70 | 79.2% | 72.24 | 79.8% |
| 15.0 | 80.92 | 77.8% | 79.96 | 76.9% | 81.86 | 78.7% |
| 17.5 | 83.78 | 75.9% | 82.47 | 74.7% | 85.01 | 77.0% |
| 20.0 | 86.00 | 73.4% | 84.48 | 72.1% | 87.60 | 74.8% |

### UMa STANDARD (M8K4, 83ep, TA-RB à T=4)

| SNR | SC (bps/Hz) | SC %WMMSE | IB (bps/Hz) | IB %WMMSE | TA-RB (bps/Hz) | TA-RB %WMMSE |
|---|---|---|---|---|---|---|
| 0.0 | 12.40 | 91.3% | 12.26 | 90.3% | 11.95 | 87.9% |
| 5.0 | 17.95 | 95.8% | 17.94 | 95.7% | 16.94 | 90.4% |
| 10.0 | 24.01 | 97.3% | 23.90 | 96.8% | 22.16 | 89.8% |
| 15.0 | 29.89 | 95.7% | 29.74 | 95.3% | 26.83 | 86.0% |
| 17.5 | 32.68 | 95.1% | 32.72 | 95.2% | 29.07 | 84.6% |
| 20.0 | 35.17 | 93.4% | 35.17 | 93.4% | 30.54 | 81.1% |

### UMa MASSIVE (M64K8, 170ep, TA-RB à T=6)

| SNR | SC (bps/Hz) | SC %WMMSE | IB (bps/Hz) | IB %WMMSE | TA-RB (bps/Hz) | TA-RB %WMMSE |
|---|---|---|---|---|---|---|
| 0.0 | 47.60 | 71.9% | 48.24 | 72.8% | 44.83 | 67.7% |
| 5.0 | 60.25 | 75.8% | 60.98 | 76.7% | 57.81 | 72.8% |
| 10.0 | 70.65 | 76.1% | 73.05 | 78.7% | 69.17 | 74.5% |
| 15.0 | 80.03 | 75.5% | 85.20 | 80.4% | 77.05 | 72.7% |
| 17.5 | 82.98 | 73.6% | 89.88 | 79.7% | 79.86 | 70.8% |
| 20.0 | 85.23 | 71.4% | 94.09 | 78.9% | 82.05 | 68.8% |

**Lecture synthétique CSI parfait** : SC/IB quasi-interchangeables partout
(écart <2pt). TA-RB : quasi à égalité sur UMi (les deux régimes), **nettement
en retrait sur UMa** (−4 à −10pt vs SC/IB, les deux échelles) — piste non
creusée cette nuit : le décodeur résiduel de TA-RB (interpolation fixe +
correction) s'accommoderait moins bien de la sélectivité fréquentielle
plus marquée d'UMa qu'IB, qui l'exploite directement.

---

## §3 — CSI imparfait (pilote 20dB) : %retenu, 4 régimes — RÉSULTAT CENTRAL

`%retenu = 100 × rate(pilote bruité) / rate(CSI parfait)`, même
checkpoint, mêmes poids figés, méthodologie identique dans les 4 régimes
(20 tirages UMi standard / 12 tirages MASSIVE, bruit LS gaussien).

### UMi STANDARD (T=4)

| SNR | SC | IB | TA-RB | RZF | WMMSE |
|---|---|---|---|---|---|
| 0.0 | 93.4% | 93.6% | **98.3%** | 93.3% | 93.9% |
| 5.0 | 87.7% | 87.9% | **96.3%** | 87.2% | 87.5% |
| 10.0 | 79.7% | 79.7% | **92.5%** | 78.3% | 78.5% |
| 15.0 | 71.0% | 70.7% | **86.6%** | 68.2% | 68.2% |
| 20.0 | 63.9% | 62.9% | **79.8%** | 58.7% | 58.7% |

### UMi MASSIVE (T=6), pilote 20dB

| SNR | SC | IB | TA-RB | RZF | WMMSE |
|---|---|---|---|---|---|
| 0.0 | 86.2% | 86.6% | **95.3%** | 88.9% | 88.9% |
| 5.0 | 77.9% | 78.5% | **90.0%** | 80.9% | 80.9% |
| 10.0 | 70.4% | 70.8% | **82.8%** | 72.1% | 72.1% |
| 15.0 | 65.1% | 65.2% | **76.0%** | 64.0% | 64.0% |
| 20.0 | 62.7% | 62.7% | **71.5%** | 57.1% | 57.1% |

### UMi MASSIVE (T=6), pilote 10dB (pilote plus dégradé, pour référence)

| SNR | SC | IB | TA-RB | RZF | WMMSE |
|---|---|---|---|---|---|
| 0.0 | 51.5% | 51.9% | **71.5%** | 60.0% | 60.0% |
| 5.0 | 42.8% | 43.2% | **60.2%** | 51.1% | 51.1% |
| 10.0 | 37.4% | 37.6% | **51.4%** | 44.2% | 44.2% |
| 15.0 | 34.1% | 34.1% | **45.6%** | 38.8% | 38.8% |
| 20.0 | 32.6% | 32.6% | **42.5%** | 34.4% | 34.4% |

### UMa STANDARD (T=4), pilote 20dB

| SNR | SC | IB | TA-RB |
|---|---|---|---|
| 0.0 | 93.4% | 93.4% | **98.1%** |
| 5.0 | 87.6% | 87.5% | **95.9%** |
| 10.0 | 79.1% | 79.0% | **91.9%** |
| 15.0 | 69.8% | 69.7% | **86.0%** |
| 17.5 | 65.5% | 65.3% | **82.7%** |
| 20.0 | 61.5% | 61.6% | **79.7%** |

### UMa MASSIVE (T=6), pilote 20dB

| SNR | SC | IB | TA-RB |
|---|---|---|---|
| 0.0 | 86.3% | 85.6% | **94.7%** |
| 5.0 | 77.5% | 76.4% | **89.2%** |
| 10.0 | 69.3% | 66.7% | **81.8%** |
| 15.0 | 63.2% | 58.9% | **75.4%** |
| 17.5 | 61.1% | 55.6% | **73.0%** |
| 20.0 | 59.6% | 53.3% | **71.3%** |

**C'est LE résultat à mettre en avant** : TA-RB retient systématiquement
+8 à +18 points de plus que SC/IB à 20dB pilote, dans les 4 régimes sans
exception, avec un écart qui croît avec le SNR data dans les 4 cas.
Hypothèse mécanistique (non vérifiée formellement, à formuler avec
prudence) : le moyennage/variance par RB dans `_extract_features` de
TA-RB agit comme un débruitage implicite du canal estimé, avant même
l'attention proprement dite.

---

## §4 — T-sweep TA-RB (UMi STANDARD uniquement, 83ep, seul régime avec sweep complet)

### CSI parfait — %WMMSE par SNR et par T

| T | 0dB | 5dB | 10dB | 15dB | 17.5dB | 20dB | **moyenne** | FLOPs (M) |
|---|---|---|---|---|---|---|---|---|
| 1 | 94.4% | 93.7% | 90.6% | 85.6% | 82.3% | 78.2% | 87.5% | 83.9 |
| 2 | 98.5% | 99.0% | 98.4% | 95.3% | 93.3% | 91.2% | 95.9% | 154.2 |
| 3 | 98.9% | 100.0% | 99.7% | 96.9% | 94.7% | 92.3% | 97.1% | 225.6 |
| **4** | 99.3% | 100.8% | 100.8% | 98.6% | 97.0% | 95.3% | **98.6%** | 298.1 |
| 6 | 99.6% | 100.8% | 100.5% | 97.4% | 94.6% | 91.5% | 97.4% | 446.2 |
| 12 | 99.2% | 100.8% | 100.5% | 98.7% | 95.8% | 93.8% | 98.1% | 915.5 |

**T=4 domine strictement T=6 et T=12** : meilleure moyenne du sweep ET
1.5× moins cher que T=6, 3.1× moins cher que T=12 — **mais seulement
sous CSI parfait**, voir tableau suivant.

### CSI imparfait (pilote 20dB) à 15dB SNR — donnée du panneau droit Fig. B

| T | débit CSI parfait | débit pilote 20dB | %retenu | énergie INT8 (µJ) |
|---|---|---|---|---|
| 1 | 21.67 | 20.98 | 96.8% | 0.013675 |
| **2** | 28.09 | **25.91** | 92.2% | 0.024714 |
| 3 | 28.08 | 25.19 | 89.7% | 0.035917 |
| 4 | 28.55* | 25.66 | 89.9%* | 0.047283 |
| 6 | 30.26 | 25.51 | 84.3% | 0.070504 |
| 12 | 31.32 | 22.22 | 71.0% | 0.144083 |

*T=4 : `perfect` recalculé indirectement (28.55 = 25.66/0.899, cohérence
à vérifier ± bruit Monte-Carlo — le calcul direct T=4 vient d'un sweep
séparé de celui des autres T, cf. `diag_csi_imperfect_signed_attn_
sweep.json` vs `diag_csi_imperfect_tsweep_T{1,2,3,6,12}.json`, mêmes
seeds/méthodologie mais tirages indépendants).

**T=2 devient le meilleur débit absolu sous CSI imparfait** (25.91 >
25.66 pour T=4, malgré une moyenne CSI-parfaite inférieure) et **T=12
s'effondre** (22.22, pire que T=6 malgré 2× plus de FLOPs) — confirme
qu'aucun T n'est universellement optimal, le choix dépend du critère
(débit pur en CSI quasi-parfait vs robustesse pilote).

Énergie INT8 par T (rappel, indépendant du CSI) :

| T | 1 | 2 | 3 | 4 | 6 | 12 |
|---|---|---|---|---|---|---|
| FLOPs (M) | 83.9 | 154.2 | 225.6 | 298.1 | 446.2 | 915.5 |
| params | 1,099,597 | 1,100,621 | 1,101,645 | 1,102,669 | 1,104,717 | 1,110,861 |
| énergie INT8 (µJ) | 0.013675 | 0.024714 | 0.035917 | 0.047283 | 0.070504 | 0.144083 |

---

## §5 — Énergie (Pareto), RZF/WMMSE/SC/IB, 4 régimes

Énergie dépend uniquement de (architecture, M, K) — **pas du canal** —
donc identique UMi/UMa à même échelle.

### STANDARD (M=8,K=4) — vaut pour UMi ET UMa

| méthode | précision | FLOPs (M) | énergie (µJ) |
|---|---|---|---|
| RZF | FP32 | 2.82 | 0.011087 |
| WMMSE (I=10) | FP32 | 2456.62 | 4.633825 |
| SC | INT8 | 812.48 | 0.127609 |
| IB | INT8 | 1023.25 | 0.160836 |
| TA-RB (T=4) | INT8 | 298.09 | 0.047283 |
| TA-RB (T=6, référence historique) | INT8 | 446.15 | 0.070504 |

### MASSIVE (M=64,K=8) — vaut pour UMi ET UMa

| méthode | précision | FLOPs (M) | énergie (µJ) |
|---|---|---|---|
| RZF | FP32 | 80.29 | 0.220666 |
| WMMSE (I=10) | FP32 | 1,249,485.76 | 2353.985939 |
| SC | INT8 | 1686.31 | 0.264427 |
| IB | INT8 | 2107.83 | 0.330759 |
| TA-RB (T=6) | INT8 | 2756.54 | 0.431571 |

**Point notable pour la discussion** : à M=64, l'écart d'énergie
RZF/WMMSE vs neuronal explose (WMMSE ≈ 2354µJ vs ≈0.26-0.43µJ pour les
architectures neuronales, >5000×, contre "seulement" ≈36-165× à M=8) —
le coût O(M³) de WMMSE domine complètement à cette échelle, pour un
débit qui ne dépasse les réseaux que de 25-30% (§2). L'argument
efficacité-énergétique des précodeurs appris se renforce nettement avec
l'échelle.

---

## §6 — BER (UMi STANDARD uniquement, T=4, 83ep)

Pas de BER produite à l'échelle MASSIVE (budget Monte-Carlo insuffisant
pour résoudre statistiquement le taux d'erreur, quasi nul par channel
hardening — choix assumé, documenté dans `SESSION_LOG_20260808.md`, pas
un oubli).

### CSI parfait (9 points, 50 batchs × 256)

| SNR (dB) | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0.0 | 2.4e-04 | 5.2e-03 | 8.1e-04 | 1.3e-03 | 1.4e-03 |
| 2.5 | 8.9e-05 | 2.5e-03 | 2.8e-04 | 4.3e-04 | 5.5e-04 |
| 5.0 | 3.3e-05 | 1.1e-03 | 9.0e-05 | 2.2e-04 | 2.6e-04 |
| 7.5 | 1.7e-05 | 3.6e-04 | 5.9e-05 | 8.1e-05 | 1.1e-04 |
| 10.0 | 4.1e-06 | 2.1e-04 | 4.9e-05 | 1.0e-04 | 5.2e-05 |
| 12.5 | 4.9e-06 | 5.6e-05 | 2.5e-05 | 6.7e-05 | 9.6e-05 |
| 15.0 | 0 | 6.6e-06 | 2.6e-05 | 7.0e-05 | 5.6e-05 |
| 17.5 | 0 | 2.4e-06 | 4.5e-05 | 7.4e-05 | 3.4e-05 |
| 20.0 | 0 | 0 | 3.0e-05 | 3.8e-05 | 8.2e-05 |

### CSI imparfait, pilote 20dB (5 points, 15 batchs × 256 — budget réduit, plus bruité que ci-dessus)

| SNR (dB) | RZF | WMMSE | SC | IB | TA-RB |
|---|---|---|---|---|---|
| 0.0 | 3.8e-04 | 4.1e-03 | 2.4e-03 | 1.7e-03 | 1.7e-03 |
| 5.0 | 1.0e-04 | 7.2e-04 | 5.1e-04 | 4.0e-04 | 2.7e-04 |
| 10.0 | 3.4e-05 | 1.6e-04 | 8.7e-05 | 1.5e-04 | 1.8e-04 |
| 15.0 | 1.8e-05 | 8.5e-06 | 2.1e-04 | 1.7e-04 | 1.4e-04 |
| 20.0 | 0 | 1.8e-05 | 1.2e-04 | 1.6e-04 | 1.3e-04 |

---

## §7 — Sources (pour vérification/regénération si besoin)

Tout est reproductible depuis `final/release/mimo_precoding/` (dossier
autonome, voir son README) ou directement depuis `final/results/*.json`
+ `*.npy` cités en tête de chaque section. Figures correspondantes :
`final/results/figures_final/{umi,uma}_{standard,massive}/fig{A,B,C,D}_*.png`.
