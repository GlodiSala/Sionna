# Synthèse des résultats — comparaison architectures × canaux

Document de synthèse scientifique (distinct des logs chronologiques
`SESSION_LOG_*.md`, qui détaillent le déroulement heure par heure).
Mis à jour au fil des résultats consolidés — objectif : narratif prêt à
réutiliser pour la rédaction du mémoire, avec les chiffres source associés.

Dernière mise à jour : 2026-08-08, après Front B (TA-RB résiduel UMa).

---

## Question centrale

Trois architectures neuronales (SingleSC, IntraRB, TA-RB) approchent WMMSE
sans l'atteindre, avec des complexités FLOPs/énergie 2.4-13× moindres
(cf. `complexity_energy_results_v3.csv`). Sous quelles conditions de canal
chaque architecture est-elle la meilleure, et pourquoi ?

## Résultat consolidé — le classement DÉPEND de la sélectivité fréquentielle du canal

| | UMi (rho 12sc/95sc = 2.2×, sélectivité modeste) | UMa (rho 12sc/95sc = 2.6×, sélectivité plus marquée) |
|---|---|---|
| **Meilleur CSI parfait** | TA-RB résiduel ≈ SingleSC (90-91%) | **IntraRB** (88.8%) |
| **Meilleur CSI imparfait** | **TA-RB résiduel** (87.7%/54.6% retenu) | TA-RB résiduel (seul testé, 87.2%/54.6%) |

### UMi (M8K4, R=20m) — CSI parfait, 83ep, protocole cosine_long

| Architecture | 15dB | 17.5dB | 20dB |
|---|---|---|---|
| SingleSC | 91.4% | 88.4% | 84.8% |
| IntraRB | 90.7% | 87.4% | 83.5% |
| TA-RB original (T=6) | 86.5% | 82.8% | 78.4% |
| **TA-RB résiduel** | **90.1%** | 86.8% | 83.0% |

Le décodeur résiduel referme quasiment tout l'écart de TA-RB original avec
SingleSC/IntraRB (+3.6 à +4.6pts) — les 3 architectures finissent
quasi ex-aequo en CSI parfait sur UMi.

### UMi — CSI imparfait (data SNR=15dB, %retenu vs CSI parfait)

| Architecture | pilote 20dB | pilote 10dB |
|---|---|---|
| RZF/WMMSE (référence classique) | ~68.5% | ~35.3% |
| SingleSC | 74.1% | 38.9% |
| IntraRB | 75.8% | 39.9% |
| TA-RB original | 85.7% | 51.0% |
| **TA-RB résiduel** | **87.7%** | **54.6%** |

**TA-RB (les deux variantes, résiduel en tête) est de loin le plus robuste
au bruit d'estimation de canal** — mécanisme identifié (moyennage dans
`_extract_features`, cf. `SESSION_LOG_20260807.md`, section PRIORITÉ 1).

### UMa (M8K4, R=20m) — CSI parfait, 83ep, même protocole

| Architecture | 15dB | 17.5dB | 20dB |
|---|---|---|---|
| SingleSC | 87.5% | 83.9% | 80.6% |
| **IntraRB** | **88.8%** | **85.2%** | **81.9%** |
| TA-RB résiduel | 81.7% | 77.8% | 73.8% |

**Le classement s'inverse et s'écarte nettement** : IntraRB prend la tête
(cohérent avec l'hypothèse causale déjà établie — la SC-attention exploite
la sélectivité fréquentielle réelle d'UMa, cf. `SESSION_LOG_20260807.md`
section UMa 20h12), et **TA-RB résiduel décroche fortement** (81.7% contre
90.1% sur UMi, -8.4pts) — plus net que la dégradation de SingleSC (-3.9pts)
ou IntraRB (-1.9pts).

### UMa — CSI imparfait (TA-RB résiduel uniquement, mesuré)

| | UMa | UMi (réf) |
|---|---|---|
| %retenu pilote 20dB | 87.2% | 87.7% |
| %retenu pilote 10dB | 54.6% | 54.6% |

L'avantage de débruitage de TA-RB résiduel **se transfère quasi à
l'identique** en magnitude relative — le canal n'affecte pas ce mécanisme
particulier.

---

## Interprétation — narratif proposé pour la thèse

**L'histoire se précise et devient cohérente** :

- **TA-RB résiduel est le meilleur choix pratique sur canal peu sélectif
  (UMi)** : compétitif en CSI parfait (90.1%, quasi ex-aequo avec
  SingleSC) ET nettement le plus robuste en CSI imparfait (+13-15pts de
  rétention vs SingleSC/IntraRB) — la compression temporelle du décodeur
  résiduel (interpolation fixe + correction) ne coûte presque rien quand
  le canal a peu de détail fréquentiel fin à préserver, et son mécanisme
  de moyennage aide activement au débruitage.

- **IntraRB devient le meilleur choix sur canal sélectif (UMa)** : la
  compression de TA-RB perd cette fois du détail fréquentiel réellement
  exploitable (le canal UMa a une sélectivité mesurée plus marquée,
  rho 12sc/95sc=2.6× contre 2.2×) — IntraRB, qui traite chaque
  sous-porteuse à pleine résolution (S=12, pas de compression), capture ce
  supplément d'information que TA-RB perd structurellement.

**Nuance importante, pas encore vérifiée** : cette synthèse combine deux
mesures faites sur des couples différents (CSI-parfait comparé sur les 3
architectures aux deux canaux ; CSI-imparfait mesuré seulement pour TA-RB
résiduel sur UMa, jamais pour SingleSC/IntraRB sur UMa). **On sait que
TA-RB résiduel reste robuste au bruit CSI sur UMa** (mécanisme préservé),
mais on ne sait PAS encore si cette robustesse suffirait à le rendre à
nouveau compétitif face à IntraRB sous CSI imparfait sur UMa (IntraRB
n'a montré aucun avantage de débruitage particulier sur UMi -- 75.8% vs
87.7% pour TA-RB résiduel -- donc l'écart pourrait se resserrer, voire
s'inverser à nouveau, sous pilote dégradé). **Prochaine étape naturelle
si on veut clore complètement cette histoire** : mesurer IntraRB (et
SingleSC) en CSI imparfait sur UMa, même protocole que Front B. Pas fait
faute de green light — à discuter.

**Conclusion en l'état** : le choix d'architecture optimal dépend
conjointement (a) de la sélectivité fréquentielle du canal et (b) de la
qualité de l'estimation de canal disponible en pratique — pas un résultat
"une architecture domine partout", ce qui est en soi un message clair et
défendable pour la thèse (le choix d'architecture est un arbitrage
canal-dépendant, pas un vainqueur universel).

---

## Décrochage haut-SNR — caractéristique transverse, canal-indépendante

Indépendamment du classement ci-dessus : toutes les architectures/canaux
perdent 6.6 à 7.2pts de %WMMSE entre 15 et 20dB (rétention 92.1-92.8%,
écart <1pt entre les 4 combinaisons mesurées) — régime MUI-limité où
l'écart RZF/réseau-vs-WMMSE grandit structurellement avec le SNR. Pas
spécifique à une architecture ni un canal (cf. `SESSION_LOG_20260808.md`).
Front A (en cours) explore des pistes pour combler cet écart
spécifiquement — résultats à intégrer ici une fois disponibles.

---

## Sources

CSI parfait UMi : `results/results_20260807_155133.npy`,
`results/diag_tarb_residual_test.json`. CSI imparfait UMi :
`results/diag_csi_imperfect.json` (+ `diag_tarb_residual_test.json` pour
le résiduel). CSI parfait/imparfait UMa : `results/diag_uma_quick_
{single_sc,intra_rb}.json`, `results/diag_tarb_residual_uma_test.json`,
référence WMMSE `results/classical_comparison_M8K4_uma.npy`. Sélectivité
fréquentielle : `results/diag_short_lag_correlation.json`.
