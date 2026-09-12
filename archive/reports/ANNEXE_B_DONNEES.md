# Annexe B — Données complémentaires (cohérence fréquentielle + SNR intermédiaires)

Calculs effectués le 10/08/2026, sur demande explicite ("Oui fais le stp;
pour pouvoir l'annexe pour toutes les données"), en complément des
tableaux principaux du mémoire. Aucun nouvel entraînement : la partie 2
réévalue des checkpoints déjà entraînés et validés (cf. vérification
signed_attn précédente) ; la partie 1 est une mesure de caractérisation
du canal (statistique sur H, indépendante de tout modèle).

---

## 1. Bande de cohérence — ρ(Δn) pour 4 scénarios

**Script** : `final/diag_coherence_annexe_b.py`
**Résultats** : `final/results/diag_coherence_annexe_b.json`
**Méthode** : identique à `diag_uma_selectivity.py` (référence déjà
utilisée dans le texte pour ρ(12) UMa) — 20 tirages de topologie
verrouillée × batch 16, `STANDARD_CONFIG` (M=8, K=4, R=20 m),
`ρ(Δn) = |E[H[:,:-Δn]·conj(H[:,Δn:])]| / E[|H|²]`, Δn ∈
{1,...,12, 24, 48, 95} — les mêmes points de lag déjà utilisés pour
ρ(12) dans le texte, étendus ici aux 4 scénarios LOS/NLOS.

Le pipeline final n'entraîne/n'évalue qu'en NLOS ; les colonnes LOS
sont une caractérisation de canal pour l'annexe, pas une config jamais
utilisée pour l'entraînement.

| Δn (sous-porteuses) | UMi-LOS | UMi-NLOS | UMa-LOS | UMa-NLOS |
|---:|---:|---:|---:|---:|
| 1  | 0.9992 | 0.9979 | 0.9867 | 0.9835 |
| 2  | 0.9971 | 0.9930 | 0.9696 | 0.9474 |
| 3  | 0.9938 | 0.9861 | 0.9533 | 0.9067 |
| 4  | 0.9896 | 0.9779 | 0.9389 | 0.8681 |
| 5  | 0.9847 | 0.9685 | 0.9277 | 0.8324 |
| 6  | 0.9792 | 0.9582 | 0.9155 | 0.7991 |
| 7  | 0.9732 | 0.9475 | 0.9053 | 0.7674 |
| 8  | 0.9668 | 0.9368 | 0.8946 | 0.7384 |
| 9  | 0.9602 | 0.9261 | 0.8888 | 0.7132 |
| 10 | 0.9535 | 0.9154 | 0.8827 | 0.6897 |
| 11 | 0.9468 | 0.9046 | 0.8745 | 0.6674 |
| **12** | **0.9400** | **0.8938** | **0.8684** | **0.6465** |
| 24 | 0.8671 | 0.7764 | 0.8093 | 0.4626 |
| 48 | 0.7795 | 0.6033 | 0.7504 | 0.3079 |
| 95 | 0.7316 | 0.4205 | 0.7329 | 0.2145 |

Décroissance ρ(12)/ρ(1) : UMi-LOS 0.941, UMi-NLOS 0.896, UMa-LOS
0.880, **UMa-NLOS 0.657** — cohérent avec la sélectivité fréquentielle
nettement plus forte du canal UMa-NLOS (étalement retard plus grand),
déjà identifiée dans le texte comme motivant IntraRB/TA-RB sur UMa.

### ⚠️ Point à vérifier avant intégration au .tex

Ma mesure **UMi-NLOS ρ(12) = 0.8938** ne correspond **pas** à une
valeur de référence **0.797** trouvée codée en dur (commentaire, pas
un fichier de résultats sauvegardé) dans `diag_uma_selectivity.py`
(variable `umi_ref`, ~ligne 112-113), présentée dans ce script comme
la valeur UMi "canal actuel" au moment de son écriture.

- Le contrôle croisé sur UMa fonctionne bien : ma mesure UMa-NLOS
  ρ(12)=0.6465 concorde avec `results/diag_uma_selectivity.json`
  (0.6323, écart ~2%, cohérent avec le bruit Monte-Carlo à 20 tirages)
  → la méthodologie du nouveau script est fiable et reproduit
  correctement une mesure déjà établie.
- Il n'existe **aucun fichier de résultats sauvegardé pour UMi**
  équivalent à `diag_uma_selectivity.json` — le "0.797" n'est qu'un
  commentaire de code, probablement une mesure antérieure à une
  révision de `channel_config.py` (le verrouillage R=20m NLOS actuel),
  donc vraisemblablement obsolète.
- **Si le texte du mémoire cite déjà 0.797 pour ρ(12) UMi**, il faut
  trancher : soit remplacer par 0.894 (nouvelle mesure, méthodologie
  validée), soit ré-exécuter l'ancien script sur l'ancienne config
  pour comprendre l'écart. Je recommande d'utiliser **0.894** (mesure
  fraîche, config verrouillée actuelle, validée croisée sur UMa) sauf
  si une raison spécifique justifie de garder l'ancienne valeur.

---

## 2. Débit somme aux SNR intermédiaires — CSI parfait, 4 régimes × 3 architectures

**Script** : `final/diag_intermediate_snr_annexe_b.py`
**Résultats** : `final/results/diag_intermediate_snr_{regime}.json`
(un fichier par régime : `umi_standard`, `uma_standard`,
`umi_massive`, `uma_massive`)
**Méthode** : `eval_perfect()`, identique à la fonction utilisée dans
tous les scripts d'entraînement (`sinr` LMMSE avec whitening
d'interférence, débit somme = Σ log2(1+SINR) moyenné sur 20 batches),
CSI parfait (pas de bruit pilote), aucun réentraînement — chargement
direct des poids des checkpoints déjà validés.

Les tableaux principaux du mémoire n'évaluent qu'à {0, 5, 10, 15,
17.5(?), 20}dB selon les scripts — ces points {2.5, 7.5, 12.5, 17.5}dB
comblent les intervalles.

### UMi STANDARD (M=8, K=4)
Checkpoints : `single_sc`→`./weights/front_a_signed_attn/best_20260808_103631`,
`intra_rb`→`./weights/IntraRB_signed_attn_4L_128d/best_20260808_113308`,
`ta_rb_residual`→`./weights/TA_RB_residual_signed_attn_T4_4L_128d/best_20260808_181436`
Source : `results/diag_intermediate_snr_umi_standard.json`

| SNR (dB) | single_sc | intra_rb | ta_rb_residual (T=4) |
|---:|---:|---:|---:|
| 2.5  | 16.78 | 16.80 | 16.80 |
| 7.5  | 22.93 | 22.98 | 22.88 |
| 12.5 | 28.97 | 29.16 | 28.88 |
| 17.5 | 34.25 | 34.78 | 34.73 |

### UMa STANDARD (M=8, K=4)
Checkpoints : `single_sc`→`./weights/UMa_SingleSC_signed_attn_4L_128d/best_20260808_231408`,
`intra_rb`→`./weights/UMa_IntraRB_signed_attn_4L_128d/best_20260809_005458`,
`ta_rb_residual`→`./weights/UMa_TA_RB_residual_signed_attn_4tok_4L_128d/best_20260809_001031`
Source : `results/diag_intermediate_snr_uma_standard.json`

| SNR (dB) | single_sc | intra_rb | ta_rb_residual (T=4) |
|---:|---:|---:|---:|
| 2.5  | 15.14 | 15.13 | 14.53 |
| 7.5  | 20.84 | 20.93 | 19.65 |
| 12.5 | 26.96 | 26.76 | 24.53 |
| 17.5 | 32.46 | 32.12 | 29.00 |

### UMi MASSIVE (M=64, K=8, budget étendu 170ep)
Checkpoints : `single_sc`→`./weights/MASSIVE_TRUE_SingleSC_signed_attn_4L_128d_extbudget/best_20260808_195627`,
`intra_rb`→`./weights/MASSIVE_TRUE_IntraRB_signed_attn_4L_128d_extbudget/best_20260808_204513`,
`ta_rb_residual`→`./weights/MASSIVE_TRUE_TA_RB_residual_signed_attn_6tok_4L_128d_extbudget/best_20260808_221356`
Source : `results/diag_intermediate_snr_umi_massive.json`

| SNR (dB) | single_sc | intra_rb | ta_rb_residual (T=6) |
|---:|---:|---:|---:|
| 2.5  | 54.33 | 54.57 | 53.54 |
| 7.5  | 65.84 | 66.43 | 66.39 |
| 12.5 | 75.21 | 76.13 | 78.16 |
| 17.5 | 81.46 | 82.75 | 86.83 |

### UMa MASSIVE (M=64, K=8, budget étendu 170ep)
Checkpoints : `single_sc`→`./weights/UMa_MASSIVE_SingleSC_signed_attn_4L_128d/best_20260808_234158`,
`intra_rb`→`./weights/UMa_MASSIVE_IntraRB_signed_attn_4L_128d/best_20260809_005701`,
`ta_rb_residual`→`./weights/UMa_MASSIVE_TA_RB_residual_signed_attn_6tok_4L_128d/best_20260809_022421`
Source : `results/diag_intermediate_snr_uma_massive.json`

| SNR (dB) | single_sc | intra_rb | ta_rb_residual (T=6) |
|---:|---:|---:|---:|
| 2.5  | 53.67 | 54.70 | 51.36 |
| 7.5  | 65.65 | 67.42 | 63.11 |
| 12.5 | 75.89 | 79.60 | 72.74 |
| 17.5 | 82.54 | 89.73 | 78.08 |

**Lecture rapide** : ces points intermédiaires sont cohérents avec les
tendances déjà établies dans le corps du texte (pas de retournement de
classement entre archis à SNR intermédiaire) — MASSIVE toutes archis
confondues > STANDARD comme attendu (channel hardening), IntraRB
généralement en tête ou proche de la tête sur les 4 régimes en CSI
parfait, TA-RB compétitif surtout à SNR élevé en MASSIVE UMi.

---

## Statut d'exécution

- 4/4 régimes SNR intermédiaires : terminés sans erreur (`exit=0` pour
  chaque régime, cf. `intermediate_snr_runner.log`).
- 4/4 scénarios cohérence fréquentielle : terminés sans erreur.
- Point ouvert : écart UMi-NLOS ρ(12) 0.894 (nouveau) vs 0.797 (ancien
  commentaire de code) — à trancher avant intégration au .tex, cf. §1.
