# Rapport d'investigation — Condition NLOS (tab:channel_config) : forcée ou probabiliste ?

**Rôle de ce document** : rapport factuel uniquement. Aucune reformulation de
texte de mémoire. Aucun fichier modifié dans le dépôt mémoire (Overleaf), rien
poussé.

**Réponse courte : (a) FORCÉE explicitement.** Ce n'est pas un résultat émergent
du modèle probabiliste TR 38.901 §7.4.2 — le code appelle l'API Sionna avec un
paramètre booléen explicite qui, par la documentation même de Sionna, désactive
le calcul de probabilité LOS et impose NLOS à 100% des tirages. Confirmé pour
UMi ET UMa, pour les deux régimes (validation M=8 et MIMO massif M=64).

---

## 1. Localisation du mécanisme

Fichier : `final/channel_config.py`

```python
# lignes 438-453
def set_locked_topology(channel_model, batch_size, num_ut,
                         cluster_radius_m=20.0, force_los=False,
                         indoor_probability=0.0, scenario=SCENARIO,
                         half_angle_deg=None):
    """Draw a topology under the locked mechanism and apply it to
    channel_model. force_los is kept as a parameter for interface stability
    but REVISION (b)'s mechanism is validated under NLOS only (los=False) --
    don't pass True without re-validating the gap first.
    [...]"""
    topology = gen_topology_locked(batch_size, num_ut, cluster_radius_m,
                                    indoor_probability, scenario)
    channel_model.set_topology(*topology, los=force_los)
```

La ligne clé est **`channel_model.set_topology(*topology, los=force_los)`**
(ligne 453), avec `force_los` par défaut = `False` (signature ligne 439).
`channel_model` est une instance Sionna `UMi` ou `UMa`
(`sionna.phy.channel.tr38901.{UMi,UMa}`) — le mécanisme est identique quel
que soit le scénario passé, `set_locked_topology` ne fait aucune distinction
UMi/UMa dans sa logique de forçage.

## 2. Comportement du paramètre `los` dans l'API Sionna elle-même (pas une supposition — docstring officielle du package installé)

Fichier : `sionna/phy/channel/tr38901/system_level_channel.py` (package
installé, `/export/tmp/sala/miniconda/envs/tf-gpu/lib/python3.11/site-packages/sionna/`),
méthode `set_topology`, docstring du paramètre `los` (lignes 134-138) :

> ```
> los : `None` (default) | `tf.bool`
>     If not `None`, all UTs located outdoor are
>     forced to be in LoS if ``los`` is set to `True`, or in NLoS
>     if it is set to `False`. If set to `None`, the LoS/NLoS states
>     of UTs is set following 3GPP specification [TR38901]_.
> ```

C'est sans ambiguïté : passer `los=False` (un booléen, PAS `None`) **désactive
complètement** le calcul probabiliste 3GPP §7.4.2 basé sur la distance 2D, et
force NLoS pour 100% des utilisateurs extérieurs, à chaque tirage. Seul
`los=None` laisserait Sionna tirer LOS/NLOS selon le modèle probabiliste
standard. Or `set_locked_topology` appelle systématiquement avec
`los=force_los` où `force_los` est un booléen concret (`False` partout dans le
dépôt, voir §3) — jamais `None`.

## 3. Tous les points d'appel du dépôt (exhaustif — `grep -rn "force_los"`)

Chaque appel a été vérifié individuellement. **Aucune exception trouvée** :

| Fichier | Valeur passée |
|---|---|
| `channel_config.py:439` (défaut de la fonction) | `False` |
| `channel_config.py` — `STANDARD_CONFIG['FORCE_LOS']` (ligne 306) | `False` |
| `channel_config.py` — `MASSIVE_CONFIG['FORCE_LOS']` (ligne 321) | `False` |
| `channel_config.py` — `MASSIVE_TRUE_CONFIG['FORCE_LOS']` (ligne 342) | `False` |
| `main_finall.py:417` (`CHOSEN_CONFIG['FORCE_LOS']`, `CHOSEN_CONFIG = STANDARD_CONFIG`, ligne 59) | `False` |
| `classical_comparison.py:51` | `False` |
| `diag_classical_comparison_massive_true.py:44` | `False` |
| `diag_classical_comparison_massive_uma.py:52` | `False` |
| `diag_uma_signed_attn_train.py:81` | `False` |
| `diag_uma_massive_train.py:77` | `False` |
| `datasets.py:180` (pipeline d'entraînement `CachedSionnaDataset`, tous scénarios) | `False` |
| `diag_csi_imperfect_massive_true_sweep.py:76`, `diag_csi_imperfect_massive_capD384L4.py:73`, `diag_csi_imperfect_massive_extbudget_sweep.py:76` | `MASSIVE_TRUE_CONFIG['FORCE_LOS']` → `False` |
| + 12 autres scripts diagnostiques (`diag_csi_imperfect*.py`, `diag_rzf_grouped_for_tsweep.py`, `diag_intrarb_gradient_layers.py`, `diag_short_lag_correlation.py`, `diag_high_snr_disadvantage.py`, `diag_tarb_residual_test.py`, `diag_coherence_annexe_b.py`, `diag_coherence_bandwidth_threshold.py`) | `False` |

**Aucun appel avec `force_los=True` ni `force_los=None` n'existe dans le
dépôt.** Le paramètre est passé positionnellement/nommément mais sa valeur
est invariablement `False`.

## 4. Historique : ce forçage est un choix délibéré, documenté comme tel

Docstring de `channel_config.py` (ligne 186, section "REVISION 2026-08-07") :

> « Switched FORCE_LOS -> forced NLOS (los=False, was True), HALF_ANGLE_DEG
> [...] »

Ceci confirme que le forçage LOS/NLOS est un **paramètre de conception
délibérément piloté** — le code a même été explicitement changé d'un forçage
LOS=True (une révision antérieure) vers un forçage NLOS=False (la révision
actuelle) ; ce n'est à aucun moment un résultat laissé au hasard du modèle
probabiliste.

Le commentaire de `set_locked_topology` lui-même (§1 ci-dessus) le confirme
une seconde fois : *« REVISION (b)'s mechanism is validated under NLOS only
(los=False) -- don't pass True without re-validating the gap first »* — la
condition NLOS est traitée comme une hypothèse de validité du mécanisme
entier, pas comme une observation empirique.

## 5. UMi et UMa — même mécanisme, confirmé dans le code (pas seulement dans le texte)

`set_locked_topology` ne contient aucune branche conditionnelle sur
`scenario` pour le traitement du LOS — le paramètre `scenario` ne sert qu'à
choisir les paramètres 3GPP internes (`set_3gpp_scenario_parameters`, pour
la géométrie du secteur), pas le forçage LOS. Confirmé directement par
inspection de code (§1) et par les points d'appel concrets :

- `diag_classical_comparison_massive_uma.py:44-52` : instancie
  `UMa(carrier_frequency=2.6e9, ...)` puis appelle
  `set_locked_topology(self.channel_model, ..., force_los=False, ...)` —
  identique à l'appel UMi du même script pour la classe soeur `LockedSystem`.
- `diag_uma_signed_attn_train.py:81` et `diag_uma_massive_train.py:77` :
  entraînement UMa (`scenario='uma'`, confirmé ligne 112/116 de ces
  fichiers pour la construction du dataset), même `force_los=False`.

**Donc oui : UMa utilise exactement le même mécanisme de forçage NLOS que
UMi, dans les deux régimes (validation M=8 et MIMO massif M=64).**

## 6. Point n'ayant pas eu besoin d'être calculé

La demande initiale prévoyait, dans le cas (b) probabiliste, de calculer la
proportion réelle NLOS/LOS observée. **Ce calcul n'est pas applicable** : le
forçage étant déterministe et explicite dans le code (`los=False` passé
directement à l'API Sionna, qui documente elle-même que cela bypasse tout
calcul probabiliste), la proportion est per construction 100% NLOS / 0% LOS
sur l'intégralité des tirages, pour tous les scripts du dépôt. Aucune mesure
empirique n'était nécessaire pour trancher la question — la lecture du code
et de la documentation Sionna suffit et est sans ambiguïté.

Pour référence, les points LOS mentionnés dans l'Annexe B (rapport du
12/08, cohérence fréquentielle) et dans `diag_coherence_bandwidth_threshold.py`
sont obtenus en appelant `force_los=True` **explicitement en dehors du
pipeline de production**, uniquement pour la caractérisation de canal
(mesure de ρ(Δn)) — jamais pour l'entraînement ou l'évaluation des
précodeurs. C'est cohérent avec le tableau `tab:channel_config` qui
n'affiche que la condition NLOS comme "retenue".

---

## Résumé pour référence rapide

| Question | Réponse |
|---|---|
| Forcé ou probabiliste ? | **Forcé** (`los=False` passé explicitement, bypass total du calcul probabiliste TR38.901 §7.4.2 — confirmé par la docstring de l'API Sionna elle-même) |
| Ligne de code exacte | `channel_config.py:453` — `channel_model.set_topology(*topology, los=force_los)` |
| UMi concerné ? | Oui |
| UMa concerné ? | Oui, mécanisme identique, confirmé par code (pas seulement par le texte) |
| Régime validation (M=8) concerné ? | Oui |
| Régime MIMO massif (M=64) concerné ? | Oui |
| Exception trouvée (un appel avec `force_los=True` en production) ? | Aucune |
| Proportion NLOS/LOS empirique nécessaire ? | Non — déterministe par construction, 100%/0% |
