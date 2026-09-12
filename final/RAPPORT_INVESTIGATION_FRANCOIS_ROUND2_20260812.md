CONTEXTE — Mémoire de maîtrise Polytechnique Montréal (Glodi Sala, dir. François
Leduc-Primeau). Vérification factuelle d'une affirmation du Chapitre 3
(7-Theme1.tex) sur la construction des canaux UMi/UMa utilisés dans le mémoire.

Dépôts :
git clone https://git:olp_3NEGLeNJX9sFF3VmtqYETsMuMAv6Xt44uLM5@git.overleaf.com/6a231f20f1b0598bbeaeada7 memoire
Projet Sionna (source des scripts, confirmé dans une investigation précédente) :
/users/sala/Documents/test_projet/Trans/freq/Sionna/final/

## RÔLE
Investigation factuelle uniquement. Pas de rédaction de texte de mémoire. Rapporter
ce qui est trouvé dans le code, avec chemins de fichiers et lignes exactes.

## QUESTION À VÉRIFIER

Le Tableau tab:channel_config (7-Theme1.tex, ligne ~224) affiche :
"Condition de propagation : NLOS"

Le paragraphe qui l'accompagne (ligne ~192-211) décrit une géométrie
utilisateurs verrouillée (rayon R, tirage 3GPP modifié), mais ne précise pas
explicitement, dans le texte, SI la condition NLOS est :
(a) FORCÉE explicitement dans le code (ex. un paramètre qui désactive le calcul
    de probabilité LOS/NLOS de Sionna et impose NLOS à 100% des tirages), ou
(b) simplement le résultat naturel du modèle 3GPP LOS/NLOS probabiliste
    standard, qui donnerait NLOS dans la quasi-totalité des tirages à cette
    configuration géométrique (R, distance BS-UT) sans intervention explicite.

C'est une distinction importante : (a) est un choix de conception explicite à
documenter comme tel, (b) est un résultat émergent de la configuration
géométrique qu'il faudrait décrire différemment dans le texte.

## À FAIRE

1. Localiser dans le dépôt Sionna le(s) script(s)/module(s) responsable(s) de la
   génération des canaux UMi et UMa pour le mémoire (probablement
   `channel_config.py`, `set_locked_topology()`, et l'appel au modèle de canal
   Sionna `sionna.phy.channel.tr38901.{UMi,UMa}`).
2. Vérifier précisément COMMENT la condition LOS/NLOS est déterminée à chaque
   tirage de canal :
   - Est-ce que le code appelle l'API Sionna standard qui calcule la
     probabilité LOS selon la distance 2D (comme spécifié par TR 38.901,
     §7.4.2) et laisse le résultat aléatoire (LOS ou NLOS selon un tirage) ?
   - OU est-ce qu'un paramètre explicite (ex. `los=False`, `always_los=False`,
     un flag custom, un override après génération) FORCE la condition NLOS
     pour 100% des réalisations, indépendamment de ce que la probabilité 3GPP
     donnerait ?
3. Si forcé : citer la ligne de code exacte qui fait ce forçage, et pour quels
   canaux (UMi seulement ? UMa aussi ? les deux régimes, validation et MIMO
   massif ?).
4. Si probabiliste (non forcé) : calculer ou retrouver dans le code/logs la
   proportion réelle de tirages NLOS vs LOS obtenue à R=20m (les deux régimes),
   pour les deux canaux (UMi, UMa), sur un échantillon représentatif (ex. les
   mêmes 1600 réalisations que la mesure de cohérence empirique récente, ou un
   nouveau tirage si nécessaire — préciser lequel).
5. Vérifier si ce comportement est identique pour UMa (le second scénario,
   introduit spécifiquement pour tester la sélectivité fréquentielle) — même
   mécanisme de construction que UMi d'après le texte (ligne ~271-274 de
   7-Theme1.tex), mais à confirmer dans le code, pas seulement dans le texte.

## LIVRABLE

Un rapport factuel : citation du code exact, réponse claire (a) forcé ou
(b) probabiliste-avec-résultat-majoritairement-NLOS, et si (b), les proportions
réelles trouvées. Aucune reformulation de texte de mémoire.