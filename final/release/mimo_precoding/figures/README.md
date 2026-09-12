# Convention de nomenclature — figures finales

Dossier structuré par régime, un sous-dossier par combinaison
canal × échelle d'antennes :

- `umi_standard/` — canal UMi, M=8/K=4 (STANDARD_CONFIG). Régime principal,
  T-sweep complet disponible (Figure C).
- `umi_massive/` — canal UMi, M=64/K=8 (MASSIVE_TRUE_CONFIG). Pas de
  T-sweep (TA-RB entraîné à T=6 fixe uniquement) -> pas de Figure C.
- `uma_standard/` — canal UMa, M=8/K=4. Pas de T-sweep (TA-RB à T=4
  fixe, référence identifiée sur umi_standard) -> pas de Figure C.
- `uma_massive/` — canal UMa, M=64/K=8.

Chaque sous-dossier suit le même schéma de figures quand les données
le permettent :
- **Figure A** — débit somme vs SNR, double panneau (CSI parfait |
  CSI imparfait pilote 20dB), zoom haut-SNR sur le panneau CSI parfait
  uniquement (umi_standard) si les courbes s'y resserrent.
- **Figure B** — Pareto débit/énergie. **Purement descriptif** : aucune
  recommandation ni mise en avant visuelle d'un T ou d'une architecture
  sur le graphique (la discussion contextuelle se fait dans le texte
  du chapitre, pas sur la figure).
- **Figure C** — débit somme vs SNR par T (TA-RB uniquement) --
  seulement pour `umi_standard/`, seul régime avec T-sweep complet.
- **Figure D** — BER vs SNR, double panneau si les données CSI
  imparfait existent, simple panneau sinon.

Chaque script `fig*.py` s'exécute depuis son propre sous-dossier
(`DATA_DIR = '../../results/'` pointe vers `results/`, où vivent les données
sources JSON/NPY -- jamais dupliquées dans `figures_final/`).
