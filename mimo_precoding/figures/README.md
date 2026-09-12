# Figures

Un sous-dossier par régime, chaque `fig*.py` régénère ses `.png` / `.pdf` /
`.json` depuis `../../results/` et écrit dans son propre dossier. À lancer
depuis le dossier du régime :

```bash
cd figures/umi_standard && $PY figA_sumrate_seedfix.py
```

## Régimes

| Dossier | Canal × échelle | Note |
|---|---|---|
| `umi_standard/` | UMi, M=8/K=4 | régime principal : seul avec le balayage T complet (figures B et C) |
| `uma_standard/` | UMa, M=8/K=4 | TA-RB à T=4 fixe |
| `umi_massive_capD384L4/` | UMi, M=64/K=8, **D=384** | régime massive de référence |
| `uma_massive_capD384L4/` | UMa, M=64/K=8, **D=384** | régime massive de référence |
| `umi_massive/`, `uma_massive/` | idem, **D=128** | état antérieur au gap-closing du 12-13 août, conservé pour comparaison |

## Conventions

- **Figure A** — débit somme vs SNR, double panneau (CSI parfait | CSI
  imparfait pilote 20 dB), zoom haut-SNR sur le panneau CSI parfait quand
  les courbes s'y resserrent.
- **Figure B** — Pareto débit / énergie. **Purement descriptive** : aucune
  recommandation ni mise en avant visuelle d'un T ou d'une architecture sur
  le graphique, la discussion se fait dans le texte du chapitre.
- **Figure C** — débit somme vs SNR par T (TA-RB seulement), `umi_standard/`
  uniquement.
- **Figure D** — BER vs SNR, double panneau si les données CSI imparfait
  existent.

## Provenance des données — à vérifier avant dépôt final

| Script | Données sources |
|---|---|
| `figA_sumrate_seedfix.py`, `figC_tsweep_seedfix.py`, `figB_pareto_energy_T_2560seedfix.py`, `figC_sumrate_snr_T_2560seedfix*.py` | campagne **seedfix du 20 août** (canonique) |
| `figA_sumrate_csi_perfect.py`, `figB_pareto_energy*.py` des régimes massive, `figD_ber_snr.py` | campagnes **antérieures** — cohérentes entre elles mais pas alignées sur les tableaux du README racine (§4.2) |

Les scripts `figB_*` consomment en plus les fichiers d'énergie
**pré-correctif** (`complexity_energy_*.json`, `energy_per_T_signed_attn.json`) ;
les versions post-correctif sont les `*_postfix_20260912.json`. Voir §6.1 et
§6.5 du README racine.

Attention : `figB_pareto_energy_T_2560seedfix.py` et
`figC_sumrate_snr_T_2560seedfix_grouped.py` écrivent sous les noms canoniques
`figB_pareto_energy_T.{png,pdf}` et `figC_sumrate_snr_T.{png,pdf}` — les
images portant ces noms sont donc déjà issues des données seedfix, contrairement
à ce que leur nom suggère.
