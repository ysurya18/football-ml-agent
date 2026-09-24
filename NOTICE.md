# Third-party attribution

## Match data — `data/raw/`

International football results, goalscorers, shootouts, and former team names.

- Source: [martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) on Kaggle
- Licence: **CC0 1.0 Universal (public domain dedication)**
- Coverage in this snapshot: 49,287 matches (1872–2024), 47,601 goal records, 675 shootouts

## Feature-engineering reference — `reference/enhanced_features_upstream.py`

The 92-feature pipeline in this project is derived from the feature engineering in
Oracle's soccer-analytics-agent workshop.

- Source: [oracle-devrel/oracle-ai-developer-hub](https://github.com/oracle-devrel/oracle-ai-developer-hub/tree/main/workshops/soccer-analytics-agent)
- Licence: **Universal Permissive License (UPL) 1.0**

The upstream file is vendored unmodified under `reference/` for provenance. The working
implementation under `football_agent/features/` is a rewrite, not a copy: it targets DuckDB
instead of Oracle Database, and its feature builder accepts scenario overrides so the agent
can answer counterfactual questions.

Everything else in this repository is original work under the MIT License (see `LICENSE`).
