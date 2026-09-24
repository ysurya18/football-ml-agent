# football-ml-agent

Ask a question about an international football match in plain English. A gradient-boosted
model answers it with real numbers — not a language model guessing.

```
> Who wins if Spain play Brazil at a neutral venue, and why?

Spain 56.7%  ·  Draw 25.5%  ·  Brazil 17.8%

Driven mainly by the Elo gap (+84 to Spain) and Spain's stronger recent
form over their last 10 competitive matches. Brazil's head-to-head record
pulls the other way but carries little weight at this sample size.
```

The language model never estimates a probability. It decides *which* part of the model to
run, runs it, and puts the output into words.

---

## Status

Early. Scaffolding and data are in place; the feature pipeline, model, tools, and agent
are being built in that order. See [Roadmap](#roadmap).

## How it works

Three layers, deliberately separated so the interesting part stays testable:

**1. Feature pipeline** — Replays all 49,287 matches in chronological order, maintaining
running trackers for each team (Elo, form, head-to-head, momentum, scoring profile, Poisson
goal rates, tournament context). Each match emits a **92-feature row built only from
pre-match state**, then the trackers update. That ordering is the whole ballgame: build the
features from the full history instead and you leak the result you're trying to predict.

**2. Model** — XGBoost over three classes (home win / draw / away win). Trained on 1990
onward, split by *time* — train on pre-2020, test on 2020+ — because a random split lets the
model peek at the future and inflates accuracy by several points.

**3. Agent** — Claude with a set of tools that map onto the pipeline. Each tool is an
ordinary, unit-tested Python function. The model picks the tool and narrates the result.

| Tool | Answers |
|---|---|
| `predict_match` | "Who wins Spain vs Brazil?" |
| `explain_prediction` | "Why does it favour Spain?" — SHAP values, actual per-match attribution |
| `what_if` | "What if it were played at altitude in Bogotá?" |
| `get_elo` / `get_form` / `get_h2h` / `get_momentum` / `get_poisson_xg` / `get_tournament_context` | One feature family each |
| `query_matches` | "How often have Brazil and Argentina drawn since 2000?" — read-only SQL |
| `simulate_tournament` | Monte Carlo a group or bracket |

## What accuracy to expect

Around **60%** on the three-class problem, time-split. That is genuinely good, and it is
worth being clear about why it sounds low:

- Draws are roughly a quarter of all results and are close to irreducible. Whether a match
  finishes 1–0 or 1–1 often turns on one deflection.
- Team sheets are unknown before kickoff. Injuries, rotation, and tactical surprises are
  simply not in the data.
- Red cards, penalties, and weather are not in the data either.

Published work on this dataset puts the ceiling near 62–65% using pre-match features alone.
A model claiming much more than that has usually leaked something.

## Data

49,287 international matches (1872–2024), 47,601 goal records, 675 shootouts, from the
Kaggle dataset maintained by martj42. Public domain (CC0), committed to this repo so a
clone runs without a Kaggle account. See [NOTICE.md](NOTICE.md).

## Getting started

Requires [uv](https://docs.astral.sh/uv/) — it manages Python 3.12 and the dependencies.

```bash
git clone git@github.com:ysurya18/football-ml-agent.git
cd football-ml-agent
uv sync

cp .env.example .env        # add your ANTHROPIC_API_KEY
uv run python -m football_agent.data.build    # load CSVs into DuckDB
uv run python -m football_agent.model.train   # build features, train, evaluate
uv run python -m football_agent.cli           # chat
```

## Roadmap

- [x] Project scaffold, data, licensing
- [ ] DuckDB loader
- [ ] Feature pipeline — 92 features, chronological replay, scenario overrides
- [ ] XGBoost training with time-based split and a calibration check
- [ ] Tool layer with unit tests
- [ ] Claude tool-use agent + CLI
- [ ] FastAPI service and web UI

## Acknowledgements

The feature design follows Oracle's soccer-analytics-agent workshop (UPL licensed); this
project reimplements it on DuckDB and Claude instead of Oracle Database and Grok.
Attribution in [NOTICE.md](NOTICE.md).

## Licence

MIT — see [LICENSE](LICENSE).
