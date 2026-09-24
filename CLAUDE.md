# football-ml-agent

Natural-language agent over an XGBoost model that predicts international football
results. The language model never estimates a probability — it chooses which part of
the model to run, runs it, and puts the output into words.

## Origin and the decisions already made

Derived from Oracle's [soccer-analytics-agent workshop](https://github.com/oracle-devrel/oracle-ai-developer-hub/tree/main/workshops/soccer-analytics-agent),
but deliberately **not** a fork. Three decisions, already settled — don't relitigate them
without a reason:

- **No Oracle Database.** Upstream uses Oracle AI Database for a three-tier memory layer,
  vector store, and hybrid retrieval. That is ~60% of upstream's complexity and serves
  Oracle's marketing, not this project's goal. DuckDB replaces it: one file, no server.
- **Claude, not Grok via OCI.** The workshop's shared OCI key expired 2026-06-11 (verified
  by request). Upstream also fakes tool calling through a prompt protocol because OCI's
  endpoint never returned a `toolCallId`; real `tool_use` blocks are strictly better.
- **Tool layer is independent of the agent.** Every tool is an ordinary, unit-tested Python
  function. The agent is a thin shell over them. Correctness lives in the tools.

## Current state

| | |
|---|---|
| Done | Scaffold, licensing, DuckDB loader, tournament tiers, feature pipeline, 83 tests |
| Next | XGBoost training + calibration check, measured against the Elo-only baseline |
| Then | Tool layer → Claude agent → FastAPI |

Roadmap checkboxes live in `README.md`.

## Facts about the data that bite

- **The CSV contains 72 unplayed fixtures.** The 2026 World Cup group stage, scores `NA`.
  They break integer type inference unless read with `nullstr='NA'`. They are loaded with
  `is_played = false` and a null `outcome`. **Anything that trains must filter on
  `outcome IS NOT NULL`** — that is the only thing keeping them out of the model.
- **Snapshot ends 2026-03-31.** Refreshing from Kaggle would turn those 72 fixtures into a
  genuine held-out validation set — results the model has never seen in any form.
- **The majority-class baseline is 48.4%, not 33%.** From 1990: home 48.4%, draw 23.5%,
  away 28.0%. Always report accuracy against 48.4%.
- **The Elo favourite alone scores 60.5% on 2020+.** Predicting home win when
  `home_expected > 0.5`, else away win — one feature, no model. That, not 48.4%, is the
  bar the trained model has to clear to justify the other 91 features. (Majority class
  over 2020+ alone is 47.4%.)
- **Expect ~60%, and no more than 62-65%.** Draws are near-irreducible; team sheets, red
  cards and weather are not in the data. A model claiming much more has leaked something.

## Two upstream defects already fixed — do not reintroduce

Both in `FootballElo.classify_tournament`, `reference/enhanced_features_upstream.py:123`.
Regression tests in `tests/test_tournaments.py`.

1. **Ordering.** Upstream scans its tournament-name dictionary before testing for
   `"qualification"`, using substring matching — so `"FIFA World Cup qualification"` matched
   the key `"FIFA World Cup"` and was graded as a final. 14,782 of 49,287 matches (30%)
   carried an inflated Elo K-factor. In `football_agent/data/tournaments.py`, the order of
   the `if` statements in `classify()` *is* the fix.
2. **Accents.** The key is `"Copa America"`; the data says `"Copa América"`. Names are now
   accent-folded before matching.

## The feature pipeline

`football_agent/features/`. `FeatureState.features(home, away, ctx)` is pure and
`update()` applies a result; the replay reads, records, then updates. Training rows and
live predictions go through the same `features()` call, and a scenario is just a
different `MatchContext`. Column names and order match upstream's `ALL_FEATURES`
exactly (tested). 76 of 92 features are numerically identical to upstream; the other 16
differ only where upstream defects were fixed — own goals, missing-data comebacks,
Poisson cache/truncation, accented cities, confederation coverage, stage substrings.
Each has a regression test in `tests/test_features.py`.

## Conventions

- **`uv` for everything.** `uv run python ...`, `uv run pytest`. Never `pip`.
- **Branch → PR → squash merge.** `main` stays working. One feature per branch, named
  `feat/…`. The user reviews every PR; that review is the point, not a formality.
- **Commit messages explain *why*.** Subject line, blank line, prose. `git log` is the
  design record.
- **Generated artifacts are gitignored.** `football.duckdb`, trained models, processed
  features. Raw CSVs are committed (public domain, 6.7 MB) so a clone runs.
- **Never commit credentials.** Scan staged diffs before committing. `.env` is gitignored;
  `.env.example` documents the variables.
- `reference/` is vendored upstream source, kept byte-identical for provenance. Excluded
  from ruff. Read it, don't edit it.

## Commands

```bash
uv sync                                        # install
uv run python -m football_agent.data.build     # build football.duckdb from the CSVs
uv run python -m football_agent.features.build # replay matches -> `features` table (~5s)
uv run pytest -q                               # 83 tests
uv run ruff check . && uv run ruff format .    # lint, format
```

## Working with the user

New to git/GitHub and to Claude Code; building this partly to learn both properly, not
just to get a working artifact. Explain the reasoning behind changes, not only the result.
Flag anything surprising found in the data or in upstream code rather than quietly
working around it — that has already turned up one real bug.
