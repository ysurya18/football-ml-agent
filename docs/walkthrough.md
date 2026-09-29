# football-ml-agent: the story so far

A plain-language walkthrough of everything built so far, from the raw dataset to the
feature pipeline. Read it top to bottom once, then use the section headings to come back
to specific parts.

---

## 1. The idea in one paragraph

You ask a question in plain English, e.g. *"Who wins Spain vs Brazil at a neutral
venue?"* A machine-learning model (XGBoost) that has studied 150 years of international
football works out the probabilities. Claude, the language model, never makes up a
number. Its job is to choose which part of the model to run, run it, and explain the
answer in words.

```
 You: "Who wins Spain vs Brazil?"
        │
        ▼
 ┌──────────────┐   picks a tool   ┌──────────────┐   asks   ┌──────────────┐
 │ Claude agent │ ───────────────► │  Tool layer  │ ───────► │ XGBoost model│
 └──────────────┘                  └──────────────┘          └──────┬───────┘
        ▲                                                           │ needs
        │ explains in words                                         ▼
        │                                                   ┌──────────────┐
        └─────────────── "Spain 57% · Draw 25% · ..." ◄──── │   Features   │ ◄── DuckDB ◄── CSVs
                                                            └──────────────┘
```

We build from the bottom of that diagram upwards. **Done so far: CSVs → DuckDB → Features.**

---

## 2. The dataset

Four CSV files in `data/raw/`, from a public Kaggle dataset (public domain, so it's fine
to commit them to the repo):

| File | What's in it | Size |
|---|---|---|
| `results.csv` | Every men's international match: date, teams, score, tournament, city, neutral venue or not | 49,287 matches, 1872 → 2026 |
| `goalscorers.csv` | Who scored, in which minute, penalty or own goal | 47,601 goals |
| `shootouts.csv` | Penalty shootout winners | 675 shootouts |
| `former_names.csv` | Old country names (Dahomey → Benin, etc.) | small |

### Four facts about the data that matter

1. **72 matches haven't been played yet.** They're the 2026 World Cup group stage, with
   the score written as `NA`. We keep them, because they're exactly the matches we want to
   predict, but they must *never* be used for training (the model can't learn from a
   result that doesn't exist).
2. **The data stops at 31 March 2026.** If we download a newer copy later, those 72
   matches will have real results: a perfect test the model has never seen.
3. **"Home team wins" is right 48.4% of the time.** So a model that always says "home
   win" already scores 48.4%. Any accuracy number means nothing unless you compare it to
   this *baseline*.
4. **About 60% accuracy is the realistic ceiling.** Draws are close to random, and team
   sheets, injuries, red cards and weather aren't in the data. A model claiming 70% has
   almost certainly cheated by accident (see "leakage" below).

---

## 3. Decisions made before writing any code

The project is based on Oracle's *soccer-analytics-agent* workshop, but it's a
**rewrite, not a copy**. Three decisions:

| Upstream (Oracle's version) | Ours | Why |
|---|---|---|
| Oracle Database (vector store, memory layers) | **DuckDB**: one file, no server | About 60% of upstream's complexity served Oracle's marketing, not the goal |
| Grok via Oracle Cloud | **Claude** | The workshop's key expired; Claude supports real tool calling |
| Agent and logic mixed together | **Tools are plain, tested Python functions** | The logic lives in code we can test; the agent is a thin layer on top |

Upstream's feature code is kept in `reference/` **unchanged**, so we can always compare
against it. We read it and never edit it.

---

## 4. What we built, step by step

Here is the project history as git shows it:

```
f414aa9  Scaffold project: data, licensing, package layout
576bf32  Lock dependencies
ac2d0df  Add DuckDB loader and fix two tournament-classification defects   ← PR #1
37d9a3c  Add CLAUDE.md so a session in this directory starts with context
         Grade the Gold Cup as a continental championship                 ← PR #2
         Add the feature pipeline: 92 pre-match features                   ← PR #3
```

### Step 1: Scaffold and lock dependencies

- Set up the folder structure (`football_agent/data`, `features`, `model`, `tools`,
  `agent`), the licence, and the attribution to Oracle (`NOTICE.md`).
- `uv` installs Python and the libraries. **`uv.lock`** records the *exact* version of
  every library, so anyone who clones the repo gets the same setup you have. "Lock
  dependencies" means committing that file.

### Step 2: The DuckDB loader (PR #1)

**What:** `uv run python -m football_agent.data.build` reads the CSVs and builds one
database file, `football.duckdb`, with tidy tables: `matches`, `goals`, `shootouts`,
`former_names`, `tournament_tiers`.

It also adds useful columns to every match:
- `outcome`: `home_win`, `draw` or `away_win`. This is what the model learns to predict,
  and it's empty for the 72 unplayed matches (that's what keeps them out of training).
- `tournament_tier` and `k_factor`: how important the match is (explained next).

**Elo ratings and K-factors, simply:** every team has a rating, starting at 1500. Beat
a stronger team and you gain a lot; beat a weaker one and you gain a little. The
**K-factor** sets how far a single result moves the rating:

| Tier | K | Example |
|---|---|---|
| World Cup finals | 60 | FIFA World Cup |
| Continental finals | 50 | Euro, Copa América, AFCON |
| Qualifiers & other competitive | 40 | World Cup qualification |
| Friendly | 20 | Friendly |

**Two bugs found in upstream's code:**
1. **Order of checks.** Upstream asked "does the name contain *FIFA World Cup*?" *before*
   asking "is it a qualifier?". So "FIFA World Cup **qualification**" was graded as a World
   Cup final. That's 30% of all matches with the wrong K-factor.
2. **Accents.** Upstream looked for "Copa America", but the data says "Copa Am**é**rica",
   so 877 matches were missed. Now names are compared without accents.

Both have regression tests: tests that fail if the bug ever comes back.

### Step 3: CLAUDE.md

A file Claude Code reads automatically at the start of every session: the decisions, the
data traps and the conventions. It means each new session doesn't start from zero.

### Step 4: The Gold Cup fix (PR #2)

Before building features, I checked every tournament name in our list against the real
data. One matched nothing: we looked for "CONCACAF Gold Cup", but the data just says
"**Gold Cup**". So 420 Gold Cup finals were graded as qualifiers.

**Lesson:** the old test passed because it used the same wrong name. It tested an input
that never actually appears. The new test checks every name against the real database.

### Step 5: The feature pipeline (PR #3)

This is the biggest piece so far.

#### What is a "feature"?

The model can't read "Spain vs Brazil". It needs numbers. A **feature** is one number
that describes a match *before it's played*, such as "Spain's Elo rating", "Brazil's win
rate over their last 10 matches" or "altitude of the stadium". Each match gets **92**
of these, grouped into families:

| Family | # | In plain words |
|---|---|---|
| Elo | 8 | How strong each team is overall, and in this type of tournament |
| Form | 11 | Results over the last 5, 10, 20 matches |
| Goals | 8 | Goals scored and conceded recently |
| Head-to-head | 3 | How these two teams have done against each other |
| Context | 10 | Neutral venue? World Cup? Friendly? Days of rest? Experience? |
| Goalscorer | 12 | Does the team rely on one star? Score late? Many penalties? |
| Momentum | 16 | Winning streaks, clean sheets, comebacks, blowouts |
| Poisson | 8 | A simple statistical estimate of expected goals and win chance |
| Venue | 5 | Altitude, same confederation or not |
| Tournament | 11 | Form in World Cups vs friendlies ("big-game" teams) |

#### The golden rule: no peeking at the future

The worst mistake in sports prediction is **leakage**: accidentally letting a match's
own result (or any later result) into its features. The model then looks brilliant in
testing and fails in real life.

So the pipeline **replays history in date order**, like watching every match since 1872
in sequence:

```
for each match, oldest first:
    1. READ    → write down the 92 numbers, using only what happened before today
    2. RECORD  → save that row
    3. UPDATE  → now let the trackers learn today's result
```

Think of a journalist writing the preview **before** kickoff, then updating their notes
**after** the final whistle. The preview can never mention the final score.

The unplayed 2026 matches get step 1 and 2 (a preview) but never step 3 (there's no
result to learn from).

#### The key design choice

Upstream did "read" and "update" in **one** function. We split them:

- `features(home, away, context)` only *reads*. It can be called a million times and
  changes nothing.
- `update(...)` is the only thing that changes the trackers.

That split gives us three things:
1. **No leakage.** The preview is always written before the result is applied.
2. **What-if questions work.** "What if Spain v Brazil were in Bogotá?" just reads the
   final state with a different city.
3. **No mismatch.** Training rows and live predictions come from the *same* function, so
   they can't quietly differ.

#### Checking against upstream

I ran upstream's own code on the same data and compared all 92 columns:

- **76 were identical** to the last decimal, so the rewrite reproduces upstream faithfully.
- **16 differed**, and only where we fixed a bug on purpose:

| Upstream bug | Plain explanation |
|---|---|
| Own goals | When a defender scores in their own net, upstream gave the goal to the wrong team when working out "who scored first" |
| Comebacks | Two-thirds of matches have no minute-by-minute goal data; upstream then guessed "conceded first" whenever a team conceded at all |
| Poisson | A shortcut cache made results depend on the *order* matches were processed; it also ignored scores above 7 goals |
| Altitude | "Bogota" vs "Bogotá": the accent bug again |
| Confederation | Upstream's hand-written list covered few teams, and treated two *unknown* teams as the same confederation. We now work it out from which tournaments each team plays (96% coverage, up from 36%) |
| Tournament stage | Any name containing "copa" or "euro" counted as a major final, including tiny cups like Copa Newton |

#### Tests

We went from 32 to **86 tests** (32 before, +3 from the Gold Cup fix, +51 for features).
The most important one is the **leakage test**: change the score of match *k* and of
every later match, then check that the rows up to *k* are unchanged. To make sure this
test actually works, I deliberately broke the code (swapped "read" and "update") and
confirmed the test failed. A test that can't fail is useless.

> Correction: PR #3's description says "35 → 83". It should say 32 → 83 on that branch,
> which becomes 86 now that both PRs are merged.

---

## 5. The most important finding so far

Picking the team with the higher Elo rating, **one feature, no machine learning**, is
right **60.5%** of the time on matches since 2020.

That's already about the accuracy we expected from the full model. So the real question
for the next step isn't "does the model beat 48.4%?" but **"does it beat 60.5%?"** If 92
features and XGBoost can't beat one Elo comparison, they aren't adding anything.

---

## 6. How we work with git

```
main ──●────────●──────────────●───────●──►   (always working)
        \                     /       /
         feat/feature-pipeline ●─────/        ← your review happens here (PR)
          \                         /
           fix/gold-cup-tier ●─────           ← one change per branch
```

1. **Branch:** make a separate copy of the code for one change (`feat/...`, `fix/...`).
2. **Commit:** save a snapshot, with a message explaining *why*.
3. **Push:** upload the branch to GitHub.
4. **Pull request (PR):** ask for the change to be merged; you review it.
5. **Squash merge:** the whole branch becomes one tidy commit on `main`.

**When you come back to the project:**

```bash
cd /Users/Socceragent
git checkout main
git pull                                         # download anything new from GitHub
uv sync                                          # make sure libraries are installed
uv run python -m football_agent.data.build       # rebuild the database
uv run python -m football_agent.features.build   # rebuild the features
uv run pytest -q                                 # expect 86 passed
```

`football.duckdb` isn't on GitHub (it's generated), which is why you rebuild it.

---

## 7. Where things live

```
data/raw/                      the 4 CSVs (committed)
football.duckdb                the database (generated, not committed)
football_agent/
  data/build.py                CSVs → DuckDB
  data/tournaments.py          tournament tiers and K-factors
  features/trackers.py         one "notebook" per feature family (Elo, form, ...)
  features/venue.py            altitude and confederations
  features/pipeline.py         the replay: read → record → update
  features/build.py            writes the `features` table
  model/ tools/ agent/         empty so far, coming next
reference/                     upstream's original code (read-only)
tests/                         86 tests
CLAUDE.md                      context for Claude Code sessions
```

---

## 8. Lessons worth keeping

- **Check your inputs against the real data.** Three of the bugs (accents, Gold Cup,
  Bogotá) came from names typed from memory that didn't match the data.
- **A test must be able to fail.** Break the code on purpose and watch the test catch it.
- **Always compare to a baseline.** 60% sounds bad until you know "always home" gets
  48%, and sounds good until you know Elo alone gets 60.5%.
- **Read before you write.** Build each preview only from the past, then learn the result.
- **Stale caches can lie.** Python once kept running an old version of a file after it
  was restored. Deleting `__pycache__` fixed it.

---

## 9. What's next

1. **Train XGBoost.** Train on matches before 2020, test on 2020 onwards (splitting by
   *time*, never randomly, which would let the model see the future). Report accuracy
   against both baselines (48.4% and 60.5%).
2. **Calibration check.** When the model says "70%", does that team actually win about
   70% of the time?
3. **Tool layer:** `predict_match`, `explain_prediction`, `what_if`, ...
4. **Claude agent + chat CLI**, then a **web API**.

---

## Glossary

| Term | Meaning |
|---|---|
| **Feature** | One number describing a match before kickoff |
| **Elo** | A strength rating that goes up when you win and down when you lose |
| **K-factor** | How much one match moves an Elo rating |
| **Leakage** | Accidentally letting future information into the model: fake-good results |
| **Baseline** | The score of a dumb strategy that any real model must beat |
| **Time split** | Train on the past, test on the future, just as it will be used |
| **Regression test** | A test that stops a fixed bug from coming back |
| **Upstream** | The original project ours is based on (Oracle's workshop) |
| **DuckDB** | A database that is just one file, with no server needed |
| **PR (pull request)** | A request to merge a branch, and the place where review happens |
| **Poisson** | A simple statistical model of how many goals a team scores |
