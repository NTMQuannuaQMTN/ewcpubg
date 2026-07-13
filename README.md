# ewcpubg

Predicts EWC 2026 PUBG team ranks from historical tournament performance,
using Twire stats/power-ranking data and Liquipedia rosters.

## Layout

- `ewcpubg/config.py` — API endpoints, credentials, tournament list, weights
- `ewcpubg/normalize.py` — team/player name normalization, weight lookups
- `ewcpubg/twire_stats.py` — Twire GraphQL client, shard discovery, team/player stats crawl
- `ewcpubg/twire_assets.py` — Twire S3 team-ranking / power-ranking JSON
- `ewcpubg/liquipedia.py` — Liquipedia EWC 2026 roster scrape
- `ewcpubg/features.py` — weighted feature aggregation (team + player -> per-team history)
- `ewcpubg/model.py` — RandomForest rank predictor
- `ewcpubg/pipeline.py` — wires everything together end to end
- `notebooks/ewcpubg.ipynb` — thin exploratory notebook on top of the package
- `output/` — generated CSVs (gitignored)

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in TWIRE_API_KEY
python -m ewcpubg.pipeline
```

`TWIRE_API_KEY` is read from the environment (via `.env`, gitignored) rather
than hardcoded, so it never ends up committed. On Kaggle, set it under
Add-ons -> Secrets instead and load it into `os.environ["TWIRE_API_KEY"]`
before running the notebook's cells.

This writes `output/team_stats.csv`, `output/player_stats.csv`,
`output/ewc2026_rosters.csv`, `output/ewc_history_features.csv`, and prints
the predicted rank table.

For interactive exploration, open `notebooks/ewcpubg.ipynb`.



================================================
FILE: requirements.txt
================================================
requests
pandas
numpy
tqdm
beautifulsoup4
scikit-learn
matplotlib
jupyter
python-dotenv



================================================
FILE: ewcpubg/__init__.py
================================================
[Empty file]


================================================
FILE: ewcpubg/config.py
================================================
"""Constants: API endpoints, credentials, tournament metadata, weights."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# --- Twire GraphQL API --------------------------------------------------------

API_URL = "https://tjjkdyimqrb7jjnc6m5rpefjtu.appsync-api.eu-west-1.amazonaws.com/graphql"

TWIRE_API_KEY = os.environ.get("TWIRE_API_KEY")

if not TWIRE_API_KEY:
    raise RuntimeError(
        "TWIRE_API_KEY is not set. Copy .env.example to .env in the project "
        "root and fill in the key, or export TWIRE_API_KEY in your shell/"
        "Kaggle secrets."
    )

HEADERS = {
    "x-api-key": TWIRE_API_KEY,
    "Content-Type": "application/json",
    "Origin": "https://twire.gg",
    "Referer": "https://twire.gg/",
}

GAME = "pubg"

# --- Twire static assets (S3) -------------------------------------------------

TEAM_RANKING_URL = "https://twire-assets.s3.eu-west-1.amazonaws.com/pubg/team-ranking/team-ranking.json"
POWER_RANKING_URL = "https://twire-assets.s3.eu-west-1.amazonaws.com/pubg/power-ranking/power-ranking.json"
POWER_RANKING_YEAR = "2026"

# Power-ranking tournament tiers: keyed by substring of the ranking's
# tournament slug (e.g. "2026-pgs-1", "pcl-26-spring", "enc-player-ranking").
POWER_TOURNAMENT_WEIGHTS = {
    "pgs": 1.0,
    "enc": 1.0,
}
POWER_TOURNAMENT_DEFAULT_WEIGHT = 0.6  # regional: pms, pec, pvs, pas, pts, pws, pcl

# --- Liquipedia ----------------------------------------------------------------

LIQUIPEDIA_URL = "https://liquipedia.net/pubg/Esports_World_Cup/2026"

LIQUIPEDIA_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TEAM_NAME_MAP = {
    "AGAL International": "Anyone's Legend",
    "GAM The Expendables": "The Expendables",
    "FULL SENSE": "Full Sense",
    "R8 Esports": "VEGA ESPORTS",
    "Shadow Esport": "SHADOW ESPORT",
    "Sharper Esports": "Sharper Esport",
    "Team Nemesis": "TEAM NEMESIS"
}

# --- Tournaments -----------------------------------------------------------------

TOURNAMENTS = {
    "PGS1": {
        "id": 2473,  # Replace with actual tournament ID from tournaments.csv
        "name": "2026 PGS 1",
        "year": 2026,
        "uuid": "66bbb080-2224-11f1-8444-6adabec81c44",
    },
    "PGS2": {
        "id": 2488,  # Replace with actual ID
        "name": "2026 PGS 2",
        "year": 2025,
        "uuid": "4702993a-28f4-11f1-9125-6adabec81c44",
    },
    "PGS3": {
        "id": 2500,  # Replace with actual ID
        "name": "2026 PGS 3",
        "year": 2025,
        "uuid": "13d20004-2e72-11f1-9dc9-6adabec81c44",
    },
    "PGS4": {
        "id": 2543,  # Replace with actual ID
        "name": "2026 PGS 4",
        "year": 2025,
        "uuid": "31054c04-540a-11f1-af22-6adabec81c44",
    },
    "PGS5": {
        "id": 2551,
        "name": "2026 PGS 5",
        "year": 2026,
        "uuid": "9c951222-5a2e-11f1-9004-6adabec81c44",
    },
    "PGS6": {
        "id": 2550,  # Replace with actual tournament ID from tournaments.csv
        "name": "2026 PGS 6",
        "year": 2026,
        "uuid": "9c702f34-5a2e-11f1-8257-6adabec81c44",
    },
    "PWS": {
        "id": 2538,
        "name": "PUBG WEEKLY SERIES 2026 Phase 1",
        "year": 2026,
        "uuid": "897c012e-4b0a-11f1-8f54-6adabec81c44",
    },
    "PCL": {
        "id": 2536,
        "name": "PUBG Champions League 2026 - Spring",
        "year": 2026,
        "uuid": "8c1c3332-4a3c-11f1-a2c0-6adabec81c44",
    },
    "PEC": {
        "id": 2512,
        "name": "PEC: Spring Playoffs & Finals",
        "year": 2026,
        "uuid": "709f733c-3a81-11f1-8bfc-6adabec81c44",
    },
    "PTS": {
        "id": 2530,
        "name": "PUBG Thailand Series 2026 - Phase 1",
        "year": 2026,
        "uuid": "513a00d2-42ee-11f1-bab5-6adabec81c44",
    },
    "PAS": {
        "id": 2513,
        "name": "PAS1 Playoffs & Finals",
        "year": 2026,
        "uuid": "c2e622ae-3aad-11f1-93de-6adabec81c44",
    },
    "PVS": {
        "id": 2510,
        "name": "PVS 2026 Phase 1",
        "year": 2026,
        "uuid": "afc03272-3810-11f1-b6ee-6adabec81c44",
    },
    "PMS": {
        "id": 2400,
        "name": "PUBG Master Series 2026: Phase 1",
        "year": 2026,
        "uuid": "ec23e36e-dc10-11f0-85d2-064f26ad4164",
    },
}

# Tournament importance, keyed by each tournament's actual `name` (matches
# team_df["tournament"] / the Twire API's tournamentName) rather than its
# short code -- short codes like "PMS"/"PWS"/"PCL"/"PTS" don't appear as
# substrings of the real tournament names ("PUBG Master Series 2026: Phase 1"
# etc.), so keying on them silently fell through to the 1.0 default weight
# for those tournaments. Deriving from TOURNAMENTS[key]["name"] keeps this
# in sync automatically.
#
# Two tiers only: PGS1-6 are the only "global" events we track (weight 4.0);
# everything else (PWS, PCL, PEC, PTS, PAS, PVS, PMS) is regional (1.5).
GLOBAL_TOURNAMENT_WEIGHT = 4.0
REGIONAL_TOURNAMENT_WEIGHT = 1.5

_PGS_WEIGHTS = {
    "PGS1": GLOBAL_TOURNAMENT_WEIGHT,
    "PGS2": GLOBAL_TOURNAMENT_WEIGHT,
    "PGS3": GLOBAL_TOURNAMENT_WEIGHT,
    "PGS4": GLOBAL_TOURNAMENT_WEIGHT,
    "PGS5": GLOBAL_TOURNAMENT_WEIGHT,
    "PGS6": GLOBAL_TOURNAMENT_WEIGHT,
}

TOURNAMENT_WEIGHTS = {
    TOURNAMENTS[key]["name"]: _PGS_WEIGHTS.get(key, REGIONAL_TOURNAMENT_WEIGHT)
    for key in TOURNAMENTS
}

# Tournament keys (short codes) considered "global" vs "regional" -- used to
# build per-team appearance counts (global_events, regional_events, etc).
GLOBAL_TOURNAMENT_KEYS = set(_PGS_WEIGHTS)
REGIONAL_TOURNAMENT_KEYS = set(TOURNAMENTS) - GLOBAL_TOURNAMENT_KEYS

# Bayesian shrinkage pseudo-count: how many "league-average" samples a team's
# own history has to outweigh before its raw average is trusted at face value.
# Higher k = more skepticism toward small samples.
SHRINKAGE_K = 5

# Stage importance
STAGE_WEIGHTS = {
    "Grand Finals": 1.30,
    "Final Stage": 1.30,
    "Winners Stage": 1.15,
    "Group Stage": 1.00,
    "Survival Stage": 0.95,
}

# --- Output --------------------------------------------------------------------
# Anchored to the project root (not the caller's cwd) so it lands in the same
# place whether run as `python -m ewcpubg.pipeline` from the repo root or from
# a notebook under notebooks/.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "output")



================================================
FILE: ewcpubg/features.py
================================================
"""Feature engineering: competition-strength-aware team/player aggregates.

Pipeline: raw stats -> tournament_strength/SOS/Twire-rating context ->
Bayesian-shrunk, strength-adjusted team history -> model-ready features.
"""

import numpy as np
import pandas as pd

from .config import GLOBAL_TOURNAMENT_KEYS, TOURNAMENTS, SHRINKAGE_K
from .normalize import normalize_team

GLOBAL_TOURNAMENT_NAMES = {TOURNAMENTS[key]["name"] for key in GLOBAL_TOURNAMENT_KEYS}


def wavg(x, value, weight_col="weight"):
    return np.average(x[value], weights=x[weight_col])


def build_player_features(player_df):
    return (
        player_df
        .groupby(["tournament", "stage", "teamName"])
        .apply(lambda x: pd.Series({
            "avg_kd": wavg(x, "kd"),
            "avg_damage": wavg(x, "damageDealt"),
            "avg_kills": wavg(x, "kills"),
            "avg_assists": wavg(x, "assists"),
            "avg_kas": wavg(x, "kas"),

            "max_damage": x["damageDealt"].max(),
            "max_kd": x["kd"].max(),

            "roster_size": x["username"].nunique()
        }))
        .reset_index()
    )


def build_dataset(team_df, player_features):
    return team_df.merge(
        player_features,
        left_on=["tournament", "stage", "team"],
        right_on=["tournament", "stage", "teamName"],
        how="left"
    )


def build_history(dataset, ewc_teams):
    return dataset[
        dataset["team"].isin(ewc_teams)
    ].copy()


def build_twire_team_ratings(team_rank_df, dataset):
    """Twire's official team ranking, scaled to (0, 1], as a prior on team
    quality (spec section 7). Teams with no official ranking get an estimate
    from their own regional results -- but capped strictly below the
    weakest *officially ranked* team, so a team that only ever dominated a
    weak regional field can never read as elite just because Twire hasn't
    rated it yet.
    """

    # Match on normalize_team(...) rather than the raw "team" column: dataset
    # still carries Twire's raw team names at this point in the pipeline
    # (normalization happens later, to match the original notebook's exact
    # sequencing), so joining on the raw column would silently miss ranked
    # teams like "17 Gaming" (raw) vs "17Gaming" (normalized, as team_rank_df
    # uses) and wrongly treat them as unranked.
    max_ranking = team_rank_df["ranking"].max()
    ranked = dict(zip(team_rank_df["team"], team_rank_df["ranking"] / max_ranking))
    min_ranked_score = min(ranked.values())

    normalized_team = dataset["team"].apply(normalize_team)
    all_teams = normalized_team.dropna().unique()
    unranked_teams = [t for t in all_teams if t not in ranked]

    if not unranked_teams:
        return ranked

    proxies = {}
    for team in unranked_teams:
        sub = dataset[normalized_team == team]
        avg_rank = np.average(sub["avgRank"], weights=sub["weight"])
        # better (lower) average finish -> higher proxy
        proxies[team] = 1 / (1 + avg_rank)

    values = np.array(list(proxies.values()))
    span = (values.max() - values.min()) or 1.0

    for team, proxy in proxies.items():
        normalized = (proxy - values.min()) / span  # 0..1 within the unranked pool
        ranked[team] = normalized * min_ranked_score * 0.9  # strictly below the weakest ranked team

    return ranked


def attach_competition_context(dataset, twire_team_ratings):
    """Per-row competition-strength columns (spec sections 2-4): a team's own
    Twire rating, the field strength of that specific tournament+stage
    (tournament_strength), and the average opponent strength excluding the
    team itself (strength_of_schedule)."""

    df = dataset.copy()
    df["twire_team_rating"] = df["team"].apply(normalize_team).map(twire_team_ratings).fillna(0)

    stage_key = ["tournament", "stage"]
    stage_sum = df.groupby(stage_key)["twire_team_rating"].transform("sum")
    stage_count = df.groupby(stage_key)["twire_team_rating"].transform("count")

    df["tournament_strength"] = stage_sum / stage_count
    df["strength_of_schedule"] = (stage_sum - df["twire_team_rating"]) / (stage_count - 1).clip(lower=1)

    df["is_global_tournament"] = df["tournament"].isin(GLOBAL_TOURNAMENT_NAMES)

    return df


def _shrink(raw, n, league_avg, k=SHRINKAGE_K):
    """Bayesian shrinkage (spec section 6): pull small-sample team averages
    toward the league-wide average until the team has enough of its own
    evidence (n >= k) to be trusted at face value."""

    return (raw * n + league_avg * k) / (n + k)


def _team_history_row(x, league_avg, global_average_strength):
    n = len(x)

    # Empirical field-strength multiplier: how strong was the competition
    # this team actually faced, relative to the season average -- replaces
    # a flat "PGS vs regional" bucket with each stage's real participant
    # quality (spec section 2).
    context = np.average(x["tournament_strength"], weights=x["weight"]) / global_average_strength

    def adjusted(col):
        raw = np.average(x[col], weights=x["weight"]) * context
        return _shrink(raw, n, league_avg[col] * context)

    global_events = x.loc[x["is_global_tournament"], "tournament"].nunique()
    regional_events = x.loc[~x["is_global_tournament"], "tournament"].nunique()
    international_matches = int(x["is_global_tournament"].sum())
    regional_matches = int((~x["is_global_tournament"]).sum())

    return pd.Series({

        "tournaments": x["tournament"].nunique(),
        "stages": n,

        "global_events": global_events,
        "regional_events": regional_events,
        "international_matches": international_matches,
        "regional_matches": regional_matches,
        "experience_score": np.log1p(n) + 2 * global_events,

        "twire_team_rating": x["twire_team_rating"].iloc[0],
        "strength_of_schedule": np.average(x["strength_of_schedule"], weights=x["weight"]),
        "tournament_strength": np.average(x["tournament_strength"], weights=x["weight"]),

        "avg_rank": _shrink(np.average(x["avgRank"], weights=x["weight"]), n, league_avg["avgRank"]),
        "std_rank": x["avgRank"].std() if n > 1 else league_avg["std_rank_prior"],

        "avg_points": adjusted("totalPoints"),
        "std_points": x["totalPoints"].std() if n > 1 else 0,

        "avg_kills": adjusted("kills"),
        "std_kills": x["kills"].std() if n > 1 else 0,

        "avg_damage": adjusted("damageDealt"),
        "std_damage": x["damageDealt"].std() if n > 1 else 0,

        "avg_damage_taken": adjusted("damageTaken"),

        "total_wins": (x["wins"] * x["weight"]).sum() * context,

        "player_avg_kd": adjusted("avg_kd"),
        "player_max_kd": adjusted("max_kd"),

        "player_avg_damage": adjusted("avg_damage"),
        "player_max_damage": adjusted("max_damage"),

        "player_avg_kills": adjusted("avg_kills"),
        "player_avg_assists": adjusted("avg_assists"),

        "player_avg_kas": adjusted("avg_kas"),

    })


def build_history_features(history, history_power, full_dataset):
    """`history` is EWC-teams-only (what we're predicting for); `full_dataset`
    is the whole competitive scene, used only to compute the league-average
    shrinkage prior so a handful of extreme EWC-team outliers can't distort
    their own baseline."""

    magnitude_cols = [
        "avgRank", "totalPoints", "kills", "damageDealt", "damageTaken",
        "avg_kd", "max_kd", "avg_damage", "max_damage",
        "avg_kills", "avg_assists", "avg_kas",
    ]
    league_avg = {col: full_dataset[col].mean() for col in magnitude_cols}
    league_avg["std_rank_prior"] = full_dataset["avgRank"].std()

    global_average_strength = full_dataset["tournament_strength"].mean()

    history_features = (
        history
        .groupby("team")
        .apply(lambda x: _team_history_row(x, league_avg, global_average_strength))
        .reset_index()
    )

    history_features = history_features.merge(
        history_power,
        on="team",
        how="left"
    )

    history_features = history_features.fillna(0)

    return history_features



================================================
FILE: ewcpubg/liquipedia.py
================================================
"""Liquipedia scraping: EWC 2026 confirmed rosters."""

import requests
import pandas as pd
from bs4 import BeautifulSoup

from .config import LIQUIPEDIA_URL, LIQUIPEDIA_HEADERS, TEAM_NAME_MAP


def fetch_ewc_rosters(html=None):
    """Scrape confirmed EWC 2026 rosters.

    Pass `html=` with a manually saved copy of the page if Liquipedia's
    Cloudflare challenge is blocking the automated request.
    """

    if html is None:
        html = requests.get(LIQUIPEDIA_URL, headers=LIQUIPEDIA_HEADERS).text

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.team-participant-card")

    if not cards:
        raise RuntimeError(
            "No team-participant-card elements found on the Liquipedia page. "
            "The site is likely blocking the automated request (e.g. a "
            "Cloudflare challenge) rather than the page layout having "
            "changed. Try again later, or open the URL in a browser, save "
            "the rendered HTML, and pass it via fetch_ewc_rosters(html=...)."
        )

    rows = []

    for card in cards:

        team_tag = card.select_one(
            ".team-participant-card__opponent-full .name a"
        )

        team = team_tag.get_text(strip=True)

        players = [
            p.get_text(strip=True)
            for p in card.select(".block-player .name a")
        ]

        for player in players[0:4]:

            rows.append({
                "team": team,
                "player": player
            })

    players_df = pd.DataFrame(rows)

    players_df["tournament"] = "EWC2026"

    players_df = players_df[
        ["tournament", "team", "player"]
    ]

    players_df["team"] = (
        players_df["team"]
        .replace(TEAM_NAME_MAP)
    )

    return players_df



================================================
FILE: ewcpubg/model.py
================================================
"""Rank prediction model: RandomForest over weighted historical team features."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "tournaments",
    "stages",

    "avg_points",
    "std_points",

    # "avg_rank",
    "std_rank",

    "avg_kills",
    "std_kills",

    "avg_damage",
    "std_damage",

    "avg_damage_taken",

    "total_wins",

    "player_avg_kd",
    "player_max_kd",

    "player_avg_damage",
    "player_max_damage",

    "player_avg_kills",
    "player_avg_assists",
    "player_avg_kas",

    "twire_power",
    "twire_peak",
    "attacker_rating",
    "survivor_rating",
    "teammate_rating",
    "utility_rating",
    "finisher_rating",
    "star_players",

    "twire_team_rating",
    "tournament_strength",
    "strength_of_schedule",

    "global_events",
    "regional_events",
    "international_matches",
    "regional_matches",
    "experience_score",
]


def train_model(history_features, features=FEATURES):
    X = history_features[features]
    y = history_features["avg_rank"]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42
    )
    model.fit(X, y)

    return model, X, y


def feature_importance(model, features=FEATURES):
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    return importance.sort_values(
        "importance",
        ascending=False
    )


def predict_ranks(model, ewc_df, features=FEATURES):
    ewc_df = ewc_df.copy()
    ewc_df["predicted_rank"] = model.predict(ewc_df[features])
    return ewc_df


def rank_predictions(ewc_df):
    return (
        ewc_df
        .sort_values("predicted_rank")
        [["team", "predicted_rank"]]
    )



================================================
FILE: ewcpubg/normalize.py
================================================
"""Name normalization and weighting helpers shared across data sources."""

import re

import pandas as pd

from .config import (
    TOURNAMENT_WEIGHTS,
    STAGE_WEIGHTS,
    POWER_TOURNAMENT_WEIGHTS,
    POWER_TOURNAMENT_DEFAULT_WEIGHT,
)


def get_tournament_weight(name):
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in str(name).lower():
            return weight
    return 1.0


def get_stage_weight(name):
    for key, weight in STAGE_WEIGHTS.items():
        if key.lower() in str(name).lower():
            return weight
    return 0.6


def get_power_tournament_weight(tournament):
    for key, weight in POWER_TOURNAMENT_WEIGHTS.items():
        if key.lower() in str(tournament).lower():
            return weight
    return POWER_TOURNAMENT_DEFAULT_WEIGHT


def normalize_team(name):
    if pd.isna(name):
        return name

    name = str(name).strip()

    # remove duplicate spaces
    name = re.sub(r"\s+", " ", name)

    # remove spaces in names beginning with numbers
    # 17 Gaming -> 17Gaming
    name = re.sub(r"^(\d+)\s+", r"\1", name)

    return name


def normalize_player(name):
    if pd.isna(name):
        return name

    name = str(name).strip()

    # Take everything after the first underscore
    if "_" in name:
        name = name.split("_", 1)[1]

    return name



================================================
FILE: ewcpubg/pipeline.py
================================================
"""End-to-end orchestration: crawl -> feature engineer -> train -> predict.

Mirrors the original notebook's data flow exactly (same order of operations,
same weighting/normalization timing), just split across modules. The one
deliberate reordering is `history_power`: it has to exist before it's merged
into `history_features`, and since it only depends on `team_power` (not on
anything computed in between), moving its computation earlier changes
nothing numerically -- it just makes the pipeline runnable top to bottom.
"""

import os

import pandas as pd

from . import config
from .normalize import normalize_team, normalize_player
from .twire_stats import (
    fetch_shard_infos,
    fetch_team_stats_raw,
    fetch_player_stats_raw,
    build_team_df,
    build_player_df,
    apply_weights,
)
from .twire_assets import (
    fetch_team_ranking,
    fetch_power_ranking,
    build_team_power,
    build_history_power,
)
from .liquipedia import fetch_ewc_rosters
from .features import (
    build_player_features,
    build_dataset,
    build_history,
    build_twire_team_ratings,
    attach_competition_context,
    build_history_features,
)
from .model import train_model, feature_importance, predict_ranks, rank_predictions


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Twire tournament stats (GraphQL) ---------------------------------
    shard_infos = fetch_shard_infos(config.TOURNAMENTS)

    team_stats_raw = fetch_team_stats_raw(shard_infos)
    player_stats_raw = fetch_player_stats_raw(shard_infos)

    team_df = build_team_df(team_stats_raw)
    player_df = build_player_df(player_stats_raw)

    team_df = apply_weights(team_df)
    player_df = apply_weights(player_df)

    team_df.to_csv(f"{config.OUTPUT_DIR}/team_stats.csv", index=False)
    player_df.to_csv(f"{config.OUTPUT_DIR}/player_stats.csv", index=False)
    print(team_df.shape)
    print(player_df.shape)

    # --- Twire team/power rankings (S3) ------------------------------------
    team_rank_df = fetch_team_ranking()

    power_df = fetch_power_ranking()
    team_power = build_team_power(power_df)
    history_power = build_history_power(team_power)

    # --- Liquipedia EWC 2026 rosters ----------------------------------------
    # Liquipedia's Cloudflare challenge blocks the scraper often enough that
    # it's worth reusing a previously saved roster rather than re-scraping.
    roster_path = f"{config.OUTPUT_DIR}/ewc2026_rosters.csv"

    if os.path.exists(roster_path):
        print(f"Using cached rosters: {roster_path}")
        players_df = pd.read_csv(roster_path)
    else:
        players_df = fetch_ewc_rosters()
        players_df.to_csv(roster_path, index=False)

    # --- Feature engineering --------------------------------------------------
    team_df = team_df[team_df["map"] == "all"].copy()
    player_features = build_player_features(player_df)
    dataset = build_dataset(team_df, player_features)

    # Twire Team Rating as a competition-quality prior (unranked teams get a
    # capped estimate, never allowed to exceed the weakest officially ranked
    # team), plus per-stage field strength / strength of schedule from it.
    twire_team_ratings = build_twire_team_ratings(team_rank_df, dataset)
    dataset = attach_competition_context(dataset, twire_team_ratings)

    for df, cols in [
        (team_df, ["team"]),
        (player_df, ["teamName"]),
        (players_df, ["team"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].apply(normalize_team)

    player_df["username"] = player_df["username"].apply(normalize_player)
    players_df["player"] = players_df["player"].apply(normalize_player)

    ewc_teams = players_df["team"].unique().tolist()
    print(f"{len(ewc_teams)} EWC teams")

    history = build_history(dataset, ewc_teams)
    print(history.shape)

    history_features = build_history_features(history, history_power, dataset)
    history_features.to_csv(f"{config.OUTPUT_DIR}/ewc_history_features.csv", index=False)

    # --- Model ------------------------------------------------------------------
    model, X, y = train_model(history_features)
    print(feature_importance(model))

    ewc_df = predict_ranks(model, history_features)
    prediction = rank_predictions(ewc_df)

    return prediction


if __name__ == "__main__":
    print(run())



================================================
FILE: ewcpubg/twire_assets.py
================================================
"""Twire's static S3 ranking assets (team ranking + player power ranking)."""

import requests
import pandas as pd

from .config import TEAM_RANKING_URL, POWER_RANKING_URL, POWER_RANKING_YEAR
from .normalize import normalize_team, get_power_tournament_weight

POWER_SCORE_COLUMNS = [
    "overall_score",
    "attacker_score",
    "survivor_score",
    "teammate_score",
    "utility_score",
    "finisher_score",
]


def fetch_team_ranking():
    data = requests.get(TEAM_RANKING_URL).json()["teams"]
    team_rank_df = pd.DataFrame(data)
    team_rank_df["team"] = team_rank_df["name"].apply(normalize_team)
    return team_rank_df


def fetch_power_ranking():
    power_data = requests.get(POWER_RANKING_URL).json()["ranking"][POWER_RANKING_YEAR]

    rows = []

    for tournament, tournament_data in power_data.items():

        for player in tournament_data["players"]:

            rows.append({
                "tournament": tournament,
                "team": player["team"],
                "nickname": player["nickname"],

                "overall_score": player["overallScore"],
                "attacker_score": player["attackerScore"],
                "survivor_score": player["survivorScore"],
                "teammate_score": player["teammateScore"],
                "utility_score": player["utilityScore"],
                "finisher_score": player["finisherScore"]
            })

    power_df = pd.DataFrame(rows)
    power_df["team"] = power_df["team"].apply(normalize_team)

    # Scale each player's scores against that tournament's own top score
    # (so a regional event's rating scale doesn't read as equal to a PGS's),
    # then weight by tournament tier (PGS/ENC 1.0, regional 0.6).
    power_df["tournament_weight"] = power_df["tournament"].apply(get_power_tournament_weight)

    # Weighted columns feed the rating aggregates below; the raw overall_score
    # is kept as-is since star_players thresholds against its natural 0-100 scale.
    for col in POWER_SCORE_COLUMNS:
        tournament_max = power_df.groupby("tournament")[col].transform("max")
        power_df[f"{col}_weighted"] = (power_df[col] / tournament_max) * power_df["tournament_weight"]

    return power_df


def build_team_power(power_df):
    return (
        power_df
        .groupby(["tournament", "team"])
        .agg(
            power_avg=("overall_score_weighted", "mean"),
            power_max=("overall_score_weighted", "max"),

            attacker_avg=("attacker_score_weighted", "mean"),
            survivor_avg=("survivor_score_weighted", "mean"),
            teammate_avg=("teammate_score_weighted", "mean"),
            utility_avg=("utility_score_weighted", "mean"),
            finisher_avg=("finisher_score_weighted", "mean"),

            star_players=("overall_score", lambda x: (x >= 85).sum())
        )
        .reset_index()
    )


def build_history_power(team_power):
    """Sum a team's (already tournament-weighted, tournament-normalized) power
    across every tournament it appeared in, rewarding repeated strong showings
    rather than averaging them away."""

    return (
        team_power
        .groupby("team")
        .agg(
            twire_power=("power_avg", "sum"),
            twire_peak=("power_max", "sum"),
            attacker_rating=("attacker_avg", "sum"),
            survivor_rating=("survivor_avg", "sum"),
            teammate_rating=("teammate_avg", "sum"),
            utility_rating=("utility_avg", "sum"),
            finisher_rating=("finisher_avg", "sum"),
            star_players=("star_players", "sum")
        )
        .reset_index()
    )



================================================
FILE: ewcpubg/twire_stats.py
================================================
"""Twire GraphQL API: tournament shard discovery and team/player stats crawling."""

import requests
import pandas as pd
from tqdm.auto import tqdm

from .config import API_URL, HEADERS, GAME
from .normalize import get_tournament_weight, get_stage_weight

GET_FILTERS = """
query ($id: Int!, $game: String!) {
  tournamentInitialData(id: $id, game: $game) {
    tournament {
      friendlyName
    }
    tournamentFilters {
      name
      value
    }
  }
}
"""

TEAM_STATS_QUERY = """
query ($shardInfo:String!, $game:String!) {
  teamStats(
    shardInfo:$shardInfo,
    token:"",
    filters:null,
    game:$game
  ) {

    tournamentName
    groupName
    matchName

    teamStats {

      teamName
      teamLogo

      stats {

        map
        rank
        kills
        assists
        damageDealt
        damageTaken
        points
        totalPoints
        wins
        avgKills
        avgDamageDealt
        avgRank
        avgPoints
      }
    }
  }
}
"""

PLAYER_STATS_QUERY = """
query ($shardInfo:String!, $game:String!) {
  platformStats(
    tournament:$shardInfo,
    token:"",
    filters:null,
    game:$game
  ) {

    tournamentName
    groupName
    matchName

    leaderboard {

      username
      teamName

      kills
      assists

      kd
      kas

      damageDealt
      damageTaken

      avgDamageDealt
      avgDamageTaken

      deaths

      dbnos

      revives

      headshotKills

      walkDistance
      rideDistance

      longestKill

      avgTimeSurvived

      numOfMatches
    }
  }
}
"""


def graphql(query, variables=None):
    payload = {
        "query": query,
        "variables": variables or {}
    }

    r = requests.post(API_URL, headers=HEADERS, json=payload)

    print("Status:", r.status_code)

    try:
        return r.json()
    except Exception:
        print(r.text)
        return None


def fetch_shard_infos(tournaments):
    """Resolve each tournament's stage filters into GraphQL shardInfo strings."""

    all_filters = {}

    for key, t in tournaments.items():

        result = graphql(GET_FILTERS, {"id": t["id"], "game": GAME})

        if result is None or "errors" in result:
            print("❌ Error")
            continue

        if result["data"]["tournamentInitialData"] is None:
            print("❌ No tournamentInitialData")
            continue

        all_filters[key] = result["data"]["tournamentInitialData"]["tournamentFilters"]

    shard_infos = {}

    for tournament, filters in all_filters.items():

        uuid = tournaments[tournament]["uuid"]

        shard_infos[tournament] = []

        for f in filters:

            shard_infos[tournament].append({
                "stage": f["name"],
                "value": f["value"],
                "shardInfo": f"{uuid}-{f['value']}"
            })

    return shard_infos


def fetch_team_stats_raw(shard_infos):
    team_stats_raw = []

    for tournament, stages in tqdm(shard_infos.items()):

        for stage in stages:

            data = graphql(
                TEAM_STATS_QUERY,
                {
                    "shardInfo": stage["shardInfo"],
                    "game": GAME
                }
            )

            team_stats_raw.append({
                "tournament": tournament,
                "stage": stage["stage"],
                "data": data["data"]["teamStats"]
            })

    return team_stats_raw


def fetch_player_stats_raw(shard_infos):
    player_stats_raw = []

    for tournament, stages in tqdm(shard_infos.items()):

        for stage in stages:

            data = graphql(
                PLAYER_STATS_QUERY,
                {
                    "shardInfo": stage["shardInfo"],
                    "game": GAME
                }
            )

            player_stats_raw.append({
                "tournament": tournament,
                "stage": stage["stage"],
                "data": data["data"]["platformStats"]
            })

    return player_stats_raw


def build_team_df(team_stats_raw):
    rows = []

    for tournament in team_stats_raw:

        info = tournament["data"]

        for team in info["teamStats"]:

            for stat in team["stats"]:

                row = {
                    "tournament": info["tournamentName"],
                    "stage": info["groupName"],
                    "team": team["teamName"],
                    **stat
                }

                rows.append(row)

    return pd.DataFrame(rows)


def build_player_df(player_stats_raw):
    rows = []

    for tournament in player_stats_raw:

        info = tournament["data"]

        for player in info["leaderboard"]:

            player["tournament"] = info["tournamentName"]
            player["stage"] = info["groupName"]

            rows.append(player)

    return pd.DataFrame(rows)


def apply_weights(df):
    """Attach tournament/stage importance weights (used by all weighted averages)."""

    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)
    df["stage_weight"] = df["stage"].apply(get_stage_weight)
    df["weight"] = (
        df["tournament_weight"]
        * df["stage_weight"]
    )
    return df


