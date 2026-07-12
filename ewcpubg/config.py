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
