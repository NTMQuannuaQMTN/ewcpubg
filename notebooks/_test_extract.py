import os
import re
import unicodedata
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns", None)


# --- Twire GraphQL API --------------------------------------------------------

import os

try:
    from dotenv import load_dotenv
    load_dotenv("../.env")
except ImportError:
    pass

API_URL = "https://tjjkdyimqrb7jjnc6m5rpefjtu.appsync-api.eu-west-1.amazonaws.com/graphql"

# Local: copy ../.env.example to ../.env and fill in the key.
# Kaggle: Add-ons -> Secrets -> TWIRE_API_KEY, then os.environ["TWIRE_API_KEY"] = user_secrets.get_secret("TWIRE_API_KEY")
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
_PGS_WEIGHTS = {
    "PGS1": 0.80,
    "PGS2": 0.90,
    "PGS3": 1.00,
    "PGS4": 1.10,
    "PGS5": 1.20,
    "PGS6": 1.30,
}
_REGIONAL_DEFAULT_WEIGHT = 0.60

TOURNAMENT_WEIGHTS = {
    TOURNAMENTS[key]["name"]: _PGS_WEIGHTS.get(key, _REGIONAL_DEFAULT_WEIGHT)
    for key in TOURNAMENTS
}

# 6 individual PGS tiers (0.80-1.30) + 1 shared regional tier (0.60) = 7.
TOTAL_TOURNAMENT_TIERS = len(set(TOURNAMENT_WEIGHTS.values()))

# Stage importance
STAGE_WEIGHTS = {
    "Grand Finals": 1.30,
    "Final Stage": 1.30,
    "Winners Stage": 1.15,
    "Group Stage": 1.00,
    "Survival Stage": 0.95,
}


# --- Output --------------------------------------------------------------------
# "../output" so it lands in the project-root output/ dir, not notebooks/output/
# (Jupyter's cwd is wherever this notebook file lives).

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

shard_infos = fetch_shard_infos(TOURNAMENTS)
shard_infos.keys()

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


team_stats_raw = fetch_team_stats_raw(shard_infos)
len(team_stats_raw)

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


player_stats_raw = fetch_player_stats_raw(shard_infos)
len(player_stats_raw)

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


team_df = build_team_df(team_stats_raw)
team_df.shape



def build_player_df(player_stats_raw):
    rows = []

    for tournament in player_stats_raw:

        info = tournament["data"]

        for player in info["leaderboard"]:

            player["tournament"] = info["tournamentName"]
            player["stage"] = info["groupName"]

            rows.append(player)

    return pd.DataFrame(rows)


player_df = build_player_df(player_stats_raw)
player_df.shape

def apply_weights(df):
    """Attach tournament/stage importance weights (used by all weighted averages)."""

    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)
    df["stage_weight"] = df["stage"].apply(get_stage_weight)
    df["weight"] = (
        df["tournament_weight"]
        * df["stage_weight"]
    )
    return df


team_df = apply_weights(team_df)
player_df = apply_weights(player_df)

team_df[["tournament", "stage", "weight"]].drop_duplicates().sort_values(["tournament", "stage"])

team_df.to_csv(f"{OUTPUT_DIR}/team_stats.csv", index=False)
player_df.to_csv(f"{OUTPUT_DIR}/player_stats.csv", index=False)

print(team_df.shape)
print(player_df.shape)

def fetch_team_ranking():
    data = requests.get(TEAM_RANKING_URL).json()["teams"]
    return pd.DataFrame(data)


team_rank_df = fetch_team_ranking()  # not used downstream, kept for parity with the original notebook
team_rank_df.head()

POWER_SCORE_COLUMNS = [
    "overall_score",
    "attacker_score",
    "survivor_score",
    "teammate_score",
    "utility_score",
    "finisher_score",
]


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


power_df = fetch_power_ranking()
power_df.head()


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


team_power = build_team_power(power_df)
team_power


def build_history_power(team_power):
    # Sum a team's (already tournament-weighted, tournament-normalized) power
    # across every tournament it appeared in, rewarding repeated strong
    # showings rather than averaging them away.
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


history_power = build_history_power(team_power)
history_power


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
            'the rendered HTML, and pass it via fetch_ewc_rosters(html=...).'
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

EWC2026_ROSTERS = [
    ("Twisted Minds", "BatulinS"),
    ("Twisted Minds", "Perfect1ks"),
    ("Twisted Minds", "xmpl"),
    ("Twisted Minds", "Lu"),
    ("Made in Thailand", "KISS"),
    ("Made in Thailand", "Jacob"),
    ("Made in Thailand", "Scappy"),
    ("Made in Thailand", "Baren"),
    ("Virtus.pro", "Lukarux"),
    ("Virtus.pro", "Beami"),
    ("Virtus.pro", "curexi"),
    ("Virtus.pro", "NIXZYEE"),
    ("17 Gaming", "Lilghost"),
    ("17 Gaming", "tiantianhaovo"),
    ("17 Gaming", "WenBo"),
    ("17 Gaming", "xwudd"),
    ("T1", "EEND"),
    ("T1", "Heather"),
    ("T1", "Rain1ng"),
    ("T1", "Type"),
    ("Natus Vincere", "Feyerist"),
    ("Natus Vincere", "Hakatory"),
    ("Natus Vincere", "boost1k-"),
    ("Natus Vincere", "spyrro"),
    ("Petrichor Road", "MMing"),
    ("Petrichor Road", "i26v6"),
    ("Petrichor Road", "Cui71"),
    ("Petrichor Road", "04NB"),
    ("SOOPers", "DIEL"),
    ("SOOPers", "Gyumin"),
    ("SOOPers", "Heaven"),
    ("SOOPers", "Rex"),
    ("Anyone's Legend", "Delwyn"),
    ("Anyone's Legend", "Himass"),
    ("Anyone's Legend", "Sololzy"),
    ("Anyone's Legend", "Destroyy"),
    ("Four Angry Men", "HSmm"),
    ("Four Angry Men", "Shen"),
    ("Four Angry Men", "SpaceMan1010"),
    ("Four Angry Men", "WINDah"),
    ("Team Falcons", "Shrimzy"),
    ("Team Falcons", "Gustav"),
    ("Team Falcons", "TGLTN"),
    ("Team Falcons", "Kickstart"),
    ("Team Liquid", "PurdyKurty"),
    ("Team Liquid", "CowBoi"),
    ("Team Liquid", "aLOW"),
    ("Team Liquid", "luke12"),
    ("JD Gaming", "Dec12th"),
    ("JD Gaming", "nanss"),
    ("JD Gaming", "Cold119"),
    ("JD Gaming", "SuZe"),
    ("TYLOO", "1ee"),
    ("TYLOO", "KKong"),
    ("TYLOO", "HaoSkr"),
    ("TYLOO", "OneDragon"),
    ("Team Vitality", "Gedrox"),
    ("Team Vitality", "hallomybad"),
    ("Team Vitality", "Lev4nte"),
    ("Team Vitality", "QWZYYY"),
    ("TEAM NEMESIS", "DIFX"),
    ("TEAM NEMESIS", "SoseD"),
    ("TEAM NEMESIS", "Staed"),
    ("TEAM NEMESIS", "Mellman"),
    ("Geekay Esports", "AKaN"),
    ("Geekay Esports", "EJ"),
    ("Geekay Esports", "Parkpro"),
    ("Geekay Esports", "Seongjang"),
    ("Gen.G Esports", "seoul"),
    ("Gen.G Esports", "Salute"),
    ("Gen.G Esports", "diyy"),
    ("Gen.G Esports", "BeaN"),
    ("Full Sense", "Thanad0l"),
    ("Full Sense", "TiGGER"),
    ("Full Sense", "Belmoth"),
    ("Full Sense", "Flash"),
    ("Sharper Esport", "ThanawatTH"),
    ("Sharper Esport", "Jdaii"),
    ("Sharper Esport", "Earthzapalui"),
    ("Sharper Esport", "Thunderz"),
    ("The Expendables", "Clories"),
    ("The Expendables", "DuCkHjeUz"),
    ("The Expendables", "TanVuu"),
    ("The Expendables", "Hoangf"),
    ("The Vicious", "YmCud"),
    ("The Vicious", "Sapauu"),
    ("The Vicious", "Taikonn"),
    ("The Vicious", "SirT"),
    ("SHADOW ESPORT", "Setsunaa"),
    ("SHADOW ESPORT", "Jekzy"),
    ("SHADOW ESPORT", "Hanzyy"),
    ("SHADOW ESPORT", "Kamalz"),
    ("VEGA ESPORTS", "AJ"),
    ("VEGA ESPORTS", "1MBOT"),
    ("VEGA ESPORTS", "Tedeeyy"),
    ("VEGA ESPORTS", "empt"),
]

roster_path = f"{OUTPUT_DIR}/ewc2026_rosters.csv"

try:
    players_df = pd.read_csv(roster_path)
    print(f"Using cached rosters: {roster_path}")
except FileNotFoundError:
    print("No cached rosters found -- using hardcoded EWC 2026 roster snapshot")
    players_df = pd.DataFrame(EWC2026_ROSTERS, columns=["team", "player"])
    players_df["tournament"] = "EWC2026"
    players_df = players_df[["tournament", "team", "player"]]

players_df

TEAM_NAME_ALIASES = {
    "AGAL INTERNATIONAL": "ANYONE'S LEGEND",
    "GAM THE EXPENDABLES": "THE EXPENDABLES",
    "FULL SENSE": "FULL SENSE",
    "R8 ESPORTS": "VEGA ESPORTS",
    "SHADOW ESPORT": "SHADOW ESPORT",
    "SHARPER ESPORTS": "SHARPER ESPORT",
    "TEAM NEMESIS": "TEAM NEMESIS",
    "FALCONS": "TEAM FALCONS",
    "DN SOOPERS": "SOOPERS",
    "GEN.G": "GEN.G ESPORTS",
    "DN FREECS": "SOOPERS",  # DN Freecs -> DN Soopers -> Soopers, same org through renames
    "NEMIGA GAMING": "TEAM VITALITY",  # 3 of 4 current Vitality players came from Nemiga
}

_APOSTROPHE_VARIANTS = ["\u2019", "\u2018", "\u02bc", "\u00b4", "`"]


def normalize_team_name(name):
    """Canonical team-name normalization. Apply this to every team column
    immediately after fetching, before any merge or groupby -- applying it
    late (as the original notebook did) silently breaks joins for any team
    whose raw name differs across data sources.

    Handles, in order: unicode quote variants (Twire's own API is
    inconsistent -- "Anyone's Legend" appears with both a straight and a
    curly apostrophe across different rows of the *same* dataset), digit-
    prefix spacing ("17 Gaming" -> "17Gaming"), known aliases, and finally
    uppercasing everything so case is never a second source of the same bug.
    """
    if pd.isna(name):
        return name

    name = str(name).strip()
    name = unicodedata.normalize("NFKC", name)

    for variant in _APOSTROPHE_VARIANTS:
        name = name.replace(variant, "'")

    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^(\d+)\s+", r"\1", name)  # "17 Gaming" -> "17Gaming"

    name = name.upper()

    return TEAM_NAME_ALIASES.get(name, name)


def normalize_player_name(name):
    """Strip a team-prefix before "_" (e.g. TL_TGLTN -> TGLTN)."""
    if pd.isna(name):
        return name

    name = str(name).strip()
    if "_" in name:
        name = name.split("_", 1)[1]

    return name


# Apply immediately, everywhere, before any merge.
team_df["team"] = team_df["team"].apply(normalize_team_name)
player_df["teamName"] = player_df["teamName"].apply(normalize_team_name)
player_df["username"] = player_df["username"].apply(normalize_player_name)
team_rank_df["team"] = team_rank_df["name"].apply(normalize_team_name)
power_df["team"] = power_df["team"].apply(normalize_team_name)
players_df["team"] = players_df["team"].apply(normalize_team_name)
players_df["player"] = players_df["player"].apply(normalize_player_name)

ewc_teams = players_df["team"].unique().tolist()
print(f"{len(ewc_teams)} EWC teams (normalized): {ewc_teams}")


TOURNAMENT_STRENGTH = {
    "PGC": 1.00,
    "PGS": 0.95,
    "EWC": 0.95,

    # Regional Championship: continental/multi-country championship-tier.
    "PCL": 0.75,
    "PAS": 0.75,
    "PEC": 0.75,
    "PWS": 0.75,
    "PMS": 0.75,  # PUBG Master Series -- treated as championship-tier; adjust if wrong

    # Regional Open: single-country/weekly-league tier.
    "PTS": 0.55,
    "PVS": 0.55,
}

UNKNOWN_TOURNAMENT_STRENGTH = 0.65

GLOBAL_TOURNAMENT_STRENGTH_THRESHOLD = 0.90  # PGC/PGS/EWC tier = "global"


def _tournament_strength_for_key(key):
    if key.startswith("PGS"):
        return TOURNAMENT_STRENGTH["PGS"]
    return TOURNAMENT_STRENGTH.get(key, UNKNOWN_TOURNAMENT_STRENGTH)


# Map each tournament's *real* display name (what team_df["tournament"]
# actually contains) to its strength -- keying on the short code directly
# silently fails, since short codes like "PMS" don't appear as substrings
# of the real names ("PUBG Master Series 2026: Phase 1").
TOURNAMENT_STRENGTH_BY_NAME = {
    TOURNAMENTS[key]["name"]: _tournament_strength_for_key(key)
    for key in TOURNAMENTS
}


def tournament_strength_for(tournament_name):
    if tournament_name in TOURNAMENT_STRENGTH_BY_NAME:
        return TOURNAMENT_STRENGTH_BY_NAME[tournament_name]
    if "ewc" in str(tournament_name).lower():
        return TOURNAMENT_STRENGTH["EWC"]
    return UNKNOWN_TOURNAMENT_STRENGTH


# Recency proxy: Twire's API doesn't expose exact tournament dates, so we
# use each tournament's numeric `id` as an ordering proxy (auto-incrementing
# IDs correlate with creation/occurrence order) and scale into the spec's
# example range (old ~0.3, recent ~1.8). Replace with real dates if/when
# Twire exposes them.
_tournament_ids_sorted = sorted(TOURNAMENTS.items(), key=lambda kv: kv[1]["id"])
_id_rank = {key: i for i, (key, _) in enumerate(_tournament_ids_sorted)}
_max_rank = max(len(_tournament_ids_sorted) - 1, 1)

RECENCY_MIN, RECENCY_MAX = 0.3, 1.8

RECENCY_BY_NAME = {
    TOURNAMENTS[key]["name"]: RECENCY_MIN + (_id_rank[key] / _max_rank) * (RECENCY_MAX - RECENCY_MIN)
    for key in TOURNAMENTS
}

print(pd.DataFrame({
    "tournament": list(TOURNAMENT_STRENGTH_BY_NAME),
    "strength": list(TOURNAMENT_STRENGTH_BY_NAME.values()),
    "recency": [RECENCY_BY_NAME[t] for t in TOURNAMENT_STRENGTH_BY_NAME],
}))

def build_twire_team_ratings(team_rank_df, team_df):
    max_ranking = team_rank_df["ranking"].max()
    ranked = dict(zip(team_rank_df["team"], team_rank_df["ranking"] / max_ranking))
    min_ranked_score = min(ranked.values())

    all_teams = team_df["team"].dropna().unique()
    unranked_teams = [t for t in all_teams if t not in ranked]

    if not unranked_teams:
        return ranked

    proxies = {}
    for team in unranked_teams:
        sub = team_df[team_df["team"] == team]
        avg_rank = np.average(sub["avgRank"], weights=sub["weight"])
        proxies[team] = 1 / (1 + avg_rank)  # better (lower) avg finish -> higher proxy

    values = np.array(list(proxies.values()))
    span = (values.max() - values.min()) or 1.0

    for team, proxy in proxies.items():
        normalized = (proxy - values.min()) / span  # 0..1 within the unranked pool
        ranked[team] = normalized * min_ranked_score * 0.9  # strictly below the weakest ranked team

    return ranked


twire_team_ratings = build_twire_team_ratings(team_rank_df, team_df)
team_df["twire_team_rating"] = team_df["team"].map(twire_team_ratings).fillna(0)

print(f"{len(twire_team_ratings)} teams rated ({len(team_rank_df)} official, "
      f"{len(twire_team_ratings) - len(team_rank_df)} estimated)")

def build_global_experience(team_df):
    df = team_df.copy()
    df["is_global"] = df["tournament"].apply(
        lambda t: tournament_strength_for(t) >= GLOBAL_TOURNAMENT_STRENGTH_THRESHOLD
    )

    def agg(x):
        global_rows = x[x["is_global"]]
        regional_rows = x[~x["is_global"]]
        return pd.Series({
            "global_events": global_rows["tournament"].nunique(),
            "global_matches": len(global_rows),
            "regional_events": regional_rows["tournament"].nunique(),
            "regional_matches": len(regional_rows),
            "global_points": global_rows["totalPoints"].sum(),
            "regional_points": regional_rows["totalPoints"].sum(),
        })

    result = df.groupby("team").apply(agg).reset_index()
    total_matches = result["global_matches"] + result["regional_matches"]
    result["global_ratio"] = result["global_matches"] / total_matches.clip(lower=1)

    return result


global_experience = build_global_experience(team_df)
global_experience.sort_values("global_ratio", ascending=False).head(10)

def build_strength_of_schedule(team_df):
    df = team_df.copy()

    stage_key = ["tournament", "stage"]
    stage_sum = df.groupby(stage_key)["twire_team_rating"].transform("sum")
    stage_count = df.groupby(stage_key)["twire_team_rating"].transform("count")

    # average opponent rating = everyone else in that stage, excluding self
    df["strength_of_schedule"] = (
        (stage_sum - df["twire_team_rating"]) / (stage_count - 1).clip(lower=1)
    )

    return df


team_df = build_strength_of_schedule(team_df)
team_df[["tournament", "stage", "team", "twire_team_rating", "strength_of_schedule"]].head()

team_df["history_weight"] = (
    team_df["tournament"].map(TOURNAMENT_STRENGTH_BY_NAME).fillna(UNKNOWN_TOURNAMENT_STRENGTH)
    * team_df["tournament"].map(RECENCY_BY_NAME).fillna(1.0)
    * team_df["strength_of_schedule"].clip(lower=0.01)
)

team_df[["tournament", "stage", "team", "history_weight"]].sort_values("history_weight", ascending=False).head()

POWER_SCORE_BASE_COLUMNS = [
    "overall_score", "attacker_score", "survivor_score",
    "teammate_score", "utility_score", "finisher_score",
]

for col in POWER_SCORE_BASE_COLUMNS:
    tournament_max = power_df.groupby("tournament")[col].transform("max")
    power_df[f"{col}_scaled"] = power_df[col] / tournament_max


def build_player_power(power_df):
    return (
        power_df
        .groupby("team")
        .agg(
            player_power_mean=("overall_score_scaled", "mean"),
            player_power_max=("overall_score_scaled", "max"),
            player_power_std=("overall_score_scaled", "std"),
            player_attacker_mean=("attacker_score_scaled", "mean"),
            player_survivor_mean=("survivor_score_scaled", "mean"),
            player_teammate_mean=("teammate_score_scaled", "mean"),
            player_utility_mean=("utility_score_scaled", "mean"),
            player_finisher_mean=("finisher_score_scaled", "mean"),
        )
        .fillna(0)
        .reset_index()
    )


player_power = build_player_power(power_df)
player_power.sort_values("player_power_mean", ascending=False).head(10)

def _team_weighted_row(x):
    w = x["history_weight"]

    most_recent_idx = x["tournament"].map(RECENCY_BY_NAME).fillna(1.0).idxmax()
    recent_avg_rank = x.loc[most_recent_idx, "avgRank"]

    return pd.Series({
        "tournaments": x["tournament"].nunique(),
        "stages": len(x),

        "avg_rank": np.average(x["avgRank"], weights=w),
        "avg_points": np.average(x["totalPoints"], weights=w),
        "avg_kills": np.average(x["kills"], weights=w),
        "avg_damage": np.average(x["damageDealt"], weights=w),
        "total_wins": (x["wins"] * w).sum(),

        "recent_avg_rank": recent_avg_rank,
        "twire_team_rating": x["twire_team_rating"].iloc[0],
        "strength_of_schedule": np.average(x["strength_of_schedule"], weights=w),
    })


team_weighted_history = (
    team_df
    .groupby("team")
    .apply(_team_weighted_row)
    .reset_index()
)

team_weighted_history.sort_values("avg_rank").head(10)

# Named, tunable coefficients -- edit these, not the formula below.
STRENGTH_WEIGHT_TWIRE_RATING = 0.30
STRENGTH_WEIGHT_HISTORICAL = 0.25
STRENGTH_WEIGHT_PLAYER = 0.20
STRENGTH_WEIGHT_RECENT_FORM = 0.15
STRENGTH_WEIGHT_EXPERIENCE = 0.10

assert abs(
    STRENGTH_WEIGHT_TWIRE_RATING + STRENGTH_WEIGHT_HISTORICAL + STRENGTH_WEIGHT_PLAYER
    + STRENGTH_WEIGHT_RECENT_FORM + STRENGTH_WEIGHT_EXPERIENCE - 1.0
) < 1e-9, "strength weights must sum to 1.0"


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0


def build_team_strength(team_weighted_history, player_power, global_experience):
    df = (
        team_weighted_history
        .merge(player_power, on="team", how="left")
        .merge(global_experience, on="team", how="left")
        .fillna(0)
    )

    historical_rating = _minmax(-df["avg_rank"])       # lower avg_rank (better) -> higher rating
    player_rating = _minmax(df["player_power_mean"])
    recent_form = _minmax(-df["recent_avg_rank"])
    experience = _minmax(df["global_ratio"])
    twire_rating = df["twire_team_rating"]             # already 0..1

    raw_strength = (
        STRENGTH_WEIGHT_TWIRE_RATING * twire_rating
        + STRENGTH_WEIGHT_HISTORICAL * historical_rating
        + STRENGTH_WEIGHT_PLAYER * player_rating
        + STRENGTH_WEIGHT_RECENT_FORM * recent_form
        + STRENGTH_WEIGHT_EXPERIENCE * experience
    )

    df["team_strength"] = _minmax(raw_strength) * 100
    return df


team_strength = build_team_strength(team_weighted_history, player_power, global_experience)
team_strength[["team", "team_strength", "twire_team_rating", "global_ratio", "stages"]].sort_values("team_strength", ascending=False).head(10)

# Roughly doubled from the initial pass: with the original values, the
# strength gap between the top 1-2 teams and the rest of the EWC field was
# ~2.3 standard deviations per match -- compounded over up to 12 Grand
# Final matches, that made the top team win almost deterministically
# (only ~5-6 of 24 teams ever won across 100k simulations). Larger
# match-level variance keeps the *ranking* the same (favorites still favored)
# but gives realistic upset odds instead of near-certainty.
UNCERTAINTY_BASE = 10.0
UNCERTAINTY_SAMPLE_SCALE = 20.0
UNCERTAINTY_REGIONAL_PENALTY = 16.0


def build_uncertainty(team_strength):
    df = team_strength.copy()

    df["strength_std"] = (
        UNCERTAINTY_BASE
        + UNCERTAINTY_SAMPLE_SCALE / np.sqrt(df["stages"].clip(lower=1))
        + (1 - df["global_ratio"]) * UNCERTAINTY_REGIONAL_PENALTY
    )

    return df


team_strength = build_uncertainty(team_strength)
team_strength[["team", "team_strength", "strength_std", "global_ratio"]].sort_values("strength_std", ascending=False).head(10)


STRENGTH_SHRINKAGE_K = 8  # pseudo-count: how many stages of evidence outweighs the prior


def shrink_team_strength(team_strength, k=STRENGTH_SHRINKAGE_K):
    df = team_strength.copy()
    global_mean = df["team_strength"].mean()
    n = df["stages"]
    df["team_strength"] = (df["team_strength"] * n + global_mean * k) / (n + k)
    return df


def calibration_check(strength_df):
    issues = []
    sos_median = strength_df["strength_of_schedule"].median()
    strength_p75 = strength_df["team_strength"].quantile(0.75)

    for _, row in strength_df.iterrows():
        if (
            row["global_events"] == 0
            and row["strength_of_schedule"] < sos_median
            and row["team_strength"] > strength_p75
        ):
            issues.append(
                f"{row['team']}: 0 global events + below-median SOS but "
                f"top-quartile strength ({row['team_strength']:.1f})"
            )
    return issues


team_strength = shrink_team_strength(team_strength)

calibration_issues = calibration_check(team_strength)
if calibration_issues:
    print("CALIBRATION ISSUES:")
    for issue in calibration_issues:
        print(" -", issue)
else:
    print("No calibration issues found.")

team_strength.sort_values("team_strength", ascending=False).head(50)

PLACEMENT_POINTS_TABLE = {1: 10, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1}  # 9th+ = 0
KILL_POINTS_PER_KILL = 1

# Expected kills-per-match, scaled by team_strength (0-100) into a plausible
# range for a 4-player squad -- adjust these two if real kill data suggests otherwise.
MIN_EXPECTED_KILLS = 1.0
MAX_EXPECTED_KILLS = 20.0


def expected_kills_for_strength(strength):
    return MIN_EXPECTED_KILLS + (np.clip(strength, 0, 100) / 100.0) * (MAX_EXPECTED_KILLS - MIN_EXPECTED_KILLS)


def simulate_match_points(strength, std, n_sims, n_teams, rng, active_mask=None):
    """One simulated match for `n_teams` teams, `n_sims` times at once.
    Returns (match_points, kills, ranks), each shaped (n_sims, n_teams).

    active_mask: optional (n_sims, n_teams) bool, True = actually competing
    in that simulation's lobby (used when the lobby composition varies per
    simulation, e.g. the Grand Finals' qualifiers). If None, every team is
    active in every simulation (the group stage's lobbies are fixed).
    """
    perf = rng.normal(strength, std, size=(n_sims, n_teams))
    if active_mask is not None:
        perf = np.where(active_mask, perf, -np.inf)

    order = np.argsort(-perf, axis=1)
    ranks = np.empty_like(order)
    placement_matrix = np.tile(np.arange(1, n_teams + 1), (n_sims, 1))
    np.put_along_axis(ranks, order, placement_matrix, axis=1)

    placement_pts_table = np.array([PLACEMENT_POINTS_TABLE.get(r, 0) for r in range(1, n_teams + 1)])
    placement_pts = placement_pts_table[ranks - 1]

    # kills correlated with, but not fully determined by, that match's placement:
    # 1st place -> ~1.0 multiplier, last place -> ~1/n_teams.
    performance_multiplier = (n_teams - ranks + 1) / n_teams
    if active_mask is not None:
        performance_multiplier = np.where(active_mask, performance_multiplier, 0)

    expected_kills = expected_kills_for_strength(strength)[None, :] * (0.6 + 0.7 * performance_multiplier)
    if active_mask is not None:
        expected_kills = np.where(active_mask, expected_kills, 0)

    kills = rng.poisson(expected_kills)
    match_points = placement_pts + kills * KILL_POINTS_PER_KILL
    if active_mask is not None:
        match_points = np.where(active_mask, match_points, 0)

    return match_points, kills, ranks


# EWC 2026 group stage: 3 fixed groups of 8 (not randomly drawn), each
# playing a cross-group day against 2 of the other 3 groups. Every team
# plays 2 days x 6 matches = 12 matches total; the top 16 of 24 by
# cumulative points (placement + kills) across all 3 days advance.
#
# This and the Grand Finals below are simulated as ONE coupled run per
# simulation, not two independent stages: each simulation's own group-stage
# outcome determines exactly which 16 teams contend in THAT simulation's
# finals, rather than aggregating "typical" qualifiers across sims into one
# fixed list and simulating finals separately on that fixed list. The
# latter would silently drop the correlation between "did well enough to
# qualify" and "how they do once qualified."
GROUPS = {
    "A": [
        "VIRTUS.PRO", "ANYONE'S LEGEND", "SOOPERS", "PETRICHOR ROAD",
        "TYLOO", "TEAM VITALITY", "FULL SENSE", "SHADOW ESPORT",
    ],
    "B": [
        "MADE IN THAILAND", "FOUR ANGRY MEN", "TWISTED MINDS", "GEEKAY ESPORTS",
        "TEAM NEMESIS", "TEAM LIQUID", "THE EXPENDABLES", "VEGA ESPORTS",
    ],
    "C": [
        "17GAMING", "NATUS VINCERE", "T1", "GEN.G ESPORTS",
        "JD GAMING", "TEAM FALCONS", "THE VICIOUS", "SHARPER ESPORT",
    ],
}

GROUP_STAGE_DAYS = [
    ("Day 1", "A", "B"),
    ("Day 2", "A", "C"),
    ("Day 3", "B", "C"),
]
MATCHES_PER_DAY = 6
GROUP_STAGE_QUALIFY = 16
GRAND_FINAL_MATCHES_PHASE1 = 12
GRAND_FINAL_MATCHES_PHASE2 = 6
SMASH_THRESHOLD_BUFFER = 10
N_SIMULATIONS = 10000000

_group_teams = sorted(sum(GROUPS.values(), []))
assert _group_teams == sorted(ewc_teams), (
    f"GROUPS doesn't match ewc_teams exactly.\n"
    f"In GROUPS but not ewc_teams: {set(_group_teams) - set(ewc_teams)}\n"
    f"In ewc_teams but not GROUPS: {set(ewc_teams) - set(_group_teams)}"
)

team_group = {team: g for g, teams in GROUPS.items() for team in teams}
ewc_strength = team_strength[team_strength["team"].isin(ewc_teams)].reset_index(drop=True)


def simulate_full_tournament(strength_df, teams, n_sims=N_SIMULATIONS, seed=42):
    """One coupled Monte Carlo run: for each of n_sims simulated tournaments,
    simulate the group stage once, take THAT run's top 16, then simulate
    the Grand Finals once among exactly those 16. Returns per-team
    qualification/champion/etc rates aggregated across all n_sims runs.
    """
    sub = strength_df.set_index("team").loc[teams]
    strength = sub["team_strength"].values
    std = sub["strength_std"].values
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # --- Group stage: fixed lobbies, everyone always active -----------------
    total_points = np.zeros((n_sims, n_teams))

    for day_i, (day_name, g1, g2) in enumerate(GROUP_STAGE_DAYS):
        lobby_teams = GROUPS[g1] + GROUPS[g2]
        lobby_idx = np.array([team_idx[t] for t in lobby_teams])
        lobby_strength = strength[lobby_idx]
        lobby_std = std[lobby_idx]
        rng = np.random.default_rng(seed + day_i)

        for match in range(MATCHES_PER_DAY):
            match_points, kills, _ = simulate_match_points(lobby_strength, lobby_std, n_sims, len(lobby_idx), rng)
            total_points[:, lobby_idx] += match_points

    group_ranks = pd.DataFrame(total_points).rank(axis=1, ascending=False, method="first").values
    qualified_mask = group_ranks <= GROUP_STAGE_QUALIFY  # (n_sims, n_teams), exactly 16 True per row
    qualification_pct = qualified_mask.mean(axis=0) * 100
    expected_group_finish = group_ranks.mean(axis=0)

    # --- Grand Finals: coupled to this same simulation's qualifiers -------
    # Phase 1: GRAND_FINAL_MATCHES_PHASE1 (12) matches, played normally --
    # smash rule cannot trigger yet, since the threshold itself isn't defined
    # until we see where the field stands after these matches.
    cumulative_points = np.zeros((n_sims, n_teams))
    rng_f = np.random.default_rng(seed + 1000)

    for match_i in range(GRAND_FINAL_MATCHES_PHASE1):
        match_points, kills, ranks = simulate_match_points(
            strength, std, n_sims, n_teams, rng_f, active_mask=qualified_mask
        )
        cumulative_points += match_points

    # Dynamic, per-simulation threshold: the leader's score after phase 1,
    # plus a fixed buffer -- not a flat constant, since "highest score" varies
    # simulation to simulation.
    leader_score = np.where(qualified_mask, cumulative_points, -np.inf).max(axis=1)
    smash_threshold = leader_score + SMASH_THRESHOLD_BUFFER

    # Phase 2: up to GRAND_FINAL_MATCHES_PHASE2 (6) more matches, smash-eligible.
    smash_winner = np.full(n_sims, -1)
    matches_played = np.full(n_sims, GRAND_FINAL_MATCHES_PHASE1, dtype=int)

    for match_i in range(GRAND_FINAL_MATCHES_PHASE2):
        still_active = smash_winner == -1
        if not still_active.any():
            break

        match_points, kills, ranks = simulate_match_points(
            strength, std, n_sims, n_teams, rng_f, active_mask=qualified_mask
        )

        # a team wins immediately if THIS match's win brings their cumulative
        # points (including this match) to/past their sim's dynamic threshold.
        new_cumulative_points = cumulative_points + match_points * still_active[:, None]
        eligible = (
            (new_cumulative_points >= smash_threshold[:, None])
            & (ranks == 1) & qualified_mask & still_active[:, None]
        )
        newly_won = eligible.any(axis=1)
        winning_team_idx = np.argmax(eligible, axis=1)
        smash_winner = np.where(newly_won & still_active, winning_team_idx, smash_winner)

        cumulative_points = new_cumulative_points
        matches_played += still_active.astype(int)

    # No smash within GRAND_FINAL_MATCHES_PHASE1 + PHASE2 (18) matches -> highest score wins.
    fallback_winner = np.where(qualified_mask, cumulative_points, -np.inf).argmax(axis=1)
    final_winner = np.where(smash_winner == -1, fallback_winner, smash_winner)

    points_for_ranking = cumulative_points.copy()
    points_for_ranking[np.arange(n_sims), final_winner] = np.inf
    points_for_ranking = np.where(qualified_mask, points_for_ranking, -np.inf)
    final_order = np.argsort(-points_for_ranking, axis=1)
    final_ranks = np.empty_like(final_order)
    placement_matrix = np.tile(np.arange(1, n_teams + 1), (n_sims, 1))
    np.put_along_axis(final_ranks, final_order, placement_matrix, axis=1)

    return {
        "teams": teams,
        "qualification_pct": qualification_pct,
        "expected_group_finish": expected_group_finish,
        "champion_pct": (final_ranks == 1).mean(axis=0) * 100,
        "top4_pct": (final_ranks <= 4).mean(axis=0) * 100,
        "top8_pct": (final_ranks <= 8).mean(axis=0) * 100,
        "bottom8_pct": ((final_ranks > n_teams - 8) & qualified_mask).mean(axis=0) * 100,
        "avg_placement": np.where(qualified_mask, final_ranks, np.nan),
        "smash_rate": (smash_winner != -1).mean() * 100,
        "avg_matches_played": matches_played.mean(),
    }


tournament_sim = simulate_full_tournament(ewc_strength, ewc_teams)

print("Groups:")
for g, teams in GROUPS.items():
    print(f"  Group {g}: {teams}")

qualification_df = pd.DataFrame({
    "team": tournament_sim["teams"],
    "group": [team_group[t] for t in tournament_sim["teams"]],
    "qualification_pct": tournament_sim["qualification_pct"],
    "expected_group_finish": tournament_sim["expected_group_finish"],
}).sort_values("qualification_pct", ascending=False)

qualification_df


print(
    f"Smash-rule finish in {tournament_sim['smash_rate']:.1f}% of simulations "
    f"(avg {tournament_sim['avg_matches_played']:.1f} of "
    f"{GRAND_FINAL_MATCHES_PHASE1 + GRAND_FINAL_MATCHES_PHASE2} matches played)"
)

grand_final_results = (
    pd.DataFrame({
        "team": tournament_sim["teams"],
        "top1_pct": tournament_sim["champion_pct"],
        "top4_pct": tournament_sim["top4_pct"],
        "top8_pct": tournament_sim["top8_pct"],
    })
    .merge(qualification_df[["team", "qualification_pct"]].rename(columns={"qualification_pct": "top16_pct"}), on="team")
    [["team", "top1_pct", "top4_pct", "top8_pct", "top16_pct"]]
    .sort_values(["top1_pct", "top4_pct", "top8_pct", "top16_pct"], ascending=False)
)

grand_final_results


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top10_champ = grand_final_results.head(10)
axes[0, 0].barh(top10_champ["team"], top10_champ["top1_pct"])
axes[0, 0].invert_yaxis()
axes[0, 0].set_title("Champion probability (top 10)")
axes[0, 0].set_xlabel("%")

top10_t4 = grand_final_results.sort_values("top4_pct", ascending=False).head(10)
axes[0, 1].barh(top10_t4["team"], top10_t4["top4_pct"])
axes[0, 1].invert_yaxis()
axes[0, 1].set_title("Top 4 probability (top 10)")
axes[0, 1].set_xlabel("%")

qual_sorted = qualification_df.sort_values("qualification_pct", ascending=False)
axes[1, 0].barh(qual_sorted["team"], qual_sorted["qualification_pct"])
axes[1, 0].invert_yaxis()
axes[1, 0].set_title("Group-stage qualification probability (all 24)")
axes[1, 0].set_xlabel("%")
axes[1, 0].tick_params(axis="y", labelsize=7)

axes[1, 1].hist(ewc_strength["team_strength"], bins=12)
axes[1, 1].set_title("Team strength distribution (EWC field)")
axes[1, 1].set_xlabel("team_strength (0-100)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ewc_simulation_charts.png", dpi=120)
plt.show()

def leave_one_tournament_out_validation(team_df):
    tournaments = team_df["tournament"].unique()
    results = []

    for held_out in tournaments:
        train_df = team_df[team_df["tournament"] != held_out]
        test_df = team_df[team_df["tournament"] == held_out]

        if test_df.empty or train_df.empty:
            continue

        train_rank = train_df.groupby("team").apply(
            lambda x: np.average(x["avgRank"], weights=x["history_weight"])
        )

        test_teams = test_df["team"].unique()
        actual = test_df.groupby("team")["avgRank"].mean().reindex(test_teams)
        predicted = train_rank.reindex(test_teams)

        valid = actual.notna() & predicted.notna()
        if valid.sum() < 3:
            continue

        y_true = actual[valid]
        y_pred = predicted[valid]

        actual_top8 = set(y_true.sort_values().head(8).index)
        pred_top8 = set(y_pred.sort_values().head(8).index)
        actual_winner = y_true.idxmin()
        pred_winner = y_pred.idxmin()

        results.append({
            "tournament": held_out,
            "n_teams": int(valid.sum()),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "spearman": float(y_true.corr(y_pred, method="spearman")),
            "top8_overlap": len(actual_top8 & pred_top8) / min(8, len(actual_top8)),
            "winner_correct": actual_winner == pred_winner,
        })

    return pd.DataFrame(results)


validation_results = leave_one_tournament_out_validation(team_df)
print(validation_results.to_string())

print("\nAggregate:")
print(f"  MAE: {validation_results['mae'].mean():.3f}")
print(f"  RMSE: {validation_results['rmse'].mean():.3f}")
print(f"  Spearman: {validation_results['spearman'].mean():.3f}")
print(f"  Top-8 overlap: {validation_results['top8_overlap'].mean():.1%}")
print(f"  Winner accuracy: {validation_results['winner_correct'].mean():.1%}")

from sklearn.ensemble import RandomForestRegressor

DIAGNOSTIC_FEATURES = [
    "twire_team_rating", "strength_of_schedule", "player_power_mean",
    "player_power_max", "player_power_std", "global_ratio",
    "global_events", "regional_events", "stages",
]

diagnostic_df = team_strength.dropna(subset=DIAGNOSTIC_FEATURES + ["avg_rank"])

diagnostic_model = RandomForestRegressor(n_estimators=300, random_state=42)
diagnostic_model.fit(diagnostic_df[DIAGNOSTIC_FEATURES], diagnostic_df["avg_rank"])

diagnostic_importance = pd.DataFrame({
    "feature": DIAGNOSTIC_FEATURES,
    "importance": diagnostic_model.feature_importances_,
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(diagnostic_importance["feature"], diagnostic_importance["importance"])
plt.gca().invert_yaxis()
plt.title("Feature importance (diagnostic RF predicting avg_rank)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=120)
plt.show()

diagnostic_importance

final_output = (
    ewc_strength[["team", "team_strength", "strength_std"]]
    .merge(grand_final_results[["team", "top1_pct", "top4_pct", "top8_pct", "top16_pct"]], on="team", how="left")
    .fillna(0)
)

final_output["expected_rank"] = (
    final_output["team_strength"].rank(ascending=False, method="min")
)

final_output = final_output.rename(columns={
    "team": "Team",
    "team_strength": "Strength",
    "strength_std": "Strength Std",
    "expected_rank": "Expected Rank",
    "top1_pct": "Top1 %",
    "top4_pct": "Top4 %",
    "top8_pct": "Top8 %",
    "top16_pct": "Top16 %",
})[[
    "Team", "Strength", "Strength Std", "Expected Rank",
    "Top1 %", "Top4 %", "Top8 %", "Top16 %",
]].sort_values(["Top1 %", "Top4 %", "Top8 %", "Top16 %"], ascending=False)

final_output.to_csv(f"{OUTPUT_DIR}/ewc_predictions.csv", index=False)

print(f"Wrote {OUTPUT_DIR}/ewc_predictions.csv\n")
print("Top 24 teams, sorted by Top1 % then Top4 % then Top8 % then Top16 %:")
final_output
