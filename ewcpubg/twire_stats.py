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
