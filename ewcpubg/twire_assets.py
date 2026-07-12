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
