"""Feature engineering: weighted team/player aggregates feeding the model."""

import numpy as np
import pandas as pd

from .config import TOURNAMENT_WEIGHTS

# 6 individual PGS tiers (0.80-1.30) + 1 shared regional tier (0.60) = 7.
TOTAL_TOURNAMENT_TIERS = len(set(TOURNAMENT_WEIGHTS.values()))


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


def _team_history_row(x):
    # A team's own average tournament tier (PGS ~0.8-1.3, regional 0.6).
    # np.average(..., weights=x["weight"]) alone only weighs a team's
    # tournaments *relative to each other* -- it cancels out for a team
    # that only ever played one tier, so a regional-only team's stats
    # would otherwise come out on the same scale as a globals team's.
    tier = np.average(x["tournament_weight"], weights=x["weight"])

    # Breadth: how many of the 7 recognized tiers (6 individual PGS majors
    # + 1 shared regional tier) has this team actually competed in. A team
    # that's only ever played one regional event gets hit twice: once for
    # low tier weight, again for having proven nothing outside that one
    # tier -- compounding into a much steeper discount than tier alone.
    coverage = x["tournament_weight"].nunique() / TOTAL_TOURNAMENT_TIERS

    discount = tier * coverage

    return pd.Series({

        "tournaments": x["tournament"].nunique(),
        "stages": len(x),

        "avg_rank": np.average(x["avgRank"], weights=x["weight"]),
        "std_rank": x["avgRank"].std(),

        "avg_points": np.average(x["totalPoints"], weights=x["weight"]) * discount,
        "std_points": x["totalPoints"].std(),

        "avg_kills": np.average(x["kills"], weights=x["weight"]) * discount,
        "std_kills": x["kills"].std(),

        "avg_damage": np.average(x["damageDealt"], weights=x["weight"]) * discount,
        "std_damage": x["damageDealt"].std(),

        "avg_damage_taken": np.average(x["damageTaken"], weights=x["weight"]) * discount,

        "total_wins": (x["wins"] * x["weight"]).sum(),

        "player_avg_kd": np.average(x["avg_kd"], weights=x["weight"]) * discount,
        "player_max_kd": np.average(x["max_kd"], weights=x["weight"]) * discount,

        "player_avg_damage": np.average(x["avg_damage"], weights=x["weight"]) * discount,
        "player_max_damage": np.average(x["max_damage"], weights=x["weight"]) * discount,

        "player_avg_kills": np.average(x["avg_kills"], weights=x["weight"]) * discount,
        "player_avg_assists": np.average(x["avg_assists"], weights=x["weight"]) * discount,

        "player_avg_kas": np.average(x["avg_kas"], weights=x["weight"]) * discount,

    })


def build_history_features(history, history_power):
    history_features = (
        history
        .groupby("team")
        .apply(_team_history_row)
        .reset_index()
    )

    history_features = history_features.merge(
        history_power,
        on="team",
        how="left"
    )

    history_features = history_features.fillna(0)

    return history_features
