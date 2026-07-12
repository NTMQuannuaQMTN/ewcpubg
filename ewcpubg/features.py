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
