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
