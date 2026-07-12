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
