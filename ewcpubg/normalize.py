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
