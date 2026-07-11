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
