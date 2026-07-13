# ewcpubg

Predicts EWC 2026 PUBG team ranks from historical tournament performance,
using Twire stats/power-ranking data and Liquipedia rosters.

## Layout

- `ewcpubg/config.py` — API endpoints, credentials, tournament list, weights
- `ewcpubg/normalize.py` — team/player name normalization, weight lookups
- `ewcpubg/twire_stats.py` — Twire GraphQL client, shard discovery, team/player stats crawl
- `ewcpubg/twire_assets.py` — Twire S3 team-ranking / power-ranking JSON
- `ewcpubg/liquipedia.py` — Liquipedia EWC 2026 roster scrape
- `ewcpubg/features.py` — weighted feature aggregation (team + player -> per-team history)
- `ewcpubg/model.py` — RandomForest rank predictor
- `ewcpubg/pipeline.py` — wires everything together end to end
- `notebooks/ewcpubg.ipynb` — thin exploratory notebook on top of the package
- `output/` — generated CSVs (gitignored)

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in TWIRE_API_KEY
python -m ewcpubg.pipeline
```

`TWIRE_API_KEY` is read from the environment (via `.env`, gitignored) rather
than hardcoded, so it never ends up committed. On Kaggle, set it under
Add-ons -> Secrets instead and load it into `os.environ["TWIRE_API_KEY"]`
before running the notebook's cells.

This writes `output/team_stats.csv`, `output/player_stats.csv`,
`output/ewc2026_rosters.csv`, `output/ewc_history_features.csv`, and prints
the predicted rank table.

For interactive exploration, open `notebooks/ewcpubg.ipynb`.

