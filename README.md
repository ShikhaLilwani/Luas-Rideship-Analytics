# Data

## Real data (CSO)

1. Go to <https://data.cso.ie/>.
2. Search for **passenger journeys by public transport** (weekly).
3. Download the table as **CSV** and save it here as `cso_weekly_passengers.csv`.
4. Run `python src/analysis.py data/cso_weekly_passengers.csv`.

The script finds the week, mode of transport and value columns by name, so a standard CSO export works without changes. Week labels should look like `2023 Week 01`.

## Sample data

`sample/sample_weekly_passengers.csv` is **synthetic**. It copies the CSO layout so the code runs without downloading anything. The numbers are made up and must not be quoted as real ridership.
