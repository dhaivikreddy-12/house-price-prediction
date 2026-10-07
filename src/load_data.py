"""Load the California Housing dataset (real data, 20,640 homes).

Source: sklearn.datasets.fetch_california_housing, originally derived from the
1990 U.S. Census. Cached to data/ so the repo runs offline after the first run.
"""
import os
import pandas as pd

CSV_PATH = "data/house_prices.csv"
URL = "https://archive.ics.uci.edu/static/public/235/california+housing.zip"


def build():
    from sklearn.datasets import fetch_california_housing

    print("Downloading California Housing dataset (first run only)...")
    bunch = fetch_california_housing()
    df = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    df["MedHouseVal"] = bunch.target * 100_000  # target is in $100k units
    df["avg_rooms"] = df["AveRooms"]
    df["avg_bedrooms"] = df["AveBedrms"]
    df["house_age"] = df["HouseAge"]
    df["population"] = df["Population"]
    df = df[
        [
            "MedHouseVal",
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
    ]
    return df


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    df = build()
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} homes -> {CSV_PATH}")
    print(df.head())
