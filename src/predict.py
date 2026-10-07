"""CLI tool to estimate a home's value using the real California Housing model."""
import argparse
from src.load_data import load
from src.train_model import FEATURES

df = load()

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)
model.fit(df[FEATURES], df["MedHouseVal"])


def main():
    parser = argparse.ArgumentParser(description="Estimate median home value")
    parser.add_argument("--medinc", type=float, required=True, help="median income in $10k units")
    parser.add_argument("--houseage", type=float, required=True)
    parser.add_argument("--averooms", type=float, required=True)
    parser.add_argument("--avebedrms", type=float, required=True)
    parser.add_argument("--population", type=float, required=True)
    parser.add_argument("--aveoccup", type=float, required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    args = parser.parse_args()

    sample = pd.DataFrame([{
        "MedInc": args.medinc,
        "HouseAge": args.houseage,
        "AveRooms": args.averooms,
        "AveBedrms": args.avebedrms,
        "Population": args.population,
        "AveOccup": args.aveoccup,
        "Latitude": args.latitude,
        "Longitude": args.longitude,
    }])
    pred = model.predict(sample)[0]
    print(f"\nEstimated median house value: ${pred:,.0f}")


if __name__ == "__main__":
    main()
