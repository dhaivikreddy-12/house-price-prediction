"""CLI tool to predict house price from your own inputs."""
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/house_prices.csv")
X = df.drop(columns=["price"])
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)


def main():
    parser = argparse.ArgumentParser(description="Predict house price")
    parser.add_argument("--sqft", type=float, required=True)
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--age", type=int, required=True)
    args = parser.parse_args()

    sample = pd.DataFrame([{
        "sqft_living": args.sqft,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "age": args.age,
    }])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]

    print(f"\nFor a {args.sqft:.0f} sqft, {args.bedrooms} bed, "
          f"{args.bathrooms} bath, {args.age}yr-old house:")
    print(f"Estimated price: ${pred:,.0f}")


if __name__ == "__main__":
    main()
