"""Generate a synthetic but realistic house prices dataset."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n = 200
sqft_living = rng.integers(600, 3500, n)
bedrooms = np.clip((sqft_living // 700) + rng.integers(0, 2, n), 1, 6)
bathrooms = np.clip((bedrooms + 1) // 2 + rng.integers(0, 2, n), 1, 5)
age = rng.integers(0, 60, n)

base = 80 * sqft_living + 12000 * bedrooms + 9000 * bathrooms
age_penalty = 500 * age
noise = rng.normal(0, 25000, n)
price = base - age_penalty + noise
price = np.maximum(price, 30000)

df = pd.DataFrame({
    "sqft_living": sqft_living,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "price": price.round(0).astype(int),
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/house_prices.csv", index=False)
print(f"Generated {len(df)} rows -> data/house_prices.csv")
print(df.head())
