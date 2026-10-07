# 🏠 House Price Prediction

> *"Every house has a story. This one teaches you how to price it."*

My first real machine learning project — predicting median house values from the **California Housing** dataset (20,640 real homes from the 1990 U.S. Census). It was my first "aha!" moment: the day a model with eight numeric columns explained ~80% of the variation in real estate prices.

Working on real data changed everything. My first attempt used made-up numbers and scored a suspiciously perfect R². The real dataset has duplicated rows, capped values, and a hard ceiling around $500k — and suddenly the problem became interesting.

## What this project does

- Loads the real California Housing dataset (8 numeric features per district).
- Compares three models: Linear Regression, Random Forest, and Gradient Boosting.
- Reports MAE / RMSE in actual dollars, plus R².
- Ranks features by importance so you can see what actually drives price.
- Includes a CLI to estimate a property from real census-style inputs.

## The dataset

[California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing) — 20,640 census tracts, derived from the 1990 U.S. Census.

| Feature      | What it means                                  |
|--------------|------------------------------------------------|
| `MedInc`     | Median income of households in the block group |
| `HouseAge`   | Median age of houses                           |
| `AveRooms`   | Average rooms per household                    |
| `AveBedrms`  | Average bedrooms per household                 |
| `Population` | Block group population                         |
| `AveOccup`   | Average household occupancy                    |
| `Latitude` / `Longitude` | Location coordinates               |
| `MedHouseVal`| Median house value in $100k units — **target** |

## How to run it

```bash
pip install -r requirements.txt

# Download the data (first run only), then train and compare models
python train.py

# Estimate a property value from census-style inputs
python -m src.predict --medinc 8.3 --houseage 41 --averooms 5.3 \
    --avebedrms 1.1 --population 322 --aveoccup 2.55 --latitude 37.88 --longitude -122.23
```

## Project structure

```
house-price-prediction/
├── data/
│   └── house_prices.csv        # cached after first download
├── src/
│   ├── load_data.py            # fetch + cache
│   ├── train_model.py          # train, compare, evaluate
│   └── predict.py              # CLI
├── tests/                      # pytest smoke tests
├── train.py
├── requirements.txt
└── README.md
```

## What I learned

- The gap between a clean synthetic dataset and a messy real one is enormous.
- Why R² alone is not enough — MAE in dollars is what a person actually cares about.
- That location features (lat/long) can carry surprising signal.
- How to compare models fairly on one split instead of guessing.

## Results

On a 20% held-out test set:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LinearRegression | $53,320 | $74,558 | 0.576 |
| **RandomForest** | **$32,656** | **$50,391** | **0.806** |
| GradientBoosting | $37,164 | $54,222 | 0.776 |

Random Forest wins, and `MedInc` dominates feature importance (0.53) — income explains house prices far better than square footage does.

---

*Built with Python, pandas, scikit-learn, matplotlib. Real data, honestly measured.*
