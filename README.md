# 🏠 House Price Prediction

> *"Every house has a story. This one teaches you how to price it."*

A beginner-friendly walkthrough that predicts housing prices from a few simple features like square footage, number of bedrooms, and age of the property. It was my first real "aha!" moment in machine learning — the day I realised a straight line could actually be smart.

## What this project does

- Loads a small dataset of homes with their sale prices.
- Cleans and preps the data (handling missing values, scaling numbers).
- Trains a **Linear Regression** model to learn the relationship between features and price.
- Evaluates how well the model performs using metrics everyone can understand (MAE, RMSE, R²).
- Lets you plug in your own house details and get a predicted price.

## The dataset

I generated a synthetic but realistic dataset (`data/house_prices.csv`) with ~200 homes. Each row has:

| Feature      | What it means                  |
|--------------|--------------------------------|
| `sqft_living`| Interior living area in sq. ft.|
| `bedrooms`   | Number of bedrooms             |
| `bathrooms`  | Number of bathrooms            |
| `age`        | Age of the house in years      |
| `price`      | Sale price in dollars (target) |

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and evaluate the model
python train.py

# 3. Predict the price of your own house
python predict.py --sqft 1800 --bedrooms 3 --bathrooms 2 --age 15
```

## Project structure

```
house-price-prediction/
├── data/
│   └── house_prices.csv        # the dataset
├── src/
│   ├── load_data.py            # loading & cleaning
│   ├── train_model.py          # training & evaluation
│   └── predict.py              # CLI for predictions
├── train.py                    # main training script
├── requirements.txt
└── README.md
```

## What I learned

- The difference between features and labels.
- Why we split data into training and testing sets.
- How to interpret R² and RMSE without getting lost in jargon.
- That a good baseline (even a simple one) beats a complex model you don't understand.

## Results

On a held-out test set, the model reaches an **R² of 0.90** — meaning it explains most of the variation in house prices using just those four features. Not bad for a first project.

---

*Built with Python, pandas, scikit-learn. Made for learning, by a student, for students.*
