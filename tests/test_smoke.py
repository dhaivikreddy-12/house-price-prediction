import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "house_prices.csv"

EXPECTED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

EXPECTED_COLUMNS = ["MedHouseVal"] + EXPECTED_FEATURES


def _require_csv():
    if not CSV.exists():
        pytest.skip(f"cached dataset missing ({CSV}); run the loader once to cache it")


def _features_from_source():
    """Read FEATURES out of src/train_model.py without executing the script.

    train_model.py trains three models and writes a plot at import time, so the
    constant is read statically here and the real import is covered separately.
    """
    tree = ast.parse((ROOT / "src" / "train_model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FEATURES":
            return ast.literal_eval(node.value)
    raise AssertionError("FEATURES is not assigned at module level in src/train_model.py")


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file(), "README.md is missing"
    assert (ROOT / "LICENSE").is_file(), "LICENSE is missing"


def test_features_constant_is_expected_list():
    features = _features_from_source()
    assert isinstance(features, list)
    assert len(features) == 8
    assert features == EXPECTED_FEATURES
    assert all(isinstance(f, str) for f in features)
    assert len(set(features)) == len(features), "FEATURES contains duplicates"


def test_load_module_declares_csv_path_under_data():
    from src import load_data

    assert load_data.CSV_PATH.startswith("data/")
    assert pathlib.PurePosixPath(load_data.CSV_PATH).name == "house_prices.csv"
    assert callable(load_data.load)


def test_cached_csv_has_expected_shape():
    _require_csv()
    df = pd.read_csv(CSV)

    assert len(df) == 20640, f"expected the full California Housing set, got {len(df)} rows"
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df["MedHouseVal"].notna().all(), "target column has NaN values"
    assert df[EXPECTED_FEATURES].notna().all().all(), "feature columns have NaN values"
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in EXPECTED_FEATURES)
    assert df["MedHouseVal"].between(14_999, 500_001).all()


def test_load_returns_dataframe_only_when_cached(monkeypatch):
    _require_csv()
    from src.load_data import load

    monkeypatch.chdir(ROOT)
    df = load()
    assert len(df) == 20640
    assert list(df.columns) == EXPECTED_COLUMNS


def test_train_model_module_exposes_features_and_load(monkeypatch):
    _require_csv()
    import matplotlib.pyplot as plt

    # train_model.py saves a plot at import time; keep the repo untouched.
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.chdir(ROOT)

    from src.train_model import FEATURES, load

    assert FEATURES == EXPECTED_FEATURES
    assert len(set(FEATURES)) == 8
    assert callable(load)
    assert len(load()) == 20640


def test_tiny_random_forest_predicts_finite_floats():
    _require_csv()
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(CSV)
    train, probe = df.iloc[:200], df.iloc[200:220]

    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(train[EXPECTED_FEATURES], train["MedHouseVal"])
    preds = model.predict(probe[EXPECTED_FEATURES])

    assert preds.shape == (20,)
    assert np.issubdtype(preds.dtype, np.floating)
    assert np.isfinite(preds).all(), "tiny forest produced non-finite predictions"
    assert (preds > 0).all()
    assert len(model.feature_importances_) == 8


def test_predict_cli_uses_features_and_target():
    src = (ROOT / "src" / "predict.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.train_model":
            imported.update(a.name for a in node.names)
    assert "FEATURES" in imported
    assert "MedHouseVal" in src
    assert "--latitude" in src and "--longitude" in src
