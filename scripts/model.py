import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neural_network import MLPRegressor


CSV_PATH = "/Users/jakubsmihula/PycharmProjects/ML/Project/terraforming-mars-corp-vp-forecast/scripts/games.csv"  # <-- zmeňte


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_onehot_dense():
    # kompatibilita medzi sklearn verziami
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_onehot_sparse():
    # kompatibilita medzi sklearn verziami
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def evaluate(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": rmse(y_test, preds),
        "R2": float(r2_score(y_test, preds))
    }


def main():
    df = pd.read_csv(CSV_PATH)

    # target
    y = df["points"]
    X = df.drop(columns=["points"])

    # Vyhoďte game_id z featur, ak nechcete (často je len identifikátor)
    if "game_id" in X.columns:
        X_no_id = X.drop(columns=["game_id"])
    else:
        X_no_id = X

    # Kategórie
    cat_cols = []
    if "corporation" in X_no_id.columns:
        cat_cols.append("corporation")

    num_cols = [c for c in X_no_id.columns if c not in cat_cols]

    # Split – ak máte viac riadkov na jedno game_id, použite group split
    if "game_id" in df.columns and df["game_id"].duplicated().any():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_no_id, y, groups=df["game_id"]))
        X_train, X_test = X_no_id.iloc[train_idx], X_no_id.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_name = "GroupShuffleSplit(game_id)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_no_id, y, test_size=0.2, random_state=42
        )
        split_name = "train_test_split"

    print(f"Split: {split_name} | Train={len(X_train)} Test={len(X_test)}")

    # Preprocess pre „skôr lineárne / NN / SVM / KNN“ (škálovanie čísel + dense onehot)
    preprocess_dense_scaled = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_dense())
            ]), cat_cols),
        ],
        remainder="drop"
    )

    # Preprocess pre stromy/boosting (škálovanie netreba, sparse OK)
    preprocess_sparse = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_sparse())
            ]), cat_cols),
        ],
        remainder="drop"
    )

    # Niektoré modely neznesú sparse => konvert na dense (bezpečné pri 1500 riadkoch)
    to_dense = FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x)

    models = []

    # Lineárne
    models += [
        ("LinearRegression", preprocess_dense_scaled, LinearRegression()),
        ("Ridge",            preprocess_dense_scaled, Ridge(alpha=1.0, random_state=42)),
        ("Lasso",            preprocess_dense_scaled, Lasso(alpha=0.01, random_state=42, max_iter=10000)),
        ("ElasticNet",       preprocess_dense_scaled, ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
        ("HuberRegressor",   preprocess_dense_scaled, HuberRegressor()),
    ]

    # SVM / KNN
    models += [
        ("KNN",              preprocess_dense_scaled, KNeighborsRegressor(n_neighbors=25)),
        ("SVR_rbf",          preprocess_dense_scaled, SVR(C=10.0, gamma="scale")),
        ("LinearSVR",        preprocess_dense_scaled, LinearSVR(random_state=42, max_iter=20000)),
    ]

    # Stromy / boosting
    models += [
        ("DecisionTree",     preprocess_sparse, DecisionTreeRegressor(random_state=42, min_samples_leaf=5)),
        ("RandomForest",     preprocess_sparse, RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ("ExtraTrees",       preprocess_sparse, ExtraTreesRegressor(n_estimators=1200, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ("GradientBoosting", preprocess_sparse, GradientBoostingRegressor(random_state=42)),
        ("HistGBR",          preprocess_sparse, HistGradientBoostingRegressor(random_state=42)),
        ("AdaBoost",         preprocess_sparse, AdaBoostRegressor(random_state=42)),
    ]

    # Neurónová sieť (MLP) – potrebuje scaled + dense
    models += [
        ("MLP_small", preprocess_dense_scaled,
         MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=2000,
                      early_stopping=True, n_iter_no_change=30)),
        ("MLP_wide", preprocess_dense_scaled,
         MLPRegressor(hidden_layer_sizes=(128, 128), random_state=42, max_iter=3000,
                      early_stopping=True, n_iter_no_change=30)),
    ]

    results = []
    for name, pre, model in models:
        try:
            # Ak preprocess je sparse a model by mohol mať problém, môžete zapnúť to_dense:
            # (stromy zvyčajne sparse zvládnu, MLP/SVR/KNN nie)
            if name in ["DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting", "HistGBR", "AdaBoost"]:
                pipe = Pipeline(steps=[("prep", pre), ("model", model)])
            else:
                pipe = Pipeline(steps=[("prep", pre), ("model", model)])

            m = evaluate(pipe, X_train, X_test, y_train, y_test)
            results.append((name, m["MAE"], m["RMSE"], m["R2"]))
        except Exception as e:
            results.append((name, np.nan, np.nan, np.nan))
            print(f"[SKIP] {name}: {e}")

    res_df = pd.DataFrame(results, columns=["model", "MAE", "RMSE", "R2"]).sort_values("MAE")
    print("\n=== Results (sorted by MAE) ===")
    print(res_df.to_string(index=False))



if __name__ == "__main__":
    main()