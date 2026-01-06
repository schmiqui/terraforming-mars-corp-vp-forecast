import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np



def make_onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _print_metrics(y_true, y_pred, label="TEST"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{label} metrics")
    print("-" * (len(label) + 8))
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")

# todo tune

def build_pipeline(X_columns, model_type="LinearRegression", n_estimators=800, random_state=42):
    cat_cols = []
    if "corporation" in X_columns:
        cat_cols.append("corporation")

    num_cols = []
    for col in X_columns:
        if col not in cat_cols:
            num_cols.append(col)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),

            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_dense()),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "Huber":
        model = HuberRegressor()
    elif model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2
        )
    else:
        model = LinearRegression()

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model)
    ])

    return pipe


def train_pipeline(df, model_type="LinearRegression", n_estimators=800, random_state=42):

    y = df["points"]
    X = df.drop(columns=["points"])

    if "game_id" in X.columns:
        X = X.drop(columns=["game_id"])

    if "game_id" in df.columns and df["game_id"].duplicated().any():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df["game_id"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    pipe = build_pipeline(
        X_columns=list(X.columns),
        model_type=model_type,
        n_estimators=n_estimators,
        random_state=random_state
    )

    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)

    _print_metrics(y_train, y_pred_train, label="TRAIN")
    _print_metrics(y_test, y_pred_test, label="TEST")

    info = {
        "model_type": model_type,
        "train_score": float(train_score),
        "test_score": float(test_score),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "feature_cols": list(X.columns)
    }

    return pipe, info


def train_from_csv_and_save(csv_path, model_path, model_type="LinearRegression", n_estimators=800, random_state=42):
    df = pd.read_csv(csv_path)

    if "points" not in df.columns:
        raise ValueError("CSV musí obsahovať stĺpec 'points'.")

    pipe, info = train_pipeline(
        df=df,
        model_type=model_type,
        n_estimators=n_estimators,
        random_state=random_state
    )

    joblib.dump(pipe, model_path)
    return info


def load_model(model_path):
    return joblib.load(model_path)