import json
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import warnings
import ydf

from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers

from .features import (
    append_avg_price,
    append_avg_price_per_sqm,
    append_price_std,
    append_room_ratio,
    append_sqm_per_bed,
    append_closest_school_gender_dummies,
    append_epc_rating_dummies,
    append_property_type_dummies,
    append_sqm_per_room,
    append_tenure_dummies,
    append_time_features,
    append_total_crime,
    append_total_rooms,
)
from .utils import (
    calculate_success_rate,
    get_train_test,
    prepare_dataset,
    save_var_importances,
)
from ..config import MISC_FOLDER, MODELS_FOLDER


def train_forest(
    threshold: float,
    save_train_dataset: bool = False,
    save_model: bool = False,
    pickle_: bool = True,
    fname: str = "forest_model.pkl",
):
    df = prepare_dataset()
    df = append_sqm_per_bed(df, regional=True)
    df = append_sqm_per_room(df, True, True)
    df = append_total_crime(df)
    df = append_total_rooms(df)
    df = append_avg_price(df, regional=True)
    df = append_avg_price_per_sqm(df, regional=True)
    df = append_room_ratio(df)
    df = df[df["city"] != "london"]

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
            "postcode",
            "street",
            "address",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "num_schools",
        ]
    )

    X_train, X_test, y_train, y_test = get_train_test(df)

    if save_train_dataset:
        df = X_train.copy()
        df["target"] = y_train
        df.to_csv(os.path.join(MISC_FOLDER, "forest.csv"), index=False)

    train_df = X_train.copy()
    train_df["target"] = y_train

    learner = ydf.GradientBoostedTreesLearner(
        "target", task=ydf.Task.REGRESSION, max_depth=100, num_trees=200
    )
    model = learner.train(pd.concat([train_df], ignore_index=True))

    save_var_importances(model.variable_importances())
    if save_model:
        model_path = os.path.join(MODELS_FOLDER, fname)

        if pickle_:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model.save(model_path)

        print(f"Model saved to {model_path}")

    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train_linear_reg(
    threshold: float,
    save_train_dataset: bool = False,
    save_model: bool = False,
    pickle_: bool = True,
    fname: str = "linear_regression_model.pkl",
):
    df = prepare_dataset()

    df = append_sqm_per_bed(df, individual=True)
    df = append_sqm_per_room(df, individual=True)
    df = append_avg_price(df, regional=True)
    df = append_avg_price(df, expanding=True, regional=True)

    df = append_price_std(df, False, False)
    df = append_total_crime(df)
    df = append_total_rooms(df)

    df = append_property_type_dummies(df)
    df = append_tenure_dummies(df)
    df = append_epc_rating_dummies(df)

    df = append_time_features(df)
    df = df[df["city"] != "london"]

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "sold_year",
            "sold_month",
            "sold_day",
            "is_spring_summer",
            "sold_date_unix_epoch",
            "city",
            "postcode",
            "street",
            "address",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "crime_violence",
            "num_pois",
            "avg_distance_all",
            "min_distance_school",
        ]
    )
    df = df.dropna()

    X_train, X_test, y_train, y_test = get_train_test(df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    if save_train_dataset:
        df = X_train.copy()
        df["target"] = y_train
        df.to_csv(os.path.join(MISC_FOLDER, "train-lr.csv"), index=False)

    if save_model:
        folder = os.path.join(MODELS_FOLDER, "linear_reg")
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        json.dump(
            X_train.columns.tolist(), open(os.path.join(folder, "features.json"), "w")
        )

        model_path = os.path.join(folder, fname)

        if pickle_:
            if not model_path.endswith(".pkl"):
                raise ValueError("For pickle models, the file must end with '.pkl'")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            if not model_path.endswith(".keras"):
                raise ValueError("For Keras models, the file must end with '.keras'")
            model.save(model_path)

        print(f"Model saved to {model_path}")

    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train_neural_net(
    threshold: float,
    save_train_dataset: bool = False,
    save_model: bool = False,
    *,
    epochs: int = 100,
    test_pct: float = 0.3,
) -> dict:
    scaler = StandardScaler()

    df = prepare_dataset()

    df = append_sqm_per_bed(df, individual=True)
    df = append_sqm_per_room(df, individual=True)
    df = append_avg_price(df, regional=True)
    df = append_avg_price(df, expanding=True, regional=True)
    df = append_price_std(df, False, False)
    df = append_total_crime(df)
    df = append_total_rooms(df)

    df = append_property_type_dummies(df)
    df = append_tenure_dummies(df)
    df = append_epc_rating_dummies(df)

    df = append_time_features(df)
    # df = df[df["city"] != "london"]

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "sold_year",
            "sold_month",
            "sold_day",
            "is_spring_summer",
            "sold_date_unix_epoch",
            "city",
            "postcode",
            "street",
            "address",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "crime_violence",
        ]
    )
    df = df[
        [
            "target",
            "price",
            "sqm",
            "bedrooms",
            "bathrooms",
            "receptions",
            "regional_avg_price",
            "total_crime",
        ]
    ]

    X_train, X_test, y_train, y_test = get_train_test(df, 1 - test_pct)
    n_feautures = len(X_train.columns)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential(
        [
            layers.Dense(
                n_feautures, activation="relu", input_shape=(X_train_scaled.shape[1],)
            ),
            layers.Dense(n_feautures // 2, activation="relu"),
            layers.Dense(1),  # Output layer for regression
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=test_pct,
        epochs=epochs,
        batch_size=32,
        verbose=1,
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    result = {
        "model": model,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    if save_train_dataset:
        try:
            X_train.to_csv(os.path.join(MISC_FOLDER, "temp.csv"), index=False)
        except Exception as e:
            msg = f"Error saving train dataset: {e}"
            warnings.warn(msg)

    if save_model:
        model_path = os.path.join(MODELS_FOLDER, "neural_net_model.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    return result


def train():
    ydf.verbose(0)
    results: list[tuple[str, tuple | dict]] = []

    for title, train_func, kwargs in (
        ("Random Forest", train_forest, {}),
        ("Linear Regression", train_linear_reg, {"save_model": True}),
        # ("Neural Network", train_neural_net, {"epochs": 300}),
    ):
        results.append(
            (
                title,
                train_func(threshold=10_000, save_train_dataset=True, **kwargs),
            )
        )

    for title, result in results:
        if isinstance(result, tuple):
            _, success_rate, _ = result
            print(f"{title} Success Rate: {success_rate}%")
        elif isinstance(result, dict):
            pprint({k: v for k, v in result.items() if k != "model"})


if __name__ == "__main__":
    train()
