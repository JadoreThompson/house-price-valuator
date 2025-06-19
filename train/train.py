import json
import os
import pandas as pd
import ydf

from sklearn.linear_model import LinearRegression
from .features import (
    append_bed_per_sqm,
    append_rooms_per_sqm,
    append_total_crime,
    append_total_rooms,
)
from .utils import calculate_success_rate, get_train_test, prepare_dataset


LEARNER = ydf.GradientBoostedTreesLearner(
    "target", task=ydf.Task.REGRESSION, max_depth=100, num_trees=1000
)


def append_features(df: pd.DataFrame) -> pd.DataFrame:
    df = append_bed_per_sqm(df)
    df = append_rooms_per_sqm(df)
    df = append_total_crime(df)
    df = append_total_rooms(df)
    return df


def train_forest(threshold: float = 50_000):
    df = prepare_dataset()
    df = append_features(df)

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "postcode",
            "street",
            "address",
            "closest_school_gender",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
            "num_schools",
        ]
    )

    X_train, X_test, y_train, y_test = get_train_test(df)

    train_df = X_train.copy()
    train_df["target"] = y_train
    model = LEARNER.train(train_df)
    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train_linear_reg(threshold: float = 50_000):
    df = prepare_dataset()
    df = append_features(df)

    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "postcode",
            "street",
            "address",
            "closest_school_gender",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
        ]
    )

    model = LinearRegression()

    for col in df.columns.tolist():
        if df[col].dtype == "object":
            df = pd.get_dummies(df, columns=[col])

    X_train, X_test, y_train, y_test = get_train_test(df)

    model.fit(X_train, y_train)
    return model, *calculate_success_rate(model, X_test, y_test, threshold)


def train():
    for title, train_func in (
        ("Forest", train_forest),
        ("Linear Regression", train_linear_reg),
    ):
        _, success_rate, _ = train_func(threshold=10_000)
        print(f"{title} Success Rate: {success_rate}%")


if __name__ == "__main__":
    train()
