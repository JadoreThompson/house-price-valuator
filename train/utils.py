import os
import pandas as pd
from json import dump
from typing import Any, Protocol, runtime_checkable
from config import DATASETS_FOLDER


@runtime_checkable
class SupportsPredict(Protocol):
    """
    Protocol for models that implement a `predict` method.
    """
    def predict(*args, **kwargs) -> Any: ...


def prepare_dataset() -> pd.DataFrame:
    """
    Loads and preprocesses the cleaned dataset.

    Returns:
        pd.DataFrame.
    """
    df = pd.read_csv(os.path.join(DATASETS_FOLDER, "cleaned.csv"))
    df = df.dropna()

    df["uprn"] = df["uprn"].astype("object")
    df["sqm"] = pd.to_numeric(df["sqm"])
    df["sold_date"] = pd.to_datetime(df["sold_date"])
    df["sold_date_unix_epoch"] = df["sold_date"].apply(lambda x: x.timestamp())
    df["sold_year"] = df["sold_date"].apply(lambda x: x.date().year)
    df["sold_month"] = df["sold_date"].apply(lambda x: x.date().month)
    df["sold_day"] = df["sold_date"].apply(lambda x: x.date().day)
    df["target"] = df.pop("price")

    return df


def get_train_test(
    df: pd.DataFrame, train_pct: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into train/test sets and separates features (X) and target (y).

    Args:
        df (pd.DataFrame): Full dataset including 'target' column.
        train_pct (float): Proportion of data to use for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    train_size = round(len(df) * train_pct)

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def calculate_success_rate(
    model: SupportsPredict, x: pd.DataFrame, y: pd.Series, threshold: float = 10000
) -> tuple[float, list[float]]:
    """
    Calculate the success rate of predictions being within the given threshold.

    Args:
        model: Trained model implementing the predict method.
        x (pd.DataFrame): Input features.
        y (pd.Series): Actual target values.
        threshold (float): Acceptable absolute error range. Default is 10,000.

    Returns:
        Tuple:
            - float: Success rate (as a percentage) to 2 decimal places.
            - list[float]: List of predictions.
    """
    y_pred = model.predict(x)
    errors = abs(y - y_pred)
    success_count = (errors <= threshold).sum()

    return round((success_count / len(x)) * 100, 2), y_pred


def save_var_importances(importances) -> None:
    """
    Saves feature importances as a JSON file to the msc/vi.json path.

    Args:
        importances: Variable importance dictionary or list to be serialized.
    """
    dump(
        importances,
        open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "msc", "vi.json"),
            "w",
        ),
    )
