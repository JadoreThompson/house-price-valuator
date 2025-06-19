import pandas as pd


def append_bed_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bedrooms_per_sqm"] = df["sqm"] / df["bedrooms"]
    return df


def append_rooms_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_sqm"] = df["sqm"] / (
        df["bedrooms"] + df["bathrooms"] + df["receptions"]
    )
    return df


def append_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["receptions"]
    return df


def append_total_crime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_crime"] = sum(df[col] for col in df.columns if col.startswith("crime"))
    return df
