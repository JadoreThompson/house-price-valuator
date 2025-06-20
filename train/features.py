import math
import pandas as pd


def append_sqm_per_bed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sqm_per_bedroom"] = df["sqm"] / df["bedrooms"]
    return df


def append_sqm_per_room(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sqm_per_room"] = df["sqm"] / (
        df["bedrooms"] + df["bathrooms"] + df["receptions"]
    )
    return df


def append_city_avg_price(df: pd.DataFrame) -> pd.DataFrame:
    city_avg_price_s = df.groupby("city")["price"].mean()
    city_avg_price_s.name = "city_average_price"
    df = df.merge(city_avg_price_s, on=["city"])
    return df


def append_city_avg_price_per_sqm(df: pd.DataFrame):
    df = df.copy()
    df["price_per_sqm"] = df["price"] / df["sqm"]
    city_avg_price_per_sqm = df.groupby("city")["price_per_sqm"].mean()
    city_avg_price_per_sqm.name = "city_avg_price_per_sqm"
    df = df.merge(city_avg_price_per_sqm, on=["city"])
    return df


def append_avg_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    df["avg_price"] = df["price"].expanding().mean()
    return df


def append_price_std(df: pd.DataFrame) -> pd.DataFrame:
    df = append_avg_price(df)
    df["diff"] = (df["price"] - df["avg_price"]) ** 2
    df["std"] = math.sqrt(sum(df["diff"]) / (len(df) - 1))
    return df.drop(columns=["diff", "std"])


def append_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["receptions"]
    return df


def append_total_crime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_crime"] = sum(df[col] for col in df.columns if col.startswith("crime"))
    return df


def append_closest_school_gender_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["closest_school_gender"])
    return df


def append_property_type_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["property_type"])
    return df


def append_tenure_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["tenure"])
    return df


def append_epc_rating_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["epc_rating"])
    return df
