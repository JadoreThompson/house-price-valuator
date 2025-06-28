import math
import pandas as pd


##############################
# Helpers
##############################
def calc_std(vals) -> float:
    l = len(vals)

    if l == 1:
        return 0

    avg = sum(vals) / l
    diffs = [(v - avg) ** 2 for v in vals]
    return math.sqrt(sum(diffs) / (l - 1))


##############################
# Features
##############################
def append_sqm_per_bed_regional(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    key = "sqm_per_bedroom"

    if expanding:
        df[key] = (
            df.groupby("city")
            .apply(lambda g: (g["sqm"] / g["bedrooms"]).expanding().mean())
            .reset_index(level=0, drop=True)
        )
    else:
        df[key] = df["sqm"] / df["bedrooms"]

    return df


def append_sqm_per_bed_individual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sqm_per_bed"] = df["sqm"] / df["bedrooms"]
    return df


def append_sqm_per_bed_global(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    key = "sqm_per_bedroom"

    if expanding:
        df[key] = (df["sqm"] / df["bedrooms"]).expanding().mean()
    else:
        df[key] = df["sqm"] / df["bedrooms"]

    return df


def append_sqm_per_bed(
    df: pd.DataFrame,
    expanding: bool = False,
    regional: bool = False,
    individual: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    if regional:
        return append_sqm_per_bed_regional(df, expanding)
    if individual:
        return append_sqm_per_bed_individual(df)
    return append_sqm_per_bed_global(df, expanding)


def append_sqm_per_room_regional(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    key = "sqm_per_room"

    if expanding:
        df[key] = (
            df.groupby("city")
            .apply(
                lambda g: (
                    g["sqm"] / (g["bedrooms"] + g["bathrooms"] + g["receptions"])
                )
                .expanding()
                .mean()
            )
            .reset_index(level=0, drop=True)
        )
    else:
        df[key] = df["sqm"] / (df["bedrooms"] + df["bathrooms"] + df["receptions"])

    return df


def append_sqm_per_room_individual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sqm_per_room"] = df["sqm"] / (df["bedrooms"] + df["receptions"])
    return df


def append_sqm_per_room_global(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    key = "sqm_per_room"

    if expanding:
        df[key] = (
            (df["sqm"] / (df["bedrooms"] + df["bathrooms"] + df["receptions"]))
            .expanding()
            .mean()
        )
    else:
        df[key] = df["sqm"] / (df["bedrooms"] + df["bathrooms"] + df["receptions"])

    return df


def append_sqm_per_room(
    df: pd.DataFrame,
    expanding: bool = False,
    regional: bool = False,
    individual: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    if regional:
        return append_sqm_per_room_regional(df, expanding)
    if individual:
        return append_sqm_per_room_individual(df)
    return append_sqm_per_room_global(df, expanding)


def append_avg_price_per_sqm_regional(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df["price_per_sqm"] = df["price"] / df["sqm"]
    df = df.sort_values(["sold_year", "sold_month"])
    key = "regional_avg_price_per_sqm"

    if expanding:
        df[key] = (
            df.groupby("city")["price_per_sqm"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df[key] = df.groupby("city")["price_per_sqm"].transform("mean")

    return df


def append_avg_price_per_sqm_global(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df["price_per_sqm"] = df["price"] / df["sqm"]
    df = df.sort_values(["sold_year", "sold_month"])

    if expanding:
        df["avg_price_per_sqm"] = df["price_per_sqm"].expanding().mean()
    else:
        df["avg_price_per_sqm"] = df["price_per_sqm"].mean()

    return df


def append_avg_price_per_sqm(
    df: pd.DataFrame, expanding: bool = False, regional: bool = False
) -> pd.DataFrame:
    if regional:
        return append_avg_price_per_sqm_regional(df, expanding=expanding)
    else:
        return append_avg_price_per_sqm_global(df, expanding=expanding)


def append_avg_price_regional(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    avg_key = "regional_avg_price"

    if expanding:
        df[avg_key] = (
            df.groupby("city")["price"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df[avg_key] = df.groupby("city")["price"].transform("mean")

    return df


def append_avg_price_global(df: pd.DataFrame, expanding: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    avg_key = "avg_price"

    if expanding:
        df[avg_key] = df["price"].expanding().mean()
    else:
        df[avg_key] = df["price"].mean()

    return df


def append_avg_price(
    df: pd.DataFrame, expanding: bool = False, regional: bool = False
) -> pd.DataFrame:
    """Delegates to the appropriate average price appender."""
    if regional:
        return append_avg_price_regional(df, expanding=expanding)
    else:
        return append_avg_price_global(df, expanding=expanding)


def append_price_std_regional(
    df: pd.DataFrame, expanding: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    std_key = "regional_std"

    if expanding:
        df[std_key] = (
            df.groupby("city")["price"]
            .expanding()
            .apply(calc_std)
            .reset_index(level=0, drop=True)
        )
    else:
        groups = []
        for _, g in df.groupby("city"):
            std = calc_std(g["price"])
            g[std_key] = std
            groups.append(g)
        df = pd.concat(groups)

    return df


def append_price_std_global(df: pd.DataFrame, expanding: bool = False) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sold_year", "sold_month"])
    std_key = "std"

    if expanding:
        df[std_key] = (
            df["price"].expanding(min_periods=1).apply(calc_std).reset_index(drop=True)
        )
    else:
        std = calc_std(df["price"])
        df[std_key] = std

    return df

def append_room_ratio_regional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["receptions"]
    
    city_avg = df.groupby("city")["total_rooms"].transform("mean")
    df["room_ratio"] = df["total_rooms"] / city_avg
    
    return df

def append_room_ratio_global(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["receptions"]
    
    global_avg = df["total_rooms"].mean()
    df["room_ratio"] = df["total_rooms"] / global_avg
    
    return df

def append_room_ratio(df: pd.DataFrame, regional: bool = False) -> pd.DataFrame:
    if regional:
        return append_room_ratio_regional(df)
    return append_room_ratio_global(df)


def append_price_std(
    df: pd.DataFrame, expanding: bool = False, regional: bool = False
) -> pd.DataFrame:
    """Delegates to the appropriate std appender."""
    if regional:
        return append_price_std_regional(df, expanding=expanding)
    else:
        return append_price_std_global(df, expanding=expanding)


def append_total_rooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["receptions"]
    return df


def append_total_crime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_crime"] = sum(df[col] for col in df.columns if col.startswith("crime"))
    return df


def append_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["sold_month"]
    df["quarter"] = (df["sold_month"] - 1) // 3 + 1
    df["is_spring_summer"] = df["sold_month"].isin([3, 4, 5, 6, 7, 8]).astype(int)
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
