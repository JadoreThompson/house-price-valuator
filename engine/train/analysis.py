import matplotlib.pyplot as plt
import numpy as np

from .features import (
    append_city_avg_price,
    append_city_avg_price_per_sqm,
    append_sqm_per_bed,
    append_sqm_per_room,
    append_total_crime,
    append_total_rooms,
)
from .utils import prepare_dataset


def simple_scatter():
    df = prepare_dataset()
    df = df[["target", "lat", "lng"]]

    plt.figure(figsize=(10, 6))
    plt.scatter(df["lat"], df["target"])
    plt.show()


def plot_corr_heatmap() -> None:
    df = prepare_dataset()
    df = append_sqm_per_bed(df)
    df = append_sqm_per_room(df)
    df = append_total_crime(df)
    df = append_total_rooms(df)
    df = append_city_avg_price(df)
    df = append_city_avg_price_per_sqm(df)
    
    df = df.drop(
        columns=[
            "uprn",
            "price_str",
            "sold_date",
            "postcode",
            "street",
            "address",
            "sold_day",
            "sold_month",
            "sold_date_unix_epoch",
            "closest_school_gender",
            "closest_school_name",
            "closest_poi_name",
            "num_schools",
            "lat",
            "lng",
        ]
    )

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(16, 16))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_corr_heatmap()
