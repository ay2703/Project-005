"""
data_loader.py
==============
Loads the master C5 dataset and merges FFA volume data.
Produces df_main: forward-filled DataFrame with VWAP columns.
"""

import pandas as pd
import numpy as np


# paths 
MASTER_PATH = "/Users/b23/Downloads/Forecast model/master_dataset_2018.xlsx"
VOLUME_PATH = (
    "/Users/b23/Downloads/Forecast model/"
    "balticCape C5 FFA Volumes Daily FFADV_C5 010118 010426 040226 (2).xlsx"
)

VWAP_WINDOWS = [7, 30, 60]   # rolling VWAP window sizes (calendar days)


def load_master(path: str = MASTER_PATH) -> pd.DataFrame:
    """Load master dataset and forward-fill."""
    df = pd.read_excel(path)
    df = df.ffill()
    return df


def merge_volume(df: pd.DataFrame, volume_path: str = VOLUME_PATH) -> pd.DataFrame:
    """
    Merge FFA volume data, compute price×volume, and add rolling VWAP columns.

    New columns added
    -----------------
    price_volume  : C5_rate_USD_tonne * Value
    vwap_{w}d     : VWAP over w calendar days (w in VWAP_WINDOWS)
    """
    volume = pd.read_excel(volume_path)
    volume["Date"] = pd.to_datetime(volume["Date"])

    df = df.merge(volume, on="Date", how="left")
    df["price_volume"] = df["C5_rate_USD_tonne"] * df["Value"]

    for w in VWAP_WINDOWS:
        df[f"vwap_{w}d"] = (
            df["price_volume"].rolling(w).sum()
            / df["Value"].rolling(w).sum()
        )

    return df


def prepare_main(
    master_path: str = MASTER_PATH,
    volume_path: str = VOLUME_PATH,
) -> pd.DataFrame:
    """
    Full load + merge + index setup pipeline.

    Returns
    -------
    df_main : DatetimeIndex DataFrame, forward-filled, no leading NaN rows.
    """
    df = load_master(master_path)
    df = merge_volume(df, volume_path)

    df["Date"] = pd.to_datetime(df["Date"]) if "Date" in df.columns else df.index
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()

    print(f"df_main shape: {df.shape}")
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()}")
    return df


if __name__ == "__main__":
    df_main = prepare_main()
    print(df_main.head())
