"""
feature_engineering.py
=======================
All feature construction logic for the C5 probabilistic forecasting pipeline.

Public API
----------
build_features(df, horizon, ...)  → enriched DataFrame
get_feature_groups(df)            → dict of feature-name lists
"""

import numpy as np
import pandas as pd


#  LAG SCHEDULES

C5_AR_LAGS = [1, 2, 3, 5, 10, 21, 42, 62]

FFA_LAGS = {
    "C5_FFA_C5_1MON": [1, 5, 21],
    "C5_FFA_C5_2MON": [1, 5, 21],
    "C3_FFA_C3_1MON": [1, 5, 21],
    "C3_FFA_C3_2MON": [1, 5, 21],
}

MACRO_MONTHLY_LAGS = {
    "China_M2_bn_USD":                [21, 42, 62],
    "China_New_Loan_bn_CNY":           [21, 42, 62],
    "China_Property_Price_YoY":        [21, 42, 62],
    "China_PMI_Official":              [21, 42],
    "China_IP_YoY_pct":                [21, 42, 62],
    "Japan_IP_YoY_pct":                [21, 42, 62],
    "SouthKorea_IP_YoY_pct":           [21, 42, 62],
    "China_PBOC_Rate_pct":             [1, 5, 21],
    "China_LPI_Value":                 [21, 42, 62],
    "China_Mainland_TPU":              [21, 42, 62],
}

STEEL_MONTHLY_LAGS = {
    "China_Steel_Prod_kt":             [21, 42, 62],
    "Japan_Steel_Prod_kt":             [21, 42],
    "SouthKorea_Steel_Prod_kt":        [21, 42],
    "China_BFI_Prod_kt":               [21, 42, 62],
    "Japan_BFI_Prod_kt":               [21, 42],
    "Korea_BFI_Prod_kt":               [21, 42],
    "China_CrudeSteel_Prod_10kt":      [21, 42, 62],
    "China_CrudeSteel_YoY_pct":        [21, 42],
    "China_PigIron_Prod_10kt":         [21, 42],
    "China_PigIron_YoY_pct":           [21, 42],
    "China_Rebar_Prod_10kt":           [21, 42],
    "China_Rebar_YoY_pct":             [21, 42],
    "China_IronOre_Prod_monthly_10kt": [21, 42, 62],
    "China_IronOre_Prod_YoY_pct":      [21, 42],
    "CNBS_IronOre_Prod_10kt":          [21, 42],
    "CNBS_IronOre_YoY_pct":            [21, 42],
    "CITIsteel_PMI":                   [21, 42],
    "CITIsteel_Production":            [21, 42],
    "CITIsteel_NewOrders":             [21, 42],
    "CITIsteel_ExportOrders":          [21, 42],
    "CITI_SteelProd_national_10mt":    [21, 42],
    "CITI_SteelProd_key_Mtpd":         [21, 42],
}

IRONORE_LAGS = {
    "IronOre_Spot_CFR_NChina_USD_t":   [1, 5, 21, 42],
    "CITI_IronOre_CFR62_USD_t":        [1, 5, 21, 42],
    "CITI_IronOreSpot_MB_USD_t":       [1, 5, 21],
    "China_IronOre_Stockpile_kt":      [5, 10, 21, 42],
    "CITI_OreInv_ports_Mt":            [5, 10, 21, 42],
    "CITI_OreInv_AUS_Mt":              [5, 10, 21],
    "CITI_OreInv_BRA_Mt":              [5, 10, 21],
    "CITI_OreInv_days_mills":          [5, 10, 21],
}

SEABORNE_LAGS = {
    "AUS_IronOre_Seaborne_Exports_Mt_SIN":   [21, 42, 62],
    "China_IronOre_Seaborne_Imports_Mt_SIN": [21, 42, 62],
    "China_Coal_Seaborne_Imports_Mt_SIN":    [21, 42],
    "China_Steel_Seaborne_Exports_Mt_SIN":   [21, 42],
    "AUS_IronOre_Exports_Mt_CLRK":           [21, 42, 62],
    "Brazil_IronOre_Exports_Mt_CLRK":        [21, 42, 62],
    "China_IronOre_Imports_Mt_CLRK":         [21, 42, 62],
    "SKorea_IronOre_Imports_Mt_CLRK":        [21, 42],
    "Japan_IronOre_Imports_Mt_CLRK":         [21, 42],
    "China_Steel_Exports_Mt_CLRK":           [21, 42],
    "TF_AUS_CHN_IronOre_MT":                 [1, 5, 10, 21, 42],
    "TF_AUS_CHN_Coal_MT":                    [1, 5, 10, 21],
    "TF_AUS_CHN_Bauxite_MT":                 [1, 5, 10, 21],
    "TF_AUS_JPN_IronOre_MT":                 [1, 5, 10, 21],
    "TF_AUS_SKorea_IronOre_MT":              [1, 5, 10],
    "TF_AUS_Taiwan_IronOre_MT":              [1, 5, 10],
    "TF_Brazil_CHN_IronOre_MT":              [1, 5, 21, 42],
    "TF_Brazil_SKorea_IronOre_MT":           [1, 5, 21],
    "TF_Brazil_Taiwan_IronOre_MT":           [1, 5, 21],
    "Guinea_CHN_Bauxite_Import_MT":          [1, 5, 21],
    "Guinea_Bauxite_Cape_Export_MT":         [1, 5, 21],
}

FLEET_LAGS = {
    "Cape_Fleet_DWT_mn_SIN":        [21, 42, 62],
    "Cape_Fleet_Growth_YoY_pct":    [21, 42, 62],
    "Cape_Deliveries_No_SIN":       [21, 42],
    "Cape_AvgSpeed_knots":          [1, 5, 10, 21],
    "Cape_Atlantic_Ships_No":       [1, 5, 10, 21],
    "Cape_Pacific_Ships_No":        [1, 5, 10, 21],
    "Cape_Atlantic_at_sea_ballast": [1, 5, 10, 21],
    "Cape_Atlantic_at_sea_laden":   [1, 5, 10],
    "Cape_Indian_at_sea_ballast":   [1, 5, 10, 21],
    "Cape_Indian_at_sea_laden":     [1, 5, 10],
    "Cape_Pacific_at_sea_ballast":  [1, 5, 10, 21],
    "Cape_Pacific_at_sea_laden":    [1, 5, 10],
}

PORT_LAGS = {
    "Cape_PortCongestion_pct_fleet": [1, 5, 10, 21],
    "Cape_PortCongestion_CHN_mDWT":  [1, 5, 10, 21],
}

BUNKER_LAGS = {
    "Platts_HFO380_SIN_ext_USD_t":  [1, 5, 21],
    "Platts_IFO380_SIN_USD_t":      [1, 5, 21],
    "Platts_VLSFO_SIN_ext_USD_t":   [1, 5, 21],
    "Platts_VLSFO_SIN_USD_t":       [1, 5, 21],
    "Bunker_HSFO380_CLRK_W_USD_t":  [1, 5, 21],
    "Bunker_VLSFO_CLRK_W_USD_t":    [1, 5, 21],
    "Bunker_MGO_CLRK_W_USD_t":      [1, 5, 21],
    "Bunker_HSFO380_SIN_USD_t":     [1, 5],
    "Bunker_VLSFO_SIN_USD_t":       [1, 5],
    "Bunker_MGO_SIN_USD_t":         [1, 5],
}

ENERGY_LAGS = {
    "Brent_Crude_USD_bbl":       [1, 5, 21],
    "Brent_1M_USD_bbl_SIN":      [1, 5, 21],
    "Coal_Price_AUS_USD_t":      [1, 5, 21, 42],
    "ThermalCoal_FOB_AUS_USD_t": [5, 21, 42],
    "Aluminum_LME_USD_t":        [1, 5, 21],
    "Wheat_CBOT_USdc_bu":        [1, 5, 21],
    "Copper_LME_USD_t":          [1, 5, 21],
    "MSCI_WorldEnergy":          [1, 5, 21],
}

STEEL_PRICE_LAGS = {
    "Steel_Rebar_CNY_t":                [1, 5, 21],
    "China_SteelPlate_CNY_t":           [1, 5, 21],
    "China_SteelPlate_USD_t":           [1, 5, 21],
    "Japan_Steel_ShipPlate_USD_t":      [1, 5, 21, 42],
    "Korea_Steel_ShipPlate_USD_t":      [1, 5, 21, 42],
    "CITI_ShanghaiCoke_CNY_t":          [1, 5, 21],
    "CITI_CokingCoal_Shanxi_Liulin_#4": [1, 5, 21],
    "CITI_Rebar_Shanghai_CNY_t":        [1, 5, 21],
    "CITI_China_Rebar12mm_CNY_t":       [1, 5, 21],
    "CITI_China_HRC3mm_CNY_t":          [1, 5, 21],
    "CITI_China_Billet_Q235_CNY_t":     [1, 5, 21],
    "CITI_Steel_SEA_USD_t":             [1, 5, 21],
    "CITI_Steel_US_USD_t":              [1, 5, 21],
    "CITI_AppConsump_Rebar_Mt":         [5, 21, 42],
    "CITI_AppConsump_Total_Mt":         [5, 21, 42],
}

FX_LAGS = {
    "AUDUSD_SIN":    [1, 5, 21],
    "EURUSD_SIN":    [1, 5, 21],
    "CNYUSD_SIN":    [1, 5, 21],
    "USD_Index_DXY": [1, 5, 21],
    "EURUSD":        [1, 5],
    "CNYUSD":        [1, 5],
    "AUDUSD":        [1, 5],
}

SENTIMENT_LAGS = {
    "VIX":         [1, 5, 21],
    "VHSI":        [1, 5, 21],
    "VSTOXX":      [1, 5, 21],
    "HangSeng":    [1, 5, 21],
    "SP500":       [1, 5, 21],
    "KOSPI":       [1, 5, 21],
    "MSCI_EM":     [1, 5, 21],
    "MSCI_Global": [1, 5, 21],
    "Nikkei225":   [1, 5, 21],
    "EuroStoxx50": [1, 5, 21],
}

# Combined dict for convenience
ALL_EXTERNAL_LAGS: dict = {}
for _d in [
    FFA_LAGS, MACRO_MONTHLY_LAGS, STEEL_MONTHLY_LAGS, IRONORE_LAGS,
    SEABORNE_LAGS, FLEET_LAGS, PORT_LAGS, BUNKER_LAGS, ENERGY_LAGS,
    STEEL_PRICE_LAGS, FX_LAGS, SENTIMENT_LAGS,
]:
    ALL_EXTERNAL_LAGS.update(_d)


#  TECHNICAL INDICATORS  (all trailing, zero look-ahead)

def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = "C5_rate_USD_tonne",
    prefix: str = "",
) -> pd.DataFrame:
    p  = df[price_col].copy()
    px = prefix if prefix else ("" if price_col == "C5_rate_USD_tonne" else f"{price_col}_")

    # EMAs
    for span in [5, 12, 26, 50]:
        df[f"{px}EMA_{span}"] = p.ewm(span=span, adjust=False).mean()

    # MACD
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df[f"{px}MACD"]        = ema12 - ema26
    df[f"{px}MACD_signal"] = df[f"{px}MACD"].ewm(span=9, adjust=False).mean()
    df[f"{px}MACD_hist"]   = df[f"{px}MACD"] - df[f"{px}MACD_signal"]

    # RSI
    delta = p.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(com=13, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df[f"{px}RSI_14"] = 100 - (100 / (1 + rs))

    # Stochastic
    for w in [14, 21]:
        lo      = p.rolling(w).min()
        hi      = p.rolling(w).max()
        stoch_k = 100 * (p - lo) / (hi - lo).replace(0, np.nan)
        df[f"{px}Stoch_K_{w}"] = stoch_k
        df[f"{px}Stoch_D_{w}"] = stoch_k.rolling(3).mean()

    # Bollinger Bands
    for w in [20, 40]:
        mid = p.rolling(w).mean()
        std = p.rolling(w).std()
        df[f"{px}BB_mid_{w}"]   = mid
        df[f"{px}BB_upper_{w}"] = mid + 2 * std
        df[f"{px}BB_lower_{w}"] = mid - 2 * std
        df[f"{px}BB_width_{w}"] = (4 * std) / mid.replace(0, np.nan)
        df[f"{px}BB_pct_{w}"]   = (p - (mid - 2 * std)) / (4 * std).replace(0, np.nan)

    # Realised volatility
    log_ret = np.log(p / p.shift(1))
    for w in [5, 10, 21, 42]:
        df[f"{px}RealVol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Momentum / ROC
    for lag in [5, 10, 21, 42, 62]:
        df[f"{px}MOM_{lag}d"] = p - p.shift(lag)
        df[f"{px}ROC_{lag}d"] = p.pct_change(lag)

    # Z-score
    for w in [21, 63]:
        roll_mean = p.rolling(w).mean()
        roll_std  = p.rolling(w).std()
        df[f"{px}Zscore_{w}d"] = (p - roll_mean) / roll_std.replace(0, np.nan)

    # 52-week high/low
    df[f"{px}PctFrom52wHigh"]     = p / p.rolling(252).max() - 1
    df[f"{px}PctFrom52wLow"]      = p / p.rolling(252).min() - 1
    df[f"{px}Spread_EMA12_EMA26"] = df[f"{px}EMA_12"] - df[f"{px}EMA_26"]
    df[f"{px}Dist_from_EMA26"]    = (p - df[f"{px}EMA_26"]) / df[f"{px}EMA_26"].replace(0, np.nan)

    return df


#  CROSS / SPREAD FEATURES

def add_cross_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["C5_FFA_C5_1MON", "C5_FFA_C5_2MON"]:
        if col in df.columns:
            df[f"C5_vs_{col}_spread"]  = df["C5_rate_USD_tonne"] - df[col]
            df[f"C5_vs_{col}_pct_gap"] = df["C5_rate_USD_tonne"] / df[col].replace(0, np.nan) - 1

    if ("IronOre_Spot_CFR_NChina_USD_t" in df.columns
            and "Platts_HFO380_SIN_ext_USD_t" in df.columns):
        df["IronOre_vs_Bunker_ratio"] = (
            df["IronOre_Spot_CFR_NChina_USD_t"]
            / df["Platts_HFO380_SIN_ext_USD_t"].replace(0, np.nan)
        )

    if ("Brent_Crude_USD_bbl" in df.columns
            and "Platts_HFO380_SIN_ext_USD_t" in df.columns):
        df["Brent_vs_HFO_spread"] = (
            df["Brent_Crude_USD_bbl"] * 7.45
            - df["Platts_HFO380_SIN_ext_USD_t"]
        )

    for region in ["Atlantic", "Pacific"]:
        l_col = f"Cape_{region}_at_sea_laden"
        b_col = f"Cape_{region}_at_sea_ballast"
        if l_col in df.columns and b_col in df.columns:
            total = df[l_col] + df[b_col]
            df[f"Cape_{region}_utilisation"] = df[l_col] / total.replace(0, np.nan)

    for col in ["TF_AUS_CHN_IronOre_MT", "TF_Brazil_CHN_IronOre_MT"]:
        if col in df.columns:
            df[f"{col}_7d_sum"]  = df[col].rolling(7).sum()
            df[f"{col}_21d_sum"] = df[col].rolling(21).sum()
            df[f"{col}_mom_5d"]  = (
                df[col].rolling(5).sum()
                / df[col].rolling(21).sum().replace(0, np.nan) - 1
            )

    if "C3_FFA_C3_1MON" in df.columns and "C5_FFA_C5_1MON" in df.columns:
        df["C3_vs_C5_FFA_spread"] = df["C3_FFA_C3_1MON"] - df["C5_FFA_C5_1MON"]

    if "VIX" in df.columns and "IronOre_Spot_CFR_NChina_USD_t" in df.columns:
        df["RiskAdj_IronOre"] = (
            df["IronOre_Spot_CFR_NChina_USD_t"] / (1 + df["VIX"] / 100)
        )

    cong_cols = [
        c for c in df.columns
        if "PortCongestion" in c or "waiting" in c or "at_port" in c
    ]
    if cong_cols:
        df["Total_Congestion_proxy"] = df[cong_cols].sum(axis=1, min_count=1)

    vol       = df["Volume_FFA"].fillna(0) if "Volume_FFA" in df.columns \
                else pd.Series(1.0, index=df.index)
    direction = np.sign(df["C5_rate_USD_tonne"].diff().fillna(0))
    df["C5_OBV_proxy"] = (direction * vol).cumsum()

    return df


#  CALENDAR FEATURES

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["DayOfWeek"]     = idx.dayofweek
    df["MonthOfYear"]   = idx.month
    df["QuarterOfYear"] = idx.quarter
    df["WeekOfYear"]    = idx.isocalendar().week.astype(int)
    df["IsMonthEnd"]    = idx.is_month_end.astype(int)
    df["IsMonthStart"]  = idx.is_month_start.astype(int)
    df["IsQtrEnd"]      = idx.is_quarter_end.astype(int)
    df["CNY_season"]    = (
        ((idx.month == 1) & (idx.day >= 15)) | (idx.month == 2)
    ).astype(int)
    return df

#  MAIN BUILD FUNCTION

def build_features(
    df: pd.DataFrame,
    horizon: int = 62,
    add_tech_on_prices: list | None = None,
    drop_port2_monthly: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a clean df_main.

    Steps
    -----
    1. Drop Port2 month-name columns
    2. C5 AR lags + log-return lags
    3. External variable lags
    4. Rolling stats on key external series
    5. Technical indicators (C5 + selected prices)
    6. Cross / spread features
    7. Calendar features
    """
    # 1. Drop Port2 month-name columns
    if drop_port2_monthly:
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        month_cols = [c for c in df.columns if any(m in c for m in month_names)]
        df.drop(columns=month_cols, inplace=True, errors="ignore")
        print(f"  Dropped {len(month_cols)} Port2 month-name columns")

    n_orig = df.shape[1]

    # 2. C5 AR lags
    print("Adding C5 AR lags...")
    for lag in C5_AR_LAGS:
        df[f"C5_lag_{lag}d"] = df["C5_rate_USD_tonne"].shift(lag)
    for lag in [1, 5, 10, 21, 42, 62]:
        df[f"C5_logret_{lag}d"] = np.log(
            df["C5_rate_USD_tonne"] / df["C5_rate_USD_tonne"].shift(lag)
        )

    # 3. External variable lags
    print("Adding external variable lags...")
    n_lag_cols = 0
    for col, lags in ALL_EXTERNAL_LAGS.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}d"] = df[col].shift(lag)
            n_lag_cols += 1
    print(f"  Added {n_lag_cols} lag columns")

    # 4. Rolling stats on key external series
    print("Adding rolling stats...")
    if "China_IronOre_Stockpile_kt" in df.columns:
        for w in [10, 21]:
            df[f"IronOreStk_chg_{w}d"] = df["China_IronOre_Stockpile_kt"].diff(w)
    for col in ["Platts_HFO380_SIN_ext_USD_t", "Bunker_HSFO380_CLRK_W_USD_t"]:
        if col in df.columns:
            for w in [21, 42]:
                df[f"{col}_roll{w}d"] = df[col].rolling(w).mean()
    if "IronOre_Spot_CFR_NChina_USD_t" in df.columns:
        for w in [21, 42]:
            df[f"IronOre_roll{w}d"] = df["IronOre_Spot_CFR_NChina_USD_t"].rolling(w).mean()
    for col in ["TF_AUS_CHN_IronOre_MT", "TF_Brazil_CHN_IronOre_MT"]:
        if col in df.columns:
            df[f"{col}_roll21d"] = df[col].rolling(21).sum()

    # 5. Technical indicators
    print("Adding technical indicators on C5...")
    df = add_technical_indicators(df, price_col="C5_rate_USD_tonne", prefix="")
    if add_tech_on_prices is None:
        add_tech_on_prices = [
            "IronOre_Spot_CFR_NChina_USD_t",
            "Brent_Crude_USD_bbl",
            "Platts_HFO380_SIN_ext_USD_t",
            "C5_FFA_C5_1MON",
        ]
    for pcol in add_tech_on_prices:
        if pcol in df.columns:
            print(f"  Technical indicators on {pcol}...")
            df = add_technical_indicators(df, price_col=pcol, prefix=f"{pcol}_")

    # 6. Cross features
    print("Adding cross features...")
    df = add_cross_indicators(df)

    # 7. Calendar features
    print("Adding calendar features...")
    df = add_calendar_features(df)

    n_added = df.shape[1] - n_orig
    print(f"\n{'='*60}")
    print(f"FEATURE BUILD COMPLETE")
    print(f"  Original columns : {n_orig}")
    print(f"  Features added   : {n_added}")
    print(f"  Total columns    : {df.shape[1]}")
    print(f"  Total rows       : {df.shape[0]}")
    print(f"{'='*60}")

    return df


#  FEATURE GROUP MAP

def get_feature_groups(df: pd.DataFrame, horizon: int = 62) -> dict:
    exclude = {"C5_rate_USD_tonne"}

    def pick(patterns):
        return [
            c for c in df.columns
            if c not in exclude and any(p in c for p in patterns)
        ]

    return {
        "c5_ar":        pick(["C5_lag_", "C5_logret_"]),
        "c5_technical": pick(["EMA_", "MACD", "RSI", "Stoch", "BB_", "RealVol",
                               "MOM_", "ROC_", "Zscore", "PctFrom52",
                               "Spread_EMA", "Dist_from"]),
        "ffa":          pick(["C5_FFA", "C3_FFA", "C3_vs_C5", "C5_vs_C5_FFA"]),
        "iron_ore":     pick(["IronOre", "OreInv"]),
        "steel":        pick(["Steel", "Rebar", "BFI", "PigIron",
                               "Coke", "Billet", "HRC"]),
        "seaborne":     pick(["TF_", "Seaborne", "_Mt_SIN", "_Mt_CLRK", "Guinea"]),
        "fleet_supply": pick(["Cape_", "utilisation", "Total_Congestion"]),
        "bunker":       pick(["Bunker", "Platts", "IFO", "VLSFO", "HFO", "MGO"]),
        "energy":       pick(["Brent", "Coal_Price", "Aluminum",
                               "Wheat", "WorldEnergy"]),
        "fx":           pick(["AUDUSD", "EURUSD", "CNYUSD", "USD_Index"]),
        "macro":        pick(["China_M2", "New_Loan", "Property", "PMI", "IP_YoY",
                               "PBOC", "Finance_Init", "LPI", "TPU"]),
        "sentiment":    pick(["VIX", "VHSI", "VSTOXX", "SP500", "HangSeng",
                               "KOSPI", "MSCI", "Nikkei", "EuroStoxx", "RiskAdj"]),
        "calendar":     pick(["DayOfWeek", "MonthOfYear", "Quarter", "WeekOfYear",
                               "IsMonth", "IsQtr", "CNY"]),
        "cross":        pick(["vs_", "ratio", "crack", "_sum",
                               "_mom_", "OBV", "Congestion_proxy"]),
    }
