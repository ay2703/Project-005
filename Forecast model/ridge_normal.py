"""
ridge_normal.py
===============
Probabilistic C5 forecasting with Ridge regression (PCA-reduced) and a
Normal(μ_h, σ_h²) noise model.

Pipeline
--------
1. build_targets     — log-returns y_{t,h} = log(C5_{t+h}/C5_t), h=1…62
2. walk_forward_forecasts — expanding-window Ridge fits, step=63 (~24 origins)
3. calibrate_sigma   — OOS residual std per horizon, monotone via isotonic
4. forecast_distribution — live 62-day distributional forecast
5. evaluate          — CRPS, Winkler, Coverage, MAE per horizon
6. run_probabilistic_pipeline — orchestrator
7. export helpers    — CSV + PCA loadings

Plotting
--------
plot_forecast              — 62-day cone with σ-growth subplot
plot_distributions_grid    — static 8-horizon grid (log-return + level)
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import Ridge
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.isotonic        import IsotonicRegression
from scipy                   import stats


#  constants 
HORIZON    = 62
MIN_TRAIN  = 252
STEP       = 63   
TARGET_COL = "C5_rate_USD_tonne"
N_PCA      = 27

Z_80 = stats.norm.ppf(0.90)    # 1.282
Z_95 = stats.norm.ppf(0.975)   # 1.960


#  helpers 

def build_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    y_{t,h} = log(C5_{t+h} / C5_t) for h = 1…horizon.
    Returns DataFrame of shape (n, horizon); future rows are NaN.
    """
    prices  = df[TARGET_COL].values
    n       = len(prices)
    targets = np.full((n, horizon), np.nan)
    for h in range(1, horizon + 1):
        targets[:n - h, h - 1] = np.log(prices[h:] / prices[:n - h])
    cols = [f"target_h{h}" for h in range(1, horizon + 1)]
    return pd.DataFrame(targets, index=df.index, columns=cols)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {TARGET_COL}
    exclude.update([c for c in df.columns if c.startswith("target_h")])
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def make_model(n_pca: int = N_PCA) -> Pipeline:
    """StandardScaler → PCA → Ridge (fits all 62 horizons jointly)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("ridge",  Ridge(alpha=1.0, fit_intercept=True)),
    ])


#  step 2: walk-forward 

def walk_forward_forecasts(
    df        : pd.DataFrame,
    targets   : pd.DataFrame,
    horizon   : int = HORIZON,
    min_train : int = MIN_TRAIN,
    step      : int = STEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward.  One model fit per origin predicts all
    62 horizons at once.  With step=63: ~24 fits, ~20-30s runtime.
    """
    feature_cols = get_feature_cols(df)
    X  = df[feature_cols].values.astype(np.float32)
    Y  = targets.values
    n  = len(df)

    origins   = list(range(min_train, n - horizon, step))
    n_origins = len(origins)
    mu_mat    = np.full((n_origins, horizon), np.nan)
    act_mat   = np.full((n_origins, horizon), np.nan)
    idx_list  = []

    print(f"Walk-forward: {n_origins} origins | step={step} | min_train={min_train}")
    print(f"Model: StandardScaler → PCA({N_PCA}) → Ridge  [all 62 horizons at once]")
    t0 = time.time()

    for i, t in enumerate(origins):
        idx_val  = df.index[t]
        date_str = idx_val.date() if hasattr(idx_val, "date") else str(idx_val)[:10]

        X_tr = X[:t];  Y_tr = Y[:t]
        valid = (~np.isnan(Y_tr).any(axis=1)) & (~np.isnan(X_tr).any(axis=1))

        if valid.sum() < 60:
            idx_list.append(idx_val)
            continue

        model = make_model()
        model.fit(X_tr[valid], Y_tr[valid])

        x_t = X[t]
        if np.isnan(x_t).any():
            idx_list.append(idx_val)
            continue

        mu_mat[i]  = model.predict(x_t.reshape(1, -1))[0]
        act_mat[i] = Y[t]

        elapsed = time.time() - t0
        eta     = (elapsed / (i + 1)) * (n_origins - i - 1)
        print(f"  [{i+1:2d}/{n_origins}] {date_str}  "
              f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")
        idx_list.append(idx_val)

    h_cols = [f"h{h}" for h in range(1, horizon + 1)]
    mu_df  = pd.DataFrame(mu_mat,  index=idx_list, columns=h_cols)
    act_df = pd.DataFrame(act_mat, index=idx_list, columns=h_cols)
    print(f"\nDone. Total: {time.time()-t0:.1f}s | Shape: {mu_df.shape}")
    return mu_df, act_df


#  step 3: calibrate σ 

def calibrate_sigma(
    mu_df  : pd.DataFrame,
    act_df : pd.DataFrame,
    horizon: int = HORIZON,
) -> np.ndarray:
    """OOS residual std per horizon, monotone-enforced via isotonic regression."""
    h_cols    = [f"h{h}" for h in range(1, horizon + 1)]
    raw_sigma = np.zeros(horizon)

    for i, col in enumerate(h_cols):
        res = (act_df[col] - mu_df[col]).dropna().values
        raw_sigma[i] = res.std(ddof=1) if len(res) > 1 else np.nan

    iso            = IsotonicRegression(increasing=True)
    sigma_monotone = iso.fit_transform(np.arange(1, horizon + 1), raw_sigma)

    print("\nσ calibration (selected horizons):")
    print(f"  {'h':>4}  {'raw σ':>8}  {'monotone σ':>10}  {'95% band ±':>10}")
    for h in [1, 5, 10, 21, 42, 62]:
        print(f"  {h:4d}  {raw_sigma[h-1]:8.4f}  "
              f"{sigma_monotone[h-1]:10.4f}  "
              f"{Z_95 * sigma_monotone[h-1]:10.4f}")
    return sigma_monotone


#  step 4: live forecast

def forecast_distribution(
    df       : pd.DataFrame,
    sigma    : np.ndarray,
    horizon  : int = HORIZON,
    origin_t : int = -1,
) -> pd.DataFrame:
    """
    Train on full history up to origin_t, predict 62-day distribution.
    Each horizon h: N(μ_h, σ_h²) in log-return space → level via exp().
    """
    feature_cols = get_feature_cols(df)
    targets      = build_targets(df, horizon)
    X            = df[feature_cols].values.astype(np.float32)
    Y            = targets.values

    t    = origin_t if origin_t >= 0 else len(df) - 1
    C5_t = df[TARGET_COL].iloc[t]

    valid = (~np.isnan(Y[:t]).any(axis=1)) & (~np.isnan(X[:t]).any(axis=1))
    model = make_model()
    model.fit(X[:t][valid], Y[:t][valid])
    mu_vec = model.predict(X[t].reshape(1, -1))[0]

    date_str = df.index[t].date() if hasattr(df.index[t], "date") else df.index[t]
    print(f"Live forecast from {date_str}, C5={C5_t:.2f}")

    rows = []
    for h in range(1, horizon + 1):
        mu_h  = mu_vec[h - 1]
        sig_h = sigma[h - 1]
        fdate = df.index[t] + pd.offsets.BDay(h)
        rows.append({
            "horizon"     : h,
            "date"        : fdate,
            "mu_logret"   : mu_h,
            "sigma_logret": sig_h,
            "C5_point"    : C5_t * np.exp(mu_h),
            "C5_lower80"  : C5_t * np.exp(mu_h - Z_80 * sig_h),
            "C5_upper80"  : C5_t * np.exp(mu_h + Z_80 * sig_h),
            "C5_lower95"  : C5_t * np.exp(mu_h - Z_95 * sig_h),
            "C5_upper95"  : C5_t * np.exp(mu_h + Z_95 * sig_h),
        })
    return pd.DataFrame(rows).set_index("horizon")


#  evaluation 

def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    if sigma <= 0 or np.isnan(y):
        return np.nan
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def winkler_score(lo, hi, y, alpha) -> float:
    width = hi - lo
    if y < lo:  return width + (2 / alpha) * (lo - y)
    if y > hi:  return width + (2 / alpha) * (y - hi)
    return width


def evaluate(
    mu_df  : pd.DataFrame,
    act_df : pd.DataFrame,
    sigma  : np.ndarray,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    records = []
    for h in range(1, horizon + 1):
        col   = f"h{h}"
        sig_h = sigma[h - 1]
        mu_v  = mu_df[col].dropna()
        act_v = act_df[col].reindex(mu_v.index).dropna()
        mu_v  = mu_v.reindex(act_v.index)
        if len(act_v) == 0:
            continue

        crps = [crps_gaussian(m, sig_h, y) for m, y in zip(mu_v, act_v)]
        lo80 = mu_v - Z_80 * sig_h;  hi80 = mu_v + Z_80 * sig_h
        lo95 = mu_v - Z_95 * sig_h;  hi95 = mu_v + Z_95 * sig_h
        w80  = [winkler_score(l, u, y, 0.20) for l, u, y in zip(lo80, hi80, act_v)]
        w95  = [winkler_score(l, u, y, 0.05) for l, u, y in zip(lo95, hi95, act_v)]

        records.append({
            "horizon"    : h,
            "sigma"      : sig_h,
            "CRPS"       : np.nanmean(crps),
            "Winkler_80" : np.nanmean(w80),
            "Winkler_95" : np.nanmean(w95),
            "Coverage_80": ((act_v >= lo80) & (act_v <= hi80)).mean(),
            "Coverage_95": ((act_v >= lo95) & (act_v <= hi95)).mean(),
            "MAE_logret" : (mu_v - act_v).abs().mean(),
            "n_obs"      : len(act_v),
        })
    return pd.DataFrame(records).set_index("horizon")


# main pipeline 

def run_probabilistic_pipeline(df: pd.DataFrame) -> dict:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)

    print("=" * 60)
    print("STEP 1 — Building targets")
    print("=" * 60)
    targets = build_targets(df)
    print(f"Target matrix shape: {targets.shape}")

    print("\n" + "=" * 60)
    print("STEP 2 — Walk-forward point forecasts (Ridge + PCA)")
    print("=" * 60)
    mu_df, act_df = walk_forward_forecasts(df, targets)

    print("\n" + "=" * 60)
    print("STEP 3 — Calibrating σ_h (monotone)")
    print("=" * 60)
    sigma = calibrate_sigma(mu_df, act_df)

    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)
    eval_df = evaluate(mu_df, act_df, sigma)
    print("\nKey horizons:")
    print(eval_df.loc[eval_df.index.isin([1, 5, 10, 21, 42, 62])].to_string())

    print("\n" + "=" * 60)
    print("STEP 4b — Live 62-day distribution forecast")
    print("=" * 60)
    forecast = forecast_distribution(df, sigma)
    print(forecast[["C5_point", "C5_lower80", "C5_upper80",
                     "C5_lower95", "C5_upper95"]].iloc[[0, 4, 9, 20, 41, 61]])

    return {
        "mu_df"   : mu_df,
        "act_df"  : act_df,
        "sigma"   : sigma,
        "eval_df" : eval_df,
        "forecast": forecast,
    }

#  export helpers

def export_forecast_csv(
    results : dict,
    filename: str = "/Users/b23/Desktop/Capstone_cleaned/result_ridge_normal/c5_forecast_62days.csv",
) -> None:
    results["forecast"].to_csv(filename)
    print(f"Saved → {filename}")


def export_pca_loadings_csv(
    df      : pd.DataFrame,
    filename: str = "/Users/b23/Desktop/Capstone_cleaned/result_ridge_normal/c5_pca_loadings.csv",
) -> pd.DataFrame:
    """Export PCA loadings (feature × PC) with explained variance."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    h_cols_y = [f"target_h{h}" for h in range(1, HORIZON + 1)]
    Y = build_targets(df)[h_cols_y].values
    n = len(df)

    origins = list(range(MIN_TRAIN, n - HORIZON, STEP))
    t = origins[-1]
    X_tr = X[:t];  Y_tr = Y[:t]
    valid = (~np.isnan(Y_tr).any(axis=1)) & (~np.isnan(X_tr).any(axis=1))

    model = make_model()
    model.fit(X_tr[valid], Y_tr[valid])

    pca        = model.named_steps["pca"]
    components = pca.components_                # (n_pca, n_features)
    var_ratio  = pca.explained_variance_ratio_  # (n_pca,)
    n_pca      = components.shape[0]

    pc_cols = [f"PC{k+1}" for k in range(n_pca)]
    df_load = pd.DataFrame(components.T, index=feature_cols, columns=pc_cols)
    df_load.index.name = "feature"

    df_load = pd.concat([
        pd.DataFrame([var_ratio * 100], index=["var_explained_%"], columns=pc_cols),
        df_load,
    ])

    df_load.to_csv(filename)
    print(f"Saved PCA loadings → {filename}  ({n_pca} PCs, {len(feature_cols)} features)")
    return df_load


def export_feature_importance_loadings(
    df     : pd.DataFrame,
    top_n  : int = 10,
    filename: str = "/Users/b23/Desktop/Capstone_cleaned/result_ridge_normal/c5_feature_importance_loadings.csv",
) -> pd.DataFrame:
    """
    Ridge coefficient × PCA loading → feature importance per horizon.
    Weights by explained variance ratio.
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    h_cols = [f"target_h{h}" for h in range(1, HORIZON + 1)]
    Y = build_targets(df)[h_cols].values
    n = len(df)

    origins = list(range(MIN_TRAIN, n - HORIZON, STEP))
    t = origins[-1]
    X_tr = X[:t];  Y_tr = Y[:t]
    valid = (~np.isnan(Y_tr).any(axis=1)) & (~np.isnan(X_tr).any(axis=1))

    model = make_model()
    model.fit(X_tr[valid], Y_tr[valid])

    pca        = model.named_steps["pca"]
    ridge      = model.named_steps["ridge"]
    var_ratio  = pca.explained_variance_ratio_
    components = pca.components_
    coef       = ridge.coef_     # (62, n_pca)

    print(f"coef shape: {coef.shape}  →  coef[0] = h=1, coef[61] = h=62")

    records = []
    for h in range(1, HORIZON + 1):
        weights    = np.abs(coef[h - 1]) * var_ratio
        importance = np.abs(weights @ components)
        pct        = importance / importance.sum()
        for rank, fidx in enumerate(np.argsort(importance)[::-1][:top_n], 1):
            records.append({
                "horizon"     : h,
                "rank"        : rank,
                "feature"     : feature_cols[fidx],
                "pct_of_total": float(pct[fidx]),
            })

    df_imp = pd.DataFrame(records)
    df_imp.to_csv(filename, index=False)
    print(f"Saved → {filename}")
    return df_imp


#  plots

def plot_forecast(
    df          : pd.DataFrame,
    forecast    : pd.DataFrame,
    history_days: int = 120,
) -> None:
    """62-day uncertainty cone with σ-growth subplot."""
    import matplotlib.pyplot as plt
    import matplotlib.dates  as mdates

    hist  = df[TARGET_COL].iloc[-history_days:]
    dates = pd.to_datetime(forecast["date"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0f1117",
    )
    origin_str = (hist.index[-1].date()
                  if hasattr(hist.index[-1], "date")
                  else str(hist.index[-1])[:10])
    fig.suptitle(
        f"C5 Capesize — 62-day probabilistic forecast  (origin: {origin_str})",
        color="#e0e0e0", fontsize=13, fontweight="bold", y=0.98,
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#888", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    ax1.fill_between(dates, forecast["C5_lower95"], forecast["C5_upper95"],
                     color="#3b5bdb", alpha=0.18, label="95% band")
    ax1.fill_between(dates, forecast["C5_lower80"], forecast["C5_upper80"],
                     color="#3b5bdb", alpha=0.32, label="80% band")
    ax1.plot(dates, forecast["C5_point"],
             color="#74c0fc", linewidth=1.8, label="Point forecast (median)")
    ax1.plot(hist.index, hist.values,
             color="#e0e0e0", linewidth=1.4, label="Historical C5")
    ax1.axvline(hist.index[-1], color="#555", linewidth=1, linestyle="--")
    ax1.scatter([hist.index[-1]], [hist.values[-1]], color="white", s=40, zorder=5)
    ax1.set_ylabel("C5 rate (USD/tonne)", color="#aaa", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.2, labelcolor="#ccc",
               fontsize=9, facecolor="#1a1a2e")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(axis="y", color="#222", linewidth=0.5)

    ax2.plot(dates, forecast["sigma_logret"], color="#f59f00", linewidth=1.6)
    ax2.fill_between(dates, 0, forecast["sigma_logret"],
                     color="#f59f00", alpha=0.15)
    ax2.set_ylabel("σ_h", color="#aaa", fontsize=10)
    ax2.set_title("Uncertainty growth  (σ_h monotone increasing)",
                  color="#888", fontsize=9, pad=4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", color="#222", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_distributions_grid(
    df      : pd.DataFrame,
    forecast: pd.DataFrame,
    horizons: list = [1, 5, 10, 21, 31, 42, 52, 62],
) -> None:
    """Static grid: N(μ_h, σ_h) in log-return and C5 level space."""
    import matplotlib.pyplot as plt

    mu_vec    = forecast["mu_logret"].values
    sigma_vec = forecast["sigma_logret"].values
    dates     = pd.to_datetime(forecast["date"])
    C5_0      = df[TARGET_COL].iloc[-1]
    n         = len(horizons)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n), facecolor="#0f1117")
    fig.suptitle(
        f"C5 Capesize — predictive distributions per horizon  "
        f"(C5 origin = {C5_0:.2f} USD/t)",
        color="#e0e0e0", fontsize=12, fontweight="bold", y=1.01,
    )

    for row, h in enumerate(horizons):
        idx  = h - 1
        mu   = mu_vec[idx]
        sig  = sigma_vec[idx]
        d    = dates.iloc[idx].date()

        lo80 = mu - Z_80 * sig;  hi80 = mu + Z_80 * sig
        lo95 = mu - Z_95 * sig;  hi95 = mu + Z_95 * sig

        # left: log-return
        ax = axes[row, 0]
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#777", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")

        x = np.linspace(mu - 4 * sig, mu + 4 * sig, 500)
        y = stats.norm.pdf(x, mu, sig)

        ax.plot(x, y, color="#74c0fc", linewidth=1.8)
        ax.fill_between(x, y, where=(x >= lo95) & (x <= hi95),
                        color="#3b5bdb", alpha=0.20, label="95%")
        ax.fill_between(x, y, where=(x >= lo80) & (x <= hi80),
                        color="#3b5bdb", alpha=0.38, label="80%")
        ax.axvline(mu, color="#74c0fc", linewidth=1, linestyle="--", alpha=0.8)
        ax.axvline(0,  color="#666",    linewidth=0.8, linestyle=":")
        ax.text(0.97, 0.88, f"μ={mu:+.3f}  σ={sig:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                color="#aaa", fontsize=8)
        ax.set_title(f"h={h}  ({d})  — log-return", color="#ccc", fontsize=9, pad=3)
        ax.set_ylabel("density", color="#777", fontsize=8)
        ax.grid(color="#1a1a1a", linewidth=0.4)
        if row == 0:
            ax.legend(fontsize=8, labelcolor="#ccc",
                      facecolor="#1a1a2e", framealpha=0.3, loc="upper left")

        # right: C5 level
        ax2 = axes[row, 1]
        ax2.set_facecolor("#0f1117")
        ax2.tick_params(colors="#777", labelsize=8)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#2a2a2a")

        C5_point = C5_0 * np.exp(mu)
        C5_sig   = C5_point * sig
        x2 = np.linspace(max(0.5, C5_point - 4.5 * C5_sig),
                         C5_point + 4.5 * C5_sig, 500)
        y2 = stats.norm.pdf(x2, C5_point, C5_sig)

        C5_lo80 = C5_0 * np.exp(lo80);  C5_hi80 = C5_0 * np.exp(hi80)
        C5_lo95 = C5_0 * np.exp(lo95);  C5_hi95 = C5_0 * np.exp(hi95)

        ax2.plot(x2, y2, color="#a9e34b", linewidth=1.8)
        ax2.fill_between(x2, y2, where=(x2 >= C5_lo95) & (x2 <= C5_hi95),
                         color="#5c940d", alpha=0.20, label="95%")
        ax2.fill_between(x2, y2, where=(x2 >= C5_lo80) & (x2 <= C5_hi80),
                         color="#5c940d", alpha=0.38, label="80%")
        ax2.axvline(C5_point, color="#a9e34b", linewidth=1, linestyle="--", alpha=0.8)
        ax2.axvline(C5_0, color="#666", linewidth=0.8, linestyle=":",
                    label=f"C5 now={C5_0:.2f}")
        ax2.text(0.97, 0.88,
                 f"point={C5_point:.2f} 80%: [{C5_lo80:.2f}, {C5_hi80:.2f}]",
                 transform=ax2.transAxes, ha="right", va="top",
                 color="#aaa", fontsize=8)
        ax2.set_title(f"h={h}  ({d})  — C5 level (USD/t)",
                      color="#ccc", fontsize=9, pad=3)
        ax2.set_xlabel("USD/tonne", color="#777", fontsize=8)
        ax2.set_ylabel("density",   color="#777", fontsize=8)
        ax2.grid(color="#1a1a1a", linewidth=0.4)
        if row == 0:
            ax2.legend(fontsize=8, labelcolor="#ccc",
                       facecolor="#1a1a2e", framealpha=0.3, loc="upper left")

    plt.tight_layout()
    plt.show()

#  entry point

if __name__ == "__main__":
    from data_loader import prepare_main
    from feature_engineering import build_features

    df_main = prepare_main()
    df_main = build_features(df_main)
    df_main = df_main.ffill().dropna()

    results = run_probabilistic_pipeline(df_main)

    plot_forecast(df_main, results["forecast"])
    plot_distributions_grid(df_main, results["forecast"])

    export_forecast_csv(results)
    export_pca_loadings_csv(df_main)
