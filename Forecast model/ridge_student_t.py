"""
ridge_student_t.py
==================
Probabilistic C5 forecasting with Ridge regression (PCA-reduced) and a
Student-t noise model: t(μ_h, σ_h², ν_h).

Differs from ridge_normal.py in
  - Step 3: joint MLE fit of (σ_h, ν_h) instead of just std
  - Step 4: exact t-distribution quantiles for prediction bands
  - Evaluation: Winkler + coverage (CRPS via Monte Carlo available)
  - Feature importance: Ridge coef × PCA loading projection

Imports make_model / get_feature_cols / build_targets / walk_forward_forecasts
from ridge_normal to avoid duplication.
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.isotonic   import IsotonicRegression
from scipy              import stats
from scipy.optimize     import minimize

# Reuse common pieces from the Normal pipeline
from ridge_normal import (
    HORIZON, MIN_TRAIN, STEP, TARGET_COL, N_PCA,
    build_targets, get_feature_cols, make_model,
    walk_forward_forecasts, winkler_score,
)


#  constants 
# nu bounds: >2 so variance exists, cap at 30 (≈ Gaussian above 30)
NU_MIN, NU_MAX = 2.1, 30.0


#  step 3: MLE calibration of (σ_h, ν_h) 

def _neg_loglik_t(params, residuals):
    """Negative log-likelihood of t(0, sigma, nu)."""
    sigma, nu = params
    if sigma <= 0 or nu <= 2:
        return 1e10
    return -stats.t.logpdf(residuals, df=nu, loc=0, scale=sigma).sum()


def calibrate_t(
    mu_df    : pd.DataFrame,
    act_df   : pd.DataFrame,
    horizon  : int = HORIZON,
    smooth_nu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each horizon h, fit t(0, σ_h, ν_h) to OOS residuals via MLE.

    Returns
    -------
    sigma_monotone : (horizon,) — monotone increasing σ
    nu_smooth      : (horizon,) — rolling-median smoothed ν
    """
    h_cols    = [f"h{h}" for h in range(1, horizon + 1)]
    raw_sigma = np.zeros(horizon)
    raw_nu    = np.zeros(horizon)

    print("\nCalibrating (sigma_h, nu_h) via MLE...")
    for i, col in enumerate(h_cols):
        res = (act_df[col] - mu_df[col]).dropna().values
        if len(res) < 5:
            raw_sigma[i] = np.nan
            raw_nu[i]    = 5.0
            continue

        x0     = [res.std(ddof=1), 5.0]
        bounds = [(1e-6, None), (NU_MIN, NU_MAX)]
        result = minimize(
            _neg_loglik_t, x0, args=(res,),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        raw_sigma[i] = result.x[0]
        raw_nu[i]    = result.x[1]

    iso            = IsotonicRegression(increasing=True)
    sigma_monotone = iso.fit_transform(np.arange(1, horizon + 1), raw_sigma)

    if smooth_nu:
        nu_smooth = (
            pd.Series(raw_nu)
            .rolling(5, min_periods=1, center=True)
            .median()
            .values
        )
    else:
        nu_smooth = raw_nu

    print(f"\n{'h':>4}  {'raw_σ':>8}  {'mono_σ':>8}  {'raw_ν':>7}  "
          f"{'smooth_ν':>9}  {'tail':>8}")
    for h in [1, 5, 10, 21, 42, 62]:
        nu_v = nu_smooth[h - 1]
        tail = "Gaussian" if nu_v > 25 else ("fat" if nu_v < 6 else "moderate")
        print(f"  {h:4d}  {raw_sigma[h-1]:8.4f}  {sigma_monotone[h-1]:8.4f}  "
              f"{raw_nu[h-1]:7.2f}  {nu_v:9.2f}  {tail:>8}")

    return sigma_monotone, nu_smooth


#  step 4: live forecast 

def forecast_distribution_t(
    df      : pd.DataFrame,
    sigma   : np.ndarray,
    nu      : np.ndarray,
    horizon : int = HORIZON,
    origin_t: int = -1,
) -> pd.DataFrame:
    """
    Live 62-day forecast. Each horizon: t(mu_h, sigma_h², nu_h).
    Quantiles via scipy.stats.t (exact).
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

    date_str = df.index[t].date() if hasattr(df.index[t], "date") else str(df.index[t])[:10]
    print(f"Live forecast from {date_str}, C5={C5_t:.2f}")

    rows = []
    for h in range(1, horizon + 1):
        mu_h  = mu_vec[h - 1]
        sig_h = sigma[h - 1]
        nu_h  = nu[h - 1]
        fdate = df.index[t] + pd.offsets.BDay(h)

        q_lo80 = stats.t.ppf(0.10,  df=nu_h, loc=mu_h, scale=sig_h)
        q_hi80 = stats.t.ppf(0.90,  df=nu_h, loc=mu_h, scale=sig_h)
        q_lo95 = stats.t.ppf(0.025, df=nu_h, loc=mu_h, scale=sig_h)
        q_hi95 = stats.t.ppf(0.975, df=nu_h, loc=mu_h, scale=sig_h)

        rows.append({
            "horizon"     : h,
            "date"        : fdate,
            "mu_logret"   : mu_h,
            "sigma_logret": sig_h,
            "nu"          : nu_h,
            "C5_point"    : C5_t * np.exp(mu_h),
            "C5_lower80"  : C5_t * np.exp(q_lo80),
            "C5_upper80"  : C5_t * np.exp(q_hi80),
            "C5_lower95"  : C5_t * np.exp(q_lo95),
            "C5_upper95"  : C5_t * np.exp(q_hi95),
        })
    return pd.DataFrame(rows).set_index("horizon")


#  evaluation 

def evaluate_t(
    mu_df  : pd.DataFrame,
    act_df : pd.DataFrame,
    sigma  : np.ndarray,
    nu     : np.ndarray,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    records = []
    for h in range(1, horizon + 1):
        col   = f"h{h}"
        sig_h = sigma[h - 1]
        nu_h  = nu[h - 1]
        mu_v  = mu_df[col].dropna()
        act_v = act_df[col].reindex(mu_v.index).dropna()
        mu_v  = mu_v.reindex(act_v.index)
        if len(act_v) == 0:
            continue

        lo80 = stats.t.ppf(0.10,  df=nu_h, loc=mu_v, scale=sig_h)
        hi80 = stats.t.ppf(0.90,  df=nu_h, loc=mu_v, scale=sig_h)
        lo95 = stats.t.ppf(0.025, df=nu_h, loc=mu_v, scale=sig_h)
        hi95 = stats.t.ppf(0.975, df=nu_h, loc=mu_v, scale=sig_h)

        w80 = [winkler_score(l, u, y, 0.20) for l, u, y in zip(lo80, hi80, act_v)]
        w95 = [winkler_score(l, u, y, 0.05) for l, u, y in zip(lo95, hi95, act_v)]

        records.append({
            "horizon"    : h,
            "sigma"      : sig_h,
            "nu"         : nu_h,
            "Winkler_80" : np.nanmean(w80),
            "Winkler_95" : np.nanmean(w95),
            "Coverage_80": ((act_v >= lo80) & (act_v <= hi80)).mean(),
            "Coverage_95": ((act_v >= lo95) & (act_v <= hi95)).mean(),
            "MAE_logret" : (mu_v - act_v).abs().mean(),
            "n_obs"      : len(act_v),
        })
    return pd.DataFrame(records).set_index("horizon")


#  main pipeline 

def run_probabilistic_pipeline_t(df: pd.DataFrame) -> dict:
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
    print("STEP 3 — MLE calibration of (sigma_h, nu_h)")
    print("=" * 60)
    sigma, nu = calibrate_t(mu_df, act_df)

    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)
    eval_df = evaluate_t(mu_df, act_df, sigma, nu)
    print("\nKey horizons:")
    print(eval_df.loc[eval_df.index.isin([1, 5, 10, 21, 42, 62])].to_string())

    print("\n" + "=" * 60)
    print("STEP 4b — Live 62-day Student-t forecast")
    print("=" * 60)
    forecast = forecast_distribution_t(df, sigma, nu)
    print(forecast[["C5_point", "C5_lower80", "C5_upper80",
                     "C5_lower95", "C5_upper95", "nu"]].iloc[[0, 4, 9, 20, 41, 61]])

    return {
        "mu_df"   : mu_df,
        "act_df"  : act_df,
        "sigma"   : sigma,
        "nu"      : nu,
        "eval_df" : eval_df,
        "forecast": forecast,
    }


#  plots 

def plot_forecast_t(df, forecast, history_days=120):
    """Cone plot with ν-per-horizon subplot (lower ν = fatter tails)."""
    import matplotlib.pyplot as plt
    import matplotlib.dates  as mdates

    hist  = df[TARGET_COL].iloc[-history_days:]
    dates = pd.to_datetime(forecast["date"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0f1117",
    )
    origin_str = (hist.index[-1].date() if hasattr(hist.index[-1], "date")
                  else str(hist.index[-1])[:10])
    fig.suptitle(
        f"C5 Capesize — 62-day Student-t forecast  (origin: {origin_str})",
        color="#e0e0e0", fontsize=13, fontweight="bold", y=0.98,
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#888", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    ax1.fill_between(dates, forecast["C5_lower95"], forecast["C5_upper95"],
                     color="#3b5bdb", alpha=0.18, label="95% band (t)")
    ax1.fill_between(dates, forecast["C5_lower80"], forecast["C5_upper80"],
                     color="#3b5bdb", alpha=0.32, label="80% band (t)")
    ax1.plot(dates, forecast["C5_point"], color="#74c0fc",
             linewidth=1.8, label="Point forecast")
    ax1.plot(hist.index, hist.values, color="#e0e0e0",
             linewidth=1.4, label="Historical C5")
    ax1.axvline(hist.index[-1], color="#555", linewidth=1, linestyle="--")
    ax1.scatter([hist.index[-1]], [hist.values[-1]], color="white", s=40, zorder=5)
    ax1.set_ylabel("C5 rate (USD/tonne)", color="#aaa", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.2, labelcolor="#ccc",
               fontsize=9, facecolor="#1a1a2e")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(axis="y", color="#222", linewidth=0.5)

    ax2.plot(dates, forecast["nu"], color="#f59f00", linewidth=1.6, label="ν_h")
    ax2.fill_between(dates, NU_MIN, forecast["nu"], color="#f59f00", alpha=0.15)
    ax2.axhline(30, color="#555", linewidth=0.8, linestyle="--", label="ν=30 (≈Gaussian)")
    ax2.set_ylabel("ν_h (d.o.f.)", color="#aaa", fontsize=10)
    ax2.set_title("Degrees of freedom per horizon  (lower = fatter tails)",
                  color="#888", fontsize=9, pad=4)
    ax2.legend(fontsize=8, labelcolor="#ccc", facecolor="#1a1a2e", framealpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", color="#222", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_distributions_grid_t(
    df      : pd.DataFrame,
    forecast: pd.DataFrame,
    horizons: list = [1, 5, 10, 21, 31, 42, 52, 62],
) -> None:
    """Static grid comparing Gaussian vs Student-t at each horizon."""
    import matplotlib.pyplot as plt

    mu_vec    = forecast["mu_logret"].values
    sigma_vec = forecast["sigma_logret"].values
    nu_vec    = forecast["nu"].values
    dates     = pd.to_datetime(forecast["date"])
    C5_0      = df[TARGET_COL].iloc[-1]
    n         = len(horizons)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.8 * n), facecolor="#0f1117")
    fig.suptitle(
        f"C5 — Student-t predictive distributions  (C5={C5_0:.2f} USD/t)",
        color="#e0e0e0", fontsize=12, fontweight="bold", y=1.01,
    )

    for row, h in enumerate(horizons):
        idx = h - 1
        mu  = mu_vec[idx];  sig = sigma_vec[idx];  nu = nu_vec[idx]
        d   = dates.iloc[idx].date()

        x    = np.linspace(mu - 5 * sig, mu + 5 * sig, 600)
        y_t  = stats.t.pdf(x, df=nu, loc=mu, scale=sig)
        y_n  = stats.norm.pdf(x, loc=mu, scale=sig)

        lo80_t = stats.t.ppf(0.10,  df=nu, loc=mu, scale=sig)
        hi80_t = stats.t.ppf(0.90,  df=nu, loc=mu, scale=sig)
        lo95_t = stats.t.ppf(0.025, df=nu, loc=mu, scale=sig)
        hi95_t = stats.t.ppf(0.975, df=nu, loc=mu, scale=sig)

        # left: log-return overlay t vs Gaussian
        ax = axes[row, 0]
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#777", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")

        ax.plot(x, y_t, color="#f59f00", linewidth=2,   label=f"t(ν={nu:.1f})")
        ax.plot(x, y_n, color="#74c0fc", linewidth=1.2,
                linestyle="--", alpha=0.6, label="Gaussian")
        ax.fill_between(x, y_t, where=(x >= lo95_t) & (x <= hi95_t),
                        color="#f59f00", alpha=0.12, label="95%")
        ax.fill_between(x, y_t, where=(x >= lo80_t) & (x <= hi80_t),
                        color="#f59f00", alpha=0.28, label="80%")
        ax.axvline(mu, color="#f59f00", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvline(0,  color="#555",    linewidth=0.8, linestyle=":")
        ax.text(0.97, 0.88, f"μ={mu:+.3f}  σ={sig:.3f}  ν={nu:.1f}",
                transform=ax.transAxes, ha="right", va="top",
                color="#aaa", fontsize=8)
        ax.set_title(f"h={h} ({d}) — log-return  t vs N",
                     color="#ccc", fontsize=9, pad=3)
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

        C5_pt  = C5_0 * np.exp(mu)
        C5_sig = C5_pt * sig
        x2     = np.linspace(max(0.5, C5_pt - 5 * C5_sig),
                              C5_pt + 5 * C5_sig, 600)
        y2_t   = stats.t.pdf(x2, df=nu, loc=C5_pt, scale=C5_sig)
        y2_n   = stats.norm.pdf(x2, loc=C5_pt, scale=C5_sig)

        C5_lo80 = C5_0 * np.exp(lo80_t);  C5_hi80 = C5_0 * np.exp(hi80_t)
        C5_lo95 = C5_0 * np.exp(lo95_t);  C5_hi95 = C5_0 * np.exp(hi95_t)

        ax2.plot(x2, y2_t, color="#f59f00", linewidth=2,   label=f"t(ν={nu:.1f})")
        ax2.plot(x2, y2_n, color="#74c0fc", linewidth=1.2,
                 linestyle="--", alpha=0.6, label="Gaussian")
        ax2.fill_between(x2, y2_t, where=(x2 >= C5_lo95) & (x2 <= C5_hi95),
                         color="#f59f00", alpha=0.12)
        ax2.fill_between(x2, y2_t, where=(x2 >= C5_lo80) & (x2 <= C5_hi80),
                         color="#f59f00", alpha=0.28)
        ax2.axvline(C5_pt, color="#f59f00", linewidth=1, linestyle="--", alpha=0.7)
        ax2.axvline(C5_0,  color="#555", linewidth=0.8, linestyle=":",
                    label=f"C5 now={C5_0:.2f}")
        ax2.text(0.97, 0.88,
                 f"point={C5_pt:.2f}\n80%: [{C5_lo80:.2f}, {C5_hi80:.2f}]",
                 transform=ax2.transAxes, ha="right", va="top",
                 color="#aaa", fontsize=8)
        ax2.set_title(f"h={h} ({d}) — C5 level (USD/t)",
                      color="#ccc", fontsize=9, pad=3)
        ax2.set_xlabel("USD/tonne", color="#777", fontsize=8)
        ax2.set_ylabel("density",   color="#777", fontsize=8)
        ax2.grid(color="#1a1a1a", linewidth=0.4)
        if row == 0:
            ax2.legend(fontsize=8, labelcolor="#ccc",
                       facecolor="#1a1a2e", framealpha=0.3, loc="upper left")

    plt.tight_layout()
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import prepare_main
    from feature_engineering import build_features

    df_main = prepare_main()
    df_main = build_features(df_main)
    df_main = df_main.ffill().dropna()

    results_t = run_probabilistic_pipeline_t(df_main)
    plot_forecast_t(df_main, results_t["forecast"])
    plot_distributions_grid_t(df_main, results_t["forecast"])

    results_t["forecast"].to_csv("/Users/b23/Desktop/Capstone_cleaned/result_ridge_t-test/c5_forecast_62days_t.csv")