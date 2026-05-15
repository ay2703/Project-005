"""
ngboost_pipeline.py
===================
Probabilistic C5 forecasting with NGBoost (Normal or LogNormal distribution).
One NGBRegressor is trained per horizon h = 1…62 at each walk-forward origin.

Pipeline
--------
1. build_targets         — log-return targets (reuses ridge_normal)
2. walk_forward_ngboost  — expanding-window, one NGBoost per horizon
3. aggregate_sigma       — mean predicted σ → monotone via isotonic
4. forecast_distribution_ngb — live 62-day distributional forecast
5. evaluate_ngb          — CRPS, Winkler, Coverage, MAE
6. extract_shap_importance — SHAP via TreeExplainer (last origin)
7. run_probabilistic_pipeline_ngb — orchestrator
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.isotonic      import IsotonicRegression
from scipy                 import stats

from ridge_normal import (
    HORIZON, MIN_TRAIN, STEP, TARGET_COL, N_PCA,
    build_targets, get_feature_cols, crps_gaussian, winkler_score,
    Z_80, Z_95,
)


#  install / import NGBoost

def _ensure_ngboost():
    try:
        import ngboost
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "ngboost", "-q"])


_ensure_ngboost()

from ngboost          import NGBRegressor
from ngboost.distns   import Normal, LogNormal
from ngboost.scores   import MLE


#  preprocessing

def make_preprocessor(n_pca: int = N_PCA) -> Pipeline:
    """StandardScaler → PCA — NGBoost is the model, not in the pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
    ])


def make_ngb_model(dist=Normal) -> NGBRegressor:
    """
    NGBoost with Normal or LogNormal distribution.
    n_estimators=200 balances speed vs accuracy for walk-forward.
    """
    return NGBRegressor(
        Dist           = dist,
        Score          = MLE,
        n_estimators   = 200,
        learning_rate  = 0.05,
        minibatch_frac = 1.0,
        col_sample     = 0.8,
        verbose        = False,
        random_state   = 42,
    )


#  step 2: walk-forward NGBoost

def walk_forward_ngboost(
    df        : pd.DataFrame,
    targets   : pd.DataFrame,
    horizon   : int  = HORIZON,
    min_train : int  = MIN_TRAIN,
    step      : int  = STEP,
    dist      : type = Normal,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns mu_df, sigma_df (NGBoost-predicted σ), act_df.
    sigma_df contains the NGBoost-predicted scale parameter, not residual std.
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(np.float64)
    Y = targets[[f"target_h{h}" for h in range(1, horizon + 1)]].values
    n = len(df)

    origins   = list(range(min_train, n - horizon, step))
    n_origins = len(origins)

    mu_mat    = np.full((n_origins, horizon), np.nan)
    sigma_mat = np.full((n_origins, horizon), np.nan)
    act_mat   = np.full((n_origins, horizon), np.nan)
    idx_list  = []

    dist_name = dist.__name__
    print(f"NGBoost walk-forward | dist={dist_name} | "
          f"{n_origins} origins | step={step}")
    print(f"Preprocessor: StandardScaler → PCA({N_PCA})")
    t0 = time.time()

    for i, t in enumerate(origins):
        idx_val  = df.index[t]
        date_str = idx_val.date() if hasattr(idx_val, "date") else str(idx_val)[:10]

        X_tr = X[:t];  Y_tr = Y[:t]
        valid = (~np.isnan(Y_tr).any(axis=1)) & (~np.isnan(X_tr).any(axis=1))

        if valid.sum() < 60:
            idx_list.append(idx_val)
            continue

        pre      = make_preprocessor()
        X_tr_pca = pre.fit_transform(X_tr[valid])
        x_t_pca  = pre.transform(X[t].reshape(1, -1))

        if np.isnan(x_t_pca).any():
            idx_list.append(idx_val)
            continue

        for h in range(1, horizon + 1):
            y_h = Y_tr[valid, h - 1]
            if np.isnan(y_h).any():
                continue

            ngb = make_ngb_model(dist=dist)
            ngb.fit(X_tr_pca, y_h)

            pred_dist = ngb.pred_dist(x_t_pca)
            mu_mat[i, h - 1]    = pred_dist.loc[0]
            sigma_mat[i, h - 1] = pred_dist.scale[0]

        act_mat[i] = Y[t]

        elapsed = time.time() - t0
        eta     = (elapsed / (i + 1)) * (n_origins - i - 1)
        print(f"  [{i+1:2d}/{n_origins}] {date_str}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        idx_list.append(idx_val)

    h_cols   = [f"h{h}" for h in range(1, horizon + 1)]
    mu_df    = pd.DataFrame(mu_mat,    index=idx_list, columns=h_cols)
    sigma_df = pd.DataFrame(sigma_mat, index=idx_list, columns=h_cols)
    act_df   = pd.DataFrame(act_mat,   index=idx_list, columns=h_cols)

    total = time.time() - t0
    print(f"\nDone. {total/60:.1f}min | Shape: {mu_df.shape}")
    return mu_df, sigma_df, act_df


# step 3: aggregate σ (monotone)

def aggregate_sigma(
    sigma_df : pd.DataFrame,
    horizon  : int = HORIZON,
) -> np.ndarray:
    """Average predicted σ across origins, enforce monotone growth."""
    h_cols    = [f"h{h}" for h in range(1, horizon + 1)]
    raw_sigma = sigma_df[h_cols].mean(axis=0).values

    iso            = IsotonicRegression(increasing=True)
    sigma_monotone = iso.fit_transform(np.arange(1, horizon + 1), raw_sigma)

    print(f"\n{'h':>4}  {'raw_σ (mean)':>13}  {'monotone_σ':>12}")
    for h in [1, 5, 10, 21, 42, 62]:
        print(f"  {h:4d}  {raw_sigma[h-1]:13.4f}  {sigma_monotone[h-1]:12.4f}")

    return sigma_monotone


# step 4: live forecast

def forecast_distribution_ngb(
    df      : pd.DataFrame,
    sigma   : np.ndarray,
    horizon : int  = HORIZON,
    origin_t: int  = -1,
    dist    : type = Normal,
) -> pd.DataFrame:
    """Train NGBoost on full history, predict 62-day distributional forecast."""
    feature_cols = get_feature_cols(df)
    targets      = build_targets(df, horizon)
    h_cols_y     = [f"target_h{h}" for h in range(1, horizon + 1)]
    X            = df[feature_cols].values.astype(np.float64)
    Y            = targets[h_cols_y].values

    t    = origin_t if origin_t >= 0 else len(df) - 1
    C5_t = df[TARGET_COL].iloc[t]

    valid    = (~np.isnan(Y[:t]).any(axis=1)) & (~np.isnan(X[:t]).any(axis=1))
    pre      = make_preprocessor()
    X_tr_pca = pre.fit_transform(X[:t][valid])
    x_t_pca  = pre.transform(X[t].reshape(1, -1))

    date_str = df.index[t].date() if hasattr(df.index[t], "date") else str(df.index[t])[:10]
    print(f"Live forecast from {date_str}, C5={C5_t:.2f}")

    rows = []
    for h in range(1, horizon + 1):
        y_h = Y[:t][valid, h - 1]
        ngb = make_ngb_model(dist=dist)
        ngb.fit(X_tr_pca, y_h)

        pred_dist = ngb.pred_dist(x_t_pca)
        mu_h      = float(pred_dist.loc[0])
        sig_h     = sigma[h - 1]       # monotone aggregated σ
        fdate     = df.index[t] + pd.offsets.BDay(h)

        lo80 = mu_h - Z_80 * sig_h;  hi80 = mu_h + Z_80 * sig_h
        lo95 = mu_h - Z_95 * sig_h;  hi95 = mu_h + Z_95 * sig_h

        rows.append({
            "horizon"     : h,
            "date"        : fdate,
            "mu_logret"   : mu_h,
            "sigma_logret": sig_h,
            "C5_point"    : C5_t * np.exp(mu_h),
            "C5_lower80"  : C5_t * np.exp(lo80),
            "C5_upper80"  : C5_t * np.exp(hi80),
            "C5_lower95"  : C5_t * np.exp(lo95),
            "C5_upper95"  : C5_t * np.exp(hi95),
        })

    return pd.DataFrame(rows).set_index("horizon")


#  evaluation

def evaluate_ngb(
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

        lo80 = mu_v - Z_80 * sig_h;  hi80 = mu_v + Z_80 * sig_h
        lo95 = mu_v - Z_95 * sig_h;  hi95 = mu_v + Z_95 * sig_h

        crps_vals = [crps_gaussian(m, sig_h, y) for m, y in zip(mu_v, act_v)]
        w80  = [winkler_score(l, u, y, 0.20) for l, u, y in zip(lo80, hi80, act_v)]
        w95  = [winkler_score(l, u, y, 0.05) for l, u, y in zip(lo95, hi95, act_v)]

        records.append({
            "horizon"    : h,
            "sigma"      : sig_h,
            "CRPS"       : np.nanmean(crps_vals),
            "Winkler_80" : np.nanmean(w80),
            "Winkler_95" : np.nanmean(w95),
            "Coverage_80": ((act_v >= lo80) & (act_v <= hi80)).mean(),
            "Coverage_95": ((act_v >= lo95) & (act_v <= hi95)).mean(),
            "MAE_logret" : (mu_v - act_v).abs().mean(),
            "n_obs"      : len(act_v),
        })
    return pd.DataFrame(records).set_index("horizon")


#  feature importance via SHAP

def _get_loc_tree(ngb_model):
    """Locate the location parameter base learner for SHAP."""
    if hasattr(ngb_model, "learners_"):
        return ngb_model.learners_[0][-1]
    for attr in ("base_models_", "estimators_", "_learners"):
        if hasattr(ngb_model, attr):
            container = getattr(ngb_model, attr)
            try:
                return container[0][-1]
            except (IndexError, TypeError):
                try:
                    return container[-1]
                except Exception:
                    pass
    raise AttributeError(
        f"Cannot locate base learner. Attrs: "
        f"{[a for a in dir(ngb_model) if not a.startswith('__')]}"
    )


def extract_shap_importance(
    df       : pd.DataFrame,
    horizon  : int  = HORIZON,
    min_train: int  = MIN_TRAIN,
    step     : int  = STEP,
    top_n    : int  = 10,
    dist     : type = Normal,
    filename : str  = "c5_shap_importance_ngboost.csv",
) -> pd.DataFrame:
    """
    Compute SHAP values for key horizons at the last walk-forward origin.
    Falls back to KernelExplainer if TreeExplainer fails.
    """
    try:
        import shap
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "shap", "-q"])
        import shap

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(np.float64)
    Y = build_targets(df)[[f"target_h{h}" for h in range(1, horizon + 1)]].values
    n = len(df)

    origins = list(range(min_train, n - horizon, step))
    t = origins[-1]
    date_str = df.index[t].date() if hasattr(df.index[t], "date") else str(df.index[t])[:10]
    print(f"SHAP importance at last origin: {date_str}")

    X_tr = X[:t];  Y_tr = Y[:t]
    valid = (~np.isnan(Y_tr).any(axis=1)) & (~np.isnan(X_tr).any(axis=1))

    pre        = make_preprocessor()
    X_tr_pca   = pre.fit_transform(X_tr[valid])
    pca        = pre.named_steps["pca"]
    components = pca.components_   # (n_pca, n_features)

    records = []
    for h in [1, 5, 10, 21, 42, 62]:
        print(f"  Computing SHAP for h={h}...")
        y_h = Y_tr[valid, h - 1]
        ngb = make_ngb_model(dist=dist)
        ngb.fit(X_tr_pca, y_h)

        try:
            loc_tree  = _get_loc_tree(ngb)
            explainer = shap.TreeExplainer(loc_tree)
            shap_vals = explainer.shap_values(X_tr_pca)   # (n, n_pca)
        except Exception as e:
            print(f"    TreeExplainer failed ({e}), falling back to KernelExplainer...")
            bg        = shap.kmeans(X_tr_pca, 20)
            explainer = shap.KernelExplainer(lambda x: ngb.predict(x), bg)
            shap_vals = explainer.shap_values(X_tr_pca[:100])

        mean_shap = np.abs(shap_vals).mean(axis=0)        # (n_pca,)
        feat_imp  = np.abs(mean_shap @ components)        # (n_features,)
        pct       = feat_imp / feat_imp.sum()

        for rank, fidx in enumerate(np.argsort(feat_imp)[::-1][:top_n], 1):
            records.append({
                "horizon"        : h,
                "rank"           : rank,
                "feature"        : feature_cols[fidx],
                "shap_importance": float(feat_imp[fidx]),
                "pct_of_total"   : float(pct[fidx]),
            })

    df_shap = pd.DataFrame(records)
    df_shap.to_csv(filename, index=False)
    print(f"\nSaved → {filename}")

    print("\n" + "=" * 65)
    print("TOP 5 FEATURES BY HORIZON (SHAP)")
    print("=" * 65)
    for h in [1, 5, 10, 21, 42, 62]:
        sub = df_shap[df_shap["horizon"] == h].head(5)
        print(f"\n  h={h:2d}:")
        for _, row in sub.iterrows():
            bar = "█" * int(row["pct_of_total"] * 200)
            print(f"    #{int(row['rank'])}  {row['feature']:<48} "
                  f"{row['pct_of_total']*100:5.1f}%  {bar}")

    return df_shap


#  main pipeline

def run_probabilistic_pipeline_ngb(
    df  : pd.DataFrame,
    dist: type = Normal,
) -> dict:
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
    print("STEP 2 — Walk-forward NGBoost forecasts")
    print("=" * 60)
    mu_df, sigma_df, act_df = walk_forward_ngboost(df, targets, dist=dist)

    print("\n" + "=" * 60)
    print("STEP 3 — Aggregate sigma (monotone)")
    print("=" * 60)
    sigma = aggregate_sigma(sigma_df)

    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)
    eval_df = evaluate_ngb(mu_df, act_df, sigma)
    print("\nKey horizons:")
    print(eval_df.loc[eval_df.index.isin([1, 5, 10, 21, 42, 62])].to_string())

    print("\n" + "=" * 60)
    print("STEP 4b — Live 62-day forecast")
    print("=" * 60)
    forecast = forecast_distribution_ngb(df, sigma, dist=dist)
    print(forecast[["C5_point", "C5_lower80", "C5_upper80",
                     "C5_lower95", "C5_upper95"]].iloc[[0, 4, 9, 20, 41, 61]])

    return {
        "mu_df"   : mu_df,
        "sigma_df": sigma_df,
        "act_df"  : act_df,
        "sigma"   : sigma,
        "eval_df" : eval_df,
        "forecast": forecast,
    }


#  plots

def plot_forecast_ngb(df, forecast, history_days=120):
    """62-day NGBoost cone with σ-growth subplot."""
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
        f"C5 Capesize — 62-day NGBoost probabilistic forecast  "
        f"(origin: {origin_str})",
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
    ax1.plot(dates, forecast["C5_point"], color="#74c0fc",
             linewidth=1.8, label="Point forecast (NGBoost μ)")
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

    ax2.plot(dates, forecast["sigma_logret"], color="#f59f00",
             linewidth=1.6, label="σ_h (NGBoost)")
    ax2.fill_between(dates, 0, forecast["sigma_logret"],
                     color="#f59f00", alpha=0.15)
    ax2.set_ylabel("σ_h", color="#aaa", fontsize=10)
    ax2.set_title("Uncertainty growth (σ_h monotone)", color="#888", fontsize=9, pad=4)
    ax2.legend(fontsize=8, labelcolor="#ccc", facecolor="#1a1a2e", framealpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", color="#222", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_distributions_grid_ngb(
    df      : pd.DataFrame,
    forecast: pd.DataFrame,
    horizons: list = [1, 5, 10, 21, 31, 42, 52, 62],
    save_png: str  = "/Users/b23/Desktop/Capstone_cleaned/result_ngboost/c5_distributions_ngboost_grid.png",
) -> None:
    """Static 8-horizon grid (log-return + C5 level) for NGBoost forecasts."""
    import matplotlib.pyplot as plt

    mu_vec    = forecast["mu_logret"].values
    sigma_vec = forecast["sigma_logret"].values
    dates     = pd.to_datetime(forecast["date"])
    C5_0      = df[TARGET_COL].iloc[-1]
    n         = len(horizons)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.8 * n), facecolor="#0f1117")
    fig.suptitle(
        f"C5 — NGBoost predictive distributions  (C5={C5_0:.2f} USD/t)",
        color="#e0e0e0", fontsize=12, fontweight="bold", y=1.01,
    )

    for row, h in enumerate(horizons):
        mu  = mu_vec[h - 1];  sig = sigma_vec[h - 1]
        d   = dates.iloc[h - 1].date()
        x   = np.linspace(mu - 4 * sig, mu + 4 * sig, 500)
        y   = stats.norm.pdf(x, mu, sig)
        lo80 = mu - Z_80 * sig;  hi80 = mu + Z_80 * sig
        lo95 = mu - Z_95 * sig;  hi95 = mu + Z_95 * sig

        for col_idx, (ax, space, color, title_sfx) in enumerate([
            (axes[row, 0], "log-return", "#74c0fc", "log-return"),
            (axes[row, 1], "level",      "#a9e34b", "C5 level"),
        ]):
            ax.set_facecolor("#0f1117")
            ax.tick_params(colors="#777", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a2a")

            if space == "log-return":
                ax.plot(x, y, color=color, linewidth=2)
                ax.fill_between(x, y, where=(x >= lo95) & (x <= hi95),
                                color=color, alpha=0.15, label="95%")
                ax.fill_between(x, y, where=(x >= lo80) & (x <= hi80),
                                color=color, alpha=0.35, label="80%")
                ax.axvline(mu, color=color, linewidth=1, linestyle="--", alpha=0.7)
                ax.axvline(0,  color="#555", linewidth=0.8, linestyle=":")
                ax.text(0.97, 0.88, f"μ={mu:+.3f}  σ={sig:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        color="#aaa", fontsize=8)
                ax.set_xlabel("log-return", color="#777", fontsize=8)
            else:
                C5_pt  = C5_0 * np.exp(mu)
                C5_sig = C5_pt * sig
                x2 = np.linspace(max(0.5, C5_pt - 4 * C5_sig),
                                  C5_pt + 4 * C5_sig, 500)
                y2 = stats.norm.pdf(x2, C5_pt, C5_sig)
                ax.plot(x2, y2, color=color, linewidth=2)
                ax.fill_between(x2, y2,
                    where=(x2 >= C5_0 * np.exp(lo95)) & (x2 <= C5_0 * np.exp(hi95)),
                    color=color, alpha=0.15, label="95%")
                ax.fill_between(x2, y2,
                    where=(x2 >= C5_0 * np.exp(lo80)) & (x2 <= C5_0 * np.exp(hi80)),
                    color=color, alpha=0.35, label="80%")
                ax.axvline(C5_pt, color=color, linewidth=1, linestyle="--", alpha=0.7)
                ax.axvline(C5_0,  color="#555", linewidth=0.8, linestyle=":")
                ax.text(0.97, 0.88,
                        f"point={C5_pt:.2f}\n"
                        f"80%: [{C5_0*np.exp(lo80):.2f}, {C5_0*np.exp(hi80):.2f}]",
                        transform=ax.transAxes, ha="right", va="top",
                        color="#aaa", fontsize=8)
                ax.set_xlabel("USD/tonne", color="#777", fontsize=8)

            ax.set_title(f"h={h} ({d}) — {title_sfx}",
                         color="#ccc", fontsize=9, pad=3)
            ax.set_ylabel("density", color="#777", fontsize=8)
            ax.grid(color="#1a1a1a", linewidth=0.4)
            if row == 0:
                ax.legend(fontsize=8, labelcolor="#ccc",
                          facecolor="#1a1a2e", framealpha=0.3, loc="upper left")

    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.show()


#  entry point

if __name__ == "__main__":
    from data_loader import prepare_main
    from feature_engineering import build_features

    df_main = prepare_main()
    df_main = build_features(df_main)
    df_main = df_main.ffill().dropna()

    results = run_probabilistic_pipeline_ngb(df_main, dist=Normal)

    plot_forecast_ngb(df_main, results["forecast"])
    plot_distributions_grid_ngb(df_main, results["forecast"])

    results["forecast"].to_csv("/Users/b23/Desktop/Capstone_cleaned/result_ngboost/c5_forecast_62days_ngboost.csv")

    df_shap = extract_shap_importance(df_main)
