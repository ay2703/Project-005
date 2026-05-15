"""
main.py
=======
Top-level entry point for the C5 probabilistic forecasting project.

Usage
-----
    python main.py --model ridge       # Normal Ridge
    python main.py --model ridge_t     # Student-t Ridge
    python main.py --model ngboost     # NGBoost (Normal dist)
    python main.py --model all         # run all three

Outputs (saved to OUTPUT_DIR)
-------------------------------
    c5_forecast_62days.csv
    c5_forecast_62days_t.csv
    c5_forecast_62days_ngboost.csv
    c5_pca_loadings.csv
    c5_feature_importance_loadings.csv
    c5_shap_importance_ngboost.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from data_loader         import prepare_main
from feature_engineering import build_features
from ridge_normal        import (
    run_probabilistic_pipeline,
    plot_forecast,
    plot_distributions_grid,
    export_forecast_csv,
    export_pca_loadings_csv,
    export_feature_importance_loadings,
)
from ridge_student_t     import (
    run_probabilistic_pipeline_t,
    plot_forecast_t,
    plot_distributions_grid_t,
)
from ngboost_pipeline    import (
    run_probabilistic_pipeline_ngb,
    plot_forecast_ngb,
    plot_distributions_grid_ngb,
    extract_shap_importance,
)

OUTPUT_DIR = "outputs"


def prepare_data() -> object:
    """Load, merge volume, build features, clean."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = prepare_main()
    df = build_features(df)
    df = df.ffill().dropna()
    print(f"Final shape: {df.shape}")
    return df


def run_ridge(df):
    print("\n" + "=" * 60)
    print("RIDGE — Normal distribution")
    print("=" * 60)
    results = run_probabilistic_pipeline(df)
    plot_forecast(df, results["forecast"])
    plot_distributions_grid(df, results["forecast"])
    export_forecast_csv(results, os.path.join(OUTPUT_DIR, "c5_forecast_62days.csv"))
    export_pca_loadings_csv(df, os.path.join(OUTPUT_DIR, "c5_pca_loadings.csv"))
    export_feature_importance_loadings(
        df, filename=os.path.join(OUTPUT_DIR, "c5_feature_importance_loadings.csv")
    )
    return results


def run_ridge_t(df):
    print("\n" + "=" * 60)
    print("RIDGE — Student-t distribution")
    print("=" * 60)
    results_t = run_probabilistic_pipeline_t(df)
    plot_forecast_t(df, results_t["forecast"])
    plot_distributions_grid_t(df, results_t["forecast"])
    results_t["forecast"].to_csv(
        os.path.join(OUTPUT_DIR, "c5_forecast_62days_t.csv")
    )
    print(f"Saved → {OUTPUT_DIR}/c5_forecast_62days_t.csv")
    return results_t


def run_ngb(df):
    from ngboost.distns import Normal
    print("\n" + "=" * 60)
    print("NGBOOST — Normal distribution")
    print("=" * 60)
    results_ngb = run_probabilistic_pipeline_ngb(df, dist=Normal)
    plot_forecast_ngb(df, results_ngb["forecast"])
    plot_distributions_grid_ngb(
        df, results_ngb["forecast"],
        save_png=os.path.join(OUTPUT_DIR, "c5_distributions_ngboost_grid.png"),
    )
    results_ngb["forecast"].to_csv(
        os.path.join(OUTPUT_DIR, "c5_forecast_62days_ngboost.csv")
    )
    df_shap = extract_shap_importance(
        df, filename=os.path.join(OUTPUT_DIR, "c5_shap_importance_ngboost.csv")
    )
    print(f"Saved → {OUTPUT_DIR}/c5_forecast_62days_ngboost.csv")
    return results_ngb


def main():
    parser = argparse.ArgumentParser(description="C5 Probabilistic Forecasting")
    parser.add_argument(
        "--model",
        choices=["ridge", "ridge_t", "ngboost", "all"],
        default="all",
        help="Which model(s) to run",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = prepare_data()

    if args.model in ("ridge", "all"):
        run_ridge(df)

    if args.model in ("ridge_t", "all"):
        run_ridge_t(df)

    if args.model in ("ngboost", "all"):
        run_ngb(df)

    print("\nAll done.")


if __name__ == "__main__":
    main()
