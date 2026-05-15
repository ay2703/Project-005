Pipeline Flow

Data Collection - Master Data Base Version.ipynb - join all available data into a big dataset, does not fill in the blank spots for non-daily data
Data Collection - AVG Master Data.ipynb - fill in the blanks with ffill() or average of monthly production

Transform & Deseason - Check for stationary of features, transform using yeo-johnson and box-cox, uses pacf and fft to remove seasonality and autoregression

Feature Engineering - Lag by the optimal lag with respect to logC5via cross-correlation

XGBoost Feature Select - XGBoost_log_C5_Predictions.ipynb - select feature sets from transformed + deseasoned dataset

XGBoost Feature Select - XGBoost_Master_Lagged_C5_feature_sets.ipynb - select feature sets from non-transformed dataset, and combo of transformed and non-transformed dataset

prophet_c5_q1_2026_forecast.ipynb - generate neuralprophet forecast

chronos2_dual... - chronos2 forecast and ensemble
