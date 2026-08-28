from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "project_data"
CHARTS = ROOT / "project_charts"
CHARTS.mkdir(exist_ok=True)

LAGS = [1, 2, 4, 13, 52]
TEST_WEEKS = 13
FORECAST_WEEKS = 8
LEAD_TIME_WEEKS = 2
SERVICE_Z = 1.645  # 95% cycle-service assumption


def weekly_series(frame: pd.DataFrame, stock_code: str) -> pd.Series:
    item = frame.loc[frame["StockCode"].astype(str).eq(str(stock_code))]
    series = item.groupby("Week")["UnitsSold"].sum().sort_index()
    complete_index = pd.date_range(frame["Week"].min(), frame["Week"].max(), freq="W-MON")
    # The source ends on Friday 9 Dec 2011, so its final Monday-based week is partial.
    return series.reindex(complete_index, fill_value=0).astype(float).iloc[:-1]


def features(series: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"y": series})
    for lag in LAGS:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    out["mean_4"] = out["y"].shift(1).rolling(4).mean()
    out["mean_13"] = out["y"].shift(1).rolling(13).mean()
    week = out.index.isocalendar().week.astype(int)
    out["week_sin"] = np.sin(2 * np.pi * week / 52)
    out["week_cos"] = np.cos(2 * np.pi * week / 52)
    return out.dropna()


def evaluate(series: pd.Series) -> tuple[str, dict[str, float], pd.Series]:
    supervised = features(series)
    cutoff = supervised.index[-TEST_WEEKS]
    train = supervised.loc[supervised.index < cutoff]
    test = supervised.loc[supervised.index >= cutoff]
    xcols = [c for c in supervised.columns if c != "y"]

    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=42
    )
    rf.fit(train[xcols], train["y"])
    predictions = {
        "Random forest": pd.Series(rf.predict(test[xcols]), index=test.index).clip(lower=0),
        "4-week moving average": test["mean_4"].clip(lower=0),
        "Seasonal naive (52 weeks)": test["lag_52"].clip(lower=0),
    }
    maes = {name: mean_absolute_error(test["y"], pred) for name, pred in predictions.items()}
    best = min(maes, key=maes.get)
    return best, maes, predictions[best]


def recursive_rf_forecast(series: pd.Series, periods: int) -> pd.Series:
    supervised = features(series)
    xcols = [c for c in supervised.columns if c != "y"]
    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=42
    )
    rf.fit(supervised[xcols], supervised["y"])
    history = series.copy()
    values = []
    dates = []
    for _ in range(periods):
        next_date = history.index[-1] + pd.Timedelta(weeks=1)
        row = {
            f"lag_{lag}": history.iloc[-lag] for lag in LAGS
        }
        row["mean_4"] = history.iloc[-4:].mean()
        row["mean_13"] = history.iloc[-13:].mean()
        week = int(next_date.isocalendar().week)
        row["week_sin"] = np.sin(2 * np.pi * week / 52)
        row["week_cos"] = np.cos(2 * np.pi * week / 52)
        pred = max(0.0, float(rf.predict(pd.DataFrame([row])[xcols])[0]))
        history.loc[next_date] = pred
        dates.append(next_date)
        values.append(pred)
    return pd.Series(values, index=dates)


def forecast_with_selected_model(series: pd.Series, model: str, periods: int) -> pd.Series:
    if model == "Random forest":
        return recursive_rf_forecast(series, periods)
    history = series.copy()
    values, dates = [], []
    for _ in range(periods):
        next_date = history.index[-1] + pd.Timedelta(weeks=1)
        if model == "4-week moving average":
            pred = float(history.iloc[-4:].mean())
        else:
            pred = float(history.iloc[-52])
        history.loc[next_date] = max(0.0, pred)
        dates.append(next_date)
        values.append(max(0.0, pred))
    return pd.Series(values, index=dates)


def main() -> None:
    weekly = pd.read_csv(DATA / "weekly_uk_product_demand.csv", parse_dates=["Week"])
    summary = pd.read_csv(DATA / "product_summary.csv")
    selected = summary.loc[(summary["ActiveWeeks"] >= 90) & (summary["Orders"] >= 500)].nlargest(5, "Revenue")

    metrics = []
    forecasts = []
    for _, product in selected.iterrows():
        code = str(product["StockCode"])
        description = product["Description"]
        series = weekly_series(weekly, code)
        best, maes, test_prediction = evaluate(series)
        future = forecast_with_selected_model(series, best, FORECAST_WEEKS)

        recent = series.iloc[-26:]
        mean_weekly = recent.mean()
        std_weekly = recent.std(ddof=1)
        safety_stock = SERVICE_Z * std_weekly * np.sqrt(LEAD_TIME_WEEKS)
        reorder_point = mean_weekly * LEAD_TIME_WEEKS + safety_stock
        wape = np.abs(series.iloc[-TEST_WEEKS:] - test_prediction).sum() / max(
            series.iloc[-TEST_WEEKS:].sum(), 1
        )
        metrics.append(
            {
                "StockCode": code,
                "Description": description,
                "BestValidationModel": best,
                "ValidationMAE": maes[best],
                "ValidationWAPE": wape,
                "AverageWeeklyDemand26W": mean_weekly,
                "WeeklyDemandStdDev26W": std_weekly,
                "LeadTimeWeeksAssumption": LEAD_TIME_WEEKS,
                "ServiceLevelAssumption": 0.95,
                "SafetyStockUnits": np.ceil(safety_stock),
                "ReorderPointUnits": np.ceil(reorder_point),
                "ForecastDemandNext8Weeks": future.sum(),
            }
        )
        for date, value in future.items():
            forecasts.append(
                {"Week": date, "StockCode": code, "Description": description, "ForecastUnits": value}
            )

        if len(metrics) == 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            series.iloc[-52:].plot(ax=ax, color="#2457A6", linewidth=2, label="Actual demand")
            future.plot(ax=ax, color="#F28E2B", linewidth=2.5, marker="o", label="8-week forecast")
            ax.axvline(series.index[-1], color="#777777", linestyle="--", linewidth=1)
            ax.set_title(f"Weekly Demand Forecast: {description.title()}", weight="bold", fontsize=14)
            ax.set_xlabel("Week")
            ax.set_ylabel("Units")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(CHARTS / "forecast_top_product.png", dpi=180)
            plt.close(fig)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(DATA / "forecast_inventory_recommendations.csv", index=False)
    pd.DataFrame(forecasts).to_csv(DATA / "eight_week_forecasts.csv", index=False)

    chart = metrics_df.sort_values("ReorderPointUnits")
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(chart))
    ax.barh(y, chart["ReorderPointUnits"], color="#2457A6", label="Reorder point")
    ax.barh(y, chart["SafetyStockUnits"], color="#F28E2B", label="Safety stock")
    ax.set_yticks(y, [x.title() for x in chart["Description"]])
    ax.set_xlabel("Units")
    ax.set_title("Inventory Policy Recommendations", weight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS / "inventory_recommendations.png", dpi=180)
    plt.close(fig)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
