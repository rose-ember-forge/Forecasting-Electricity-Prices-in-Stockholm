"""Static charts for the README, drawn from ``data/processed``.

Usage::

    python -m se3forecast.figures
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import FIGURES, PROCESSED
from .backtest import FINAL, TEST, load_features
from .models import GBM, Naive

# Validated categorical order (colour-blind safe on adjacent pairs), shared with the other portfolio projects.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL_BAR = "#c9c7bd"

EXAMPLE_WEEKS = ("2026-01-12", "2026-01-25")

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "legend.frameon": False,
    }
)


def _title(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()  # place text in inches from the top so spacing is the same on every chart
    fig.text(0.01, 1 - 0.12 / h, title, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.01, 1 - 0.45 / h, subtitle, fontsize=10, va="top", color=INK_2)


def _source(fig, extra: str = "") -> None:
    note = "Prices: elprisetjustnu.se (Nord Pool day-ahead). Weather: SMHI, Open-Meteo. " + extra
    fig.text(0.01, 0.01, note.strip(), fontsize=8, color=MUTED, va="bottom")


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def predictions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "backtest_predictions.parquet")


def model_comparison() -> None:
    m = pd.read_csv(PROCESSED / "backtest_metrics.csv", index_col="model")
    m = m.drop(index=[i for i in m.index if "observed weather" in i]).sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.33, right=0.95)
    colors = [SERIES[0] if name == FINAL else NEUTRAL_BAR for name in m.index]
    ax.barh(range(len(m)), m["mae"], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ci = pd.read_csv(PROCESSED / "bootstrap_ci.csv", index_col="model").reindex(m.index)
    ax.hlines(range(len(m)), ci["mae_low"], ci["mae_high"], color=INK_2, lw=1.2)
    for y, (name, row) in enumerate(m.iterrows()):
        change = 1 - row["rmae"]
        label = f"{row['mae']:.1f}"
        if name != Naive.name:
            label += f"   ({abs(change):.0%} {'lower' if change > 0 else 'higher'} than naive)"
        ax.text(ci.loc[name, "mae_high"] + 0.5, y, label, va="center", fontsize=9, color=INK if name == FINAL else INK_2,
                fontweight="bold" if name == FINAL else "normal")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlim(0, ci["mae_high"].max() * 1.4)
    ax.set_xlabel("Mean absolute error, EUR/MWh (lower is better)")
    _title(fig, "Averaging a Lasso model and LightGBM cuts the error by 44%",
           "Day-ahead forecasts of every hour from October 2025 to September 2026, retrained monthly")
    _source(fig, "Whiskers: 95% block bootstrap interval.")
    _save(fig, "model_comparison")


SHORT_NAMES = {
    "Naive (yesterday / last week)": "Naive",
    "Same hour last week": "Last week",
    "LEAR (Lasso, one model per hour)": "LEAR",
    "LightGBM, prices + calendar": "LightGBM,\nno weather",
    "LightGBM, prices + calendar + weather": "LightGBM",
    FINAL: "Average",
}


def significance() -> None:
    """Diebold-Mariano chessboard: is the row model significantly more accurate than the column?"""
    p = pd.read_csv(PROCESSED / "dm_pvalues.csv", index_col="model")
    keep = [m for m in SHORT_NAMES if m in p.index]
    p = p.loc[keep, keep]
    bins = [(0.001, "#0d3b73", "< 0.001"), (0.01, "#2a78d6", "< 0.01"), (0.05, "#8fb8ea", "< 0.05"), (1.01, "#ecebe6", "not significant")]
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.2, right=0.97)
    for i, row in enumerate(keep):
        for j, col in enumerate(keep):
            if i == j:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=SURFACE))
                continue
            v = p.loc[row, col]
            color = next(c for limit, c, _ in bins if v < limit)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=color, lw=0))
            if v < 0.05 or v < 0.2:
                text = "<0.001" if v < 0.001 else f"{v:.3f}"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8.5,
                        color="white" if v < 0.01 else INK)
    ax.set_xlim(0, len(keep))
    ax.set_ylim(len(keep), 0)
    ax.set_xticks(np.arange(len(keep)) + 0.5)
    ax.set_xticklabels([SHORT_NAMES[k] for k in keep])
    ax.set_yticks(np.arange(len(keep)) + 0.5)
    ax.set_yticklabels([SHORT_NAMES[k] for k in keep])
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xlabel("... is more accurate than this model", labelpad=8)
    ax.set_ylabel("This model ...", labelpad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in bins]
    ax.legend(handles, [f"p {lab}" if lab != "not significant" else lab for _, _, lab in bins], loc="upper left",
              bbox_to_anchor=(0, -0.2), ncols=4, fontsize=9, handlelength=1.2)
    _title(fig, "The average is significantly better than every other model",
           "One-sided Diebold-Mariano tests on the daily mean absolute error, test year (365 days)")
    _source(fig)
    _save(fig, "significance")


def example_weeks() -> None:
    p = predictions()
    lo, hi = EXAMPLE_WEEKS
    final = p[(p["model"] == FINAL) & p["date"].between(lo, hi)].copy()
    band = p[(p["model"] == "LightGBM, prices + calendar + weather") & p["date"].between(lo, hi)]
    t = final["date"] + pd.to_timedelta(final["hour"], unit="h")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.07, right=0.97)
    ax.fill_between(t, band["lo"], band["hi"], color=SERIES[0], alpha=0.14, lw=0, label="80% prediction interval")
    ax.plot(t, final["y"], color=INK, lw=1.6, label="Actual price")
    ax.plot(t, final["pred"], color=SERIES[0], lw=2, label="Forecast, made the day before")
    ax.set_ylabel("EUR/MWh")
    ax.set_xlim(t.min(), t.max())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%a %d %b"))
    ax.legend(loc="upper left", ncols=3, fontsize=9, bbox_to_anchor=(0, 1.02))
    mae = (final["pred"] - final["y"]).abs().mean()
    _title(fig, "Two winter weeks: the shape is right, the peaks are hard",
           f"SE3 hourly price and the final model's forecast, {pd.Timestamp(lo):%d %b} to {pd.Timestamp(hi):%d %b %Y} (MAE {mae:.1f} EUR/MWh)")
    _source(fig)
    _save(fig, "example_weeks")


def error_by_hour() -> None:
    p = predictions()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.08, right=0.8)
    for name, color, label in [(Naive.name, SERIES[1], "Naive"), (FINAL, SERIES[0], "Final model")]:
        part = p[p["model"] == name]
        mae = (part["pred"] - part["y"]).abs().groupby(part["hour"]).mean()
        ax.plot(mae.index, mae.values, color=color, lw=2, marker="o", ms=4)
        ax.annotate(label, (23, mae.iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK_2, fontsize=9)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylim(0)
    ax.set_ylabel("MAE, EUR/MWh")
    _title(fig, "Evening peaks are the hardest hours to forecast",
           "Mean absolute error by hour of the day, October 2025 to September 2026")
    _source(fig)
    _save(fig, "error_by_hour")


def error_by_month() -> None:
    p = predictions()
    p = p[p["model"].isin([Naive.name, FINAL])]
    p = p.assign(month=p["date"].dt.to_period("M"), ae=(p["pred"] - p["y"]).abs())
    mae = p.pivot_table(index="month", columns="model", values="ae", aggfunc="mean")
    price = p[p["model"] == FINAL].groupby("month")["y"].mean()
    x = np.arange(len(mae))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35})
    fig.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=0.97)
    w = 0.38
    ax.bar(x - w / 2, mae[Naive.name], w, color=SERIES[1], label="Naive", edgecolor=SURFACE, linewidth=2)
    ax.bar(x + w / 2, mae[FINAL], w, color=SERIES[0], label="Final model", edgecolor=SURFACE, linewidth=2)
    ax.set_ylabel("MAE, EUR/MWh")
    ax.set_title("Forecast error", fontsize=10)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    ax2.bar(x, price, 0.6, color=NEUTRAL_BAR)
    ax2.set_ylabel("EUR/MWh")
    ax2.set_title("Average SE3 price", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.strftime("%b\n%Y") if m.month in (1, 10) else m.strftime("%b") for m in mae.index])
    _title(fig, "The model beats the naive forecast in every month",
           "Mean absolute error per month (top) next to the average price that month (bottom)")
    _source(fig)
    _save(fig, "error_by_month")


def price_history() -> None:
    h = pd.read_parquet(PROCESSED / "hourly.parquet")
    weekly = h.set_index("date")[["se1", "se3", "se4"]].resample("W").mean().iloc[:-1]  # drop the unfinished week
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.07, right=0.86)
    series = [("se4", SERIES[1], "SE4 Malmö"), ("se3", SERIES[0], "SE3 Stockholm"), ("se1", SERIES[2], "SE1 Luleå")]
    label_y = sorted(((weekly[col].iloc[-1], col) for col, _, _ in series), reverse=True)
    placed = {}
    for value, col in label_y:  # keep end labels at least 20 EUR/MWh apart
        placed[col] = min(value, min(placed.values(), default=np.inf) - 20)
    for col, color, label in series:
        ax.plot(weekly.index, weekly[col], color=color, lw=2 if col == "se3" else 1.4)
        ax.annotate(label, (weekly.index[-1], placed[col]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.axvspan(pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1]), color=GRID, alpha=0.6, lw=0)
    ax.text(pd.Timestamp(TEST[0]) + pd.Timedelta(days=10), ax.get_ylim()[1] * 0.95, "test year", fontsize=9, color=INK_2, va="top")
    ax.set_ylabel("EUR/MWh, weekly average")
    ax.set_ylim(0)
    _title(fig, "The 2022 energy crisis dwarfs everything since",
           "Weekly average day-ahead price in three of Sweden's four bidding zones")
    _source(fig)
    _save(fig, "price_history")


def feature_importance(top: int = 15) -> pd.Series:
    """Gain importance of the LightGBM model trained on everything before the test year."""
    train, _ = load_features()
    train = train[train["date"] < TEST[0]].dropna()
    model = GBM(day_lags=True).fit(train)
    imp = pd.Series(model.model.booster_.feature_importance("gain"), index=model.features)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(PROCESSED / "feature_importance.csv", header=["share_of_gain"])
    labels = {
        "p_lag1": "Price, same hour yesterday",
        "p_lag2": "Price, same hour 2 days ago",
        "p_lag7": "Price, same hour last week",
        "p_hour_mean7": "Price, same hour, 7-day mean",
        "p_hour_std7": "Price, same hour, 7-day spread",
        "d1_mean": "Yesterday's mean price",
        "d1_min": "Yesterday's lowest price",
        "d1_max": "Yesterday's highest price",
        "d1_std": "Yesterday's price spread",
        "d1_last": "Yesterday's 23:00 price",
        "d7_mean": "Last week's mean price",
        "se1_d1_mean": "Yesterday's mean price in SE1",
        "se2_d1_mean": "Yesterday's mean price in SE2",
        "se4_d1_mean": "Yesterday's mean price in SE4",
        "spread_se3_se1_d1": "Yesterday's SE3 minus SE1",
        "spread_se4_se3_d1": "Yesterday's SE4 minus SE3",
        "wind": "Wind, this hour",
        "wind_mean": "Wind, daily mean",
        "wind_max": "Wind, daily max",
        "wind_mean_change": "Wind, change from yesterday",
        "temp": "Temperature, this hour",
        "temp_mean": "Temperature, daily mean",
        "temp_mean_change": "Temperature, change from yesterday",
        "hour": "Hour of day",
        "dow": "Day of week",
        "month": "Month",
        "doy_sin": "Season (sine)",
        "doy_cos": "Season (cosine)",
        "holiday": "Public holiday",
    }
    show = imp.head(top)[::-1]
    names = [labels.get(f, f"Price {f[3:].replace('h', '')}:00, {f[1]} day{'s' if f[1] != '1' else ''} ago") for f in show.index]
    weather = {"wind", "wind_mean", "wind_max", "wind_mean_change", "temp", "temp_mean", "temp_mean_change"}
    colors = [SERIES[2] if f in weather else NEUTRAL_BAR for f in show.index]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.36, right=0.95)
    ax.barh(range(len(show)), show.values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(show.values):
        ax.text(v + 0.003, y, f"{v:.0%}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of the model's total split gain")
    _title(fig, "Recent prices matter most, then the wind",
           f"Top {top} LightGBM features; weather features in green")
    _source(fig, "Model trained on Nov 2021 to Sep 2025.")
    _save(fig, "feature_importance")
    return imp


def main() -> None:
    price_history()
    model_comparison()
    significance()
    example_weeks()
    error_by_hour()
    error_by_month()
    print(feature_importance().head(10))


if __name__ == "__main__":
    main()
