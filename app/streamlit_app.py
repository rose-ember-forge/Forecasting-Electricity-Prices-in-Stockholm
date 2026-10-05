"""Dashboard for the SE3 price forecasts: tomorrow's forecast and the backtest.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "forecasts"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, GRID = "#0b0b0b", "#e1e0d9"
FINAL = "Average of LEAR and LightGBM"
MODEL_COLORS = {  # colour follows the model, whatever else is selected
    FINAL: SERIES[0],
    "Naive (yesterday / last week)": SERIES[1],
    "LEAR (Lasso, one model per hour)": SERIES[2],
    "LightGBM, prices + calendar + weather": SERIES[3],
    "LightGBM, prices + calendar": SERIES[4],
    "Same hour last week": SERIES[5],
    FINAL + " (observed weather)": SERIES[6],
}

st.set_page_config(page_title="SE3 electricity price forecast", layout="wide")


@st.cache_data
def load():
    preds = pd.read_parquet(PROCESSED / "backtest_predictions.parquet")
    preds["time"] = preds["date"] + pd.to_timedelta(preds["hour"], unit="h")
    hourly = pd.read_parquet(PROCESSED / "hourly.parquet")
    return preds, hourly


def style(fig: go.Figure, y_title: str = "EUR/MWh") -> go.Figure:
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=y_title,
    )
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


preds, hourly = load()
models = list(dict.fromkeys(preds["model"]))

st.title("Forecasting Stockholm's electricity price")
st.caption(
    "Day-ahead prices for bidding zone SE3, forecast the morning before the noon auction. "
    "Prices from elprisetjustnu.se, weather from SMHI and Open-Meteo."
)

tab_next, tab_backtest, tab_history = st.tabs(["Next day", "Backtest", "Price history"])

with tab_next:
    latest_path = FORECASTS / "latest.csv"
    if latest_path.exists():
        latest = pd.read_csv(latest_path)
        day = latest["date"].iloc[0]
        st.subheader(f"Forecast for {pd.Timestamp(day):%A %d %B %Y}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Daily mean", f"{latest['forecast_eur_mwh'].mean():.0f} EUR/MWh")
        cheapest = latest.loc[latest["forecast_eur_mwh"].idxmin()]
        dearest = latest.loc[latest["forecast_eur_mwh"].idxmax()]
        c2.metric("Cheapest hour", f"{int(cheapest['hour']):02d}:00", f"{cheapest['forecast_eur_mwh']:.0f} EUR/MWh", delta_color="off")
        c3.metric("Most expensive hour", f"{int(dearest['hour']):02d}:00", f"{dearest['forecast_eur_mwh']:.0f} EUR/MWh", delta_color="off")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=latest["hour"], y=latest["high_80_eur_mwh"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=latest["hour"], y=latest["low_80_eur_mwh"], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                 line=dict(width=0), name="80% interval", hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=latest["hour"], y=latest["forecast_eur_mwh"], line=dict(color=SERIES[0], width=2),
                                 name="Forecast", hovertemplate="%{y:.1f}"))
        fig.update_xaxes(title="Hour", tickmode="linear", dtick=2)
        st.plotly_chart(style(fig), width="stretch")
        st.caption(f"Made {latest['made_at_utc'].iloc[0]} UTC. Updated every morning by a scheduled GitHub Action.")
        scores_path = FORECASTS / "live_scores.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            st.write(f"Live track record: {len(scores)} days, average MAE {scores['mae'].mean():.1f} EUR/MWh.")
            st.dataframe(scores.sort_values("date", ascending=False), hide_index=True)
    else:
        st.info("No live forecast yet. Run `python -m se3forecast.forecast`.")

with tab_backtest:
    st.subheader("October 2025 to September 2026, retrained every month")
    left, right = st.columns([2, 1])
    chosen = left.multiselect("Models", models, default=[FINAL, "Naive (yesterday / last week)"])
    dates = sorted(preds["date"].dt.date.unique())
    start, end = right.select_slider("Period", options=dates, value=(dates[103], dates[116]), format_func=lambda d: f"{d:%d %b %Y}")
    window = preds[preds["date"].dt.date.between(start, end)]
    if not chosen:
        st.warning("Pick at least one model.")
    else:
        fig = go.Figure()
        actual = window[window["model"] == models[0]]
        fig.add_trace(go.Scatter(x=actual["time"], y=actual["y"], name="Actual", line=dict(color=INK, width=1.6), hovertemplate="%{y:.1f}"))
        for name in chosen:
            part = window[window["model"] == name]
            fig.add_trace(go.Scatter(x=part["time"], y=part["pred"], name=name, line=dict(color=MODEL_COLORS.get(name, SERIES[7]), width=2),
                                     hovertemplate="%{y:.1f}"))
        st.plotly_chart(style(fig), width="stretch")

        err = window[window["model"].isin(chosen)].assign(abs_err=lambda d: (d["pred"] - d["y"]).abs())
        table = err.groupby("model").agg(MAE=("abs_err", "mean"), bias=("pred", "mean")).reindex(chosen)
        table["bias"] = table["bias"] - actual["y"].mean()
        st.dataframe(table.round(1).rename(columns={"MAE": "MAE (EUR/MWh)", "bias": "Bias (EUR/MWh)"}))

        st.markdown("**Error by hour of day, whole test year**")
        full = preds[preds["model"].isin(chosen)].assign(abs_err=lambda d: (d["pred"] - d["y"]).abs())
        by_hour = full.groupby(["model", "hour"])["abs_err"].mean().reset_index()
        fig = go.Figure()
        for name in chosen:
            part = by_hour[by_hour["model"] == name]
            fig.add_trace(go.Scatter(x=part["hour"], y=part["abs_err"], name=name, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS.get(name, SERIES[7]), width=2), hovertemplate="%{y:.1f}"))
        fig.update_xaxes(title="Hour", tickmode="linear", dtick=2)
        st.plotly_chart(style(fig, "MAE, EUR/MWh"), width="stretch")

with tab_history:
    st.subheader("Daily average price by bidding zone")
    daily = hourly.groupby("date")[["se1", "se2", "se3", "se4"]].mean()
    fig = go.Figure()
    for i, col in enumerate(["se1", "se2", "se3", "se4"]):
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], name=col.upper(), line=dict(color=SERIES[[2, 3, 0, 1][i]], width=1.2),
                                 hovertemplate="%{y:.1f}"))
    st.plotly_chart(style(fig), width="stretch")
    st.caption("SE1 Luleå, SE2 Sundsvall, SE3 Stockholm, SE4 Malmö. Since 1 October 2025 prices are set per 15 minutes; they are averaged to hours here.")
