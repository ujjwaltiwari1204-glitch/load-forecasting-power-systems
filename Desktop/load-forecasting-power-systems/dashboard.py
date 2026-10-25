from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score


st.set_page_config(page_title="AEP Load Forecasting", layout="wide")

MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
DATA_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_predictions(horizon):
    path = DATA_DIR / f"test_predictions_{horizon}h.csv"

    if not path.exists():
        st.error(f"Missing {path.name}. Run the export cell in the notebook.")
        st.stop()

    data = pd.read_csv(
        path,
        index_col="Target timestamp",
        parse_dates=["Target timestamp"],
    ).sort_index()

    required = ["Actual", *MODELS]
    missing = [column for column in required if column not in data]

    if missing or data.index.has_duplicates or data[required].isna().any().any():
        st.error(f"{path.name} has missing columns, timestamps, or values.")
        st.stop()

    return data


def metrics(actual, predicted):
    actual = actual.to_numpy(dtype=float)
    predicted = predicted.to_numpy(dtype=float)
    error = actual - predicted

    return {
        "RMSE (MW)": np.sqrt(np.mean(error ** 2)),
        "MAE (MW)": np.mean(np.abs(error)),
        "MAPE (%)": 100 * np.mean(np.abs(error / actual)),
        "R²": r2_score(actual, predicted),
    }


h1 = load_predictions(1)
h24 = load_predictions(24)

st.title("AEP Regional Electricity Load Forecasting")
st.caption(
    "Historical test-set evaluation. Load is measured in MW. "
    "The date control selects past predictions; it does not generate "
    "forecasts for arbitrary future dates."
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [1, 24],
    format_func=lambda hours: f"{hours}-hour ahead",
)

data = h1 if horizon == 1 else h24

selected_models = st.sidebar.multiselect(
    "Models to plot",
    MODELS,
    default=["XGBoost", "LSTM"],
)

available_baselines = [
    column for column in data.columns if column.startswith("Naive lag")
]
show_baselines = st.sidebar.checkbox("Include naive baselines in table", value=True)

st.subheader(f"{horizon}-hour-ahead prediction")
if horizon == 24:
    st.info(
        "Each prediction targets one load reading 24 hours after its "
        "forecast origin. This is not a forecast of all 24 intervening hours."
    )

st.write(f"Test timestamps: **{len(data):,}**")
st.write(
    f"Test period: **{data.index.min():%d %b %Y %H:%M}** to "
    f"**{data.index.max():%d %b %Y %H:%M}**"
)

columns = MODELS + (available_baselines if show_baselines else [])
score_table = pd.DataFrame({
    name: metrics(data["Actual"], data[name])
    for name in columns
}).T.sort_values("RMSE (MW)")
st.subheader("Full test-period results")
st.dataframe(score_table.style.format({
    "RMSE (MW)": "{:.3f}",
    "MAE (MW)": "{:.3f}",
    "MAPE (%)": "{:.3f}",
    "R²": "{:.4f}",
}))

st.subheader("Model performance comparison")
chart_scores = score_table.loc[MODELS]

left, right = st.columns(2)
with left:
    st.write("RMSE (MW) — lower is better")
    st.bar_chart(chart_scores["RMSE (MW)"])

with right:
    st.write("R² — higher is better")
    st.bar_chart(chart_scores["R²"])

if horizon == 24:
    st.caption(
        "XGBoost was selected using validation RMSE. "
        "The seed-42 LSTM has a slightly lower RMSE on this test set; "
        "the notebook also reports results across additional LSTM seeds."
    )

first_date = data.index.min().date()
last_date = data.index.max().date()
default_end = min(
    (data.index.min() + pd.Timedelta(days=14)).date(),
    last_date,
)

dates = st.sidebar.date_input(
    "Historical dates to plot",
    value=(first_date, default_end),
    min_value=first_date,
    max_value=last_date,
)

if len(dates) == 2:
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
    visible = data[(data.index >= start) & (data.index < end)]
    st.subheader("Actual versus predicted")
    if selected_models and not visible.empty:
        import matplotlib.pyplot as plt

        hourly_index = pd.date_range(
            visible.index.min(), visible.index.max(), freq="h"
        )
        plot_data = visible.reindex(hourly_index)

        fig, ax = plt.subplots(figsize=(12, 4))
        for name in ["Actual", *selected_models]:
            ax.plot(
                plot_data.index, plot_data[name],
                label=name, linewidth=1.2
            )

        ax.set_ylabel("Load (MW)")
        ax.set_xlabel("Target timestamp")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select at least one model and a date range with test data.")
st.subheader("Horizon comparison on shared test timestamps")
shared = h1.index.intersection(h24.index).sort_values()
rows = []

for name in MODELS:
    for hours, frame in [(1, h1), (24, h24)]:
        rows.append({
            "Model": name,
            "Horizon": f"{hours} hour{'s' if hours != 1 else ''}",
            **metrics(
                frame.loc[shared, "Actual"],
                frame.loc[shared, name],
            ),
        })

shared_scores = pd.DataFrame(rows).set_index(["Model", "Horizon"])
st.write(f"Shared target timestamps: **{len(shared):,}**")
st.dataframe(shared_scores.round(3))
import matplotlib.pyplot as plt

importance = pd.read_csv(DATA_DIR / "feature_importances.csv")

st.subheader(f"Feature importance — {horizon}-hour-ahead models")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model_name in zip(axes, ["Random Forest", "XGBoost"]):
    top = (
        importance[
            (importance["Horizon"] == horizon)
            & (importance["Model"] == model_name)
        ]
        .nlargest(12, "Importance")
        .sort_values("Importance")
    )
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title(model_name)
    ax.set_xlabel("Model-native importance")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "These are rankings within each fitted model. Importance values "
    "should not be compared numerically between Random Forest and XGBoost; "
    "correlated lag features can share importance."
)