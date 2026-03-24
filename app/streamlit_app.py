import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Celes Supply Logistics Grocery Stores | Retail Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

ARTIFACTS_DIR = "artifacts"
HIST_PATH = f"{ARTIFACTS_DIR}/historical.parquet"
FCST_PATH = f"{ARTIFACTS_DIR}/forecast_next_365_days.parquet"
METRICS_PATH = f"{ARTIFACTS_DIR}/metrics.json"
IMPORTANCE_PATH = f"{ARTIFACTS_DIR}/feature_importance.csv"

STORE_NAME = "Celes Grocery Store"


@st.cache_data
def load_data():
    historical = pd.read_parquet(HIST_PATH)
    forecast = pd.read_parquet(FCST_PATH)

    historical["Date"] = pd.to_datetime(historical["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    try:
        importance = pd.read_csv(IMPORTANCE_PATH)
    except FileNotFoundError:
        importance = pd.DataFrame(columns=["feature", "importance"])

    if "Revenue" not in historical.columns:
        historical["Revenue"] = historical["Quantity"] * historical["Price"]

    return historical, forecast, metrics, importance


def apply_filters(df, start_date, end_date, store_ids, product_ids, product_names):
    filtered = df.copy()

    filtered = filtered[
        (filtered["Date"] >= pd.to_datetime(start_date)) &
        (filtered["Date"] <= pd.to_datetime(end_date))
    ]

    if store_ids:
        filtered = filtered[filtered["StoreID"].astype(str).isin(store_ids)]

    if product_ids:
        filtered = filtered[filtered["ProductID"].astype(str).isin(product_ids)]

    if product_names and "ProductName" in filtered.columns:
        filtered = filtered[filtered["ProductName"].astype(str).isin(product_names)]

    return filtered


def plot_history_forecast(hist_df, fcst_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 5))

    if not hist_df.empty:
        hist_daily = hist_df.groupby("Date", as_index=False)["Quantity"].sum()
        ax.plot(hist_daily["Date"], hist_daily["Quantity"], linewidth=2, label="Historical Quantity")

    if not fcst_df.empty:
        fcst_daily = fcst_df.groupby("Date", as_index=False)["ForecastQuantity"].sum()
        ax.plot(fcst_daily["Date"], fcst_daily["ForecastQuantity"], linewidth=2, label="Forecast Quantity")

    chart_title = "Historical vs Forecast Quantity"
    if title_suffix:
        chart_title = f"{chart_title} — {title_suffix}"

    ax.set_title(chart_title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()

    return fig


def plot_revenue(hist_df, fcst_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 5))

    if not hist_df.empty:
        hist_daily = hist_df.groupby("Date", as_index=False)["Revenue"].sum()
        ax.plot(hist_daily["Date"], hist_daily["Revenue"], linewidth=2, label="Historical Revenue")

    if not fcst_df.empty:
        fcst_daily = fcst_df.groupby("Date", as_index=False)["ForecastRevenue"].sum()
        ax.plot(fcst_daily["Date"], fcst_daily["ForecastRevenue"], linewidth=2, label="Forecast Revenue")

    chart_title = "Historical vs Forecast Revenue"
    if title_suffix:
        chart_title = f"{chart_title} — {title_suffix}"

    ax.set_title(chart_title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()

    return fig


def monthly_forecast_summary(fcst_df):
    if fcst_df.empty:
        return pd.DataFrame(columns=["Month", "ForecastQuantity", "ForecastRevenue"])

    tmp = fcst_df.copy()
    tmp["Month"] = tmp["Date"].dt.to_period("M").astype(str)

    return (
        tmp.groupby("Month", as_index=False)[["ForecastQuantity", "ForecastRevenue"]]
        .sum()
        .sort_values("Month")
    )


def top_products_table(fcst_df, n=10):
    if fcst_df.empty:
        return pd.DataFrame(columns=["StoreID", "ProductID", "ProductName", "ForecastQuantity", "ForecastRevenue"])

    group_cols = ["StoreID", "ProductID"]
    if "ProductName" in fcst_df.columns:
        group_cols.append("ProductName")

    return (
        fcst_df.groupby(group_cols, as_index=False)[["ForecastQuantity", "ForecastRevenue"]]
        .sum()
        .sort_values("ForecastRevenue", ascending=False)
        .head(n)
    )


historical, forecast, metrics, importance = load_data()

st.markdown(
    f"""
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0;">📈 {STORE_NAME}</h1>
        <p style="margin-top: 0.25rem; color: #666;">Retail Demand Forecast Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Filters")

all_store_ids = sorted(historical["StoreID"].astype(str).dropna().unique().tolist())
all_product_ids = sorted(historical["ProductID"].astype(str).dropna().unique().tolist())
all_product_names = sorted(historical["ProductName"].astype(str).dropna().unique().tolist())

min_hist_date = historical["Date"].min().date()
max_fcst_date = forecast["Date"].max().date()
default_hist_start = max(min_hist_date, historical["Date"].max().date() - pd.Timedelta(days=120))
default_fcst_end = min(max_fcst_date, historical["Date"].max().date() + pd.Timedelta(days=90))

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_hist_start, default_fcst_end),
    min_value=min_hist_date,
    max_value=max_fcst_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_hist_date
    end_date = max_fcst_date

selected_store_ids = st.sidebar.multiselect("StoreID", all_store_ids)
selected_product_ids = st.sidebar.multiselect("ProductID", all_product_ids)

product_name_search = st.sidebar.text_input("Search ProductName")

if product_name_search:
    matching_names = [x for x in all_product_names if product_name_search.lower() in x.lower()]
else:
    matching_names = all_product_names

selected_product_names = st.sidebar.multiselect("ProductName", matching_names)

hist_filtered = apply_filters(
    historical,
    start_date=start_date,
    end_date=end_date,
    store_ids=selected_store_ids,
    product_ids=selected_product_ids,
    product_names=selected_product_names
)

fcst_filtered = apply_filters(
    forecast,
    start_date=start_date,
    end_date=end_date,
    store_ids=selected_store_ids,
    product_ids=selected_product_ids,
    product_names=selected_product_names
)

selection_label = "All Data"
if len(selected_store_ids) == 1:
    selection_label = f"Store {selected_store_ids[0]}"
elif len(selected_product_ids) == 1:
    selection_label = f"Product {selected_product_ids[0]}"
elif len(selected_product_names) == 1:
    selection_label = selected_product_names[0]

st.markdown(f"### Current View: {selection_label}")

hist_qty = hist_filtered["Quantity"].sum() if not hist_filtered.empty else 0
hist_rev = hist_filtered["Revenue"].sum() if not hist_filtered.empty else 0
fcst_qty = fcst_filtered["ForecastQuantity"].sum() if not fcst_filtered.empty else 0
fcst_rev = fcst_filtered["ForecastRevenue"].sum() if not fcst_filtered.empty else 0

hist_avg_daily_qty = hist_filtered.groupby("Date")["Quantity"].sum().mean() if not hist_filtered.empty else 0
fcst_avg_daily_qty = fcst_filtered.groupby("Date")["ForecastQuantity"].sum().mean() if not fcst_filtered.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Historical Quantity", f"{hist_qty:,.0f}")
k2.metric("Forecast Quantity", f"{fcst_qty:,.0f}")
k3.metric("Historical Revenue", f"${hist_rev:,.2f}")
k4.metric("Forecast Revenue", f"${fcst_rev:,.2f}")

k5, k6, k7 = st.columns(3)
k5.metric("Avg Daily Historical Qty", f"{hist_avg_daily_qty:,.1f}")
k6.metric("Avg Daily Forecast Qty", f"{fcst_avg_daily_qty:,.1f}")
k7.metric("WAPE", f"{metrics['WAPE']:.2%}")

st.markdown("---")

left_top, right_top = st.columns([2, 1])

with left_top:
    st.subheader("Quantity Trend")
    st.pyplot(plot_history_forecast(hist_filtered, fcst_filtered, selection_label))

with right_top:
    st.subheader("Model Metrics")
    metric_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "WAPE", "Cutoff Date"],
            "Value": [
                f"{metrics['MAE']:.2f}",
                f"{metrics['RMSE']:.2f}",
                f"{metrics['WAPE']:.2%}",
                metrics.get("cutoff_date", "N/A")
            ]
        }
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

st.subheader("Revenue Trend")
st.pyplot(plot_revenue(hist_filtered, fcst_filtered, selection_label))

mid_left, mid_right = st.columns(2)

with mid_left:
    st.subheader("Monthly Forecast Summary")
    st.dataframe(monthly_forecast_summary(fcst_filtered), use_container_width=True, height=320)

with mid_right:
    st.subheader("Top Forecasted Products")
    st.dataframe(top_products_table(fcst_filtered, n=10), use_container_width=True, height=320)

bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("Historical Data")
    st.dataframe(
        hist_filtered.sort_values("Date", ascending=False),
        use_container_width=True,
        height=420
    )

with bottom_right:
    st.subheader("Forecast Data")
    st.dataframe(
        fcst_filtered.sort_values("Date", ascending=False),
        use_container_width=True,
        height=420
    )

st.subheader("Feature Importance")
if not importance.empty:
    st.dataframe(
        importance.sort_values("importance", ascending=False),
        use_container_width=True,
        height=350
    )
else:
    st.info("No feature importance file found.")