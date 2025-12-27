# Streamlit TCS Stock Analysis Dashboard
# Save this file as `streamlit_tcs_dashboard.py`
# Run with: streamlit run streamlit_tcs_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime

st.set_page_config(page_title="TCS Stock Dashboard", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------

def load_csv(uploaded_file=None, default_path=None):
    """Load CSV from uploaded file or default path."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(default_path)
    return df


def preprocess(df):
    """Standard preprocessing: parse dates, sort, ensure numeric columns."""
    df = df.copy()
    # Try to find a date column
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        dcol = date_cols[0]
    else:
        # fallback to first column
        dcol = df.columns[0]

    df.rename(columns={dcol: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # normalize common column names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'open' in lc:
            col_map[c] = 'Open'
        if 'high' in lc:
            col_map[c] = 'High'
        if 'low' in lc:
            col_map[c] = 'Low'
        if 'close' in lc or 'adj close' in lc or 'close_price' in lc:
            col_map[c] = 'Close'
        if 'volume' in lc:
            col_map[c] = 'Volume'

    df.rename(columns=col_map, inplace=True)

    # Ensure numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing Date or Close
    df = df.dropna(subset=['Date'])
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close'])

    df = df.reset_index(drop=True)
    return df


def compute_moving_averages(df, windows=[20, 50]):
    df = df.copy()
    for w in windows:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
    return df

def compute_basic_stats(df):
    stats = {}

    # convert date objects into strings so Streamlit metric() accepts them
    stats['Start Date'] = df['Date'].min().strftime('%Y-%m-%d')
    stats['End Date'] = df['Date'].max().strftime('%Y-%m-%d')

    stats['Records'] = int(len(df))
    stats['Max Close'] = float(df['Close'].max())
    stats['Min Close'] = float(df['Close'].min())
    stats['Mean Close'] = float(df['Close'].mean())
    stats['Latest Close'] = float(df['Close'].iloc[-1])
    stats['Latest Date'] = df['Date'].iloc[-1].strftime('%Y-%m-%d')

    return stats



def generate_insights(df):
    insights = []
    # recent trend
    if len(df) >= 2:
        last = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = (last - prev) / prev * 100
        insights.append(f"Latest day change: {change:.2f}% ({prev:.2f} -> {last:.2f})")

    # longest upward streak (simple)
    diffs = np.sign(df['Close'].diff().fillna(0))
    streak = 0
    max_up = 0
    for d in diffs:
        if d > 0:
            streak += 1
        else:
            max_up = max(max_up, streak)
            streak = 0
    max_up = max(max_up, streak)
    insights.append(f"Longest recent upward streak: {int(max_up)} days")

    # volatility (std of daily returns)
    df['Return'] = df['Close'].pct_change()
    vol = df['Return'].std() * np.sqrt(252)  # annualized
    insights.append(f"Estimated annualized volatility: {vol:.2%}")

    # best month
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Close'].last().pct_change()
    if not monthly.dropna().empty:
        best = monthly.idxmax().strftime('%Y-%m')
        insights.append(f"Best performing month (by close-to-close): {best}")

    return insights


def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------------------------
# App layout
# ---------------------------

st.title("📈 TCS Stock Analysis — Streamlit Dashboard")
st.markdown("Upload your TCS stock CSV or use the provided dataset. The dashboard shows summary cards, interactive charts, and simple automated insights.")

# Sidebar controls
st.sidebar.header("Data & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample CSV from project (if available)", value=True)

default_csv_path = r"C:\Users\jlaks\OneDrive\Desktop\SEM 5\INDUSTRIAL TARINGE PROJECT\TCS_stock_history.csv"


if uploaded_file is None and not use_sample:
    st.sidebar.info("Upload a CSV or enable sample CSV to proceed.")

# Load data
try:
    df_raw = load_csv(uploaded_file if uploaded_file is not None else None,
                      default_path=(default_csv_path if use_sample else None))
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.warning("No data available. Please upload a CSV or enable sample CSV in the sidebar.")
    st.stop()

# Preprocess
df = preprocess(df_raw)

# Date filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(start_date, datetime):
    start_date = start_date.date()
if isinstance(end_date, datetime):
    end_date = end_date.date()

mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
df = df.loc[mask].reset_index(drop=True)

# Moving averages
st.sidebar.subheader("Moving Averages")
ma_windows = st.sidebar.multiselect("Select MA windows", options=[5,10,20,50,100,200], default=[20,50])

# Chart options
show_candlestick = st.sidebar.checkbox("Show candlestick chart", value=True)
show_volume = st.sidebar.checkbox("Show volume panel", value=True)

# Compute MAs
if 'Close' not in df.columns:
    st.error("Could not find 'Close' column after preprocessing. Please check your CSV headers.")
    st.stop()

if ma_windows:
    df = compute_moving_averages(df, windows=ma_windows)

# Compute stats & insights
stats = compute_basic_stats(df)
insights = generate_insights(df)

# Top row: summary cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Start Date", stats['Start Date'])
col2.metric("End Date", stats['End Date'])
col3.metric("Records", stats['Records'])
col4.metric("Latest Close", f"{stats['Latest Close']:.2f}")

# Main charts
st.markdown("---")
chart_col1, chart_col2 = st.columns((3,1))

with chart_col1:
    st.subheader("Price Chart")
    if show_candlestick and all(c in df.columns for c in ['Open','High','Low','Close']):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
        # add moving averages
        for w in ma_windows:
            colname = f'MA_{w}'
            if colname in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[colname], mode='lines', name=f'MA {w}'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback to line chart of Close
        fig = px.line(df, x='Date', y='Close', title='Close Price')
    for w in ma_windows:
     colname: str = f'MA_{w}'
     if colname in df.columns:
        fig.add_scatter(x=df['Date'], y=df[colname], mode='lines', name=f'MA {w}')

        st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Key Stats & Insights")
    st.write(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
    st.markdown("**Automated insights:**")
    for it in insights:
        st.write("- ", it)

# Volume chart
if show_volume and 'Volume' in df.columns:
    st.subheader("Volume")
    fig_vol = px.bar(df, x='Date', y='Volume', title='Volume')
    st.plotly_chart(fig_vol, use_container_width=True)

# Data table and download
st.markdown("---")
with st.expander("View data table"):
    st.dataframe(df)

st.subheader("Export")
col_a, col_b = st.columns(2)
with col_a:
    csv_bytes = to_csv_bytes(df)
    st.download_button("Download filtered CSV", data=csv_bytes, file_name='tcs_filtered.csv', mime='text/csv')

with col_b:
    # Small summary text file
    summary_text = f"TCS Stock Analysis Summary\nStart: {stats['Start Date']}\nEnd: {stats['End Date']}\nRecords: {stats['Records']}\nLatest Close: {stats['Latest Close']:.2f}\n\nInsights:\n" + '\n'.join(insights)
    st.download_button("Download summary.txt", data=summary_text, file_name='tcs_summary.txt')

st.markdown("---")
st.caption("Built with Streamlit — modify the code to add more analysis (e.g., forecasting, technical indicators, or pptx export).")

# EOF
