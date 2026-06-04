import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin: 0;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('superstore_updated.csv', encoding='latin-1')
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Quarter'] = df['Order_Date'].dt.quarter
    return df

df = load_data()

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=80)
st.sidebar.title("Filters")

year_filter = st.sidebar.multiselect(
    "Select Year",
    options=sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique())
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

filtered_df = df[
    (df['Year'].isin(year_filter)) &
    (df['Region'].isin(region_filter)) &
    (df['Category'].isin(category_filter))
]

# ─── Header ────────────────────────────────────────────────
st.title("🛒 Superstore Sales Dashboard")
st.markdown("*Interactive analysis of sales, profit, and forecasting*")
st.markdown("---")

# ─── KPI Cards ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = filtered_df['Order_ID'].nunique()
avg_discount = filtered_df['Discount'].mean() * 100

with k1:
    st.metric("💰 Total Sales", f"${total_sales:,.0f}")
with k2:
    st.metric("📈 Total Profit", f"${total_profit:,.0f}")
with k3:
    st.metric("📦 Total Orders", f"{total_orders:,}")
with k4:
    st.metric("🏷️ Avg Discount", f"{avg_discount:.1f}%")

st.markdown("---")

# ─── Sales Trend ───────────────────────────────────────────
st.subheader("📊 Monthly Sales Trend")

monthly_sales = filtered_df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
monthly_sales = monthly_sales.sort_values('Date')

fig_trend = px.line(
    monthly_sales,
    x='Date', y='Sales',
    title='Monthly Sales Over Time',
    color_discrete_sequence=['#667eea']
)
fig_trend.update_traces(line_width=2.5, mode='lines+markers')
fig_trend.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    hovermode='x unified',
    xaxis_title="Month",
    yaxis_title="Sales ($)"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ─── Category & Region ─────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("🗂️ Sales by Category")
    cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    fig_cat = px.pie(
        cat_sales, values='Sales', names='Category',
        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
    )
    fig_cat.update_traces(textposition='inside', textinfo='percent+label')
    fig_cat.update_layout(showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)

with c2:
    st.subheader("🗺️ Sales by Region")
    reg_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
    fig_reg = px.bar(
        reg_sales.sort_values('Sales', ascending=True),
        x='Sales', y='Region',
        orientation='h',
        color='Sales',
        color_continuous_scale='Purples'
    )
    fig_reg.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ─── Sub-Category Profit ───────────────────────────────────
st.markdown("---")
st.subheader("💹 Profit by Sub-Category")

subcat = filtered_df.groupby('Sub-Category')['Profit'].sum().reset_index()
subcat = subcat.sort_values('Profit', ascending=True)
subcat['Color'] = subcat['Profit'].apply(lambda x: '🔴 Loss' if x < 0 else '🟢 Profit')

fig_sub = px.bar(
    subcat, x='Profit', y='Sub-Category',
    orientation='h',
    color='Color',
    color_discrete_map={'🔴 Loss': '#ff6b6b', '🟢 Profit': '#667eea'}
)
fig_sub.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    legend_title="Status"
)
st.plotly_chart(fig_sub, use_container_width=True)

# ─── Prophet Forecast ──────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Sales Forecast — Next 6 Months")

@st.cache_data
def run_forecast(data):
    monthly = data.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    monthly['ds'] = pd.to_datetime(monthly[['Year', 'Month']].assign(day=1))
    monthly['y'] = monthly['Sales']
    monthly = monthly[['ds', 'y']].sort_values('ds')

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(monthly)
    future = model.make_future_dataframe(periods=6, freq='MS')
    forecast = model.predict(future)
    return monthly, forecast

monthly_data, forecast = run_forecast(df)

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=monthly_data['ds'], y=monthly_data['y'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='#667eea', width=2.5)
))

fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='#f093fb', width=2.5, dash='dash')
))

fig_forecast.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(240,147,251,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

fig_forecast.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    hovermode='x unified',
    xaxis_title="Month",
    yaxis_title="Sales ($)",
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast table
st.markdown("**📅 Forecasted Sales — Next 6 Months**")
future_only = forecast[forecast['ds'] > monthly_data['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
future_only.columns = ['Month', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
future_only['Month'] = future_only['Month'].dt.strftime('%B %Y')
future_only[['Predicted Sales', 'Lower Bound', 'Upper Bound']] = future_only[['Predicted Sales', 'Lower Bound', 'Upper Bound']].applymap(lambda x: f"${x:,.0f}")
st.dataframe(future_only, use_container_width=True, hide_index=True)

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem'>Built with using Python · Streamlit · Prophet · Plotly</p>",
    unsafe_allow_html=True
)