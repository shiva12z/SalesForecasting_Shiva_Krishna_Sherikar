import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    return df

@st.cache_data
def get_monthly_sales(df):
    return df.groupby(pd.Grouper(key='Order Date', freq='MS')).agg(
        Sales=('Sales', 'sum'),
        Orders=('Order ID', 'nunique')
    ).reset_index()

@st.cache_data
def get_weekly_sales(df):
    return df.groupby([pd.Grouper(key='Order Date', freq='W')])['Sales'].sum().reset_index()

df = load_data()
monthly_sales = get_monthly_sales(df)
weekly_sales = get_weekly_sales(df)

st.sidebar.title("Sales Forecasting Dashboard")
page = st.sidebar.radio("Navigate to:", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# ============================================================
# PAGE 1: SALES OVERVIEW
# ============================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Unique Customers", f"{df['Customer ID'].nunique():,}")
    col4.metric("Date Range", f"{df['Order Date'].min().year}-{df['Order Date'].max().year}")

    st.subheader("Total Sales by Year")
    yearly = df.groupby('Year')['Sales'].sum().reset_index()
    fig_bar = px.bar(yearly, x='Year', y='Sales', color='Sales', color_continuous_scale='Blues')
    fig_bar.update_layout(xaxis_title='Year', yaxis_title='Sales ($)', showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    fig_line = px.line(monthly_sales, x='Order Date', y='Sales')
    fig_line.update_layout(xaxis_title='Date', yaxis_title='Sales ($)')
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Sales by Region and Category")
    col_a, col_b = st.columns(2)
    with col_a:
        region_filter = st.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
    with col_b:
        cat_filter = st.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(cat_filter)]
    region_cat = filtered.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig_sun = px.sunburst(region_cat, path=['Region', 'Category'], values='Sales', color='Sales', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_sun, use_container_width=True)

# ============================================================
# PAGE 2: FORECAST EXPLORER
# ============================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    forecast_type = st.selectbox("Select Forecast Type:", ["Category", "Region"])
    if forecast_type == "Category":
        segments = df['Category'].unique().tolist()
    else:
        segments = df['Region'].unique().tolist()

    selected_segment = st.selectbox(f"Select {forecast_type}:", segments)
    horizon = st.slider("Forecast Horizon (months):", 1, 3, 3)

    if forecast_type == "Category":
        seg_data = df[df['Category'] == selected_segment]
    else:
        seg_data = df[df['Region'] == selected_segment]

    seg_monthly = seg_data.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().asfreq('MS', fill_value=0)

    prophet_df = pd.DataFrame({'ds': seg_monthly.index, 'y': seg_monthly.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg_monthly.index, y=seg_monthly.values, mode='lines+markers', name='Actual', line=dict(color='#2196F3')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Forecast', line=dict(color='#FF9800', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper CI', line=dict(color='gray', width=0.5)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower CI', line=dict(color='gray', width=0.5), fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))
    fig.update_layout(title=f'{selected_segment} — {horizon}-Month Forecast', xaxis_title='Date', yaxis_title='Sales ($)')
    st.plotly_chart(fig, use_container_width=True)

    # Metrics using backtesting
    if len(seg_monthly) > horizon + 12:
        train_df = prophet_df[:-horizon]
        test_df = prophet_df[-horizon:]
        
        eval_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        eval_model.fit(train_df)
        eval_future = eval_model.make_future_dataframe(periods=horizon, freq='MS')
        eval_forecast = eval_model.predict(eval_future)
        
        eval_pred = eval_forecast.tail(horizon)['yhat'].values
        eval_actual = test_df['y'].values
        
        mae_val = mean_absolute_error(eval_actual, eval_pred)
        rmse_val = np.sqrt(mean_squared_error(eval_actual, eval_pred))
        mape_val = np.mean(np.abs((eval_actual - eval_pred) / np.where(eval_actual == 0, 1, eval_actual))) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (Validation)", f"${mae_val:,.2f}")
        col2.metric("RMSE (Validation)", f"${rmse_val:,.2f}")
        col3.metric("MAPE (Validation)", f"{mape_val:.2f}%")
    else:
        st.warning("Not enough data to perform robust metric validation.")

    st.subheader("Forecasted Values")
    future_vals = forecast.tail(horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_vals.columns = ['Date', 'Forecast', 'Lower CI', 'Upper CI']
    st.dataframe(future_vals.style.format({'Forecast': '${:,.2f}', 'Lower CI': '${:,.2f}', 'Upper CI': '${:,.2f}'}), use_container_width=True)

# ============================================================
# PAGE 3: ANOMALY REPORT
# ============================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly_data = weekly_sales.set_index('Order Date')
    weekly_data['Week_Index'] = range(len(weekly_data))

    # 1. Isolation Forest Anomaly Detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    weekly_data['Anomaly_IF'] = iso_forest.fit_predict(weekly_data[['Sales']])
    weekly_data['Anomaly_IF'] = weekly_data['Anomaly_IF'].map({1: 0, -1: 1})

    # 2. Z-Score Based Anomaly Detection (using 12-week rolling window so standard deviations are mathematically valid)
    window = 12
    weekly_data['Rolling_Mean'] = weekly_data['Sales'].rolling(window=window, center=True).mean()
    weekly_data['Rolling_Std'] = weekly_data['Sales'].rolling(window=window, center=True).std()
    weekly_data['Z_Score'] = (weekly_data['Sales'] - weekly_data['Rolling_Mean']) / weekly_data['Rolling_Std'].replace(0, np.nan).fillna(1)
    weekly_data['Anomaly_ZS'] = (weekly_data['Z_Score'].abs() > 2.0).astype(int)

    method = st.radio("Select Anomaly Detection Method:", 
                      ["Isolation Forest", "Z-Score (12-Week Rolling)", "Compare Both Methods"], 
                      horizontal=True)

    def get_explanation(dt):
        month = dt.month
        if month in [11, 12]:
            return "Festive/holiday sale period (Black Friday, Christmas peaks)"
        elif month in [1, 2]:
            return "Post-holiday demand slump (low consumer spending)"
        elif month in [9, 10]:
            return "Back-to-school or early holiday prep order spikes"
        else:
            return "Unusual pattern — possible corporate bulk order or mid-year promotion"

    if method == "Isolation Forest":
        anomalies = weekly_data[weekly_data['Anomaly_IF'] == 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data['Sales'], mode='lines', name='Weekly Sales', line=dict(color='#2196F3')))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Sales'], mode='markers', name='Anomaly (IF)',
                                 marker=dict(color='#F44336', size=10, symbol='x')))
        fig.update_layout(title='Anomaly Detection — Isolation Forest (Contamination = 5%)', xaxis_title='Date', yaxis_title='Weekly Sales ($)')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detected Anomalies (Isolation Forest)")
        if len(anomalies) > 0:
            anomaly_table = anomalies.reset_index()[['Order Date', 'Sales']]
            anomaly_table['Date'] = anomaly_table['Order Date'].dt.strftime('%Y-%m-%d')
            anomaly_table['Explanation'] = anomaly_table['Order Date'].apply(get_explanation)
            st.dataframe(anomaly_table[['Date', 'Sales', 'Explanation']], use_container_width=True)
        else:
            st.write("No anomalies detected by Isolation Forest.")

    elif method == "Z-Score (12-Week Rolling)":
        anomalies = weekly_data[weekly_data['Anomaly_ZS'] == 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data['Sales'], mode='lines', name='Weekly Sales', line=dict(color='#2196F3')))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Sales'], mode='markers', name='Anomaly (Z-Score)',
                                 marker=dict(color='#FF9800', size=10, symbol='diamond')))
        fig.update_layout(title='Anomaly Detection — Z-Score (Rolling std dev > 2.0)', xaxis_title='Date', yaxis_title='Weekly Sales ($)')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detected Anomalies (Z-Score)")
        if len(anomalies) > 0:
            anomaly_table = anomalies.reset_index()[['Order Date', 'Sales', 'Z_Score']]
            anomaly_table['Date'] = anomaly_table['Order Date'].dt.strftime('%Y-%m-%d')
            anomaly_table['Z-Score'] = anomaly_table['Z_Score'].round(2)
            anomaly_table['Explanation'] = anomaly_table['Order Date'].apply(get_explanation)
            st.dataframe(anomaly_table[['Date', 'Sales', 'Z-Score', 'Explanation']], use_container_width=True)
        else:
            st.write("No anomalies detected by Z-Score.")

    else: # Compare Both Methods
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data['Sales'], mode='lines', name='Weekly Sales', line=dict(color='#2196F3'), opacity=0.5))
        
        if_anom = weekly_data[weekly_data['Anomaly_IF'] == 1]
        zs_anom = weekly_data[weekly_data['Anomaly_ZS'] == 1]
        both_anom = weekly_data[(weekly_data['Anomaly_IF'] == 1) & (weekly_data['Anomaly_ZS'] == 1)]
        
        if_only_anom = weekly_data[(weekly_data['Anomaly_IF'] == 1) & (weekly_data['Anomaly_ZS'] == 0)]
        zs_only_anom = weekly_data[(weekly_data['Anomaly_IF'] == 0) & (weekly_data['Anomaly_ZS'] == 1)]

        fig.add_trace(go.Scatter(x=if_only_anom.index, y=if_only_anom['Sales'], mode='markers', name='Isolation Forest Only',
                                 marker=dict(color='#F44336', size=8, symbol='x')))
        fig.add_trace(go.Scatter(x=zs_only_anom.index, y=zs_only_anom['Sales'], mode='markers', name='Z-Score Only',
                                 marker=dict(color='#FF9800', size=8, symbol='diamond')))
        fig.add_trace(go.Scatter(x=both_anom.index, y=both_anom['Sales'], mode='markers', name='Flagged by Both',
                                 marker=dict(color='#4CAF50', size=12, symbol='circle-open', line=dict(width=3))))
        
        fig.update_layout(title='Comparison of Anomaly Detection Methods', xaxis_title='Date', yaxis_title='Weekly Sales ($)')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Comparison Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Isolation Forest Total", f"{len(if_anom)}")
        col2.metric("Z-Score Total", f"{len(zs_anom)}")
        col3.metric("Overlap (Both Methods)", f"{len(both_anom)}")

        st.markdown("""
        ### Method Comparison & Insights:
        * **Isolation Forest (ML-based)** isolates points by randomly partitioning features. It finds multi-dimensional global anomalies and is robust to various distributions.
        * **Z-Score (Statistical)** calculates distance from a rolling local mean. It is sensitive to sudden changes relative to its local neighborhood (last 12 weeks).
        * **Overlap**: Both methods successfully agree on extreme outliers (e.g. Q4 holiday sales spikes).
        """)

# ============================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    subcat = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().reset_index()

    subcat_features = []
    for name, group in subcat.groupby('Sub-Category'):
        group = group.sort_values('Order Date')
        monthly_s = group.set_index('Order Date')['Sales']
        if len(monthly_s) < 12:
            continue
        total_sales = monthly_s.sum()
        avg_order_value = monthly_s.mean()
        yearly = monthly_s.resample('YS').sum()
        growth_rate = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100) if len(yearly) > 1 else 0
        volatility = monthly_s.std()
        subcat_features.append({'Sub-Category': name, 'Total_Sales': total_sales, 'Growth_Rate': growth_rate,
                                'Volatility': volatility, 'Avg_Order_Value': avg_order_value})

    features_df = pd.DataFrame(subcat_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[['Total_Sales', 'Growth_Rate', 'Volatility', 'Avg_Order_Value']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features_df['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    features_df['PC1'] = X_pca[:, 0]
    features_df['PC2'] = X_pca[:, 1]

    cluster_profiles = features_df.groupby('Cluster')[['Total_Sales', 'Growth_Rate', 'Volatility']].mean()
    
    # Robust cluster labeling using ranking to prevent duplicates and ensure clean business categories
    cluster_labels = {}
    available_clusters = list(range(3))
    
    high_vol_c = cluster_profiles['Total_Sales'].idxmax()
    cluster_labels[high_vol_c] = 'High Volume, Stable'
    available_clusters.remove(high_vol_c)
    
    remaining_growth = cluster_profiles.loc[available_clusters, 'Growth_Rate']
    growing_c = remaining_growth.idxmax()
    cluster_labels[growing_c] = 'Growing Demand'
    available_clusters.remove(growing_c)
    
    for c in available_clusters:
        cluster_labels[c] = 'Low Volume, Volatile'
            
    features_df['Cluster_Label'] = features_df['Cluster'].map(cluster_labels)

    fig = px.scatter(features_df, x='PC1', y='PC2', color='Cluster_Label', hover_data=['Sub-Category', 'Total_Sales', 'Growth_Rate'],
                     title='Product Demand Segments (PCA)', labels={'Cluster_Label': 'Segment'})
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category Clusters")
    display_df = features_df[['Sub-Category', 'Cluster_Label', 'Total_Sales', 'Growth_Rate', 'Volatility']].sort_values('Cluster_Label')
    display_df.columns = ['Sub-Category', 'Segment', 'Total Sales', 'Growth Rate (%)', 'Volatility']
    st.dataframe(display_df.style.format({'Total Sales': '${:,.0f}', 'Growth Rate (%)': '{:.1f}', 'Volatility': '${:,.0f}'}), use_container_width=True)

    st.subheader("Stocking Strategy per Segment")
    strategies = {
        'High Volume, Stable': 'Maintain high safety stock, use automated reorder. These are bread-and-butter products.',
        'Low Volume, Volatile': 'Use just-in-time ordering, monitor demand signals closely. Consider bundle promotions.',
        'Growing Demand': 'Increase inventory gradually, negotiate better supplier terms. Invest in marketing.',
        'Mature/Stable': 'Maintain current levels, focus on cost optimization. Monitor for signs of decline.'
    }
    for label, strategy in strategies.items():
        count = len(features_df[features_df['Cluster_Label'] == label])
        st.markdown(f"**{label}** ({count} sub-categories): {strategy}")
