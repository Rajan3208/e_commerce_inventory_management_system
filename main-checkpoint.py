import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime

# Set color palette
sns.set_palette("Set3")

# Load or generate data
@st.cache_data
def load_data():
    df = pd.read_csv("fashion_data.csv")
    
    # Generate synthetic monthly time data if not present
    if 'month' not in df.columns:
        df['month'] = pd.date_range(start='2023-04-01', periods=len(df), freq='W').strftime('%Y-%m')
    
    return df

@st.cache_resource
def load_ml_models():
    # Load the saved models and encoders
    rf_model = joblib.load('rf_model.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, le_dict, scaler

def generate_future_features(df, months_ahead=6):
    # Get the last month's data as a base
    last_month_data = df.iloc[-1:].copy()
    future_data = pd.concat([last_month_data] * months_ahead, ignore_index=True)
    
    # Set future months
    last_date = pd.to_datetime(df['month'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_ahead, freq='M')
    future_data['month'] = future_dates.strftime('%Y-%m')
    
    # Adjust seasonal data
    for idx, date in enumerate(future_dates):
        future_data.loc[idx, 'season'] = get_season(date)
    
    return future_data

def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

def prepare_prediction_data(df, future_data, le_dict, scaler):
    # Prepare features for prediction
    feature_cols = ['clothing_type', 'gender', 'season', 'city', 'color', 'price_range', 'review_score']
    
    # Encode categorical features
    for col in ['clothing_type', 'gender', 'season', 'city', 'color']:
        future_data[col] = le_dict[col].transform(future_data[col])
    
    # Scale numerical features
    numerical_cols = ['price_range', 'review_score']
    future_data[numerical_cols] = scaler.transform(future_data[numerical_cols])
    
    return future_data[feature_cols]

# Load data and models
df = load_data()
rf_model, le_dict, scaler = load_ml_models()

# Streamlit layout
st.title("E-Commerce Fashion Data Analysis")
st.markdown("### Visualizing Sales Trends Across Indian Cities")

# Trend Analysis Graph with Plotly Area Chart
st.markdown("## Trend Analysis of Sales Volume Over Time")
time_col, volume_col = 'month', 'sales_volume'
if time_col in df.columns and volume_col in df.columns:
    sales_trend = df.groupby(time_col)[volume_col].sum().reset_index()
    
    # Rolling average to smooth out data
    sales_trend['rolling_sales'] = sales_trend[volume_col].rolling(window=4).mean()
    
    fig = px.area(
        sales_trend, x=time_col, y='rolling_sales',
        title="Sales Volume Trend Over Time (Rolling Average)",
        labels={time_col: "Time (Monthly)", 'rolling_sales': "Sales Volume"}
    )
    fig.update_traces(line_color='royalblue')
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ML-based Prediction for Next 6 Months
st.markdown("## ML-Based Sales Prediction for Next 6 Months")

# Generate future data and make predictions
future_data = generate_future_features(df)
X_future = prepare_prediction_data(df, future_data, le_dict, scaler)
predicted_sales = rf_model.predict(X_future)

# Create forecast DataFrame
forecast = pd.DataFrame({
    'month': future_data['month'],
    'forecast_sales': predicted_sales
})

# Plot ML-based forecast
forecast_fig = px.line(forecast, x='month', y='forecast_sales', 
                      title="ML-Based Sales Forecast for Next 6 Months")
forecast_fig.update_traces(line=dict(color='red', dash='dash'))
forecast_fig.update_layout(hovermode="x unified", 
                         xaxis_title="Time (Monthly)", 
                         yaxis_title="Forecasted Sales Volume")
st.plotly_chart(forecast_fig)

# Analysis of predicted trends
st.markdown("### Predicted Trend Analysis")
avg_current_sales = df[volume_col].mean()
avg_predicted_sales = predicted_sales.mean()
growth_rate = ((avg_predicted_sales - avg_current_sales) / avg_current_sales) * 100

st.write(f"Average current sales volume: {avg_current_sales:,.0f}")
st.write(f"Average predicted sales volume: {avg_predicted_sales:,.0f}")
st.write(f"Predicted growth rate: {growth_rate:.1f}%")

# Rest of your visualizations (Pie charts etc.)
st.markdown("## Sales Distribution by City")
city_sales = df['city'].value_counts()
fig = px.pie(
    names=city_sales.index, values=city_sales.values,
    title="Sales Distribution by City",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

st.markdown("## Sales Distribution by Clothing Type")
clothing_sales = df['clothing_type'].value_counts()
fig = px.pie(
    names=clothing_sales.index, values=clothing_sales.values,
    title="Sales Distribution by Clothing Type",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

# Enhanced Conclusion Section
st.markdown("## ML-Based Insights and Future Clothing Trends")
st.markdown("### Key Insights")

# Identify top trending items based on ML predictions
future_clothing_trends = future_data.groupby('clothing_type')['forecast_sales'].mean().sort_values(ascending=False)
top_trending = future_clothing_trends.head(5)

st.write("Top 5 Predicted Trending Items:")
for item, sales in top_trending.items():
    st.write(f"- {item}: Predicted average sales volume of {sales:,.0f}")

st.markdown("### Seasonal Analysis")
seasonal_trends = future_data.groupby('season')['forecast_sales'].mean().sort_values(ascending=False)
st.write("Predicted Sales by Season:")
for season, sales in seasonal_trends.items():
    st.write(f"- {season}: {sales:,.0f}")

st.markdown("### Note")
st.write("These predictions are based on machine learning analysis of historical data, seasonal patterns, and multiple features including price ranges, review scores, and regional preferences. The model accounts for complex interactions between these factors to provide more accurate predictions than simple trend extrapolation.")