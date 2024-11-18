import streamlit as st
st.set_page_config(layout="wide", page_title="Inventory Management Dashboard")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import joblib

# Custom CSS
st.markdown("""
    <style>
    .stTitle {
        font-size: 42px !important;
        text-align: center;
        padding-bottom: 20px;
    }
    .product-chart {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved model and preprocessors
@st.cache_resource
def load_ml_components():
    rf_model = joblib.load('rf_model.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, le_dict, scaler

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

def generate_overall_analysis(df):
    """Generate comprehensive overall analysis of the fashion retail data"""
    historical_data = df[df['is_historical']]
    future_data = df[~df['is_historical']]
    
    # Time-based metrics
    total_historical_sales = historical_data['sales_volume'].sum()
    total_future_sales = future_data['sales_volume'].sum()
    sales_growth = ((total_future_sales - total_historical_sales) / total_historical_sales) * 100
    
    # Product performance
    product_performance = df.groupby('clothing_type').agg({
        'sales_volume': ['sum', 'mean'],
        'review_score': 'mean',
        'price_range': 'mean'
    }).round(2)
    
    # Seasonal patterns
    seasonal_performance = df.groupby('season')['sales_volume'].mean().sort_values(ascending=False)
    
    # Gender-based insights
    gender_performance = df.groupby('gender').agg({
        'sales_volume': 'sum',
        'review_score': 'mean'
    }).round(2)
    
    # City performance
    city_performance = df.groupby('city').agg({
        'sales_volume': 'sum',
        'review_score': 'mean'
    }).sort_values('sales_volume', ascending=False).round(2)
    
    # Color preferences
    color_performance = df.groupby('color')['sales_volume'].sum().sort_values(ascending=False)
    
    # Price analysis
    price_correlation = df['price_range'].corr(df['sales_volume'])
    
    # Key takeaways
    key_takeaways = {
        'best_performing_product': product_performance.index[0],
        'highest_rated_product': product_performance[('review_score', 'mean')].idxmax(),
        'top_market': city_performance.index[0],
        'peak_season': seasonal_performance.index[0],
        'price_sensitivity': 'high' if price_correlation < -0.5 else 'moderate' if price_correlation < 0 else 'low',
        'top_colors': color_performance.head(3).index.tolist()
    }
    
    return {
        'sales_metrics': {
            'total_historical_sales': total_historical_sales,
            'total_future_sales': total_future_sales,
            'projected_growth': sales_growth
        },
        'product_performance': product_performance,
        'seasonal_performance': seasonal_performance,
        'gender_performance': gender_performance,
        'city_performance': city_performance,
        'color_performance': color_performance,
        'price_correlation': price_correlation,
        'key_takeaways': key_takeaways
    }

def generate_product_analysis(df, selected_product):
    """Generate comprehensive analysis for a specific product"""
    product_data = df[df['clothing_type'] == selected_product]
    historical_data = product_data[product_data['is_historical']]
    future_data = product_data[~product_data['is_historical']]
    
    # Sales metrics
    historical_sales = historical_data['sales_volume'].sum()
    future_sales = future_data['sales_volume'].sum()
    sales_growth = ((future_sales - historical_sales) / historical_sales) * 100
    
    # Price metrics
    avg_price = product_data['price_range'].mean()
    price_trend = product_data.groupby('date')['price_range'].mean().rolling(7).mean()
    
    # Review metrics
    avg_review = product_data['review_score'].mean()
    review_trend = product_data.groupby('date')['review_score'].mean().rolling(7).mean()
    
    # Seasonal performance
    seasonal_sales = product_data.groupby('season')['sales_volume'].mean().sort_values(ascending=False)
    
    # Color preferences
    color_sales = product_data.groupby('color')['sales_volume'].sum().sort_values(ascending=False)
    
    # City performance
    city_sales = product_data.groupby('city')['sales_volume'].sum().sort_values(ascending=False)
    
    # Gender distribution
    gender_sales = product_data.groupby('gender')['sales_volume'].sum()
    
    return {
        'sales_metrics': {
            'historical_sales': historical_sales,
            'future_sales': future_sales,
            'growth_rate': sales_growth
        },
        'price_metrics': {
            'average_price': avg_price,
            'price_trend': price_trend
        },
        'review_metrics': {
            'average_review': avg_review,
            'review_trend': review_trend
        },
        'seasonal_sales': seasonal_sales,
        'color_sales': color_sales,
        'city_sales': city_sales,
        'gender_sales': gender_sales
    }

# Generate synthetic data with future predictions
@st.cache_data
def generate_fashion_data(n_samples=365):  # Changed to 365 for 1 year of daily data
    current_date = datetime.now()
    start_date = current_date - timedelta(days=365)  # 1 year historical
    future_date = current_date + timedelta(days=365)  # 1 year forecast
    
    # Historical dates
    historical_dates = pd.date_range(start=start_date, end=current_date, freq='D')
    # Future dates
    future_dates = pd.date_range(start=current_date + timedelta(days=1), end=future_date, freq='D')
    
    def create_data_segment(dates, is_historical):
        n_records = len(dates)
        
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
        colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Yellow', 'Pink', 'Grey']
        clothing_types = ['T-Shirt', 'Hoodie', 'Pants', 'Kurta', 'Dress', 'Jacket']
        genders = ['Male', 'Female']

        np.random.seed(42 if is_historical else 43)
        
        data = {
            'date': dates,
            'clothing_type': np.random.choice(clothing_types, n_records),
            'gender': np.random.choice(genders, n_records),
            'city': np.random.choice(cities, n_records),
            'color': np.random.choice(colors, n_records),
            'is_historical': [is_historical] * n_records
        }
        
        df = pd.DataFrame(data)
        
        
        # Add seasons based on dates
        df['season'] = df['date'].apply(get_season)
        
        # Add price ranges
        price_ranges = {
            'T-Shirt': (15, 50), 'Hoodie': (30, 100), 'Pants': (25, 80),
            'Kurta': (40, 150), 'Dress': (45, 180), 'Jacket': (50, 200)
        }
        
        df['price_range'] = df['clothing_type'].map(
            lambda x: np.random.uniform(price_ranges[x][0], price_ranges[x][1])
        )
        
        # Generate review scores (weighted towards positive)
        df['review_score'] = np.random.beta(8, 2, n_records) * 2 + 3
        
        # Generate sales with patterns
        base_sales = np.random.normal(3000, 200, n_records)
        season_multiplier = {
            'Winter': 1.3, 'Summer': 1.2, 'Spring': 1.0, 'Fall': 1.1
        }
        
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Weekend effect
        weekend_effect = np.where(df['day_of_week'].isin([5, 6]), 1.2, 1.0)
        
        # Review score effect
        review_effect = 1 + 0.2 * (df['review_score'] - 4.0)
        
        # Seasonal effect
        seasonal_effect = df['season'].map(season_multiplier)
        
        df['sales_volume'] = (base_sales * 
                            seasonal_effect * 
                            weekend_effect * 
                            review_effect).astype(int)
        
        return df

    # Generate historical and future data separately
    df_hist = create_data_segment(historical_dates, True)
    df_future = create_data_segment(future_dates, False)
    
    # Combine and sort the data
    df = pd.concat([df_hist, df_future], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

# Main app
st.title("🛍️ Fashion Analytics Dashboard")

try:
    # Load ML components
    rf_model, le_dict, scaler = load_ml_components()
    
    # Generate data
    df = generate_fashion_data()
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Overall Analysis", "Product Analysis"])
    
    with tab1:
        # Overall Analysis Section
        st.markdown("## 📊 Overall Business Analysis")
        
        # Generate overall analysis
        analysis_results = generate_overall_analysis(df)
        
        with st.expander("View Comprehensive Business Analysis", expanded=True):
            # Sales Overview
            st.markdown("### 📈 Sales Overview")
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric(
                    "Historical Sales Volume",
                    f"{analysis_results['sales_metrics']['total_historical_sales']:,.0f}",
                    "Past Year"
                )
            with metrics_cols[1]:
                st.metric(
                    "Projected Sales Volume",
                    f"{analysis_results['sales_metrics']['total_future_sales']:,.0f}",
                    "Next 12 Months"
                )
            with metrics_cols[2]:
                st.metric(
                    "Projected Growth",
                    f"{analysis_results['sales_metrics']['projected_growth']:.1f}%",
                    "vs Previous Period"
                )
            
            # Product Performance
            st.markdown("### 👕 Product Category Analysis")
            product_fig = go.Figure()
            product_fig.add_trace(go.Bar(
                x=analysis_results['product_performance'].index,
                y=analysis_results['product_performance'][('sales_volume', 'sum')],
                name='Total Sales'
            ))
            product_fig.add_trace(go.Scatter(
                x=analysis_results['product_performance'].index,
                y=analysis_results['product_performance'][('review_score', 'mean')],
                name='Avg Review Score',
                yaxis='y2'
            ))
            product_fig.update_layout(
                title='Product Performance Overview',
                yaxis2=dict(
                    title='Average Review Score',
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            st.plotly_chart(product_fig, use_container_width=True)
            
            # Market Insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌍 Geographic Insights")
                city_fig = px.bar(
                    analysis_results['city_performance'].reset_index(),
                    x='city',
                    y='sales_volume',
                    color='review_score',
                    title='City-wise Performance'
                )
                city_fig.update_layout(height=400)
                st.plotly_chart(city_fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Customer Segments")
                gender_fig = px.pie(
                    values=analysis_results['gender_performance']['sales_volume'],
                    names=analysis_results['gender_performance'].index,
                    title='Sales Distribution by Gender'
                )
                gender_fig.update_layout(height=400)
                st.plotly_chart(gender_fig, use_container_width=True)
            
            # Seasonal and Price Analysis
            st.markdown("### 🌤️ Seasonal & Pricing Insights")
            season_price_cols = st.columns(2)
            
            with season_price_cols[0]:
                seasonal_fig = px.bar(
                    x=analysis_results['seasonal_performance'].index,
                    y=analysis_results['seasonal_performance'].values,
                    title='Average Sales by Season'
                )
                seasonal_fig.update_layout(height=350)
                st.plotly_chart(seasonal_fig, use_container_width=True)
            
            with season_price_cols[1]:
                st.markdown(f"""
                #### Price Impact Analysis
                - Price to Sales Correlation: {analysis_results['price_correlation']:.2f}
                - Top Performing Price Ranges:
                    - Budget: ₹200-350
                    - Mid-range: ₹350-600
                    - Premium: ₹800+
                """)
            
            # Color Preferences
            st.markdown("#### Top Selling Colors")
            top_colors = analysis_results['color_performance'].head(5)
            color_fig = px.pie(
                values=top_colors,
                names=top_colors.index,
                title='Top 5 Color Preferences'
            )
            color_fig.update_layout(height=300)
            st.plotly_chart(color_fig, use_container_width=True)
            
            # Key Takeaways Section
            st.markdown("### 🎯 Key Business Takeaways")
            takeaways = analysis_results['key_takeaways']
            st.markdown(f"""
            1. **Sales Trajectory**
               - Projected growth of {analysis_results['sales_metrics']['projected_growth']:.1f}% indicates{' positive' if analysis_results['sales_metrics']['projected_growth'] > 0 else ' challenging'} market conditions.
               - Historical performance shows steady growth with seasonal variations.

            2. **Product Strategy**
               - Best performing category: {takeaways['best_performing_product']}.
               - Highest rated category: {takeaways['highest_rated_product']}.
               - Focus on expanding successful product lines while optimizing underperforming categories.

            3. **Market Opportunities**
               - Top market: {takeaways['top_market']}.
               - Peak season: {takeaways['peak_season']}.
               - Price sensitivity: {takeaways['price_sensitivity']}.

            4. **Color Preferences**
               - Top selling colors: {', '.join(takeaways['top_colors'])}.
            """)

            

            # Create the analysis report text
            overall_analysis_text = f"""# Fashion Retail Analysis Report

## Overall Performance Summary
- Total Historical Sales: {analysis_results['sales_metrics']['total_historical_sales']:,.0f}
- Projected Growth: {analysis_results['sales_metrics']['projected_growth']:.1f}%
- Best Performing Product: {takeaways['best_performing_product']}
- Top Market: {takeaways['top_market']}
- Peak Season: {takeaways['peak_season']}

## Product Performance
- Highest Rated Category: {takeaways['highest_rated_product']}
- Price Sensitivity: {takeaways['price_sensitivity']}
- Top Colors: {', '.join(takeaways['top_colors'])}

## Market Analysis
- City Performance: {', '.join(analysis_results['city_performance'].index[:3])} (top 3 cities)
- Seasonal Trends: {', '.join(analysis_results['seasonal_performance'].index)}
- Gender Distribution: {', '.join(f"{index}: {value['sales_volume']:,.0f}" for index, value in analysis_results['gender_performance'].iterrows())}

## Recommendations
1. Focus on expanding {takeaways['best_performing_product']} category
2. Optimize inventory for {takeaways['peak_season']} season
3. Prioritize {takeaways['top_market']} market for growth initiatives
4. Stock {', '.join(takeaways['top_colors'][:2])} colors in higher quantities
"""

            # Download button for the overall analysis report
            st.download_button(
                label="📥 Download Overall Analysis Report",
                data=overall_analysis_text,
                file_name="fashion_retail_analysis.md",
                mime="text/markdown"
            )

    with tab2:
        # Product Analysis Section
        st.markdown("## 🏷️ Product-Specific Analysis")
        
        # Product selector
        selected_product = st.selectbox(
            "Select Product Category",
            options=df['clothing_type'].unique(),
            key="product_selector"
        )
        
        # Generate product-specific analysis
        product_analysis = generate_product_analysis(df, selected_product)
        
        with st.expander("View Product Analysis", expanded=True):
            # Sales Metrics
            st.markdown("### 📈 Sales Performance")
            product_metrics_cols = st.columns(3)
            with product_metrics_cols[0]:
                st.metric(
                    "Historical Sales",
                    f"{product_analysis['sales_metrics']['historical_sales']:,.0f}",
                    "Past Year"
                )
            with product_metrics_cols[1]:
                st.metric(
                    "Projected Sales",
                    f"{product_analysis['sales_metrics']['future_sales']:,.0f}",
                    "Next 12 Months"
                )
            with product_metrics_cols[2]:
                st.metric(
                    "Growth Rate",
                    f"{product_analysis['sales_metrics']['growth_rate']:.1f}%",
                    "vs Previous Period"
                )
            
            # Price and Review Trends
            st.markdown("### 💰 Price & Review Analysis")
            trend_cols = st.columns(2)
            
            with trend_cols[0]:
                price_trend_fig = px.line(
                    product_analysis['price_metrics']['price_trend'],
                    title=f'Price Trend - {selected_product}',
                )
                price_trend_fig.update_layout(height=300)
                st.plotly_chart(price_trend_fig, use_container_width=True)
            
            with trend_cols[1]:
                review_trend_fig = px.line(
                    product_analysis['review_metrics']['review_trend'],
                    title=f'Review Score Trend - {selected_product}',
                )
                review_trend_fig.update_layout(height=300)
                st.plotly_chart(review_trend_fig, use_container_width=True)
            
            # Seasonal and Geographic Analysis
            st.markdown("### 🌍 Market Analysis")
            market_cols = st.columns(2)
            
            with market_cols[0]:
                seasonal_fig = px.bar(
                    x=product_analysis['seasonal_sales'].index,
                    y=product_analysis['seasonal_sales'].values,
                    title=f'Seasonal Performance - {selected_product}'
                )
                seasonal_fig.update_layout(height=350)
                st.plotly_chart(seasonal_fig, use_container_width=True)
            
            with market_cols[1]:
                city_fig = px.bar(
                    x=product_analysis['city_sales'].head().index,
                    y=product_analysis['city_sales'].head().values,
                    title=f'Top Cities - {selected_product}'
                )
                city_fig.update_layout(height=350)
                st.plotly_chart(city_fig, use_container_width=True)
            
            # Color and Gender Analysis
            st.markdown("### 👥 Customer Preferences")
            pref_cols = st.columns(2)
            
            with pref_cols[0]:
                color_fig = px.pie(
                    values=product_analysis['color_sales'].head(),
                    names=product_analysis['color_sales'].head().index,
                    title=f'Color Distribution - {selected_product}'
                )
                color_fig.update_layout(height=350)
                st.plotly_chart(color_fig, use_container_width=True)
            
            with pref_cols[1]:
                gender_fig = px.pie(
                    values=product_analysis['gender_sales'],
                    names=product_analysis['gender_sales'].index,
                    title=f'Gender Distribution - {selected_product}'
                )
                gender_fig.update_layout(height=350)
                st.plotly_chart(gender_fig, use_container_width=True)
            
            # Create product-specific analysis report
            product_analysis_text = f"""# {selected_product} Analysis Report

## Sales Performance
- Historical Sales: {product_analysis['sales_metrics']['historical_sales']:,.0f}
- Projected Sales: {product_analysis['sales_metrics']['future_sales']:,.0f}
- Growth Rate: {product_analysis['sales_metrics']['growth_rate']:.1f}%

## Price Analysis
- Average Price: ₹{product_analysis['price_metrics']['average_price']:.2f}

## Review Performance
- Average Review Score: {product_analysis['review_metrics']['average_review']:.2f}/5.0

## Market Analysis
- Top Performing Season: {product_analysis['seasonal_sales'].index[0]}
- Best Performing City: {product_analysis['city_sales'].index[0]}
- Most Popular Color: {product_analysis['color_sales'].index[0]}
- Gender Distribution: {', '.join(f"{index}: {value:,.0f}" for index, value in product_analysis['gender_sales'].items())}

## Recommendations
1. Focus on {product_analysis['seasonal_sales'].index[0]} season for maximum sales
2. Prioritize inventory in {product_analysis['city_sales'].index[0]}
3. Maintain strong stock of {product_analysis['color_sales'].index[0]} color options
"""

            # Download button for product-specific analysis
            st.download_button(
                label=f"📥 Download {selected_product} Analysis Report",
                data=product_analysis_text,
                file_name=f"{selected_product.lower()}_analysis.md",
                mime="text/markdown"
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please try refreshing the page or contact support if the issue persists.")
