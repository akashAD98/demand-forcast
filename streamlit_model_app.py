"""
Streamlit App for Model Inference
Supports both ML and Time Series models for demand forecasting

Features:
- Load saved models (ML, Prophet, ARIMA)
- Interactive prediction interface
- Model performance comparison
- Visualization of predictions
- Download results

Author: Demand Forecasting Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
from model_training_pipeline import AdvancedForecastingPipeline

# Page configuration
st.set_page_config(
    page_title="🔮 Demand Forecasting Models",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .best-model {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_metadata():
    """Load model performance metadata."""
    try:
        pipeline = AdvancedForecastingPipeline(model_save_dir='saved_models')
        metadata = pipeline.load_model_metadata()
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

@st.cache_resource
def load_pipeline():
    """Load the forecasting pipeline."""
    return AdvancedForecastingPipeline(model_save_dir='saved_models')

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🔮 Demand Forecasting Models</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive interface for ML and Time Series model predictions**")
    
    # Check if models exist
    if not os.path.exists('saved_models'):
        st.error("⚠️ No saved models found! Please run the training pipeline first.")
        st.code("python advanced_forecasting_pipeline.py")
        return
    
    # Load data
    pipeline = load_pipeline()
    metadata = load_model_metadata()
    
    if metadata is None:
        st.error("❌ Could not load model metadata.")
        return
    
    # Sidebar for model selection and settings
    st.sidebar.title("🎛️ Model Settings")
    
    # Show best model
    best_model = metadata['best_model']
    st.sidebar.markdown(f'<div class="best-model"><strong>🏆 Best Model</strong><br>{best_model}<br>MAE: {metadata["models"][best_model]["mae"]:.2f}</div>', unsafe_allow_html=True)
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "📊 Select Model Type",
        ["🤖 ML Models", "📈 Time Series Models", "🏆 Best Model", "📋 Model Comparison"],
        index=2
    )
    
    # Main content based on selection
    if model_type == "🤖 ML Models":
        show_ml_models_interface(pipeline, metadata)
    elif model_type == "📈 Time Series Models":
        show_time_series_interface(pipeline, metadata)
    elif model_type == "🏆 Best Model":
        show_best_model_interface(pipeline, metadata)
    else:
        show_model_comparison(metadata)

def show_ml_models_interface(pipeline, metadata):
    """Interface for ML model predictions."""
    
    st.header("🤖 ML Model Predictions")
    st.write("ML models use features like SKU, location, date, and rolling averages to make predictions.")
    
    # ML model selection
    ml_models = [name for name, info in metadata['models'].items() if info['type'] == 'ML']
    selected_ml_model = st.selectbox("Select ML Model", ml_models)
    
    # Show model performance
    model_info = metadata['models'][selected_ml_model]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{model_info['mae']:.2f}")
    with col2:
        st.metric("R²", f"{model_info['r2']:.3f}")
    with col3:
        st.metric("MAPE", f"{model_info['mape']:.1f}%")
    with col4:
        st.metric("RMSE", f"{model_info['rmse']:.2f}")
    
    # Input form for predictions
    st.subheader("📝 Input Data for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range
        start_date = st.date_input("Start Date", datetime.now().date())
        num_days = st.number_input("Number of Days", min_value=1, max_value=30, value=7)
        
        # SKU and Location
        sku_id = st.selectbox("SKU ID", ["apple456", "banana789", "mango123", "SKU_001", "SKU_002"])
        location = st.selectbox("Location", ["Delhi", "Mumbai", "Location_A", "Location_B"])
    
    with col2:
        # Rolling average
        rolling_7d = st.number_input("Rolling 7-day Average", min_value=0.0, value=50.0, step=1.0)
        
        # Additional options
        show_confidence = st.checkbox("Show prediction details", value=True)
    
    if st.button("🚀 Generate ML Predictions", type="primary"):
        
        # Create prediction data
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        prediction_data = pd.DataFrame({
            'order_date': pd.to_datetime(dates),
            'sku_id': [sku_id] * num_days,
            'location': [location] * num_days,
            'rolling_7d': [rolling_7d] * num_days
        })
        
        try:
            with st.spinner(f"Generating predictions with {selected_ml_model}..."):
                predictions = pipeline.predict_with_ml_model(selected_ml_model, prediction_data)
            
            # Display results
            st.success(f"✅ Generated {len(predictions)} predictions!")
            
            # Create results dataframe
            results_df = prediction_data.copy()
            results_df['predicted_quantity'] = predictions
            results_df['prediction_date'] = results_df['order_date'].dt.strftime('%Y-%m-%d')
            
            # Show results table
            st.subheader("📊 Prediction Results")
            st.dataframe(results_df[['prediction_date', 'sku_id', 'location', 'predicted_quantity']].round(2))
            
            # Visualization
            fig = px.line(
                results_df, 
                x='order_date', 
                y='predicted_quantity',
                title=f'{selected_ml_model} Predictions for {sku_id} at {location}',
                labels={'predicted_quantity': 'Predicted Quantity', 'order_date': 'Date'}
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            if show_confidence:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Prediction", f"{np.mean(predictions):.2f}")
                with col2:
                    st.metric("Min Prediction", f"{np.min(predictions):.2f}")
                with col3:
                    st.metric("Max Prediction", f"{np.max(predictions):.2f}")
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"{selected_ml_model}_predictions_{start_date}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

def show_time_series_interface(pipeline, metadata):
    """Interface for time series model predictions."""
    
    st.header("📈 Time Series Model Predictions")
    st.write("Time series models (Prophet/ARIMA) use historical patterns to forecast future demand.")
    
    # Time series model selection
    ts_models = [name for name, info in metadata['models'].items() if info['type'] == 'TimeSeries']
    selected_ts_model = st.selectbox("Select Time Series Model", ts_models)
    
    # Show model performance
    model_info = metadata['models'][selected_ts_model]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{model_info['mae']:.2f}")
    with col2:
        st.metric("R²", f"{model_info['r2']:.3f}")
    with col3:
        st.metric("MAPE", f"{model_info['mape']:.1f}%")
    with col4:
        st.metric("Time Series", f"{model_info['num_series']}")
    
    # Input form for predictions
    st.subheader("📝 Input Data for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Available SKU-location combinations
        if selected_ts_model == 'Prophet':
            try:
                prophet_models = pipeline.load_prophet_models()
                available_combinations = list(prophet_models.keys())
            except:
                available_combinations = ["apple456_Delhi", "apple456_Mumbai", "banana789_Delhi", "banana789_Mumbai", "mango123_Delhi", "mango123_Mumbai"]
        else:  # ARIMA
            try:
                arima_models = pipeline.load_arima_models()
                available_combinations = list(arima_models.keys())
            except:
                available_combinations = ["apple456_Delhi", "apple456_Mumbai", "banana789_Delhi", "banana789_Mumbai", "mango123_Delhi", "mango123_Mumbai"]
        
        selected_combinations = st.multiselect(
            "Select SKU-Location Combinations",
            available_combinations,
            default=available_combinations[:3] if len(available_combinations) >= 3 else available_combinations
        )
        
        forecast_steps = st.number_input("Forecast Steps (Days)", min_value=1, max_value=30, value=7)
    
    with col2:
        start_date = st.date_input("Forecast Start Date", datetime.now().date())
        show_all_series = st.checkbox("Show all time series in one chart", value=False)
    
    if st.button("🚀 Generate Time Series Predictions", type="primary"):
        
        if not selected_combinations:
            st.warning("⚠️ Please select at least one SKU-Location combination.")
            return
        
        # Create prediction data from selected combinations
        prediction_data_list = []
        for combo in selected_combinations:
            sku, location = combo.split('_')
            prediction_data_list.append({
                'order_date': start_date,
                'sku_id': sku,
                'location': location
            })
        
        prediction_data = pd.DataFrame(prediction_data_list)
        prediction_data['order_date'] = pd.to_datetime(prediction_data['order_date'])
        
        try:
            with st.spinner(f"Generating predictions with {selected_ts_model}..."):
                if selected_ts_model == 'Prophet':
                    predictions = pipeline.predict_with_prophet_models(prediction_data, forecast_steps)
                else:  # ARIMA
                    predictions = pipeline.predict_with_arima_models(prediction_data, forecast_steps)
            
            if not predictions:
                st.warning("⚠️ No predictions generated. Check if models exist for selected combinations.")
                return
            
            st.success(f"✅ Generated predictions for {len(predictions)} time series!")
            
            # Display results
            st.subheader("📊 Time Series Prediction Results")
            
            # Create visualization
            if show_all_series:
                # Single chart with all series
                fig = go.Figure()
                
                for ts_key, preds in predictions.items():
                    dates = [start_date + timedelta(days=i) for i in range(len(preds))]
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=preds,
                        mode='lines+markers',
                        name=ts_key,
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title=f'{selected_ts_model} Predictions - All Series',
                    xaxis_title='Date',
                    yaxis_title='Predicted Quantity',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Separate charts for each series
                for ts_key, preds in predictions.items():
                    st.subheader(f"📈 {ts_key}")
                    
                    dates = [start_date + timedelta(days=i) for i in range(len(preds))]
                    
                    fig = px.line(
                        x=dates,
                        y=preds,
                        title=f'{selected_ts_model} Predictions - {ts_key}',
                        labels={'x': 'Date', 'y': 'Predicted Quantity'}
                    )
                    fig.update_traces(line=dict(width=3))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average", f"{np.mean(preds):.2f}")
                    with col2:
                        st.metric("Min", f"{np.min(preds):.2f}")
                    with col3:
                        st.metric("Max", f"{np.max(preds):.2f}")
            
            # Create downloadable results
            all_results = []
            for ts_key, preds in predictions.items():
                sku, location = ts_key.split('_')
                for i, pred in enumerate(preds):
                    all_results.append({
                        'date': start_date + timedelta(days=i),
                        'sku_id': sku,
                        'location': location,
                        'predicted_quantity': pred,
                        'model': selected_ts_model
                    })
            
            results_df = pd.DataFrame(all_results)
            
            # Show summary table
            st.subheader("📋 Summary Table")
            summary_df = results_df.groupby(['sku_id', 'location'])['predicted_quantity'].agg(['mean', 'min', 'max']).round(2)
            st.dataframe(summary_df)
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"{selected_ts_model}_predictions_{start_date}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

def show_best_model_interface(pipeline, metadata):
    """Interface for using the best performing model."""
    
    best_model = metadata['best_model']
    best_model_info = metadata['models'][best_model]
    
    st.header(f"🏆 Best Model: {best_model}")
    st.write(f"Using the best performing model based on MAE: {best_model_info['mae']:.2f}")
    
    # Show performance
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 MAE", f"{best_model_info['mae']:.2f}")
    with col2:
        st.metric("📊 R²", f"{best_model_info['r2']:.3f}")
    with col3:
        st.metric("📈 MAPE", f"{best_model_info['mape']:.1f}%")
    with col4:
        st.metric("🔧 Type", best_model_info['type'])
    
    # Quick prediction interface
    st.subheader("🚀 Quick Prediction")
    
    if best_model_info['type'] == 'ML':
        # ML model interface (simplified)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sku_id = st.selectbox("SKU", ["apple456", "banana789", "mango123"])
            location = st.selectbox("Location", ["Delhi", "Mumbai"])
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now().date())
            num_days = st.number_input("Days to Predict", 1, 14, 7)
        
        with col3:
            rolling_7d = st.number_input("Rolling Average", 0.0, 100.0, 50.0)
        
        if st.button("🎯 Predict with Best Model", type="primary"):
            dates = [start_date + timedelta(days=i) for i in range(num_days)]
            prediction_data = pd.DataFrame({
                'order_date': pd.to_datetime(dates),
                'sku_id': [sku_id] * num_days,
                'location': [location] * num_days,
                'rolling_7d': [rolling_7d] * num_days
            })
            
            try:
                predictions = pipeline.predict_with_ml_model(best_model, prediction_data)
                
                # Quick visualization
                fig = px.line(
                    x=dates, 
                    y=predictions,
                    title=f'{best_model} Predictions - {sku_id} at {location}',
                    labels={'x': 'Date', 'y': 'Predicted Quantity'}
                )
                fig.update_traces(line=dict(width=4, color='#28a745'))
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"📈 Average predicted demand: {np.mean(predictions):.2f}")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    else:
        # Time Series model interface (simplified)
        available_combinations = ["apple456_Delhi", "apple456_Mumbai", "banana789_Delhi", "banana789_Mumbai", "mango123_Delhi", "mango123_Mumbai"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_combo = st.selectbox("SKU-Location", available_combinations)
            forecast_days = st.number_input("Forecast Days", 1, 14, 7)
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now().date())
        
        if st.button("🎯 Predict with Best Model", type="primary"):
            sku, location = selected_combo.split('_')
            prediction_data = pd.DataFrame({
                'order_date': [pd.to_datetime(start_date)],
                'sku_id': [sku],
                'location': [location]
            })
            
            try:
                if best_model == 'Prophet':
                    predictions = pipeline.predict_with_prophet_models(prediction_data, forecast_days)
                else:
                    predictions = pipeline.predict_with_arima_models(prediction_data, forecast_days)
                
                if predictions:
                    preds = predictions[selected_combo]
                    dates = [start_date + timedelta(days=i) for i in range(len(preds))]
                    
                    # Quick visualization
                    fig = px.line(
                        x=dates, 
                        y=preds,
                        title=f'{best_model} Predictions - {selected_combo}',
                        labels={'x': 'Date', 'y': 'Predicted Quantity'}
                    )
                    fig.update_traces(line=dict(width=4, color='#28a745'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"📈 Average predicted demand: {np.mean(preds):.2f}")
                else:
                    st.warning("⚠️ No predictions generated.")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

def show_model_comparison(metadata):
    """Show detailed model comparison."""
    
    st.header("📋 Model Performance Comparison")
    st.write("Compare all trained models and their performance metrics.")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, model_info in metadata['models'].items():
        comparison_data.append({
            'Model': model_name,
            'Type': model_info['type'],
            'MAE': model_info['mae'],
            'R²': model_info['r2'],
            'MAPE (%)': model_info['mape'],
            'RMSE': model_info['rmse'],
            'Time Series': model_info.get('num_series', 1)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MAE')
    
    # Display comparison table
    st.subheader("📊 Performance Table")
    
    # Highlight best model
    def highlight_best(row):
        if row['Model'] == metadata['best_model']:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1).format({
            'MAE': '{:.2f}',
            'R²': '{:.3f}',
            'MAPE (%)': '{:.1f}',
            'RMSE': '{:.2f}'
        }),
        use_container_width=True
    )
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE comparison
        fig_mae = px.bar(
            comparison_df, 
            x='Model', 
            y='MAE',
            color='Type',
            title='Mean Absolute Error (MAE) Comparison',
            color_discrete_map={'ML': '#1f77b4', 'TimeSeries': '#ff7f0e'}
        )
        fig_mae.update_layout(height=400)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        # R² comparison
        fig_r2 = px.bar(
            comparison_df, 
            x='Model', 
            y='R²',
            color='Type',
            title='R² Score Comparison',
            color_discrete_map={'ML': '#1f77b4', 'TimeSeries': '#ff7f0e'}
        )
        fig_r2.update_layout(height=400)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Model type analysis
    st.subheader("🔍 Model Type Analysis")
    
    type_summary = comparison_df.groupby('Type').agg({
        'MAE': ['mean', 'min', 'max'],
        'R²': ['mean', 'min', 'max'],
        'Model': 'count'
    }).round(3)
    
    st.dataframe(type_summary, use_container_width=True)
    
    # Training info
    st.subheader("ℹ️ Training Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Training Date:** {metadata['training_timestamp'][:19]}")
        st.info(f"**Best Model:** {metadata['best_model']}")
    
    with col2:
        st.info(f"**Total Models:** {len(metadata['models'])}")
        ml_count = len([m for m in metadata['models'].values() if m['type'] == 'ML'])
        ts_count = len([m for m in metadata['models'].values() if m['type'] == 'TimeSeries'])
        st.info(f"**ML Models:** {ml_count} | **Time Series:** {ts_count}")

if __name__ == "__main__":
    main() 