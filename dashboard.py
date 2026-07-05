import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Carbon-Aware Workload Optimizer", layout="wide", page_icon="🌍")

# Color scheme
COLORS = {'Europe': '#2ecc71', 'USA': '#3498db', 'India': '#e67e22'}

@st.cache_data
def load_data():
    """Load all CSV data files"""
    try:
        predictions = pd.read_csv('results/predictions.csv', parse_dates=['timestamp'])
        scheduling = pd.read_csv('results/scheduling_results.csv', parse_dates=['timestamp'])
        scaling = pd.read_csv('results/scaling_results.csv', parse_dates=['timestamp'])
        scores = pd.read_csv('results/full_region_scores.csv', parse_dates=['timestamp'])
        federated = pd.read_csv('results/federated_metrics.csv')
        model_metrics = pd.read_csv('results/model_metrics.csv')
        return predictions, scheduling, scaling, scores, federated, model_metrics
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        carbon_model = joblib.load('models/carbon_model.pkl')
        temp_model = joblib.load('models/temp_model.pkl')
        return carbon_model, temp_model
    except:
        return None, None

def create_kpi_cards(scores_df, federated_df):
    """Display KPI metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_carbon = scores_df['predicted_carbon_intensity'].mean()
        st.metric("Avg Carbon Intensity", f"{avg_carbon:.1f} gCO₂/kWh", delta=None)
    
    with col2:
        avg_temp = scores_df['predicted_temperature'].mean()
        st.metric("Avg Predicted Temp", f"{avg_temp:.2f}°C", delta=None)
    
    with col3:
        avg_score = scores_df['score'].mean()
        st.metric("Avg Region Score", f"{avg_score:.2f}", delta="Lower is better", delta_color="inverse")
    
    with col4:
        st.metric("Supported Regions", "3", delta="US, India, Europe")

def overview_page(predictions, scheduling, scaling, scores, federated, model_metrics):
    """Overview dashboard page"""
    st.title("🌍 Carbon-Aware Federated Workload Optimizer")
    st.markdown("### System Overview")
    
    create_kpi_cards(scores, federated)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Model Performance Summary")
        carbon_r2 = federated[federated['target'] == 'carbon_intensity']['r2_score'].max()
        temp_r2 = federated[federated['target'] == 'temperature']['r2_score'].max()
        carbon_mae = federated[federated['target'] == 'carbon_intensity']['mae'].min()
        temp_mae = federated[federated['target'] == 'temperature']['mae'].min()
        
        perf_data = pd.DataFrame({
            'Model': ['Carbon Intensity', 'Temperature'],
            'R² Score': [carbon_r2, temp_r2],
            'MAE': [carbon_mae, temp_mae]
        })
        st.dataframe(perf_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### ✅ Project Phase Status")
        phases = [
            "Data Generation & Preprocessing",
            "Feature Engineering",
            "Traditional ML Training",
            "Federated Learning",
            "Prediction & Forecasting",
            "Workload Scheduling",
            "Dynamic Scaling",
            "Dashboard & Visualization"
        ]
        for phase in phases:
            st.success(f"✓ {phase}")

def trends_page(predictions, scores):
    """Trends and analysis page"""
    st.title("📈 Trends & Analysis")
    
    regions = ['All'] + sorted(scores['region'].unique().tolist())
    selected_region = st.selectbox("Select Region", regions)
    
    if selected_region != 'All':
        scores = scores[scores['region'] == selected_region]
    
    tab1, tab2, tab3 = st.tabs(["Carbon Intensity", "Temperature", "Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(scores, x='timestamp', y='predicted_carbon_intensity', 
                         color='region', color_discrete_map=COLORS,
                         title="Carbon Intensity Over Time")
            fig.update_layout(yaxis_title="Carbon Intensity (gCO₂/kWh)", xaxis_title="Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_carbon = scores.groupby('region')['predicted_carbon_intensity'].mean().reset_index()
            fig = px.bar(avg_carbon, x='region', y='predicted_carbon_intensity',
                        color='region', color_discrete_map=COLORS,
                        title="Average Carbon Intensity by Region")
            fig.update_layout(yaxis_title="Avg Carbon Intensity (gCO₂/kWh)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(scores, x='timestamp', y='predicted_temperature',
                         color='region', color_discrete_map=COLORS,
                         title="Temperature Over Time")
            fig.update_layout(yaxis_title="Temperature (°C)", xaxis_title="Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_temp = scores.groupby('region')['predicted_temperature'].mean().reset_index()
            fig = px.bar(avg_temp, x='region', y='predicted_temperature',
                        color='region', color_discrete_map=COLORS,
                        title="Average Temperature by Region")
            fig.update_layout(yaxis_title="Avg Temperature (°C)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        metric = st.selectbox("Select Metric", 
                             ['predicted_carbon_intensity', 'predicted_temperature', 
                              'energy_consumption', 'cost', 'score'])
        fig = px.box(scores, x='region', y=metric, color='region',
                    color_discrete_map=COLORS,
                    title=f"{metric.replace('_', ' ').title()} Distribution by Region")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def scheduling_page(scheduling, scaling, scores):
    """Scheduling results page"""
    st.title("🎯 Scheduling Results")
    
    tab1, tab2 = st.tabs(["Scheduling Decisions", "Workload Scaling"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            region_counts = scheduling['selected_region'].value_counts()
            fig = px.pie(values=region_counts.values, names=region_counts.index,
                        color=region_counts.index, color_discrete_map=COLORS,
                        title="Region Selection Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(scheduling, x='score', nbins=30,
                             title="Score Distribution (Lower is Better)")
            fig.update_layout(xaxis_title="Score", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Recent Scheduling Decisions")
        recent = scheduling.tail(10)[['timestamp', 'selected_region', 'score', 
                                      'carbon_intensity', 'energy_consumption', 'cost']]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            latest = scaling.groupby('region').last().reset_index()
            fig = px.bar(latest, x='region', y='workload_percentage',
                        color='region', color_discrete_map=COLORS,
                        title="Current Workload Distribution")
            fig.update_layout(yaxis_title="Workload %", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(scaling, x='timestamp', y='score', color='region',
                         color_discrete_map=COLORS,
                         title="Score Trends Over Time")
            fig.update_layout(yaxis_title="Score (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)

def model_performance_page(federated, model_metrics, scores):
    """Model performance page"""
    st.title("🤖 Model Performance")
    
    tab1, tab2, tab3 = st.tabs(["Federated Learning Metrics", "Traditional ML Metrics", "Prediction Accuracy"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        for target, col in zip(['carbon_intensity', 'temperature'], [col1, col2]):
            with col:
                st.markdown(f"#### {target.replace('_', ' ').title()}")
                data = federated[federated['target'] == target]
                
                local_avg = data[data['model_type'].str.contains('local')]['r2_score'].mean()
                global_r2 = data[data['model_type'] == 'global_federated']['r2_score'].values[0]
                
                comparison = pd.DataFrame({
                    'Model Type': ['Local (Avg)', 'Global Federated'],
                    'R² Score': [local_avg, global_r2],
                    'MAE': [data[data['model_type'].str.contains('local')]['mae'].mean(),
                           data[data['model_type'] == 'global_federated']['mae'].values[0]]
                })
                
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                
                if global_r2 > local_avg:
                    st.success(f"✓ Global model outperforms by {((global_r2/local_avg - 1)*100):.1f}%")
                else:
                    st.info("Local models perform better")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            carbon_data = model_metrics[model_metrics['target'] == 'carbon_intensity']
            fig = px.bar(carbon_data, x='model', y='r2_score',
                        title="Carbon Intensity Model Comparison",
                        color='model')
            fig.update_layout(yaxis_title="R² Score", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            temp_data = model_metrics[model_metrics['target'] == 'temperature']
            fig = px.bar(temp_data, x='model', y='r2_score',
                        title="Temperature Model Comparison",
                        color='model')
            fig.update_layout(yaxis_title="R² Score", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detailed Metrics")
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
    
    with tab3:
        # Note: For actual vs predicted, we'd need test data. Using scores as proxy
        st.info("Prediction accuracy visualization requires test set data with actual values")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(scores, x='predicted_carbon_intensity', y='score',
                           color='region', color_discrete_map=COLORS,
                           title="Carbon Intensity vs Score",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(scores, x='predicted_temperature', y='score',
                           color='region', color_discrete_map=COLORS,
                           title="Temperature vs Score",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

def predictor_page(carbon_model, temp_model):
    """Interactive predictor page"""
    st.title("🔮 Interactive Predictor")
    
    if carbon_model is None or temp_model is None:
        st.warning("Models not loaded. Please train models first.")
        return
    
    st.markdown("### Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Europe")
        eu_cpu = st.slider("CPU Usage (%)", 0, 100, 50, key='eu_cpu')
        eu_temp = st.slider("Temperature (°C)", -10, 40, 15, key='eu_temp')
        eu_cost = st.slider("Cost ($/kWh)", 0.0, 0.5, 0.15, key='eu_cost')
        eu_workload = st.slider("Workload", 0, 100, 50, key='eu_workload')
    
    with col2:
        st.markdown("#### USA")
        us_cpu = st.slider("CPU Usage (%)", 0, 100, 50, key='us_cpu')
        us_temp = st.slider("Temperature (°C)", -10, 40, 20, key='us_temp')
        us_cost = st.slider("Cost ($/kWh)", 0.0, 0.5, 0.20, key='us_cost')
        us_workload = st.slider("Workload", 0, 100, 50, key='us_workload')
    
    with col3:
        st.markdown("#### India")
        in_cpu = st.slider("CPU Usage (%)", 0, 100, 50, key='in_cpu')
        in_temp = st.slider("Temperature (°C)", -10, 40, 25, key='in_temp')
        in_cost = st.slider("Cost ($/kWh)", 0.0, 0.5, 0.10, key='in_cost')
        in_workload = st.slider("Workload", 0, 100, 50, key='in_workload')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Time Parameters")
        hour = st.slider("Hour of Day", 0, 23, 12)
        day = st.slider("Day of Week", 0, 6, 3, help="0=Monday, 6=Sunday")
        month = st.slider("Month", 1, 12, 6)
    
    with col2:
        st.markdown("#### Scheduling Weights")
        w_carbon = st.slider("Carbon Weight", 0.0, 1.0, 0.4)
        w_energy = st.slider("Energy Weight", 0.0, 1.0, 0.3)
        w_cost = st.slider("Cost Weight", 0.0, 1.0, 0.2)
        w_temp = st.slider("Temperature Weight", 0.0, 1.0, 0.1)
    
    if st.button("Calculate Optimal Region", type="primary"):
        regions_data = {
            'Europe': {'cpu': eu_cpu, 'temp': eu_temp, 'cost': eu_cost, 'workload': eu_workload, 'region_India': 0, 'region_USA': 0},
            'USA': {'cpu': us_cpu, 'temp': us_temp, 'cost': us_cost, 'workload': us_workload, 'region_India': 0, 'region_USA': 1},
            'India': {'cpu': in_cpu, 'temp': in_temp, 'cost': in_cost, 'workload': in_workload, 'region_India': 1, 'region_USA': 0}
        }
        
        results = []
        for region, data in regions_data.items():
            try:
                # Calculate derived features
                is_weekend = 1 if day >= 5 else 0
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                energy = 30 + data['cpu'] * 0.2
                cpu_temp_interaction = data['cpu'] * data['temp']
                workload_cost_interaction = data['workload'] * data['cost']
                
                # Use average values for lag/rolling features (simplified)
                carbon_estimate = 250
                carbon_lag_1h = carbon_estimate
                carbon_lag_24h = carbon_estimate
                temp_lag_1h = data['temp']
                carbon_rolling_mean_24h = carbon_estimate
                temp_rolling_mean_24h = data['temp']
                
                # Create feature vector matching model's expected 20 features
                # ['carbon_intensity', 'cpu_usage', 'workload', 'energy_consumption', 'cost',
                #  'hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos',
                #  'carbon_lag_1h', 'carbon_lag_24h', 'temp_lag_1h', 'carbon_rolling_mean_24h',
                #  'temp_rolling_mean_24h', 'cpu_temp_interaction', 'workload_cost_interaction',
                #  'region_India', 'region_USA']
                
                features = np.array([[
                    carbon_estimate,  # carbon_intensity (placeholder)
                    data['cpu'],  # cpu_usage
                    data['workload'],  # workload
                    energy,  # energy_consumption
                    data['cost'],  # cost
                    hour,  # hour
                    day,  # day_of_week
                    month,  # month
                    is_weekend,  # is_weekend
                    hour_sin,  # hour_sin
                    hour_cos,  # hour_cos
                    carbon_lag_1h,  # carbon_lag_1h
                    carbon_lag_24h,  # carbon_lag_24h
                    temp_lag_1h,  # temp_lag_1h
                    carbon_rolling_mean_24h,  # carbon_rolling_mean_24h
                    temp_rolling_mean_24h,  # temp_rolling_mean_24h
                    cpu_temp_interaction,  # cpu_temp_interaction
                    workload_cost_interaction,  # workload_cost_interaction
                    data['region_India'],  # region_India
                    data['region_USA']  # region_USA
                ]])
                
                # Predict using actual models
                carbon = carbon_model.predict(features)[0]
                temp_pred = temp_model.predict(features)[0]
                
                score = (w_carbon * carbon + w_energy * energy + 
                        w_cost * data['cost'] * 100 + w_temp * abs(temp_pred))
                
                results.append({
                    'Region': region,
                    'Carbon (gCO₂/kWh)': round(carbon, 2),
                    'Temperature (°C)': round(temp_pred, 2),
                    'Energy (kWh)': round(energy, 2),
                    'Cost ($)': data['cost'],
                    'Score': round(score, 2)
                })
            except Exception as e:
                st.error(f"Error predicting for {region}: {str(e)}")
        
        if results:
            results_df = pd.DataFrame(results).sort_values('Score')
            
            st.markdown("### 🏆 Results")
            st.success(f"**Optimal Region: {results_df.iloc[0]['Region']}** (Score: {results_df.iloc[0]['Score']:.2f})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(results_df, x='Region', y='Score',
                           color='Region', color_discrete_map=COLORS,
                           title="Region Score Comparison (Lower is Better)")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Overview", "Trends & Analysis", "Scheduling Results", 
                            "Model Performance", "Interactive Predictor"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Carbon-Aware Federated Workload Optimization**\n\n"
                   "Optimizing workload placement across geo-distributed data centers "
                   "using federated learning and carbon intensity predictions.")
    
    # Load data
    predictions, scheduling, scaling, scores, federated, model_metrics = load_data()
    carbon_model, temp_model = load_models()
    
    if predictions is None:
        st.error("Failed to load data. Please ensure all CSV files exist in results/ directory.")
        return
    
    # Route to pages
    if page == "Overview":
        overview_page(predictions, scheduling, scaling, scores, federated, model_metrics)
    elif page == "Trends & Analysis":
        trends_page(predictions, scores)
    elif page == "Scheduling Results":
        scheduling_page(scheduling, scaling, scores)
    elif page == "Model Performance":
        model_performance_page(federated, model_metrics, scores)
    elif page == "Interactive Predictor":
        predictor_page(carbon_model, temp_model)

if __name__ == "__main__":
    main()
