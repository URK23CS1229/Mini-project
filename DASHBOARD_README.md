# Carbon-Aware Federated Workload Optimizer Dashboard

## Overview
Professional interactive dashboard for monitoring and optimizing workload placement across 3 geo-distributed data centers (US, India, Europe) using federated learning predictions.

## Features

### 1. Overview Page
- **KPI Cards**: Real-time metrics for carbon intensity, temperature, region scores, and supported regions
- **Model Performance Summary**: R² scores and MAE for carbon and temperature models
- **Project Phase Status**: Visual completion tracker for all 8 project phases

### 2. Trends & Analysis Page
- **Region Filter**: Analyze specific regions or all regions combined
- **Carbon Intensity Tab**: Time series trends and average comparisons
- **Temperature Tab**: Temperature trends and regional averages
- **Comparison Tab**: Box plot distributions for any metric

### 3. Scheduling Results Page
- **Scheduling Decisions**: 
  - Pie chart showing region selection distribution
  - Score distribution histogram
  - Recent scheduling decisions table
- **Workload Scaling**:
  - Current workload distribution across regions
  - Score trends over time

### 4. Model Performance Page
- **Federated Learning Metrics**: Local vs global model comparison
- **Traditional ML Metrics**: RandomForest and XGBoost performance
- **Prediction Accuracy**: Scatter plots with trendlines

### 5. Interactive Predictor
- **Region Parameters**: CPU, temperature, and cost sliders for each region
- **Time Parameters**: Hour, day of week, and month inputs
- **Scheduling Weights**: Customizable weights for carbon, energy, cost, and temperature
- **Live Calculation**: Real-time optimal region recommendation
- **Visual Comparison**: Bar charts comparing region scores

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Windows
```bash
run_production_dashboard.bat
```

### Command Line
```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

## Data Requirements

The dashboard expects the following files:

```
results/
├── predictions.csv
├── scheduling_results.csv
├── scaling_results.csv
├── full_region_scores.csv
├── federated_metrics.csv
└── model_metrics.csv

models/
├── carbon_model.pkl
└── temp_model.pkl
```

## Color Scheme
- **Europe**: Green (#2ecc71)
- **USA**: Blue (#3498db)
- **India**: Orange (#e67e22)

## Key Metrics

### Score Interpretation
- **Lower scores = Better performance**
- Scores combine carbon intensity, energy consumption, cost, and temperature
- Optimal region has the lowest score

### Model Performance
- **R² Score**: Closer to 1.0 indicates better predictions
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

## Navigation
Use the sidebar to switch between pages:
1. Overview - System summary and KPIs
2. Trends & Analysis - Historical data visualization
3. Scheduling Results - Workload distribution insights
4. Model Performance - ML model evaluation
5. Interactive Predictor - Real-time region optimization

## Features

### Caching
- Data and models are cached for optimal performance
- Automatic refresh when source files change

### Interactivity
- All charts support zoom, pan, and hover tooltips
- Filters and selectors update visualizations in real-time
- Responsive design adapts to screen size

### Error Handling
- Graceful degradation if data files are missing
- Clear error messages for troubleshooting
- Fallback values for predictions

## Technical Details

### Performance Optimization
- `@st.cache_data` for CSV files
- `@st.cache_resource` for ML models
- Efficient data loading and processing

### Compatibility
- Supports both old (5 features) and new (11 features) model formats
- Backward compatible with existing data files
- Safe fallback for missing columns

## Troubleshooting

### Dashboard won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory

### Data not loading
- Verify all CSV files exist in `results/` directory
- Check file permissions
- Ensure CSV files have correct column names

### Models not loading
- Verify `.pkl` files exist in `models/` directory
- Retrain models if necessary using `scripts/train_model_improved.py`

### Predictions failing
- Check model feature compatibility
- Verify input parameter ranges
- Review console for error messages

## Best Practices

1. **Regular Updates**: Refresh data files periodically for accurate insights
2. **Weight Tuning**: Adjust scheduling weights based on organizational priorities
3. **Trend Monitoring**: Review trends page regularly to identify patterns
4. **Model Evaluation**: Check model performance page after retraining

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the dashboard
3. Verify data file integrity
4. Ensure models are properly trained

## Version
Dashboard v1.0 - Production Ready
