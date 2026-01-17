

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Food Price Volatility Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# MODERN CSS STYLING

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with soft gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 25%, #f0e8f5 75%, #e8f5f1 100%);
    }
    
    /* Sidebar styling */
    /* 
       RADIO BUTTONS STYLED AS BUTTONS - KEY STYLING
        */
    
    /* Hide the radio button circle */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Style the radio button labels as buttons */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: #ffffff !important;
        padding: 14px 20px !important;
        border-radius: 12px !important;
        margin: 8px 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #4a5568 !important;
    }
    
    /* Make sure the text/span inside is also visible */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label span {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    /* Hover effect for radio buttons */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: linear-gradient(135deg, #ede9fe 0%, #e0e7ff 100%) !important;
        border-color: rgba(139, 92, 246, 0.5) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover p,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover span {
        color: #5b21b6 !important;
    }
    
    /* Selected/Active state for radio buttons */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
        font-weight: 600 !important;
    }
    
    /* Selected state text color */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] p,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] span,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) span {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Alternative selected state selector for better compatibility */
    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%) !important;
    }
    
    /* Hide the radio group label if needed */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 50%, #7c3aed 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(139, 92, 246, 0.2);
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #1a202c !important;
        font-weight: 700;
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        color: #2d3748 !important;
    }
    
    /* Cards and containers */
    .element-container {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    }
    
    /* Input fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 10px !important;
        color: #2d3748 !important;
    }
    
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stMultiSelect label {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Table styling */
    table {
        color: #2d3748 !important;
    }
    
    thead tr th {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    tbody tr {
        background: #ffffff !important;
    }
    
    tbody tr:hover {
        background: rgba(167, 139, 250, 0.1) !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 0, 0, 0.1), transparent);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #2d3748;
    }
    
    /* Code blocks */
    code {
        background: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 6px;
        padding: 2px 6px;
        color: #7c3aed !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: rgba(45, 55, 72, 0.6);
        font-size: 0.9rem;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    /* Prediction result cards */
    .prediction-card {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
    }
    
    .prediction-card.volatile {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-color: rgba(239, 68, 68, 0.3);
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.2);
    }
    
    .prediction-card.stable {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# CHECK FOR REQUIRED FILES

REQUIRED_FILES = [
    'catboost_model.cbm',
    'feature_info.json',
    'model_metrics.json',
    'processed_data.csv',
    'feature_importance.csv',
    'classification_report.json'
]

def check_required_files():
    """Check if all required files exist"""
    missing_files = []
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            missing_files.append(file)
    return missing_files


# LOAD PRE-TRAINED MODEL AND DATA (CACHED)

@st.cache_resource
def load_model():
    """Load the pre-trained CatBoost model (cached - loads only once)"""
    model = CatBoostClassifier()
    model.load_model('catboost_model.cbm')
    return model

@st.cache_data
def load_feature_info():
    """Load feature configuration"""
    with open('feature_info.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    """Load model metrics"""
    with open('model_metrics.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_processed_data():
    """Load processed data"""
    df = pd.read_csv('processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    return pd.read_csv('feature_importance.csv')

@st.cache_data
def load_classification_report():
    """Load classification report from training"""
    with open('classification_report.json', 'r') as f:
        return json.load(f)


# FEATURE ENGINEERING FOR NEW PREDICTIONS

def create_features_for_prediction(input_data, historical_data, feature_info):
    """Create features for a new prediction based on input data"""
    commodity_hist = historical_data[historical_data['commodity'] == input_data['commodity']]
    
    if len(commodity_hist) == 0:
        return None, "No historical data found for this commodity"
    
    features = {}
    
    # Categorical features
    features['admin1'] = input_data['province']
    features['admin2'] = input_data.get('district', commodity_hist['admin2'].mode().iloc[0])
    features['market'] = input_data['market']
    features['category'] = input_data['category']
    features['commodity'] = input_data['commodity']
    
    # Temporal features
    features['year'] = input_data['year']
    features['month'] = input_data['month']
    features['quarter'] = (input_data['month'] - 1) // 3 + 1
    features['is_maha_season'] = 1 if input_data['month'] in [10, 11, 12, 1, 2] else 0
    features['is_yala_season'] = 1 if input_data['month'] in [5, 6, 7, 8, 9] else 0
    features['is_festive_period'] = 1 if input_data['month'] in [4, 12] else 0
    
    # Price features
    features['price'] = input_data['current_price']
    
    # Derived features from historical data
    recent_hist = commodity_hist.tail(6)
    features['price_lag_1'] = input_data['previous_price']
    features['price_change_lag_1'] = ((input_data['current_price'] - input_data['previous_price']) / 
                                       input_data['previous_price'] * 100) if input_data['previous_price'] > 0 else 0
    
    features['rolling_mean_3'] = recent_hist['price'].tail(3).mean() if len(recent_hist) >= 3 else input_data['current_price']
    features['rolling_std_3'] = recent_hist['price'].tail(3).std() if len(recent_hist) >= 3 else 0
    features['rolling_volatility_3'] = recent_hist['price_change_pct'].abs().tail(3).mean() if len(recent_hist) >= 3 else 0
    features['price_momentum'] = ((input_data['current_price'] - features['rolling_mean_3']) / 
                                   (features['rolling_mean_3'] + 0.001) * 100)
    
    # Z-score
    rolling_mean_6 = recent_hist['price'].mean() if len(recent_hist) > 0 else input_data['current_price']
    rolling_std_6 = recent_hist['price'].std() if len(recent_hist) > 0 else 1
    features['z_score'] = (input_data['current_price'] - rolling_mean_6) / (rolling_std_6 + 0.001)
    
    # Annual average comparison
    annual_avg = commodity_hist[commodity_hist['year'] == input_data['year']]['price'].mean()
    if pd.isna(annual_avg):
        annual_avg = commodity_hist['price'].mean()
    features['price_vs_annual_avg'] = (input_data['current_price'] - annual_avg) / (annual_avg + 0.001) * 100
    
    # Price percentile
    features['price_percentile'] = (commodity_hist['price'] < input_data['current_price']).mean()
    
    # Geographic features
    market_data = commodity_hist[commodity_hist['market'] == input_data['market']]
    if len(market_data) > 0:
        features['distance_from_colombo'] = market_data['distance_from_colombo'].iloc[0]
    else:
        features['distance_from_colombo'] = 50
    
    features['is_conflict_region'] = 1 if input_data['province'] in ['Northern', 'Eastern'] else 0
    
    # Essential commodity flag
    essential = ['Rice (red nadu)', 'Rice (white)', 'Rice (medium grain)', 'Wheat flour', 
                 'Potatoes (local)', 'Potatoes (imported)', 'Coconut', 'Onions (red)', 'Lentils']
    features['is_essential'] = 1 if input_data['commodity'] in essential else 0
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features])
    feature_df = feature_df[feature_info['all_features']]
    
    return feature_df, None


# PLOTLY THEME

def get_plotly_template():
    """Return modern light plotly template"""
    return {
        'layout': {
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': 'rgba(248, 249, 251, 0.5)',
            'font': {'color': '#2d3748', 'family': 'Inter'},
            'xaxis': {'gridcolor': 'rgba(0, 0, 0, 0.05)', 'linecolor': 'rgba(0, 0, 0, 0.1)'},
            'yaxis': {'gridcolor': 'rgba(0, 0, 0, 0.05)', 'linecolor': 'rgba(0, 0, 0, 0.1)'},
        }
    }


# MAIN APP

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Sri Lanka Food Price Volatility Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for required files
    missing_files = check_required_files()
    if missing_files:
        st.error(f"""
        ### Required Files Missing!
        
        The following files are missing: **{', '.join(missing_files)}**
        
        Please run the training script first:
        ```bash
        python catboost_train_model.py
        ```
        """)
        return
    
    # Load everything (cached - only loads once)
    try:
        model = load_model()
        feature_info = load_feature_info()
        metrics = load_metrics()
        df = load_processed_data()
        feat_imp = load_feature_importance()
        class_report = load_classification_report()
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar navigation with radio buttons
    st.sidebar.header("Navigation")
    
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Model Performance", "Feature Importance", "SHAP Analysis", "Make Predictions", "Data Explorer"],
        label_visibility="collapsed"
    )
    
    
    # PAGE: OVERVIEW
    
    if page == "Overview":
        st.header("Dataset & Model Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Volatile Records", f"{df['high_volatility'].sum():,}")
        with col3:
            st.metric("Commodities", f"{df['commodity'].nunique()}")
        with col4:
            st.metric("Markets", f"{df['market'].nunique()}")
        
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            test_m = metrics['test']
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Test Accuracy | {test_m['accuracy']:.1%} |
            | Test F1-Score | {test_m['f1']:.4f} |
            | Test ROC-AUC | {test_m['roc_auc']:.4f} |
            | Features | {len(feature_info['all_features'])} |
            """)
        
        with col2:
            st.subheader("Volatility by Category")
            cat_vol = df.groupby('category')['high_volatility'].mean() * 100
            fig = px.bar(
                x=cat_vol.values, 
                y=cat_vol.index, 
                orientation='h',
                color=cat_vol.values,
                color_continuous_scale='Purples',
                template=get_plotly_template()
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    
    # PAGE: MODEL PERFORMANCE
    
    elif page == "Model Performance":
        st.header("Model Performance")
        
        test_m = metrics['test']
        train_m = metrics['train']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{test_m['accuracy']:.1%}")
        with col2:
            st.metric("Test F1-Score", f"{test_m['f1']:.3f}")
        with col3:
            st.metric("Test ROC-AUC", f"{test_m['roc_auc']:.3f}")
        
        # Overfitting check
        gap = train_m['accuracy'] - test_m['accuracy']
        if gap < 0.05:
            st.success(f"Model generalization is EXCELLENT (Train-Test gap: {gap:.4f})")
        elif gap < 0.10:
            st.warning(f"Slight overfitting detected (Train-Test gap: {gap:.4f})")
        else:
            st.error(f"Overfitting detected (Train-Test gap: {gap:.4f})")
        
        
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Train vs Test Metrics")
            train_image_path = 'metrics_comparison.png'
            if os.path.exists(train_image_path):
                st.image(train_image_path, use_container_width=True)
            else:
                st.warning("Train vs Test Metrics image not found")
        
        with col2:
            st.subheader("Confusion Matrix")
            confusion_image_path = 'confusion_matrix.png'
            if os.path.exists(confusion_image_path):
                st.image(confusion_image_path, use_container_width=True)
            else:
                st.warning("Confusion Matrix image not found")
        
        
        
        # ROC and PR curves
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            roc_image_path = 'roc_curve.png'
            if os.path.exists(roc_image_path):
                st.image(roc_image_path, use_container_width=True)
            else:
                st.warning("ROC curve image not found")
        
        with col2:
            st.subheader("Precision-Recall Curve")
            pr_image_path = 'precision_recall_curve.png'
            if os.path.exists(pr_image_path):
                st.image(pr_image_path, use_container_width=True)
            else:
                st.warning("Precision-Recall curve image not found")
        
        
        # Classification Report
        st.subheader("Classification Report (Test Set)")
        stable_metrics = class_report['Stable']
        volatile_metrics = class_report['Volatile']
        accuracy = class_report['accuracy']
        
        st.markdown(f"""
        | Class | Precision | Recall | F1-Score | Support |
        |-------|-----------|--------|----------|---------|
        | **Stable** | {stable_metrics['precision']:.2f} | {stable_metrics['recall']:.2f} | {stable_metrics['f1-score']:.2f} | {int(stable_metrics['support'])} |
        | **Volatile** | {volatile_metrics['precision']:.2f} | {volatile_metrics['recall']:.2f} | {volatile_metrics['f1-score']:.2f} | {int(volatile_metrics['support'])} |
        
        **Overall Accuracy:** {accuracy:.2f}
        """)
    
    
    # PAGE: FEATURE ANALYSIS
    
    elif page == "Feature Importance":
        st.header("Feature Importance")
        
        fig = px.bar(
            feat_imp.head(15), 
            x='importance', 
            y='feature', 
            orientation='h',
            color='importance',
            color_continuous_scale='Viridis',
            template=get_plotly_template()
        )
        fig.update_layout(height=500, showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(feat_imp, use_container_width=True)
    
    
    # PAGE: SHAP ANALYSIS
    
    elif page == "SHAP Analysis":
        st.header("SHAP Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 15 Features by Importance")
            fig = px.bar(
                feat_imp.head(15),
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Plasma',
                template=get_plotly_template()
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Key Insights")
            top_features = feat_imp.head(5)['feature'].tolist()
            st.markdown(f"""
            **Top 5 Most Important Features:**
            
            1. **{top_features[0]}** - Most influential
            2. **{top_features[1]}**
            3. **{top_features[2]}**
            4. **{top_features[3]}**
            5. **{top_features[4]}**
            """)
        
        
        
        # Interactive Feature Impact
        st.subheader("Explore Individual Feature Impact")
        selected_feature = st.selectbox(
            "Select a feature to explore:",
            feat_imp['feature'].tolist()
        )
        
        feature_rank = feat_imp[feat_imp['feature'] == selected_feature].index[0] + 1
        feature_importance = feat_imp[feat_imp['feature'] == selected_feature]['importance'].values[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Feature Rank", f"#{feature_rank} of {len(feat_imp)}")
        with col2:
            st.metric("Importance Score", f"{feature_importance:.2f}")
        with col3:
            impact = "Strong" if feature_importance > 10 else "Moderate" if feature_importance > 5 else "Mild"
            st.metric("Impact Level", impact)
        
        # Show feature distribution
        if selected_feature in df.columns:
            st.markdown(f"**Distribution of `{selected_feature}` by Volatility Class:**")
            fig = px.histogram(
                df,
                x=selected_feature,
                color='high_volatility',
                color_discrete_map={0: '#a78bfa', 1: '#ef4444'},
                labels={'high_volatility': 'Volatile'},
                title=f"Distribution of {selected_feature}",
                marginal='box',
                barmode='overlay',
                opacity=0.7,
                template=get_plotly_template()
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    
    # PAGE: MAKE PREDICTIONS
    
    elif page == "Make Predictions":
        st.header("Predict Price Volatility")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_commodity = st.selectbox("Commodity", sorted(df['commodity'].unique()))
            commodity_cat = df[df['commodity'] == input_commodity]['category'].mode()
            default_cat = commodity_cat.iloc[0] if len(commodity_cat) > 0 else df['category'].mode().iloc[0]
            input_category = st.selectbox("Category", sorted(df['category'].unique()),
                                         index=list(sorted(df['category'].unique())).index(default_cat))
            input_market = st.selectbox("Market", sorted(df['market'].unique()))
        
        with col2:
            input_province = st.selectbox("Province", sorted(df['admin1'].unique()))
            input_year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
            input_month = st.slider("Month", 1, 12, 6)
        
        with col3:
            commodity_prices = df[df['commodity'] == input_commodity]['price']
            default_price = commodity_prices.median() if len(commodity_prices) > 0 else 500.0
            input_current_price = st.number_input("Current Price (LKR)", min_value=1.0, 
                                                 value=float(default_price), step=10.0)
            input_previous_price = st.number_input("Previous Price (LKR)", min_value=1.0, 
                                                  value=float(default_price * 0.95), step=10.0)
        
        if st.button("Predict Volatility", use_container_width=True, type="primary"):
            input_data = {
                'commodity': input_commodity,
                'category': input_category,
                'market': input_market,
                'province': input_province,
                'year': input_year,
                'month': input_month,
                'current_price': input_current_price,
                'previous_price': input_previous_price
            }
            
            feature_df, error = create_features_for_prediction(input_data, df, feature_info)
            
            if error:
                st.error(f"Error: {error}")
            else:
                prediction = model.predict(feature_df)[0]
                probability = model.predict_proba(feature_df)[0][1]
                price_change = ((input_current_price - input_previous_price) / input_previous_price * 100)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price Change", f"{price_change:.1f}%")
                with col2:
                    st.metric("Volatility Probability", f"{probability:.1%}")
                with col3:
                    st.metric("Prediction", "VOLATILE" if prediction == 1 else "STABLE")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card volatile">
                        <h2 style="color: #ef4444; margin: 0;">⚠️ HIGH VOLATILITY PREDICTED</h2>
                        <p style="font-size: 1.2rem; margin-top: 1rem; color: #2d3748;">Probability: {probability:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card stable">
                        <h2 style="color: #10b981; margin: 0;">✓ STABLE PRICE EXPECTED</h2>
                        <p style="font-size: 1.2rem; margin-top: 1rem; color: #2d3748;">Volatility Probability: {probability:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    
    # PAGE: DATA EXPLORER
    
    elif page == "Data Explorer":
        st.header("Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_years = st.multiselect(
                "Year",
                sorted(df['year'].unique()),
                default=sorted(df['year'].unique())[-3:]
            )
        
        with col2:
            selected_categories = st.multiselect(
                "Category",
                df['category'].unique(),
                default=list(df['category'].unique())[:3]
            )
        
        filtered = df[(df['year'].isin(selected_years)) & (df['category'].isin(selected_categories))]
        
        st.markdown(f"**Showing {len(filtered):,} records**")
        
        display_cols = ['date', 'market', 'admin1', 'commodity', 'price', 'high_volatility']
        st.dataframe(filtered[display_cols].head(500), use_container_width=True)
    
    # Footer
    
    st.markdown("""
    <div class="footer">
        MSc AI - ML Assignment | CatBoost
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()