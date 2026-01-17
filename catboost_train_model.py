
#################################### IMPORTS ##########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from catboost import CatBoostClassifier, Pool
import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

######################## 1. DATA LOADING & INITIAL EXPLORATION ########################

print("1. DATA LOADING & INITIAL EXPLORATION")

# Load data (skip the second header row which contains metadata)
df = pd.read_csv('wfp_food_prices_lka.csv', skiprows=[1])

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Basic info
print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nBasic Statistics")
print(df[['price', 'usdprice']].describe())


######################## 2. DATA PREPROCESSING ########################

print("2. DATA PREPROCESSING")

# Handle missing values
print(f"\nMissing values before cleaning: {df.isnull().sum().sum()}")

# Drop rows with missing geographic data (only 111 rows - 0.46%)
df_clean = df.dropna(subset=['admin1', 'admin2', 'latitude', 'longitude'])
print(f"Rows after removing missing geographic data: {len(df_clean)}")

# Filter to retail prices only (98% of data) for consistency
df_clean = df_clean[df_clean['pricetype'] == 'Retail']
print(f"Rows after filtering to retail prices: {len(df_clean)}")

# Remove "National Average" market (aggregated data)
df_clean = df_clean[df_clean['market'] != 'National Average']
print(f"Rows after removing National Average: {len(df_clean)}")

# Sort by commodity, market, and date for proper lag calculation
df_clean = df_clean.sort_values(['commodity', 'market', 'date']).reset_index(drop=True)
print(f"Final dataset size: {len(df_clean)}")


######################### 3. TARGET VARIABLE CREATION #########################

print("3. TARGET VARIABLE CREATION")

# Calculate price change percentage (month-over-month)
df_clean['price_change_pct'] = df_clean.groupby(['commodity', 'market'])['price'].pct_change() * 100

# Define target: High volatility = |price change| > 20%
VOLATILITY_THRESHOLD = 20
df_clean['high_volatility'] = (abs(df_clean['price_change_pct']) > VOLATILITY_THRESHOLD).astype(int)

# Remove first observation per group (no previous price to compare)
df_clean = df_clean.dropna(subset=['price_change_pct'])

print(f"\nVolatility Threshold: {VOLATILITY_THRESHOLD}%")
print(f"\nTarget Variable Distribution ---")
print(df_clean['high_volatility'].value_counts())
print(f"\nClass Balance:")
print(f"  Stable (0): {(df_clean['high_volatility']==0).mean()*100:.1f}%")
print(f"  Volatile (1): {(df_clean['high_volatility']==1).mean()*100:.1f}%")


######################### 4. FEATURE ENGINEERING #########################

print("4. FEATURE ENGINEERING")

df_features = df_clean.copy()

# 4.1 Temporal Features 
print("\nCreating temporal features...")
df_features['year'] = df_features['date'].dt.year
df_features['month'] = df_features['date'].dt.month
df_features['quarter'] = df_features['date'].dt.quarter
df_features['day_of_year'] = df_features['date'].dt.dayofyear

# Sri Lanka specific seasonal features
df_features['is_maha_season'] = df_features['month'].isin([10, 11, 12, 1, 2]).astype(int)
df_features['is_yala_season'] = df_features['month'].isin([5, 6, 7, 8, 9]).astype(int)
df_features['is_festive_period'] = df_features['month'].isin([4, 12]).astype(int)
df_features['is_sw_monsoon'] = df_features['month'].isin([5, 6, 7, 8, 9]).astype(int)
df_features['is_ne_monsoon'] = df_features['month'].isin([12, 1, 2]).astype(int)

min_date = df_features['date'].min()
df_features['months_since_start'] = ((df_features['date'] - min_date).dt.days / 30.44).astype(int)

# 4.2 Lag Features (Critical for Volatility Prediction)
print("Creating lag features...")

def create_lag_features(group):
    """Create lag and rolling features for each commodity-market combination"""
    group = group.sort_values('date')
    
    # Price lags
    group['price_lag_1'] = group['price'].shift(1)
    group['price_lag_2'] = group['price'].shift(2)
    group['price_lag_3'] = group['price'].shift(3)
    
    # Price change lag (previous volatility)
    group['price_change_lag_1'] = group['price_change_pct'].shift(1)
    group['price_change_lag_2'] = group['price_change_pct'].shift(2)
    
    # Rolling statistics
    group['rolling_mean_3'] = group['price'].rolling(window=3, min_periods=1).mean()
    group['rolling_mean_6'] = group['price'].rolling(window=6, min_periods=1).mean()
    group['rolling_std_3'] = group['price'].rolling(window=3, min_periods=1).std()
    group['rolling_std_6'] = group['price'].rolling(window=6, min_periods=1).std()
    
    # Rolling volatility (historical volatility indicator)
    group['rolling_volatility_3'] = group['price_change_pct'].abs().rolling(window=3, min_periods=1).mean()
    
    # Price momentum (current vs 3-month average)
    group['price_momentum'] = (group['price'] - group['rolling_mean_3']) / (group['rolling_mean_3'] + 0.001) * 100
    
    # Trend direction
    group['price_trend'] = group['price'].diff(3)
    
    return group

df_features = df_features.groupby(['commodity', 'market'], group_keys=False).apply(create_lag_features)

# 4.3 Price-Based Features
print("Creating price-based features...")

# Z-score (standardized deviation from rolling mean)
df_features['z_score'] = (df_features['price'] - df_features['rolling_mean_6']) / (df_features['rolling_std_6'] + 0.001)

# Price vs annual average
annual_avg = df_features.groupby(['commodity', 'year'])['price'].transform('mean')
df_features['price_vs_annual_avg'] = (df_features['price'] - annual_avg) / (annual_avg + 0.001) * 100

# Price vs category average
category_avg = df_features.groupby(['category', 'year', 'month'])['price'].transform('mean')
df_features['price_vs_category_avg'] = (df_features['price'] - category_avg) / (category_avg + 0.001) * 100

# Price percentile within commodity
df_features['price_percentile'] = df_features.groupby('commodity')['price'].transform(
    lambda x: x.rank(pct=True)
)

# 4.4 Geographic Features
print("Creating geographic features...")

colombo_lat, colombo_lon = 6.93, 79.85

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df_features['distance_from_colombo'] = haversine_distance(
    df_features['latitude'], df_features['longitude'],
    colombo_lat, colombo_lon
)

df_features['is_conflict_region'] = df_features['admin1'].isin(['Northern', 'Eastern']).astype(int)

# 4.5 Commodity Features
print("Creating commodity features...")

essential_commodities = ['Rice (red nadu)', 'Rice (white)', 'Rice (medium grain)', 
                         'Rice (red)', 'Rice (long grain)', 'Wheat flour', 
                         'Potatoes (local)', 'Potatoes (imported)', 'Coconut', 
                         'Onions (red)', 'Lentils', 'Sugar']
df_features['is_essential'] = df_features['commodity'].isin(essential_commodities).astype(int)

# 4.6 Handle Missing Values from Lag Features
print("\nHandling missing values from lag features...")

lag_columns = ['price_lag_1', 'price_lag_2', 'price_lag_3', 
               'price_change_lag_1', 'price_change_lag_2',
               'rolling_std_3', 'rolling_std_6', 'rolling_volatility_3',
               'price_momentum', 'price_trend', 'z_score']

for col in lag_columns:
    if col in df_features.columns:
        df_features[col] = df_features[col].fillna(df_features[col].median())

df_features = df_features.replace([np.inf, -np.inf], np.nan)
df_features = df_features.fillna(df_features.median(numeric_only=True))

print(f"\nFeatures created: {len(df_features.columns)} columns")
print(f"Remaining missing values: {df_features.isnull().sum().sum()}")


######################### 5. PREPARE DATA FOR CATBOOST #########################

print("5. PREPARE DATA FOR CATBOOST")

# Define feature columns - REDUCED to prevent overfitting
# Remove highly correlated and redundant features
categorical_features = ['admin1', 'admin2', 'market', 'category', 'commodity']

numerical_features = [
    # Temporal (reduced)
    'year', 'month', 'quarter',
    'is_maha_season', 'is_yala_season', 'is_festive_period',
    
    # Lag features (most important, but reduced)
    'price_lag_1',
    'price_change_lag_1',
    'rolling_mean_3',
    'rolling_std_3',
    'rolling_volatility_3', 
    'price_momentum',
    
    # Price-based (reduced)
    'price', 
    'z_score',
    'price_vs_annual_avg',
    'price_percentile',
    
    # Geographic
    'distance_from_colombo', 
    'is_conflict_region',
    
    # Commodity
    'is_essential'
]

all_features = categorical_features + numerical_features

# Get indices of categorical features for CatBoost
cat_feature_indices = [all_features.index(f) for f in categorical_features]

print(f"\nTotal features: {len(all_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"\nFeatures used: {all_features}")

# Prepare X and y
X = df_features[all_features].copy()
y = df_features['high_volatility'].copy()


# 6. TRAIN/VALIDATION/TEST SPLIT (Using train_test_split)

print("6. TRAIN/VALIDATION/TEST SPLIT (Using train_test_split)")

# Prepare X and y
X = df_features[all_features].copy()
y = df_features['high_volatility'].copy()

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution
)

# Second split: 75% train, 25% val (of the 80% = 60% train, 20% val overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=0.25, 
    random_state=42,
    stratify=y_train_val  # Maintain class distribution
)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nClass Distribution")
print(f"Training - Volatile: {y_train.mean()*100:.1f}%")
print(f"Validation - Volatile: {y_val.mean()*100:.1f}%")
print(f"Test - Volatile: {y_test.mean()*100:.1f}%")


######################### 7. MODEL TRAINING - CATBOOST (WITH REGULARIZATION) #########################

print("7. MODEL TRAINING - CATBOOST (WITH REGULARIZATION)")

# Create CatBoost Pools
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_feature_indices
)

val_pool = Pool(
    data=X_val,
    label=y_val,
    cat_features=cat_feature_indices
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_feature_indices
)

# Initialize CatBoost with STRONG REGULARIZATION to prevent overfitting
catboost_model = CatBoostClassifier(
    # Tree parameters - MORE REGULARIZED
    iterations=300,               # Further reduced
    learning_rate=0.02,           # Even slower learning
    depth=3,                      # Further reduced from 4 to limit complexity
    
    # STRONGER REGULARIZATION parameters
    l2_leaf_reg=10,               # Increased from 5 (stronger L2 regularization)
    min_data_in_leaf=100,         # Increased from 50 (prevents small leaves)
    random_strength=3,            # More randomization for scoring splits
    
    # Subsampling for regularization
    subsample=0.7,                # Use 70% of data per tree (more aggressive)
    bootstrap_type='MVS',         # MVS supports subsample
    
    # Feature subsampling
    rsm=0.8,                      # Use 80% of features per tree
    
    # Other parameters
    border_count=64,              # Further reduced
    grow_policy='SymmetricTree',
    loss_function='Logloss',
    eval_metric='AUC',
    
    # Class weights for imbalance
    class_weights={0: 1, 1: 1.3},  # Further reduced weight
    
    # Training control
    random_seed=42,
    verbose=100,
    early_stopping_rounds=20,     # More aggressive early stopping
    use_best_model=True
)

print("\nTraining CatBoost model with regularization...")


catboost_model.fit(
    train_pool,
    eval_set=val_pool,
    plot=False
)

print("\nModel training complete!")
print(f"Best iteration: {catboost_model.get_best_iteration()}")
print(f"Best validation AUC: {catboost_model.get_best_score()['validation']['AUC']:.4f}")


######################### 8. MODEL EVALUATION #########################

print("8. MODEL EVALUATION")

# Predictions
y_train_pred = catboost_model.predict(X_train)
y_val_pred = catboost_model.predict(X_val)
y_test_pred = catboost_model.predict(X_test)

y_train_prob = catboost_model.predict_proba(X_train)[:, 1]
y_val_prob = catboost_model.predict_proba(X_val)[:, 1]
y_test_prob = catboost_model.predict_proba(X_test)[:, 1]

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_prob, set_name):
    print(f"\n{set_name} Set Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'pr_auc': pr
    }

train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob, "Test")

# Check for overfitting

print("OVERFITTING CHECK")

train_test_gap = train_metrics['accuracy'] - test_metrics['accuracy']
print(f"Training Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
print(f"Gap:                {train_test_gap:.4f}")

if train_test_gap > 0.1:
    print("WARNING: Model may still be overfitting (gap > 0.1)")
elif train_test_gap > 0.05:
    print("NOTICE: Slight overfitting detected (gap 0.05-0.1)")
else:
    print("GOOD: Model generalization looks healthy (gap < 0.05)")

# Classification Report
print("\nDetailed Classification Report (Test Set)")
print(classification_report(y_test, y_test_pred, target_names=['Stable', 'Volatile']))

# Confusion Matrix
print("\nConfusion Matrix (Test Set)")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")


#########################  9. SAVE MODEL AND ARTIFACTS FOR STREAMLIT #########################

print("9. SAVE MODEL AND ARTIFACTS FOR STREAMLIT")

# Save the trained model
model_path = 'catboost_model.cbm'
catboost_model.save_model(model_path)
print(f"Model saved to: {model_path}")

# Save feature information
feature_info = {
    'all_features': all_features,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'cat_feature_indices': cat_feature_indices,
    'volatility_threshold': VOLATILITY_THRESHOLD
}

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print("Feature info saved to: feature_info.json")

# Save metrics
all_metrics = {
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics
}

with open('model_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("Metrics saved to: model_metrics.json")

# Save classification report for Streamlit
classification_rep = classification_report(y_test, y_test_pred, target_names=['Stable', 'Volatile'], output_dict=True)
with open('classification_report.json', 'w') as f:
    json.dump(classification_rep, f, indent=2)
print("Classification report saved to: classification_report.json")

# Save processed data for reference
df_features.to_csv('processed_data.csv', index=False)
print("Processed data saved to: processed_data.csv")

# Save feature importance
feature_importance = catboost_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
importance_df.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to: feature_importance.csv")


#########################  10. VISUALIZATION - SAVE AS SEPARATE IMAGES #########################

print("10. VISUALIZATION - SAVE AS SEPARATE IMAGES")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable', 'Volatile'],
            yticklabels=['Stable', 'Volatile'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {test_metrics["roc_auc"]:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve.png")

# 3. Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
plt.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {test_metrics["pr_auc"]:.3f})')
plt.axhline(y=y_test.mean(), color='navy', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_curve.png")

# 4. Feature Importance (Top 15)
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'], color='teal')
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance (CatBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# 5. Prediction Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test_prob[y_test == 0], bins=50, alpha=0.5, label='Stable', color='blue')
plt.hist(y_test_prob[y_test == 1], bins=50, alpha=0.5, label='Volatile', color='red')
plt.xlabel('Predicted Probability of High Volatility')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
plt.tight_layout()
plt.savefig('prediction_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: prediction_probability_distribution.png")

# 6. Metrics Comparison (Train vs Test)
plt.figure(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_vals = [train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], 
              train_metrics['f1'], train_metrics['roc_auc']]
test_vals = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], 
             test_metrics['f1'], test_metrics['roc_auc']]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = plt.bar(x - width/2, train_vals, width, label='Train', color='steelblue')
bars2 = plt.bar(x + width/2, test_vals, width, label='Test', color='coral')

plt.ylabel('Score')
plt.title('Model Performance Comparison (Train vs Test)')
plt.xticks(x, metrics_names, rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: metrics_comparison.png")

# Also save the combined visualization (optional - for backward compatibility)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Recreate all plots in the combined figure
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Stable', 'Volatile'],
            yticklabels=['Stable', 'Volatile'])
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix (Test Set)')

ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {test_metrics["roc_auc"]:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve (Test Set)')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
ax3.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {test_metrics["pr_auc"]:.3f})')
ax3.axhline(y=y_test.mean(), color='navy', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve (Test Set)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 0]
top_features = importance_df.head(15)
ax4.barh(top_features['feature'], top_features['importance'], color='teal')
ax4.set_xlabel('Feature Importance')
ax4.set_title('Top 15 Feature Importance (CatBoost)')
ax4.invert_yaxis()

ax5 = axes[1, 1]
ax5.hist(y_test_prob[y_test == 0], bins=50, alpha=0.5, label='Stable', color='blue')
ax5.hist(y_test_prob[y_test == 1], bins=50, alpha=0.5, label='Volatile', color='red')
ax5.set_xlabel('Predicted Probability of High Volatility')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Probability Distribution')
ax5.legend()
ax5.axvline(x=0.5, color='black', linestyle='--', label='Threshold')

ax6 = axes[1, 2]
x = np.arange(len(metrics_names))
bars1 = ax6.bar(x - width/2, train_vals, width, label='Train', color='steelblue')
bars2 = ax6.bar(x + width/2, test_vals, width, label='Test', color='coral')
ax6.set_ylabel('Score')
ax6.set_title('Model Performance Comparison (Train vs Test)')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names, rotation=45, ha='right')
ax6.legend()
ax6.set_ylim(0, 1)
ax6.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax6.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax6.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('catboost_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: catboost_evaluation_results.png (combined)")

print("\nAll visualization files saved successfully!")


#########################  11. EXPLAINABILITY - SHAP ANALYSIS #########################

print("11. EXPLAINABILITY - SHAP ANALYSIS")

# Sample for SHAP
sample_size = min(1000, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = X_test.iloc[sample_indices]

print(f"\nCalculating SHAP values for {sample_size} samples...")

# Get SHAP values
shap_values = catboost_model.get_feature_importance(
    Pool(X_sample, cat_features=cat_feature_indices),
    type='ShapValues'
)

shap_values_no_bias = shap_values[:, :-1]
print("SHAP values calculated!")

# SHAP Summary Plot
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values_no_bias, 
    X_sample, 
    feature_names=all_features,
    show=False,
    max_display=15
)
plt.title('SHAP Feature Importance Summary', fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP summary plot saved as 'shap_summary_plot.png'")


#########################  12. FINAL SUMMARY #########################

print("12. FINAL SUMMARY")

print(f"""
MODEL PERFORMANCE SUMMARY:
==========================
                Train     Test      Gap
Accuracy:       {train_metrics['accuracy']:.4f}    {test_metrics['accuracy']:.4f}    {train_metrics['accuracy']-test_metrics['accuracy']:.4f}
F1-Score:       {train_metrics['f1']:.4f}    {test_metrics['f1']:.4f}    {train_metrics['f1']-test_metrics['f1']:.4f}
ROC-AUC:        {train_metrics['roc_auc']:.4f}    {test_metrics['roc_auc']:.4f}    {train_metrics['roc_auc']-test_metrics['roc_auc']:.4f}

Best Iteration: {catboost_model.get_best_iteration()}
""")

print("TRAINING COMPLETE! Now run the Streamlit app.")
