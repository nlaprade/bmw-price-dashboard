
"""
Author: Nicholas Laprade
Date: 2025-10-08
Topic: BMW Worldwide Sales Records (Kaggle Dataset) - Preprocessing/Data Exploration
Dataset: https://www.kaggle.com/datasets/wardabilal/bmw-cars-dataset-analysis-with-visualizations
"""

import os
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "bmw.csv")
graphs_dir = os.path.join(BASE_DIR, "graphs")

# Read CSV
df = pd.read_csv(data_path)

#print(df.head())
#print(df.info())

"""
RangeIndex: 10781 entries, 0 to 10780
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   model         10781 non-null  object  Name of model
 1   year          10781 non-null  int64   Year of manufacture
 2   price         10781 non-null  int64   GBP selling price
 3   transmission  10781 non-null  object  Type of transmission
 4   mileage       10781 non-null  int64   Total miles traveled
 5   fuelType      10781 non-null  object  Fuel type
 6   tax           10781 non-null  int64   Road tax per year in GBP
 7   mpg           10781 non-null  float64 Miles per gallon
 8   engineSize    10781 non-null  float64 Engine capactity in litres
dtypes: float64(2), int64(4), object(3)
memory usage: 758.2+ KB
None
"""

#df.groupby('model')['price'].describe()

"""
           count          mean           std  ...      50%       75%       max
model                                         ...                             
1 Series  1969.0  15821.670391   5957.713264  ...  14800.0  19761.00   38555.0
2 Series  1229.0  19539.371847   6080.250595  ...  19980.0  22875.00  123456.0
3 Series  2443.0  19880.857962   8394.421652  ...  18299.0  25938.00   71990.0
4 Series   995.0  22498.418090   6401.674559  ...  21975.0  26775.00   48155.0
5 Series  1056.0  22537.428030   8729.997867  ...  22996.5  29457.50   54845.0
6 Series   108.0  24354.370370   8590.169421  ...  22854.0  29904.00   52500.0
7 Series   106.0  36934.320755  14491.225974  ...  37224.5  49000.00   59995.0
8 Series    39.0  63997.794872   8307.191251  ...  61898.0  67444.50   88980.0
M2          21.0  43140.333333   6197.695921  ...  44990.0  46995.00   49999.0
M3          27.0  30229.777778  12689.168538  ...  31000.0  37780.50   62995.0
M4         125.0  43274.232000  10920.580243  ...  43480.0  48630.00   99950.0
M5          29.0  57760.000000  19403.790552  ...  63980.0  70000.00   89900.0
M6           8.0  32190.000000   9807.652901  ...  33823.5  36172.25   46995.0
X1         804.0  19816.564677   5892.125822  ...  19042.5  24481.25   37320.0
X2         288.0  28486.663194   4108.537160  ...  27989.5  30452.50   44980.0
X3         551.0  27758.310345  11152.770424  ...  29357.0  34995.00   69991.0
X4         179.0  32721.335196   9870.424905  ...  34995.0  40008.00   60995.0
X5         468.0  39651.196581  13222.679569  ...  44151.0  49990.00   73000.0
X6         106.0  43838.613208  15990.381070  ...  40207.0  58357.50   69995.0
X7          55.0  69842.763636   5101.521240  ...  69146.0  73970.00   79566.0
Z3           7.0   5826.428571   4109.930772  ...   3995.0   4972.50   14995.0
Z4         108.0  27001.935185  10619.681063  ...  30995.0  33885.00   50800.0
i3          43.0  18667.116279   2663.548172  ...  19300.0  19999.00   23751.0
i8          17.0  57012.588235  11135.665321  ...  57870.0  64980.00   74226.0
"""

# Feature Engineering Section

# Using median price per model to assign tiers
model_medians = df.groupby('model')['price'].median()
Q1 = model_medians.quantile(0.25)
Q3 = model_medians.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_medians = model_medians[(model_medians >= lower_bound) & (model_medians <= upper_bound)]
q1 = filtered_medians.quantile(0.33)
q2 = filtered_medians.quantile(0.66)

def assign_tier(price):
    if price <= q1:
        return 'Entry'
    elif price <= q2:
        return 'Mid'
    else:
        return 'High'

df['model_tier'] = df['price'].apply(assign_tier)

def flag_model_outliers(group):
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ~group['price'].between(lower, upper)

df['is_outlier'] = df.groupby('model', group_keys=False).apply(flag_model_outliers)

#print(df['is_outlier'].value_counts())

# Rare Models < 100
model_counts = df['model'].value_counts()
rare_models = model_counts[model_counts < 100].index

df['rare_model'] = df['model'].isin(rare_models)

#print(df['model'].unique())
"""
[' 5 Series' ' 6 Series' ' 1 Series' ' 7 Series' ' 2 Series' ' 4 Series'
 ' X3' ' 3 Series' ' X5' ' X4' ' i3' ' X1' ' M4' ' X2' ' X6' ' 8 Series'
 ' Z4' ' X7' ' M5' ' i8' ' M2' ' M3' ' M6' ' Z3']
"""

#print(df['model'].value_counts())
"""
3 Series    2443
1 Series    1969
2 Series    1229
5 Series    1056
4 Series     995
X1           804
X3           551
X5           468
X2           288
X4           179
M4           125
6 Series     108
Z4           108
7 Series     106
X6           106
X7            55
i3            43
8 Series      39
M5            29
M3            27
M2            21
i8            17
M6             8
Z3             7
Name: count, dtype: int64
"""

# OneHotEncoding

# Define categorical columns
cat_columns = ['model', 'transmission', 'fuelType', 'model_tier']
df['rare_model'] = df['rare_model'].astype(int)
df['is_outlier'] = df['is_outlier'].astype(int)
# Create and fit the encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical columns
encoded_array = encoder.fit_transform(df[cat_columns])

# Get the new column names from the encoder
encoded_cols = encoder.get_feature_names_out(cat_columns)

# Convert the encoded array to a DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Drop original categorical columns and concatenate the encoded ones
df_encoded = pd.concat([df.drop(columns=cat_columns), encoded_df], axis=1)

# Boxplot: Price vs Model
plt.figure(figsize=(10, 6))
sns.boxplot(x='model', y='price', data=df)
plt.xticks(rotation=45)
plt.title('Price Distribution by Model')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "price_model_boxplot.png"), bbox_inches='tight')

plt.show()

# Boxplot: Price vs Color
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission', y='price', data=df)
plt.xticks(rotation=45)
plt.title('Price Distribution by Color')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "price_color_boxplot.png"), bbox_inches='tight')

plt.show()

# Boxplot: Price vs Fuel Type
plt.figure(figsize=(8, 6))
sns.boxplot(x='fuelType', y='price', data=df)
plt.xticks(rotation=45)
plt.title('Price Distribution by Fuel Type')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "price_fueltype_boxplot.png"), bbox_inches='tight')

plt.show()

"""
Explaining Boxplots:
    Bottom Whisker: Extends to the minimum value within 1.5 * IQR below Q1
    Bottom of the Box: Q1 (25th percentile)
    Middle Line: Median/Q2 (50th percentile)
    Top of the Box: Q3 (75th percentile)
    Top Whisker: Extends to the maximum value within 1.5 * IQR above Q3
    Dots: Outliers
"""

# Split features and target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary of models to benchmark
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "R²": round(r2, 4),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2)
    })


results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
print(results_df)

"""
4            XGBoost  0.9645   4573107.50  1407.07
1   Ridge Regression  0.9330   8639322.43  1991.52
0  Linear Regression  0.9326   8683479.68  1996.62
2   Lasso Regression  0.9326   8680951.12  1996.60
3      Random Forest  0.9135  11144271.88  2423.62
"""

# Plotting models r2 score and mae
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# R² plot
ax[0].bar(results_df['Model'], results_df['R²'], color='skyblue')
ax[0].set_title("R² Score by Model")
ax[0].set_ylabel("R²")
ax[0].set_xticks(range(len(results_df['Model'])))
ax[0].set_xticklabels(results_df['Model'], rotation=45)

# MAE plot
ax[1].bar(results_df['Model'], results_df['MAE'], color='salmon')
ax[1].set_title("MAE by Model")
ax[1].set_ylabel("Mean Absolute Error")
ax[1].set_xticks(range(len(results_df['Model'])))
ax[1].set_xticklabels(results_df['Model'], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "price_fueltype_boxplot.png"), bbox_inches='tight')
plt.show()

# SHAP
shap.initjs()

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Global importance summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(graphs_dir, "SHAP_summary.png"), bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(os.path.join(graphs_dir, "SHAP_summary_bar.png"), bbox_inches='tight')
plt.close()

"""
SHAP Explanation:
    Color (Red - Blue)
    Red = High feature value
    Blue = Low feature value
    
    X-axis (SHAP value)
    Positive SHAP value = pushes prediction higher
    Negative SHAP value = pushes prediction lower
    
    Example: model_tier_Entry column
    High amount of red dots (high feature values) on the left side (negative SHAP values)
    = When this feature is high, it tends to decrease the predicted price (target)!
"""


