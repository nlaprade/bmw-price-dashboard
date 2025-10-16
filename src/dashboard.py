"""
Author: Nicholas Laprade
Date: 2025-10-12
Finish: 2025-10-16
Topic: BMW Worldwide Sales Records (Kaggle Dataset) - Dashboard
Dataset: https://www.kaggle.com/datasets/wardabilal/bmw-cars-dataset-analysis-with-visualizations
"""

import streamlit as st
import io
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Streamlit Page Config
st.set_page_config(
    page_title="BMW Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)
st.title("BMW Price Prediction Dashboard")
st.markdown("A modular, production-grade dashboard for model benchmarking, SHAP interpretability, and stakeholder-ready reporting.")

# --- Pathing and Data Loading Section ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "bmw.csv")

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = df[df['price'] > 1000]
    return df

df = load_data(data_path)

# Cleaning 0 engineSize values for Petrol + Diesel and replacing by model average
df.loc[
    (df['fuelType'].isin(['Petrol', 'Diesel'])) & (df['engineSize'] == 0),
    'engineSize'
] = np.nan

df['engineSize'] = df.groupby('model')['engineSize'].transform(
    lambda x: x.fillna(x.mean())
)

# --- Preprocessing Section ---

# Clean input values to match training format
for col in ["model", "transmission", "fuelType"]:
    df[col] = df[col].str.strip()

df['mileage'] = np.log1p(df['mileage'])

def preprocess(df):
    cat_cols = ["model", "transmission", "fuelType"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df[cat_cols])

    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    df_encoded = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    X = df_encoded.drop(columns="price")
    y = df_encoded["price"]

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist(), encoder


(X_train, X_test, y_train, y_test), feature_names, encoder = preprocess(df)

# --- Model Training and Evauluation Section ---
@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, X_test, y_test):
    models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=50000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=5, verbose=0, random_state=42
    )
}

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        results.append((name, f"{r2:.4f}", f"{mse:.2f}", f"{mae:.2f}"))

    return models, pd.DataFrame(results, columns=["Model", "R²", "MSE", "MAE"])

models, results_df = train_models(X_train, y_train, X_test, y_test)

# --- Model Comparison Section ---
st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df, width="stretch")

best_model_name = results_df.iloc[results_df["R²"].idxmax()]["Model"]
st.success(f"🏆 Best Model: **{best_model_name}!**")
best_model = models[best_model_name]

default_model_name = results_df.iloc[results_df["R²"].idxmax()]["Model"]
selected_model_name = st.selectbox("🔀 Select model for SHAP analysis", results_df["Model"], index=results_df["Model"].tolist().index(default_model_name))
selected_model = models[selected_model_name]

with st.expander("📘 Dataset Column Description"):
    st.markdown("""
    **Data columns (total 9):**

    | # | Column       | Dtype    | Description                        |
    |---|--------------|----------|------------------------------------|
    | 0 | model        | object   | Name of model                      |
    | 1 | year         | int64    | Year of manufacture                |
    | 2 | price        | int64    | GBP selling price                  |
    | 3 | transmission | object   | Type of transmission               |
    | 4 | mileage      | int64    | Total miles traveled               |
    | 5 | fuelType     | object   | Fuel type                          |
    | 6 | tax          | int64    | Road tax per year in GBP          |
    | 7 | mpg          | float64  | Miles per gallon                   |
    | 8 | engineSize   | float64  | Engine capacity in litres          |

    **Dtypes summary**: `float64(2)`, `int64(4)`, `object(3)`
    """)

# --- Manual Input Section ---
st.subheader("🧮 Predict Price from Manual Input")

# --- Dictionaries for valid model combinations + valid years ---
valid_combinations = {
    "5 Series": {
        "fuelType": ["Diesel", "Petrol", "Hybrid", "Other"],
        "engineSize": [2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.4],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "6 Series": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [2.0, 3.0, 3.4],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "1 Series": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [1.5, 2.0, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "7 Series": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [2.0, 2,5, 3.0, 4.4, 6.6],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "2 Series": {
        "fuelType": ["Diesel", "Petrol", "Hybrid", "Other"],
        "engineSize": [1.5, 2.0, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Automatic"]
    },
    "4 Series": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [2.0, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "X3": {
        "fuelType": ["Diesel", "Petrol", "Hybrid"],
        "engineSize": [2.0, 2.5, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "3 Series": {
        "fuelType": ["Diesel", "Petrol", "Hybrid", "Other"],
        "engineSize": [1.5, 1.6, 2.0, 2.2, 2.5, 2.8, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "X5": {
        "fuelType": ["Diesel", "Petrol", "Hybrid", "Other"],
        "engineSize": [2.0, 3.0, 4.4],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "X4": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [2.0, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "i3": {
        "fuelType": ["Electric", "Hybird", "Other"],
        "engineSize": [0.0, 0.6, 1.0],
        "transmission": ["Automatic"]
    },
    "X1": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [1.5, 2.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "X2": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [1.5, 2.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "X6": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [3.0, 4.4],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "8 Series": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [3.0, 4.4],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "Z4": {
        "fuelType": ["Petrol"],
        "engineSize": [2.0, 2.2, 2.5, 3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "X7": {
        "fuelType": ["Diesel", "Petrol"],
        "engineSize": [3.0],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "M5": {
        "fuelType": ["Petrol"],
        "engineSize": [4.4],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "i8": {
        "fuelType": ["Hybrid", "Other"],
        "engineSize": [1.5],
        "transmission": ["Automatic"]
    },
    "M2": {
        "fuelType": ["Petrol"],
        "engineSize": [3.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "M3": {
        "fuelType": ["Petrol"],
        "engineSize": [3.0, 3.2, 4.0],
        "transmission": ["Automatic", "Manual", "Semi-Auto"]
    },
    "M6": {
        "fuelType": ["Petrol"],
        "engineSize": [4.4, 5.0],
        "transmission": ["Automatic", "Semi-Auto"]
    },
    "Z3": {
        "fuelType": ["Petrol"],
        "engineSize": [1.9, 2.2],
        "transmission": ["Automatic", "Manual"]
    },
}

valid_years_by_model = {
    "5 Series": list(range(1996, 2020 + 1)),
    "6 Series": list(range(2005, 2020 + 1)),
    "1 Series": list(range(2001, 2020 + 1)),
    "7 Series": list(range(2006, 2020 + 1)),
    "2 Series": list(range(2014, 2020 + 1)),
    "4 Series": list(range(2014, 2020 + 1)),
    "X3": list(range(2004, 2020 + 1)),
    "3 Series": list(range(1999, 2020 + 1)),
    "X5": list(range(2005, 2020 + 1)),
    "X4": list(range(2014, 2020 + 1)),
    "i3": list(range(2013, 2020 + 1)),
    "X1": list(range(2010, 2020 + 1)),
    "X2": list(range(2017, 2020 + 1)),
    "X6": list(range(2007, 2020 + 1)),
    "8 Series": list(range(2000, 2020 + 1)),
    "Z4": list(range(2003, 2020 + 1)),
    "X7": list(range(2018, 2020 + 1)),
    "M5": list(range(2000, 2020 + 1)),
    "i8": list(range(2014, 2020 + 1)),
    "M2": list(range(2016, 2020 + 1)),
    "M3": list(range(2000, 2020 + 1)),
    "M6": list(range(2005, 2020 + 1)),
    "Z3": list(range(1997, 2002 + 1))
    }

#print(df.loc[df['model'] == ' Z3', 'fuelType'].unique())
#print(df.loc[df['model'] == ' Z3', 'engineSize'].unique())
#print(df.loc[df['model'] == ' Z3', 'transmission'].unique())

# Select model first (outside the form)
model_input = st.selectbox("Model", sorted(valid_combinations.keys()))

# Dynamically filter options based on selected model
allowed_fuels = valid_combinations[model_input]["fuelType"]
allowed_transmissions = valid_combinations[model_input]["transmission"]
allowed_engines = valid_combinations[model_input]["engineSize"]
allowed_years = valid_years_by_model.get(model_input, list(range(1990, 2021)))

# --- Manual input form ---
with st.form("manual_input_form"):
    st.markdown("Enter feature values below to predict BMW price:")

    fuel_input = st.selectbox("Fuel Type", allowed_fuels)
    transmission_input = st.selectbox("Transmission", allowed_transmissions)
    year_input = st.selectbox("Year", allowed_years)
    engine_input = st.selectbox("Engine Size", allowed_engines)

    mileage_input = st.number_input("Mileage", min_value=0, max_value=200000, value=60000)
    tax_input = st.number_input("Tax", min_value=0, max_value=600, value=150)
    mpg_input = st.number_input("MPG", min_value=0.0, max_value=650.0, value=float(df["mpg"].median()), step=0.1)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_dict = {
        "model": model_input,
        "transmission": transmission_input,
        "fuelType": fuel_input,
        "year": year_input,
        "mileage": mileage_input,
        "tax": tax_input,
        "mpg": mpg_input,
        "engineSize": engine_input
    }
    input_df = pd.DataFrame([input_dict])

    # Match training format
    input_df["mileage"] = np.log1p(input_df["mileage"])
    for col in ["model", "transmission", "fuelType"]:
        input_df[col] = input_df[col].str.strip()

    # Use trained encoder
    encoded = encoder.transform(input_df[["model", "transmission", "fuelType"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["model", "transmission", "fuelType"]))

    final_input = pd.concat([input_df.drop(columns=["model", "transmission", "fuelType"]), encoded_df], axis=1)

    # Align columns
    missing_cols = set(X_train.columns) - set(final_input.columns)
    for col in missing_cols:
        final_input[col] = 0
    final_input = final_input[X_train.columns]

    # Prediction
    predicted_price = selected_model.predict(final_input)[0]

    # Display compact prediction metric
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.metric(label="💰 Predicted Price", value=f"£{predicted_price:,.0f}")

    # Mini report summary
    st.markdown("### 📝 Prediction Summary Report")

    # Input preview as dataframe
    st.markdown("#### 🔢 Input Features")
    input_preview_df = pd.DataFrame.from_dict(input_dict, orient="index", columns=["Value"]).reset_index()
    input_preview_df.columns = ["Feature", "Value"]
    input_preview_df["Value"] = input_preview_df["Value"].astype(str)
    st.dataframe(input_preview_df, width="stretch")

    # --- Downloadable report ---
    report_df = input_preview_df.copy()
    report_df.loc[len(report_df.index)] = ["Predicted Price", f"£{predicted_price:,.0f}"]
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_bytes,
        file_name="bmw_prediction_report.csv",
        mime="text/csv"
    )

    # SHAP values (if available)
    if "shap_values" in locals():
        st.markdown("#### 📊 SHAP Feature Contributions")
        explainer = shap.Explainer(selected_model, X_train)
        manual_shap = explainer(final_input)
        shap_df = pd.DataFrame({
            "Feature": X_train.columns,
            "SHAP Value": manual_shap.values[0]
        })
        shap_df = shap_df[shap_df["SHAP Value"].abs() > 0.01].sort_values(by="SHAP Value", key=abs, ascending=False)
        shap_df["SHAP Value"] = shap_df["SHAP Value"].astype(str)
        st.dataframe(shap_df, width="stretch")

# --- Shap Explain Section ---
with st.expander("❗ What are SHAP Values?"):
    st.markdown("""
    **SHAP** (**SH**apley **A**dditive ex**P**lanations) is a powerful method for interpreting machine learning models. It assigns each feature a contribution value showing how much that feature pushed the prediction up or down.

    🔍 **Why use SHAP?**
    - It helps you understand *why* a model made a specific prediction.
    - It reveals which features are most influential for each individual prediction.
    - It supports trust and transparency in ML systems — especially important for stakeholders and decision-makers.

    🧠 **How does it work?**
    SHAP is based on game theory. Imagine each feature as a player in a game, and the prediction as the payout. SHAP calculates how much each feature contributes to the final prediction by comparing all possible combinations of features.

    📊 **In this dashboard**, SHAP values show how your input features (like mileage, engine size, fuel type) influence the predicted price — positively or negatively.

    For example, a high SHAP value for mileage means it's significantly lowering the predicted price. A positive SHAP value for engine size might mean it's increasing the price.

    """)

if "X_sample" not in st.session_state:
    st.session_state.X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

# Ensuring column names are preserved
X_sample = st.session_state.X_sample.copy()
X_sample.columns = X_test.columns

explainer = shap.Explainer(selected_model, X_sample)
shap_values = explainer(X_sample)

# --- SHAP Interpretability Section ---
st.subheader("📊 SHAP Interpretability")

with st.expander("📈 SHAP Summary & Feature Importance"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### SHAP Summary Plot")
        fig_summary, ax = plt.subplots(figsize=(5, 3))
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig_summary)
        plt.close()

    with col2:
        st.markdown("##### SHAP Feature Importance")
        fig_bar, ax = plt.subplots(figsize=(5, 3))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig_bar)
        plt.close()

    with st.expander("ℹ️ What do these plots show?"):
        st.markdown("""
        **Summary Plot**  
        - **Color** = feature value (**red** = **high**, **blue** = **low**, **purple** = **mid**)  
        - **Position** = SHAP value (**left** = **negative**, **right** = **positive**)  
        - **Density** = importance across samples  

        **Feature Importance Plot**  
        - Ranks features by average absolute SHAP value  
        - **Longer** bars = **higher** influence  
        - Helps identify top drivers of prediction
        """)

# --- SHAP Dependence & Waterfall Section ---
with st.expander("🔍 SHAP Dependence & Waterfall Analysis"):
    col1, col2 = st.columns(2)

    # --- Dependence Plot ---
    with col1:
        st.markdown("##### SHAP Dependence Plot")

        # Label mapping for display
        label_map = {"mileage": "Mileage (log)"}
        display_labels = [label_map.get(col, col) for col in X_sample.columns]

        # Session state setup
        if "selected_feature" not in st.session_state:
            st.session_state.selected_feature = display_labels[0]
        if "color_feature" not in st.session_state:
            st.session_state.color_feature = display_labels[1]

        # Display selectboxes with mapped labels
        selected_label = st.selectbox("Feature to analyze", display_labels)
        color_label = st.selectbox("Color by feature", display_labels)

        # Map back to original column names
        selected_feature = X_sample.columns[display_labels.index(selected_label)]
        color_feature = X_sample.columns[display_labels.index(color_label)]

        st.session_state.selected_feature = selected_label
        st.session_state.color_feature = color_label

        fig_dep, ax = plt.subplots(figsize=(5, 3))
        shap.dependence_plot(
            selected_feature,
            shap_values.values,
            X_sample,
            interaction_index=color_feature,
            show=False,
            ax=ax
        )
        st.pyplot(fig_dep)
        plt.close()

    # --- Waterfall Plot ---
    with col2:
        st.markdown("##### SHAP Waterfall Plot")
        sample_index = st.slider("Select test sample index", 0, min(499, len(X_test) - 1), 0)

        fig_waterfall, ax = plt.subplots(figsize=(5, 3))
        shap.plots.waterfall(shap_values[sample_index], show=False)
        st.pyplot(fig_waterfall)
        plt.close()

        # --- Car Stats ---
        with st.expander("🚗 Car Stats for Selected Sample"):
            sample_data = shap_values.data[sample_index]
            feature_names = X_test.columns

            non_zero_data = {
                feature: value
                for feature, value in zip(feature_names, sample_data)
                if abs(value) > 0.01
            }

            for feature, value in non_zero_data.items():
                if feature == "mileage":
                    raw_mileage = np.expm1(value)
                    st.markdown(f"- **Mileage**: {int(raw_mileage):,}")
                elif feature == "year":
                    st.markdown(f"- **Year**: {value:.0f}")
                elif feature == "mpg":
                    st.markdown(f"- **MPG**: {value:.1f}")
                elif feature == "tax":
                    st.markdown(f"- **Tax**: £{int(value)}")
                elif feature == "engineSize":
                    st.markdown(f"- **Engine Size**: {value:.1f} L")
                else:
                    st.markdown(f"- **{feature}**: {value}")

    # --- Plot Explanation ---
    with st.expander("ℹ️ What do these plots show?"):
        st.markdown(f"""
        ### 📈 Dependence Plot  
        - Shows how **{selected_label}** affects its SHAP value  
        - Color = value of **{color_label}** (interaction feature)  
        - Reveals nonlinear effects and feature interactions  
        
        ### 🧮 Waterfall Plot  
        - Breaks down a single prediction  
        - Starts from base value  
        - Shows how each feature pushes the prediction up or down  
        - Ideal for explaining individual predictions to stakeholders
        """)


# --- Download SHAP Values ---
st.subheader("📥 Download SHAP Values")
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
csv = shap_df.to_csv(index=False).encode("utf-8")
st.download_button("Download SHAP values as CSV", data=csv, file_name=f"{selected_model_name}_shap_values.csv", mime="text/csv")

# --- Streamlit Footer ---
st.markdown("""
<hr style="margin-top: 50px;">

<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Built by Nicholas Laprade · 
    <a href='https://www.linkedin.com/in/nicholas-laprade/' target='_blank'>LinkedIn</a> · 
    <a href='https://github.com/nlaprade' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)