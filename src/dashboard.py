# ─────────────────────────────────────────────
# 🚀 Imports & Setup
# ─────────────────────────────────────────────
import streamlit as st
import io
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# ⚙️ Streamlit Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BMW Price Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)
st.title("🚗 BMW Price Prediction Dashboard")
st.markdown("A professional-grade dashboard for model comparison, interpretability, and stakeholder insights.")

# ─────────────────────────────────────────────
# 📁 Paths & Data Loading
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "bmw.csv")

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = df[df['price'] > 1000]
    return df

df = load_data(data_path)

# ─────────────────────────────────────────────
# 🧼 Preprocessing
# ─────────────────────────────────────────────
def preprocess(df):
    cat_cols = ['model', 'transmission', 'fuelType']
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    df_encoded = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    X = df_encoded.drop(columns='price')
    y = df_encoded['price']
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()

(X_train, X_test, y_train, y_test), feature_names = preprocess(df)

# ─────────────────────────────────────────────
# 🧠 Model Training & Evaluation
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        results.append((name, round(r2, 4), round(mse, 2), round(mae, 2)))

    return models, pd.DataFrame(results, columns=["Model", "R²", "MSE", "MAE"])

models, results_df = train_models(X_train, y_train, X_test, y_test)

# ─────────────────────────────────────────────
# 📊 Model Comparison
# ─────────────────────────────────────────────
st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df, width="stretch")

best_model_name = results_df.iloc[results_df["R²"].idxmax()]["Model"]
st.success(f"🏆 Best Model: {best_model_name}")
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


# ─────────────────────────────────────────────
# 🧮 Manual Input Prediction
# ─────────────────────────────────────────────
st.subheader("🧮 Predict Price from Manual Input")

with st.form("manual_input_form"):
    st.markdown("Enter feature values below to predict BMW price:")

    # Categorical inputs
    model_input = st.selectbox("Model", df["model"].unique())
    transmission_input = st.selectbox("Transmission", df["transmission"].unique())
    fuel_input = st.selectbox("Fuel Type", df["fuelType"].unique())

    # Numeric inputs
    year_input = st.number_input("Year", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=int(df["year"].median()))
    mileage_input = st.number_input("Mileage", min_value=0, max_value=int(df["mileage"].max()), value=int(df["mileage"].median()))
    tax_input = st.number_input("Tax", min_value=0, max_value=600, value=150)
    mpg_input = st.number_input("MPG", min_value=0.0, max_value=650.0, value=float(df["mpg"].median()), step=0.1)
    engine_input = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=float(df["engineSize"].median()), step=0.18)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Create input dictionary
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

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[["model", "transmission", "fuelType"]])
    encoded = encoder.transform(input_df[["model", "transmission", "fuelType"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["model", "transmission", "fuelType"]))

    # Combine with numeric features
    final_input = pd.concat([input_df.drop(columns=["model", "transmission", "fuelType"]), encoded_df], axis=1)

    # Align columns with training set
    missing_cols = set(X_train.columns) - set(final_input.columns)
    for col in missing_cols:
        final_input[col] = 0
    final_input = final_input[X_train.columns]

    # Predict
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

    # Downloadable report
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



# ─────────────────────────────────────────────
# 💡 SHAP Explainability Setup
# ─────────────────────────────────────────────
if "X_sample" not in st.session_state:
    st.session_state.X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

X_sample = st.session_state.X_sample.copy()
X_sample.columns = X_test.columns  # Ensure column names are preserved

explainer = shap.Explainer(selected_model, X_sample)
shap_values = explainer(X_sample)

# ─────────────────────────────────────────────
# 📊 SHAP Interpretability Section
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 🔍 SHAP Dependence & Waterfall Section
# ─────────────────────────────────────────────
with st.expander("🔍 SHAP Dependence & Waterfall Analysis"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### SHAP Dependence Plot")
        selected_feature = st.selectbox("Feature to analyze", X_sample.columns, index=0)
        color_feature = st.selectbox("Color by feature", X_sample.columns, index=1)

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

    with col2:
        st.markdown("##### SHAP Waterfall Plot")
        sample_index = st.slider("Select test sample index", 0, min(499, len(X_test) - 1), 0)

        fig_waterfall, ax = plt.subplots(figsize=(5, 3))
        shap.plots.waterfall(shap_values[sample_index], show=False)
        st.pyplot(fig_waterfall)
        plt.close()

        with st.expander("🚗 Car Stats for Selected Sample"):
            # Pull raw feature values from SHAP object to ensure alignment
            sample_data = shap_values.data[sample_index]
            feature_names = X_test.columns

            non_zero_data = {
                feature: value
                for feature, value in zip(feature_names, sample_data)
                if value != 0
            }

            for feature, value in non_zero_data.items():
                st.markdown(f"- **{feature}**: {value}")

    with st.expander("ℹ️ What do these plots show?"):
        st.markdown(f"""
        **Dependence Plot**  
        - Shows how **{selected_feature}** affects its SHAP value  
        - Color = value of **{color_feature}** (interaction feature)  
        - Reveals nonlinear effects and interactions  

        **Waterfall Plot**  
        - Breaks down a single prediction  
        - Starts from base value  
        - Shows how each feature pushes prediction up/down  
        - Ideal for explaining individual predictions
        """)



# ─────────────────────────────────────────────
# 📥 Download SHAP Values
# ─────────────────────────────────────────────
st.subheader("📥 Download SHAP Values")
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
csv = shap_df.to_csv(index=False).encode("utf-8")
st.download_button("Download SHAP values as CSV", data=csv, file_name=f"{selected_model_name}_shap_values.csv", mime="text/csv")

# ─────────────────────────────────────────────
# 📌 Footer
# ─────────────────────────────────────────────
st.markdown("""
<hr style="margin-top: 50px;">

<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Built by Nicholas Laprade · 
    <a href='https://www.linkedin.com/in/nicholas-laprade/' target='_blank'>LinkedIn</a> · 
    <a href='https://github.com/nlaprade' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
