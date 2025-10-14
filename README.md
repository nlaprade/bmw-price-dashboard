
# BMW Price Predication Dashboard

An interactive dashboard for predicting BMW vehicle prices using machine learning and SHAP-based interpretability. Built for transparency, modularity, and stakeholder insights.

## Description

This projects showcases a clean ML pipeline and dashboard for estimating BMW vehicle prices based on user-defined features. It includes model comparison, SHAP visualizations, and downloadable prediction summaries.

## Dataset

The dataset used in this project contains 10,781 BMW listings.

- **Source**: [Kaggle BMW Used Car Dataset](https://www.kaggle.com/datasets/wardabilal/bmw-cars-dataset-analysis-with-visualizations)  
- **License**: Publicly available for educational and non-commercial use  
- **Preprocessing**: Categorical encoding, missing value handling, feature alignment

## Features

- Manual input prediction with real-time feedback
- SHAP summary, dependence, and waterfall plots
- Model comparison with R² and RMSE metrics
- Downloadable prediction report and SHAP values
- Clean UI with expandable sections and metric highlights

## Installation


```bash
  git clone https://github.com/nlaprade/bmw-price-dashboard.git
  cd bmw-price-dashboard
  pip install -r requirements.txt
```
    
## Tools Stack

| Tool         | Purpose                                      |
|--------------|----------------------------------------------|
| **Streamlit**| Interactive dashboard and UI                 |
| **scikit-learn** | Model training, evaluation, preprocessing |
| **SHAP**     | Model interpretability and feature impact    |
| **pandas**   | Data manipulation and cleaning               |
| **matplotlib** | Plotting SHAP visuals                      |

All dependencies are listed in `requirements.txt`.

## Preview

![BMW Price Prediction Dashboard](images/dashboard-preview.jpg)
## Authors

Nicholas Laprade 
- [LinkedIn](https://www.linkedin.com/in/nicholas-laprade/)
- [GitHub](https://github.com/nlaprade)


