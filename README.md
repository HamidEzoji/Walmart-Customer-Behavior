# 🛒 Walmart Customer Purchase Behavior Analysis

Final MSc project for **IT for Business Data Analytics**  
**Author:** Hamid Ezoji | IBS, Budapest

---

## 📄 Project Overview

This project analyzes Walmart customer purchase data to uncover behavioral trends, segment shoppers based on patterns, and predict repeat customer likelihood using machine learning models.

The analysis includes:
- Exploratory data analysis (EDA)
- Customer segmentation using KMeans
- Classification of repeat buyers using Logistic Regression and Random Forest
- Clean modular code and visual output

---

## 📁 Repository Structure

walmart-customer-behavior/
- ├── code/
│ ├── 01_data_preparation.py
│ ├── 02_eda.py
│ ├── 03_segmentation.py
│ ├── 04_modelling/
│ │ ├── 01_classification.py
│ │ ├── 02_logistic_regression.py
│ │ └── 03_customer‐level aggregation (rfm + repe).py
│ └── full_analysis.py
- ├── data/
│ └── walmart_customer_purchases.csv
- ├── visualizations/
│ └── *.png


---

## 🧪 How to Run

> Requires: Python 3.11+  
> Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`

**Option 1 – Run each part manually:**
```bash
python code/01_data_preparation.py
python code/02_eda.py
python code/03_segmentation.py
python code/04_modelling/01_classification.py
python code/04_modelling/02_logistic_regression.py
python code/04_modelling/03_customer‐level aggregation (rfm + repe).py
```
**Option 2 – Run the full project in one go:**
```bash
python code/full_analysis.py

```
**📊 Project Highlights**
- 🛍 Top Product Categories: Identified and visualized most frequently purchased categories
- 📅 Time-Based Trends: Purchases analyzed by day of week, month, and season
- 👥 Customer Segmentation: Applied KMeans clustering on behavioral features (RFM-style)
- 🔁 Repeat Customer Prediction: Used Logistic Regression and Random Forest models with SMOTE balancing and hyperparameter tuning

  **📬 Contact**
  *Hamid Ezoji*
- 📧 HEZOJI@IBS-b.hu
- 🔗 [GitHub Repository](<<<(https://github.com/HamidEzoji/Walmart-Customer-Behavior)>>>)
- `[Walmart_Customer_Behavior DATASET ON KAGGLE]<<<(https://www.kaggle.com/datasets/logiccraftbyhimanshi/walmart-customer-purchase-behavior-dataset)>>>`  
