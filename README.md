# Diabetic Readmission Prediction

This project focuses on predicting diabetic patient readmissions using a comprehensive dataset. The goal is to identify factors contributing to readmission and build predictive models to help healthcare providers intervene proactively.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Models and Evaluation](#models-and-evaluation)
5.  [Results](#results)
6.  [Usage](#usage)
7.  [Dependencies](#dependencies)

## Project Overview

Diabetic readmissions are a significant concern in healthcare, impacting patient well-being and increasing healthcare costs. This notebook presents an end-to-end machine learning pipeline to predict whether a diabetic patient will be readmitted to the hospital within 30 days, or at all.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository (Diabetic Readmission Dataset). It contains de-identified patient records from 130 US hospitals over 10 years (1999-2008), encompassing clinical and demographic features such as:

*   **Patient demographics**: race, gender, age, weight.
*   **Hospitalization details**: admission type, discharge disposition, admission source, time in hospital.
*   **Medical information**: number of lab procedures, number of procedures, number of medications, diagnoses (ICD-9 codes), lab results (max_glu_serum, A1Cresult).
*   **Medication details**: dosage changes for various diabetic medications.
*   **Readmission status**: indicating if a patient was readmitted within 30 days, after 30 days, or not at all.

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration**: Loading the dataset, inspecting data types, unique values, and initial statistical summaries.
2.  **Missing Data Handling**: Identifying and visualizing missing values. Imputing categorical features with 'unknown' or mode, and continuous features with appropriate strategies. Highly missing columns like 'weight', 'max_glu_serum', and 'A1Cresult' were specifically addressed.
3.  **Feature Engineering**: Converting categorical age and weight ranges into numerical values. Grouping detailed ICD-9 codes into broader categories for better interpretability and model performance.
4.  **Encoding Categorical Variables**: Converting all remaining categorical features into numerical formats using Label Encoding for most, and binary mapping for certain medication features and the target variable.
5.  **Exploratory Data Visualization (EDA)**: Visualizing distributions of demographics, numeric features (time in hospital, lab procedures, medications), and lab results. Exploring relationships between features using correlation heatmaps. Analyzing admission/discharge patterns and medication usage across subgroups.
6.  **Model Training and Evaluation**: Splitting the data into training and testing sets. Training and evaluating several classification models:
    *   Logistic Regression (with and without gradient descent)
    *   Random Forest Classifier
    *   Gradient Boosting Classifier
    *   AdaBoost Classifier (with Decision Tree and Random Forest estimators)
    *   XGBoost Classifier
    *   CatBoost Classifier

Each model is evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score, along with Confusion Matrices.

## Models and Evaluation

The following models were trained and evaluated:

*   **Logistic Regression**: A baseline linear model.
*   **Random Forest Classifier**: An ensemble tree-based method.
*   **Gradient Boosting Classifier**: Another powerful ensemble technique.
*   **AdaBoost Classifier**: An adaptive boosting method with different base estimators.
*   **XGBoost Classifier**: An optimized distributed gradient boosting library.
*   **CatBoost Classifier**: A gradient boosting library specifically designed for categorical features.

Model performance was assessed primarily using the F1-Score, which provides a balance between precision and recall, crucial for imbalanced classification tasks like readmission prediction.

## Results

After training and evaluating multiple models, the performance metrics were compiled and sorted by F1-Score.

```
# Model Comparison Table (Example, actual values are in the notebook)
                                      Model  Accuracy  Precision    Recall  F1-Score
0                         CatBoost (Initial)  0.653604   0.618198  0.656497  0.636772
1                           CatBoost (Final)  0.648809   0.614102  0.647659  0.630434
2                        Gradient Boosting  0.651521   0.641858  0.557746  0.596853
3                      AdaBoost with Random Forest  0.638708   0.634493  0.516189  0.569259
4                            Random Forest  0.643385   0.634808  0.539050  0.583023
5  Logistic Regression (Gradient Descent)  0.618544   0.636992  0.407411  0.496968
6                    AdaBoost with Decision Tree  0.623850   0.641724  0.422708  0.509683
7                                  XGBoost  0.650892   0.644583  0.546528  0.591519
8                      Logistic Regression  0.616343   0.633733  0.403926  0.493382
```

The initial CatBoost model and Gradient Boosting generally showed strong performance, achieving the highest F1-Scores. The confusion matrices provide further insight into the models' ability to correctly classify readmitted and non-readmitted patients.

## Usage

To run this notebook:

1.  **Clone the Repository**: Download or clone this GitHub repository to your local machine.
2.  **Open in Google Colab**: Upload the `.ipynb` file to Google Colab or open it directly from GitHub.
3.  **Run Cells**: Execute the cells sequentially. Ensure all necessary libraries are installed (see Dependencies).
4.  **Dataset**: The notebook expects `diabetic_data.csv` to be available, usually by mounting Google Drive or placing it in the Colab environment.

## Dependencies

This project requires the following Python libraries:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `catboost`
*   `xgboost`
*   `plotly`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost plotly
```
