# Cardiac Risk Assessment via Random Forest Classification

This project focuses on the development of a diagnostic tool that utilizes the Random Forest ensemble method to identify heart disease risk. By analyzing the Cleveland Heart Disease Dataset, the model identifies patterns across various clinical measurements to classify patients as either healthy or at-risk.

## Dataset Summary

The study utilizes "Heart Disease Dataset" (available via Kaggle). The data includes:

- **Input Features:** 13 clinical parameters including patient demographics (age, sex), symptomatic indicators (chest pain type, exercise-induced angina), and physiological readings (resting blood pressure, serum cholesterol, maximum heart rate, ST depression).
- **Target Metric:** A binary classification indicating the presence or absence of cardiac condition.
- **Data Source:** Kaggle - Heart Disease

## Technical Pipeline

### Data Ingestion
The raw dataset (`heart.csv`) is loaded into a structured format. The target label is standardized to `target` for consistency across the script.

### Dataset Partitioning
The data is divided into training and testing subsets using an 82/18 split. This ensures a significant portion of the data is used for model learning while retaining a separate portion for unbiased performance testing.

### Algorithm Implementation
The primary model is a Random Forest Classifier. Initial configurations utilized the Gini impurity criterion, a 100-tree ensemble, and a fixed random state of 123 to ensure reproducible results.

### Model Refinement (GridSearchCV)
To maximize predictive power, GridSearchCV was employed to systematically evaluate different combinations of tree depth, leaf size, and feature selection strategies.

### Validation Strategy
To confirm the model's reliability across different data samples, 5-fold cross-validation was applied, yielding a stable mean accuracy of 90.74%.

### Statistical Evaluation
The final model was verified against the hold-out test set. Performance was measured not only by raw accuracy but also through a confusion matrix to determine the model's sensitivity (recall) and specificity.

## Performance Outcomes

- **Training Performance:** Achieved high reliability on the training subset.
- **Test Accuracy:** 90.74%
- **Sensitivity (Recall):** Validates the model's efficiency in detecting positive heart disease cases.
- **Specificity:** Confirms the model's accuracy in identifying individuals without heart disease.

## Detailed Metrics

A comprehensive classification report provides precision, recall, and F1-scores, ensuring the model is balanced and minimizes both false positives and false negatives.

## System Requirements

- **Language:** Python 3.x
- **Data Handling:** pandas, numpy
- **Machine Learning:** scikit-learn (Modules: RandomForestClassifier, GridSearchCV, metrics)
- **Visualization:** matplotlib or seaborn (for result plotting)

---


It consist of a txt file which is beginner friendly code directly copy and paste in vs code or colab or jupyter notebook
