# Human Activity Classification Using Sensor Data

This project focuses on building and evaluating machine learning models to classify human activities (like walking, sitting, standing, etc.) based on sensor data. The goal is to tune the hyperparameters of several classification models using GridSearchCV and identify the best-performing classification model using classification report.

---

## Dataset

- **Source**: Kaggle
- **Size**: 10299 rows x 562 columns
- **Classes**: 6 (Sitting, Standing, Walking, Laying, walking upstairs, walking downstairs)

---

## Problem Statement

To accurately classify a person's activity using sensor measurements like accelerometer and gyroscope data.

---

## Models Used

All models were evaluated using pipelines, and `GridSearchCV` for hyperparameter tuning:

1. **Logistic Regression**
2. **K-Nearest Neighbors**
3. **Support Vector Machine**
4. **Decision Trees**
5. **Bagging Classifier**
6. **Random Forest**
7. **AdaBoost**
8. **Gradient Boosting**
9. **Stacking Classifier**
10. **Voting Classifier**

---

## Methodology

- **Preprocessing**:
  - Handled large data by stratified sampling 
  - Encoded target labels

- **Model Evaluation**:
  - Cross-validation (cv=5)
  - Metrics: Accuracy, F1-Score, Precision, Recall
 

- **Visualization**:
  - Classification Report
  - Confusion matrices
  - Model comparison bar chart

---


## Model Performance Comparison

| Model               | Precision | Recall | F1-Score | Accuracy |
|----------------------|-----------|--------|----------|----------|
| Logistic Regression  | 0.88      | 0.87   | 0.87     | 0.87     |
| K-Nearest Neighbors  | 0.71      | 0.73   | 0.71     | 0.73     |
| Support Vector Machine | 0.87    | 0.83   | 0.82     | 0.83     |
| Decision Tree        | 0.84      | 0.77   | 0.76     | 0.77     |
| Bagging Classifier   | 0.84      | 0.80   | 0.80     | 0.80     |
| Random Forest        | 0.92      | 0.90   | 0.89     | **0.90** |
| AdaBoost             | 0.84      | 0.80   | 0.80     | 0.80     |
| Gradient Boosting    | 0.72      | 0.70   | 0.69     | 0.70     |
| Stacking Classifier  | 0.87      | 0.83   | 0.83     | 0.83     |
| Voting Classifier    | 0.87      | 0.83   | 0.82     | 0.83     |

**Key Insight:** Random Forest outperformed other models, achieving the highest accuracy (~90%) and best overall balance across metrics.



---

## Conclusion
The multiclass classification model effectively predicted six human activities based on sensor data using machine learning techniques.<br>
Among various models tested, Random Forest achieved high classification accuracy.<br>
The inclusion of GridSearchCV helped in fine-tuning hyperparameters to optimize performance.

---

## Limitations & Next Steps

1. The dataset was sampled to only 100 due to memory or compute constraints, which may limit generalization.
2. Sensor data preprocessing (e.g., noise reduction, signal segmentation, feature extraction) was not deeply explored.

---


## Libraries Used

```bash
scikit-learn
matplotlib
pandas
numpy
seaborn

