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

## Results

| Model                        | Precision | Recall | F1-Score | Accuracy |
| ---------------------------- | --------- | ------ | -------- | -------- |
| Logistic Regression          | 0.881     | 0.867  | 0.865    | 0.867    |
| K-Nearest Neighbors          | 0.715     | 0.733  | 0.706    | 0.733    |
| Support Vector Machine (SVM) | 0.872     | 0.833  | 0.821    | 0.833    |
| Decision Tree                | 0.817     | 0.767  | 0.765    | 0.767    |
| Bagging Classifier           | 0.863     | 0.833  | 0.832    | 0.833    |
| Random Forest                | 0.869     | 0.833  | 0.824    | 0.833    |
| AdaBoost                     | 0.841     | 0.800  | 0.795    | 0.800    |
| Gradient Boosting            | 0.767     | 0.733  | 0.713    | 0.733    |
| Stacking Classifier          | 0.841     | 0.800  | 0.795    | 0.800    |
| Voting Classifier            | 0.869     | 0.833  | 0.820    | 0.833    |


---

## Conclusion
The multiclass classification model effectively predicted six human activities based on sensor data using machine learning techniques.<br>
Among various models tested, logistic regression achieved high classification accuracy.<br>
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

