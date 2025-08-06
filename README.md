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
 

STarts here

- **Visualization**:
  - Correlation plot of features vs. target
  - Confusion matrices
  - Feature importance (tree-based models)
  - Model comparison bar chart

---

## 🧪 Results

| Model               | Accuracy | F1 Score | Time Taken |
|---------------------|----------|----------|-------------|
| Logistic Regression |  |  |  |
| KNN                 |  |  |  |
| SVM                 |  |  |  |
| Decision Tree       |  |  |  |
| Bagging             |  |  |  |
| Random Forest       |  |  |  |
| AdaBoost            |  |  |  |
| Gradient Boost      |  |  |  |
| Stacking            |  |  |  |
| Voting              |  |  |  |

Ends here

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

