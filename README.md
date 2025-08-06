# Human Activity Classification Using Sensor Data

This project focuses on building and evaluating machine learning models to classify human activities (like walking, sitting, standing, etc.) based on sensor data. The goal is to identify the best-performing classification model using systematic evaluation and hyperparameter tuning.

---

## 📂 Dataset

- **Source**: [Mention source – e.g., UCI HAR, Kaggle, etc.]
- **Size**: [Number of samples], [Number of features]
- **Classes**: 6 (e.g., Sitting, Standing, Walking, Lying Down, Upstairs, Downstairs)

---

## 🎯 Problem Statement

To accurately classify a person's activity using sensor measurements like accelerometer and gyroscope data.

---

## 🧠 Models Used

All models were evaluated using pipelines, scaling, and `GridSearchCV` for hyperparameter tuning:

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

## ⚙️ Methodology

- **Preprocessing**:
  - Handled large data by stratified sampling (20%)
  - Encoded target labels
  - Feature scaling using `StandardScaler`

- **Model Evaluation**:
  - Cross-validation (cv=5)
  - Metrics: Accuracy, F1-Score, Confusion Matrix, ROC-AUC (if binary), Precision, Recall

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

*Fill in actual values after running.*

---

## ✅ Conclusion

- **Best Model**: [e.g., Random Forest or Stacking Classifier]
- **Key Takeaways**:
  - Tree-based models handled feature interactions better.
  - Distance-based models like KNN were sensitive to scaling and dimensionality.
  - Ensemble methods (Stacking/Voting) improved accuracy marginally.

---

## ⚠️ Flaws & Limitations

- Some models were computationally heavy on large datasets.
- Feature engineering was minimal — model relies heavily on raw data.
- No real-time validation on unseen sensor streams.

---

## 🔜 Next Steps

- Incorporate time-series models (e.g., LSTM, HMM)
- Add more features like mean, std, FFT components over sliding windows
- Deploy as a lightweight mobile model for activity detection
- Try auto-feature selection or dimensionality reduction (PCA)

---

## 🛠️ Libraries Used

```bash
scikit-learn
matplotlib
pandas
numpy
seaborn

