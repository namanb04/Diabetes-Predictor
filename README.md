# Diabetes Prediction

A machine learning project aimed at predicting diabetes outcomes using various classification algorithms, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and Support Vector Classifier (SVC).

This project demonstrates the end-to-end pipeline of data preprocessing, feature selection, model training, evaluation, and visualization using Python libraries like Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

## Features
- **Data Preprocessing**: Handles missing values by imputing with mean values and applies feature scaling using MinMaxScaler.
- **Feature Selection**: Uses key attributes such as Glucose, Insulin, BMI, and Age to train the models.
- **Model Training and Evaluation**:  
  - Implements multiple machine learning algorithms for comparison.  
  - Identifies the best-performing model based on accuracy and other metrics.
- **Visualization**:  
  - Generates correlation heatmaps, histograms, scatter plots, and pair plots to explore data relationships.  
  - Visualizes the confusion matrix and classification metrics for model evaluation.

---

## Dataset
The project uses the **PIMA Indians Diabetes Database** from the UCI Machine Learning Repository.
- **Total Records**: 768
- **Features**:  
  - Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
  - Target variable: Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## Results
- **Best Model**: K-Nearest Neighbors (KNN)  
  - **Accuracy**: 78.57%  
  - **Precision**: 83% (Non-Diabetic)  
  - **Recall**: 85% (Non-Diabetic)  
  - **F1-Score**: 84% (Non-Diabetic)  
- Comparative results for other models:  
  - Logistic Regression: 72.08%  
  - Decision Tree: 68.18%  
  - Random Forest: 75.97%  
  - Naive Bayes: 71.43%  
  - Support Vector Classifier: 73.38%

---


## Technologies Used
- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Tools**: Google Colab, Jupyter Notebook

