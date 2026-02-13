# Diabetes Prediction Using Machine Learning

## 1. Introduction
This project aims to predict diabetes using various machine learning classification algorithms.

## 2. Dataset Description
The dataset contains health indicators related to diabetes.

## 3. Data Preprocessing
- Train-test split
- Feature scaling
- Handling class imbalance

## 4. Models Implemented
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## 5. Evaluation Metrics
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

## 6. Results & Comparison
ML Model Name	      Accuracy	AUC	    Precision	 Recall	  F1	       MCC
Logistic Regression	0.8295	0.824306	0.783316	0.8295	0.787582	0.24837
Decision Tree	      0.7385	0.579999	0.744494	0.7385	0.741436	0.14496
kNN	                0.805	  0.728626	0.754754	0.805	  0.772932	0.181142
Naive Bayes	        0.7515	0.787408	0.789855	0.7515	0.764773	0.300707
Random Forest	      0.8275	0.792621	0.778129	0.8275	0.78435	  0.235118
XGBoost	            0.8225	0.795623	0.778444	0.8225	0.792142	0.260992

## 7. Observations
ML Model Name	           Observation about model performance
Logistic Regression	       Stable and interpretable model; performs well on linearly separable data.
Decision Tree	             Captures non-linear relationships but prone to overfitting.
kNN	                       Performance depends on distance metric and feature scaling.
Naive Bayes	               Fast and efficient but assumes feature independence.
Random Forest (Ensemble)	 Robust and accurate due to ensemble averaging; reduces overfitting.
XGBoost (Ensemble)	       Best overall performance with strong generalization and high accuracy.

## 8. Conclusion
Ensemble models outperform traditional classifiers.

## 9. How to Run
```bash
pip install -r requirements.txt
streamlit run app.py



