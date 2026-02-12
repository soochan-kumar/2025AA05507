#Question 1 - Size of the DataSet,Missing values & 
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

data = pd.read_csv("C:\\Users\\sooch\\Downloads\\diabetes_indicators.csv")

print("Size of the Dataset is (rows, columns) = ")
print(data.shape)

print(data.info())

print("Missing values from the dataSets = ")
print(data.isnull().sum)

print("Feature types from the dataSets = ")
print(data.dtypes)

X = data.drop("Diabetes_012", axis=1)
y = data["Diabetes_012"]


#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

