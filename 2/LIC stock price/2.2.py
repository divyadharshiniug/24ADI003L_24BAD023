print("Divyadharshini U G  24BAD023")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

path = r"C:\Users\divya\Documents\python\2\LIC stock price\LICI - 10 minute data.csv"

df = pd.read_csv(path, engine="python")

if df.shape[1] == 1:
    df = pd.read_csv(path, sep=';', engine="python")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df['price_movement'] = np.where(df['close'] > df['open'], 1, 0)

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df[['open', 'high', 'low', 'volume']].copy()
y = df['price_movement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000, solver='liblinear'),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Best Params:", grid.best_params_)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

for col, coef in zip(['open','high','low','volume'], model.coef_[0]):
    print(col, ":", coef)

features = ['open','high','low','volume']
importance = model.coef_[0]

plt.figure(figsize=(6,4))
plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance (Logistic Regression)")
plt.axhline(0)
plt.show()
