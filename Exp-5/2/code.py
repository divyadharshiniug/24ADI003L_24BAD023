print("Divyadharshini U G 24BAD023")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\divya\Documents\Experiments\Exp-5\2\train_u6lujuX_CVtuZ9i (1).csv")

df = df.ffill()
df = df.bfill()
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Property_Area']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=X.columns, class_names=True, filled=True)
plt.show()
