print("U G Divyadharshini 24BAD023")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\divya\Desktop\experiments\3\student academic performance\StudentsPerformance.csv")

le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

df['final_score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3

np.random.seed(10)
df['study_hours'] = np.random.randint(1, 6, len(df))
df['attendance'] = np.random.randint(60, 101, len(df))
df['sleep_hours'] = np.random.randint(5, 9, len(df))

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df[['study_hours',
        'attendance',
        'parental level of education',
        'test preparation course',
        'sleep_hours']]

y = df['final_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=10
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.show()

plt.figure()
plt.bar(X.columns, model.coef_)
plt.xticks(rotation=45)
plt.show()

residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=20)
plt.show()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge R2:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R2:", r2_score(y_test, lasso_pred))


train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, y_pred)

print("Training R2:", train_r2)
print("Testing R2:", test_r2)

difference = train_r2 - test_r2

if train_r2 < 0.5 and test_r2 < 0.5:
    print("Model is UNDERFITTING (High Bias).")

elif difference > 0.15:
    print("Model is OVERFITTING (High Variance).")

else:
    print("Model is WELL-FITTED (Good Generalization).")

