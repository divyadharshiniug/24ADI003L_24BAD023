print("U G Divyadharshini 24BAD023")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv(r"C:\Users\divya\Documents\python\3\vehicle fuel efficiency\auto-mpg.csv")

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df[['horsepower', 'mpg']]
df.fillna(df.mean(), inplace=True)


X = df[['horsepower']].values
y = df['mpg'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


degrees = [2, 3, 4]

train_errors = []
test_errors = []
models = {}


for d in degrees:
    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

    print("Degree:", d)
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("R2:", r2_score(y_test, y_test_pred))
    print()

    models[d] = (model, poly)


ridge = Ridge(alpha=1.0)
poly4 = PolynomialFeatures(degree=4)

X_train_poly4 = poly4.fit_transform(X_train)
X_test_poly4 = poly4.transform(X_test)

ridge.fit(X_train_poly4, y_train)
ridge_pred = ridge.predict(X_test_poly4)

print("Ridge Degree 4 R2:", r2_score(y_test, ridge_pred))


x_range = np.linspace(X_scaled.min(), X_scaled.max(), 200).reshape(-1, 1)


# Shows polynomial regression curves of degree 2, 3, 4 fitted to the data
plt.figure()

for d in degrees:
    model, poly = models[d]
    x_poly = poly.transform(x_range)
    y_curve = model.predict(x_poly)
    plt.plot(x_range, y_curve, label=f"Degree {d}")

plt.scatter(X_scaled, y, s=10)
plt.xlabel("Horsepower (Scaled)")
plt.ylabel("MPG")
plt.title("Polynomial Curve Fitting for Different Degrees")
plt.legend()
plt.show()



# Compares training error and testing error to detect overfitting/underfitting
plt.figure()

plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Training vs Testing Error Comparison")
plt.legend()
plt.show()



# Visualizes how model complexity affects performance (bias-variance tradeoff)
plt.figure()

plt.bar(['Deg2', 'Deg3', 'Deg4'], test_errors)

plt.xlabel("Polynomial Degree")
plt.ylabel("Testing Error")
plt.title("Underfitting vs Overfitting Demonstration")
plt.show()
