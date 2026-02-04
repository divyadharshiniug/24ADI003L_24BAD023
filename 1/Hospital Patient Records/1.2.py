print("Divyadharshini U G 24BAD023\n")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\divya\Documents\python\1\diabetes.csv", encoding="latin1")

print("top 5 rows of dataset\n")
print(df.head())
print("\n")

print("last 5 rows\n")
print(df.tail())
print("\n")

print("dataset information\n")
print(df.info())
print("\n")

print("statistical summary\n")
print(df.describe())
print("\n")

print("check for missing values\n")
print(df.isnull().sum())
print("\n")

print("histogram\n")
df['Glucose'].plot(kind='hist')
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")
plt.show()

print("boxplot\n")
df['Age'].plot(kind='box')
plt.title("Age Distribution of Patients")
plt.show()

print("health matrics\n")
print(df[['Glucose','Age']].describe())








