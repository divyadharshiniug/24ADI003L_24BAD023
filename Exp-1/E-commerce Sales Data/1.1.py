print("Divyadharshini U G 24BAD023\n")

import pandas as pd
import matplotlib.pyplot as plt

print("scenario 1\n")

df = pd.read_csv(r"C:\Users\divya\Documents\python\1\data.csv\data.csv", encoding="latin1")

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

print("plotting\n")
df.groupby('InvoiceDate')['Quantity'].sum().head(10).plot(kind='line')
plt.title("Sales Trend Over Time")
plt.xlabel("Invoice Date")
plt.ylabel("Total Quantity Sold")
plt.show()

df.groupby('InvoiceDate')['Quantity'].sum().head(10).plot(kind='bar')
plt.title("Sales Trend Over Time")
plt.xlabel("Invoice Date")
plt.ylabel("Total Quantity Sold")
plt.show()







