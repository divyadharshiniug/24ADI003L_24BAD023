print("Divyadharshini U G 24BAD023\n")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\divya\Documents\python\marketing_campaign.csv", encoding="latin1")

print("top 5 rows of dataset\n")
print(df.head())
print("\n")

print("dataset information\n")
print(df.info())
print("\n")

print("check for missing values\n")
print(df.isnull().sum())
print("\n")

df['Age'] = 2026 - df['Year_Birth']

print("bar graph\n")
df['Age'].value_counts().head(10).plot(kind='bar')
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

print("boxplot for income distribution\n")
df['Income'].plot(kind='box')
plt.title("Income Distribution of Customers")
plt.show()

df['TotalSpending'] = (
    df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
)

print("boxplot for spending pattern\n")
df['TotalSpending'].plot(kind='box')
plt.title("Customer Spending Pattern")
plt.show()










