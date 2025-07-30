#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Read Dataset
file_path = 'water_potability.csv'  # Path to the dataset file
df = pd.read_csv(file_path)  # Reading the dataset into a DataFrame

print("Head of the dataset:")
print(df.head())  # Displaying the first 5 rows of the dataset

print("\nTail of the dataset:")
print(df.tail())  # Displaying the last 5 rows of the dataset

print("\nShape of the dataset:")
print(df.shape)  # Displaying the shape (number of rows and columns) of the dataset

print("\nInfo of the dataset:")
print(df.info())  # Displaying information about the dataset, such as column names, non-null counts, and data types

print("\nDescription of the dataset:")
print(df.describe())  # Displaying statistical summary of the dataset

print("\nUniqueness of values in each column:")
print(df.nunique())  # Displaying the number of unique values in each column

for column in df.columns:
    print(f"\nValue counts for {column}:")
    print(df[column].value_counts())  # Displaying the count of unique values in each column

# Step 2: Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='Potability', data=df)
plt.title("Distribution of Potability")
plt.show()

# Step 3: Preprocessing
df['ph'].fillna(df['ph'].mean(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].mean(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)

df.drop_duplicates(inplace=True)

print("\nNumber of missing values in each column after preprocessing:")
print(df.isna().sum())

# Step 4: Feature Engineering
print("\nCorrelation Matrix:")
correlation_matrix = df.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(15, 10))
df.boxplot()
plt.title("Box Plot of All Features")
plt.xticks(rotation=90)
plt.show()

for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution Plot of {column}")
    plt.show()

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\nFirst few rows of the scaled dataset:")
print(df_scaled.head())

# Step 5: Implementation of Model
X = df_scaled.drop('Potability', axis=1)
y = df_scaled['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Step 6: Evaluation of Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the KNN model: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Visualization of Results
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

