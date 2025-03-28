# from zipfile import ZipFile

# dataset = 'C:/Users/user/Desktop/machine learning/tuberculosis data.zip'

# with ZipFile(dataset, 'r') as zip:
#     zip.extractall()
#     print('The dataset is extracted')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# read csv //
# dataset = pd.read_csv('C:/Users/user/Desktop/machine learning/5- tuberculosis-patients-with-hiv-share.csv')
# print(dataset.head())
# print(dataset.shape)

# import pandas as pd

# List of CSV files
# file_list = ["C:/Users/user/Desktop/machine learning/1- incidence-of-tuberculosis-sdgs.csv",
#              "C:/Users/user/Desktop/machine learning/2- tuberculosis-deaths-by-age.csv",
#              "C:/Users/user/Desktop/machine learning/3- tuberculosis-case-detection-rate.csv", 
#              "C:/Users/user/Desktop/machine learning/4- tuberculosis-treatment-success-rate-by-type.csv", 
#              "C:/Users/user/Desktop/machine learning/5- tuberculosis-patients-with-hiv-share.csv",
#              "C:/Users/user/Desktop/machine learning/6- tuberculosis-deaths-under-five-ihme.csv"]

# # Read and concatenate files while aligning columns
# merged_df = pd.concat([pd.read_csv(file) for file in file_list], axis=0, ignore_index=True)

# # Save the merged file
# merged_df.to_csv("merged_output.csv", index=False)

# print("Concatenation completed successfully!")

dataset = pd.read_excel("C:/Users/user/Desktop/machine learning/tuberculosis dataset.xlsx")

# print(dataset.isnull().sum())
# print(dataset.shape)

# sp_mapping = {'bloody' : 0, 'colorless' : 1, 'green' : 2}
# dataset['SP'] = dataset['SP'].map(sp_mapping)
print(dataset.head())


fig, ax = plt.subplots(figsize=(8,8))

sns.displot(dataset["SP"])
plt.show()

 
dataset['SP'] = dataset['SP'].fillna(dataset['SP'].mode()[0])
print(dataset.isnull().sum())
print(dataset.head())
print(dataset.shape)
# drop_dataset = pd.read_csv("C:/Users/user/Desktop/machine learning/merged_output.csv")
# drop_dataset = drop_dataset.dropna(how ='any')
# print(drop_dataset.shape)
# drop_dataset['Estimated incidence of all forms of tuberculosis'].fillna(dataset['Estimated incidence of all forms of tuberculosis'].median(), inplace=True)


x = dataset.drop(columns=['SP'])
y = dataset['SP']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)
print(x.std())

scaler = StandardScaler()

scaler.fit(x_train)
x_train_standerdized = scaler.transform(x_train)
print(x_train_standerdized.std())

x_test_standardizes = scaler.transform(x_test)
print(x_test_standardizes.std())

print(x_train_standerdized.std())
model = RandomForestClassifier(n_estimators=100, random_state=42)
model = model.fit(x_train, y_train)

predictions = model.predict(x_test)

importance = model.feature_importances_
print(f"Feature importance for sputum: {importance[0]}")
