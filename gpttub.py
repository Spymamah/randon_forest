import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib

# Load Dataset
dataset = pd.read_excel("C:/Users/user/Desktop/machine learning/tuberculosis dataset.xlsx")

# Check for missing values
print("Missing values before handling:\n", dataset.isnull().sum())
#distribution of the three classes
print(dataset['SP'].value_counts())

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=dataset['SP'], palette="viridis")  # 'viridis' gives a nice color gradient
plt.xlabel("Class Labels (SP)")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()
# Map categorical values

print(dataset.head())  # Check if dataset is correctly loaded
print(f"Dataset contains {len(dataset)} rows.")
if len(dataset) >= 500:
    dataset_sample = dataset.sample(n=500, replace=False)
else:
    dataset_sample = dataset.copy()  # Use full dataset if it's smaller


 
# dataset['SP'] = dataset['SP'].fillna(dataset['SP'].mode()[0])

# Check again
# print("Missing values after handling:\n", dataset.isnull().sum())

# Visualize Distribution
sns.displot(dataset["SP"])
plt.show()

# Split Features and Target
x = dataset.drop(columns=['SP'])
y = dataset['SP']

# Train-test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Print Shapes
print("X Shape:", x.shape, "| X_train:", x_train.shape, "| X_test:", x_test.shape)
print("Y Shape:", y.shape, "| Y_train:", y_train.shape, "| Y_test:", y_test.shape)



# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)




# Save the trained model
joblib.dump(model, "random_forest_tb.pkl")

print("Model saved successfully!")

# Predictions
predictions = model.predict(x_test)

# Feature Importance
importance = model.feature_importances_
feature_names = x.columns
for name, imp in zip(feature_names, importance):
    print(f"Feature: {name}, Importance: {imp}")

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names, palette="viridis")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()



# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()


# Simulate accuracy scores over multiple runs
runs = np.arange(1, 11)  # 10 runs
accuracies = [0.91, 0.92, 0.93, 0.94, 0.95, 0.94, 0.93, 0.92, 0.91, 0.94]  # Example data

plt.figure(figsize=(8, 5))
plt.plot(runs, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel("Run Number")
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy Over Multiple Runs")
plt.ylim(0.9, 1)  # Set y-axis range for better visualization
plt.grid()
plt.show()


input_data = [0,0,	2,	0,	1,	0,	1,	0,	2,	2,	0,	0,	0,	2,	1]
# Assume 'model' is your trained model and 'new_sample' is a single instance
new_sample = np.array(input_data)  # 1D array

# Reshape to match (1, num_features) before prediction
new_sample = new_sample.reshape(1, -1)  

# Predict
prediction = model.predict(new_sample)
print("Predicted class:", prediction)
