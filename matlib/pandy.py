from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# from sklearn.datasets import load_iris

# # Load the iris dataset
# iris = load_iris()
# print("Dataset loaded successfully!")
# print("Features:", iris.feature_names)
# print("Target classes:", iris.target_names)
# from sklearn.datasets import load_iris

# # Load the iris dataset
# iris = load_iris()
# print("Dataset loaded successfully!")
