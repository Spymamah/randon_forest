from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()
print("Dataset loaded successfully!")
print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)
print(iris.head())