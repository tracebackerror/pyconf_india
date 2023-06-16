import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into features (X) and target variable (y)
X = data.data
y = data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=data.feature_names, class_names=data.target_names)

# Choose a sample from the test set for explanation
sample_idx = 0
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

# Explain the prediction using LIME
exp = explainer.explain_instance(X_sample, clf.predict_proba, num_features=len(data.feature_names))

# Print the explanation
print('Explanation for the prediction:')
print(exp.as_list())
print('True Label:', data.target_names[y_sample])
print('Predicted Probability:', clf.predict_proba([X_sample])[0])
