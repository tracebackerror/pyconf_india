"""
In this example, we load the iris dataset from the sklearn library, split it into training and test sets, and create a RandomForestClassifier model. We then train the classifier on the training data.

Next, we create a LIME explainer using lime_tabular.LimeTabularExplainer. The explainer takes the training features (X_train), feature names (iris.feature_names), and class names (iris.target_names) as inputs.

We select an instance from the test set (X_test[0]) and explain the prediction for that instance using LIME's explain_instance method. We provide the instance, the predict_proba function of the classifier, and the number of features (len(iris.feature_names)) as inputs to the explain_instance method.

The explanation is printed using exp.show_in_notebook(show_table=True), which displays a table summarizing the contributions of each feature to the prediction.

Additionally, we plot the LIME explanation using exp.as_pyplot_figure() and plt.show(), which visualizes the contributions of each feature in a bar chart.

This example showcases the use of LIME for local interpretability in machine learning models. LIME helps us understand how individual features contribute to predictions by providing interpretable explanations. Such explainable AI techniques are crucial for building trust, understanding model behavior, and identifying potential biases or errors in the decision-making process.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# Split the dataset into features (X) and target variable (y)
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
classifier = RandomForestClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=False)

# Select an instance from the test set for explanation
instance = X_test[0]

# Explain the prediction using LIME
exp = explainer.explain_instance(instance, classifier.predict_proba, num_features=len(iris.feature_names))

# Print the explanation
exp.show_in_notebook(show_table=True)

# Plot the LIME explanation
fig = exp.as_pyplot_figure()
plt.show()
