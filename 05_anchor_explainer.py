import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from anchor import anchor_tabular

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Define the predict function for the model
predict_fn = lambda x: rf.predict(x)

# Create an Anchor Explainer
explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, X_train, categorical_names=None)

# Generate an anchor explanation for a test instance
idx = 0  # Index of the test instance
instance = X_test[idx]
explanation = explainer.explain_instance(instance, predict_fn, threshold=0.95)

# Print the anchor explanation
print('Anchor: %s' % (' AND '.join(explanation.names())))
print('Precision: %.2f' % explanation.precision)

# Visualize the anchor explanation
explanation.show_in_notebook()

