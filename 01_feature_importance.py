"""
In this example, we load the diabetes dataset from the sklearn library, split it into training and test sets, and create a RandomForestRegressor model. We then train the regressor on the training data.

Next, we calculate feature importance using permutation importance. The permutation_importance function takes the trained regressor, test features (X_test), and target variable (y_test) as inputs. The n_repeats parameter specifies the number of times to permute each feature, and the random_state parameter ensures reproducibility.

The result.importances_mean attribute provides the mean importance score for each feature. We store these scores in a DataFrame, sort them in descending order, and print the feature importance scores.

This example demonstrates how to use permutation importance to assess the importance of features in a regression problem. You can adapt this code to different datasets and models to explore feature importance in various scenarios.

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the dataset into features (X) and target variable (y)
X = diabetes.data
y = diabetes.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest regressor
regressor = RandomForestRegressor()

# Train the regressor on the training data
regressor.fit(X_train, y_train)

# Calculate feature importance using permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean

# Create a DataFrame to display the feature importance scores
feature_importance = pd.DataFrame({'Feature': diabetes.feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print(feature_importance)
