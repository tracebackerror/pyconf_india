import shap
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# California Housing Prices
dataset = fetch_california_housing(as_frame = True)
X = dataset['data']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Prepares a default instance of the random forest regressor
model = RandomForestRegressor()
# Fits the model on the data
model.fit(X_train, y_train)


# Fits the explainer
explainer = shap.Explainer(model.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

# Evaluate SHAP values
shap_values = explainer.shap_values(X)
"""
Here the features are ordered from the highest to the lowest effect on the prediction. It takes in account the absolute SHAP value, so it does not matter if the feature affects the prediction in a positive or negative way.
"""
shap.plots.bar(shap_values)

# Summary plot: beeswarm
"""
On the beeswarm the features are also ordered by their effect on prediction, but we can also see how higher and lower values of the feature will affect the result.

All the little dots on the plot represent a single observation. The horizontal axis represents the SHAP value, while the color of the point shows us if that observation has a higher or a lower value, when compared to other observations.

In this example, higher latitudes and longitudes have a negative impact on the prediction, while lower values have a positive impact.
"""
shap.summary_plot(shap_values)
# or
shap.plots.beeswarm(shap_values)


"""
Summary plot: violin

Another way to see the information of the beeswarm is by using the violin plot:

"""
shap.summary_plot(shap_values, plot_type='violin')

"""
Local bar plot
This plot shows us what are the main features affecting the prediction of a single observation, and the magnitude of the SHAP value for each feature.
"""
shap.plots.bar(shap_values[0])


"""
Force plot
The force plot is another way to see the effect each feature has on the prediction, for a given observation. In this plot the positive SHAP values are displayed on the left side and the negative on the right side, as if competing against each other. The highlighted value is the prediction for that observation.

"""
shap.plots.force(shap_test[0])