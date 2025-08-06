# Grid-and-Random-Search-CV
In machine learning, hyperparameter tuning is a crucial step to optimize model performance. Scikit-learn offers two powerful tools for this purpose: GridSearchCV and RandomizedSearchCV.

 GridSearchCV
GridSearchCV performs an exhaustive search over all possible combinations of specified hyperparameter values.

Approach: Tries every combination in the defined parameter grid

Best For: Small search spaces where you want to try all options

Drawback: Can be very slow and computationally expensive for large grids

Example Use Case:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
 RandomizedSearchCV
RandomizedSearchCV selects a fixed number of random combinations from a specified hyperparameter distribution.

Approach: Randomly selects combinations from a given range/distribution

Best For: Large or continuous search spaces where GridSearch would be too slow

Advantage: Often finds good results much faster

Example Use Case:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X_train, y_train)
