#%% Define Library
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import string
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV
import sklearn.metrics as metrics

#%%
# *********Task1: Generate Random Dataset******************

# Sample Count


n_sample = 1000

# Generate Random Numeric Columns
_X, _y= datasets.make_regression(n_samples=n_sample,#number of samples
                                      n_features=2,#number of features
                                      noise=10,#bias and standard deviation of the guassian noise
                                      random_state=0) #set for same data points for each run
# Generate Random Letter Column
alphabet = list(string.ascii_lowercase)
letter_col = random.choices(alphabet,k=n_sample)

# Data Set
df = pd.concat([pd.DataFrame(_X,columns=["est_num1","est_num2"])
                   ,pd.DataFrame(letter_col,columns=["est_cat"])
                   ,pd.DataFrame(_y,columns=["target"])]
               , axis=1)

#%%

# Preprocess
# Convert Categorical Column to Numeric Columns with One-Hot Encoding
df["est_cat"] = df["est_cat"].astype(pd.CategoricalDtype(categories=alphabet))
df = pd.get_dummies(df)

# X , y
X = df.loc[:, df.columns != 'target'].values
y = df['target'].values

# Split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%%
# Find Best Model
# Check Modeling
models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver='lbfgs')))  # neural network
models.append(('KNN', KNeighborsRegressor()))
models.append(('RF', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto')))  # kernel = linear

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # evaluate model
    cv_results = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)

    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#%%
# LinearRegression and RandomForest perform equally well.
# But I personally prefer RF since this ensemble model (combine multiple ‘individual’ (diverse) models together
# and deliver superior prediction power.)
# can almost work out of the box and that is one reason why they are very popular.

# Grid Searching Hyperparameters to find best RandomForest Model

model = RandomForestRegressor()
param_search = {
    'n_estimators': [20, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [i for i in range(5,15)]
}

gsearch = GridSearchCV(estimator=model, cv=cv, param_grid=param_search, scoring = "r2",n_jobs=-1)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_

print("Best Model is RandomForestRegressor with:")
print("n_estimators: " , best_model.n_estimators )
print("max_features: " , best_model.max_features)
print("max_depth   : " , best_model.max_depth)


#%%
# Checking best model performance on test data

y_true = y_test
y_pred = best_model.predict(X_test)

# Regression metrics
explained_variance=metrics.explained_variance_score(y_true, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
mse=metrics.mean_squared_error(y_true, y_pred)
median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
r2=metrics.r2_score(y_true, y_pred)

print('explained_variance: ', round(explained_variance,4))
print('r2: ', round(r2,4))
print('MAE: ', round(mean_absolute_error,4))
print('MSE: ', round(mse,4))
print('RMSE: ', round(np.sqrt(mse),4))

#%%