#%% Define Library
import joblib
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
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

# Final Sample Data Set
df = pd.concat([pd.DataFrame(_X,columns=["est_num1","est_num2"])
                   ,pd.DataFrame(letter_col,columns=["est_letter"])
                   ,pd.DataFrame(_y,columns=["target"])]
               , axis=1)

#%%

# Preprocessing
# Convert Categorical Column to Numeric Columns with One-Hot Encoding
df["est_letter"] = df["est_letter"].astype(pd.CategoricalDtype(categories=alphabet))
df = pd.get_dummies(df)

# Define X as estimators and y as target
X = df.loc[:, df.columns != 'target'].values
y = df['target'].values

# Split Data to train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# GETTING Correlation matrix
corr_mat=df.loc[:, df.columns != 'target'].corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.show()

# !! there is any correlation in estimators

#%%
# Find the best model

# Check Modeling
models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('BayesianRidge', BayesianRidge()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('RandomForest', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto')))  # kernel = linear

# Evaluate each model in turn
results = []
names = []

print("***Models Evaluations: ")
for name, model in models:
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # evaluate model
    cv_results = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)

    results.append(cv_results)
    names.append(name)
    print('   %s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.figure()
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison (R2)')
plt.show()

#%%
# !! LinearRegression, Rigide and RandomForest perform equally well.
# !! But I personally prefer RandomForestRegression since this ensemble model (combine multiple ‘individual’ (diverse) models together
# !! and deliver superior prediction power.)
# !! can almost work out of the box and that is one reason why they are very popular.

# Grid Searching Hyperparameters to find the best RandomForest Model
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

print("\n****Best Model is RandomForestRegressor with this parameters:")
print("    n_estimators: " , best_model.n_estimators )
print("    max_features: " , best_model.max_features)
print("    max_depth   : " , best_model.max_depth)


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

print("\n****Best Model Performance****")
print('   explained_variance: ', round(explained_variance,4))
print('   r2: ', round(r2,4))
print('   MAE: ', round(mean_absolute_error,4))
print('   MSE: ', round(mse,4))
print('   RMSE: ', round(np.sqrt(mse),4))

#%%
# Save Model
# serialize model
joblib.dump(best_model, 'models/final_prediction_model.mdl')

