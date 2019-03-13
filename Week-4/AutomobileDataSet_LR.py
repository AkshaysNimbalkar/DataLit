import pandas as pd
import numpy as np

df = pd.read_csv("E:/DataScience_Study/3months/Data-Lit/week4-Regression/4.2_Linear_regression_python/Automobile_data.csv")

df.tail()
# list the columns
df.columns

# check the data types for all columns: as we want Numeric values to important types
df.dtypes

# errors : {‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
# If ‘raise’, then invalid parsing will raise an exception
# If ‘coerce’, then invalid parsing will be set as NaN
# If ‘ignore’, then invalid parsing will return the inpu

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Now will remove Na's
# df.dropna(how='all', inplace=True)
df.dropna(subset=['horsepower', 'price'], inplace=True)

# Now will calculate pearson co-relation cofficient by using scipy
from scipy.stats.stats import pearsonr
pearsonr(x=df['horsepower'], y=df['price'])


import seaborn as sns

# before applying Linear regression
sns.regplot(x=df['horsepower'], y=df['price'], data=df, scatter=True, fit_reg=True)

# split data into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.25)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# the linear regression model expects a 2d array, so we add an extra dimension with reshape
# input : [1 2, 3] output : [[1], [2], [3]]
# this allows us to regress multiple variable later
# OR we can do like this: train_data = train[['horsepower']]
train_x = np.array(train['horsepower']).reshape(-1, 1)
train_y = np.array(train['price'])

# perform linear Regression
lin_reg.fit(train_x, train_y)

slope = np.asscalar(np.squeeze(lin_reg.coef_))
intercept = lin_reg.intercept_

# after linear Regression: # line_kws for customized reg. line
sns.regplot(x=df['horsepower'], y=df['price'], data=df,
            line_kws={'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept), 'color':'red'})

# prediction
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def predict_metrics(lr, x, y):
    pred = lr.predict(x)
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    return mse, mae, r2


train_mae, train_mse, train_r2 = predict_metrics(lin_reg, train_x, train_y)

print("train mae: ", train_mae, "train mse: ", train_mae, "train_r2: ", train_r2)

# calculate with test data so we can compare:
test_x = np.array(test['horsepower']).reshape(-1, 1)
test_y = np.array(test['price'])

test_mae, test_mse, test_r2 = predict_metrics(lin_reg, test_x, test_y)

print("train mae: ", train_mae, "train mse: ", train_mae, "train_r2: ", train_r2)
print("test mae: ", test_mae, "test mse: ", test_mae, "test_r2: ", test_r2)

############################### Multiple Linear regression ####################

df.columns
cols = ['horsepower', 'engine-size', 'peak-rpm', 'length', 'width', 'height']

# data preprocessing
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['horsepower', 'price'], inplace=True)

# Now will calculate pearson co-relation cofficient by using scipy
from scipy.stats.stats import pearsonr

for col in cols:
    print(col, pearsonr(df[col], df['price']))
# by seeing co-relation cofficients we can eliminate peak-rpm and height as they have less corelation

# split data in train and test
from sklearn.model_selection import train_test_split

model_cols = ['horsepower', 'engine-size', 'length', 'width']
multi_x = np.column_stack(tuple(df[col] for col in model_cols))
target = df.loc[:, 'price']


multi_x.shape
target.shape


multi_train_x, multi_test_x, multi_train_y, multi_test_y = train_test_split(multi_x, target, test_size=0.25)

# Fit the model as before
from sklearn.linear_model import LinearRegression


linR = LinearRegression(normalize=True).fit(multi_train_x, multi_train_y)
multi_intercept = linR.intercept_
multi_coeffs = dict(zip(model_cols, linR.coef_))

print("Intercept: ", multi_intercept)
print("coefficients: ", multi_coeffs)

# calculate error matrix:
mae_train, mse_train, r2_train = predict_metrics(linR, multi_train_x, multi_train_y)
mae_test, mse_test, r2_test = predict_metrics(linR, multi_test_x, multi_test_y)

print("train mae:", mae_train," train mse:", mse_train," R2-train", r2_train)
print("test mae:", mae_test, " test mse:",mse_test, " R2 test", r2_train)


### Now will apply Regularization  to tune our model

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(multi_train_x, multi_train_y)

ridge_regressor.best_params_
ridge_regressor.best_score_

ridge_regressor.score(multi_test_x, multi_test_y)

### LAsso
from sklearn.linear_model import Lasso

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso = Lasso()

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5 )
lasso_regressor.fit(multi_test_x, multi_test_y)

lasso_regressor.best_params_   # {'alpha': 1}

lasso_regressor.best_score_

lasso_regressor.score(multi_test_x, multi_test_y)

# we can see that there is hugh improvement in our model.