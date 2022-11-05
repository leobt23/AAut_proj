import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


#################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Step 1 - Get data
X = np.load("Xtrain_Regression1.npy")

y = np.load("Ytrain_Regression1.npy")

x_test = np.load("Xtest_Regression1.npy")

# Análise dos dados
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(y)
data=pd.concat([df_y,df_x.reindex(df_y.index)],axis=1)
data.set_axis(['Y','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'], axis='columns', inplace=True)

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Plot the dataframe
ax = data[['Y','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].plot(kind='box')

# Save the plot
plt.savefig('boxplot1.png')

X = StandardScaler(with_std=False).fit_transform(X)
y = StandardScaler(with_std=False).fit_transform(y)
x_test = StandardScaler(with_std=False).fit_transform(x_test)


# Step 2 - Define alphas for Ridge and Lasso:

# Ridge
ridge=Ridge()
parameters={'alpha':[0.000000001,0.001,0.05,0.1,0.1633066,0.17,0.18,0.19,0.2,0.3,0.4,1,20]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)

alphas = np.linspace(0.15, 0.23, 500)
regressorL = RidgeCV(alphas=alphas, cv=10)
regressorL.fit(X, y.ravel())
print("Alpha Ridge:" )
print(regressorL.alpha_) 

# Lasso
lasso=Lasso()
parameters={'alpha':[0.000000001,0.00099,0.001,0.0014,0.002,0.003,0.005,0.05,0.1,0.2,0.3,0.4,1,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)

alphas = np.linspace(0.0001, 0.004, 500)
regressorL = LassoCV(alphas=alphas, cv=10)
regressorL.fit(X, y.ravel())
print(regressorL.alpha_) 

#Polinomial 
x_poly = PolynomialFeatures(2).fit_transform(X)


# Auxiliar functions
def Predict_val(model, X_train, X_validation, y_train):
    model.fit(X_train, y_train)
    return model.predict(X_validation)

def Average(argumento):
    return sum(argumento) / len(argumento)

# Arrays of errors
array_mse_linear_regr = [];
array_mse_ridge = [];
array_mse_lasso = [];
array_mse_poli = [];
outcomes_predicted = [];

# Selection models 
kfold = KFold(10)
linear_regression = LinearRegression()
ridge = Ridge(alpha = 0.1633066132264529)
lasso = Lasso(alpha = 0.0007486973947895793)
poli = LinearRegression()


# Step 3 - Split data into data for training and data for tests
# Calculate mse of each iteration of each linear model

for train_index, validate_index in kfold.split(X, y):
    X_train, X_validation = X[train_index], X[validate_index]
    y_train, y_validation = y[train_index], y[validate_index]
    
    y_predicted_LR = Predict_val(linear_regression, X_train, X_validation, y_train)
    y_predicted_ridge = Predict_val(ridge, X_train, X_validation, y_train)
    y_predicted_lasso = Predict_val(lasso, X_train, X_validation, y_train)

    array_mse_linear_regr.append(mse(y_validation, y_predicted_LR))
    array_mse_ridge.append(mse(y_validation, y_predicted_ridge))
    array_mse_lasso.append(mse(y_validation, y_predicted_lasso))

for train_index, validate_index in kfold.split(x_poly, y):
    X_train, X_validation = x_poly[train_index], x_poly[validate_index]
    y_train, y_validation = y[train_index], y[validate_index]
    y_test = poli.fit(X_train, y_train).predict(X_validation)
    array_mse_poli.append(mse(y_validation, y_test))


# Step 4 - Calculate SSE of each Linear Model
print("SSE Linear Regression:", Average(array_mse_linear_regr))
print("SSE Ridge:", Average(array_mse_ridge))
print("SSE Lasso:", Average(array_mse_lasso))
print("SSE Polin:", Average(array_mse_poli))
print("SSE Polin array:", array_mse_poli)

# Coefficients of models
linear_regression = LinearRegression()
ridge = Ridge(alpha = 0.1633066132264529)
lasso = Lasso(alpha = 0.0007486973947895793)
poli = LinearRegression()

np.set_printoptions(precision=3,suppress=True)
print("linear",linear_regression.fit(X, y).coef_)
print("ridge",ridge.fit(X, y).coef_)
print("lasso",lasso.fit(X, y).coef_)

print("linear",linear_regression.fit(X, y).intercept_)
print("ridge",ridge.fit(X, y).intercept_)
print("lasso",lasso.fit(X, y).intercept_)

# Step 5 - Deliver results of best Linear Model for the problem

choosed_linear_model = Lasso(alpha = 0.0019757515030060123)
choosed_linear_model.fit(X, y)
final_results = choosed_linear_model.predict(x_test)
betas = choosed_linear_model.coef_
intercept = choosed_linear_model.intercept_
np.save('y_results.npy', final_results)
