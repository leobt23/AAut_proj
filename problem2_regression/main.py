import numbers
from tkinter.tix import DirSelectDialog
import numpy as np
from numpy import array, mean, std
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.stats as ss
from pca import pca
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


# Step 1 - Get data
X = np.load("Xtrain_Regression2.npy")
X_df = pd.DataFrame(X)
y = np.load("Ytrain_Regression2.npy")
y_df = pd.DataFrame(y)
x_test = np.load("Xtest_Regression2.npy")


# Identify outliers


###################################################################################
# Nota: Verificar se a distribuição é normal
# Z-SCORE 

mean = np.mean(X)
std = np.std(X)
#print('mean of the dataset is', mean)
#print('std. deviation is', std)

array_ot = []
z = np.abs(ss.zscore(X))
array_outliers = np.where(z > 2)
array_ot = array_outliers[0]

zscore_outliers = {array_outliers[0][i]: array_outliers[1][i] for i in range(len(array_outliers[0]))}
 
#print(f"Z-Score results (Std = 2): {zscore_outliers}" )


###################################################################################
# IQR

dfx = pd.DataFrame(X)
dfy = pd.DataFrame(y)

def find_iqr(x):
    return np.subtract(*np.percentile(x,[75,25]))

def find_q1(x):
    return np.percentile(x, 25)

def find_q3(x):
    return np.percentile(x, 75)

iqr = dfx.apply(find_iqr)
q1 = dfx.apply(find_q1)
q3 = dfx.apply(find_q3)

outliers_key = [];
outliers_value = [];
for i in range(0, 10):
    upper_bound = q3[i]+(1.0*iqr[i])
    lower_bound = q1[i]-(1.0*iqr[i])
    for j in range(0, 99):
        if (dfx[i][j] <= lower_bound) | (dfx[i][j] >= upper_bound):
            outliers_value.append(i)
            outliers_key.append(j)
            #outliers.append([j, i, dfx[i][j]]) #outliers.append(j) 
iqr_outliers = {outliers_key[i]: outliers_value[i] for i in range(len(outliers_key))}
iqr_outliers = dict(sorted(iqr_outliers.items()))

#print(f"IQR results: {iqr_outliers}")

###################################################################################
# PCA 

#model = pca(alpha=0.05)
# Fit transform
#out = model.fit_transform(X)
#print(out['outliers'])
#model.biplot(legend=True, SPE=True, hotellingt2=False)

"""
# calculate summary statistics
data_mean, data_std = mean(X), std(X)
print(f"Mean: {data_mean}      Devivation: {data_std}")
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in X if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))
# remove outliers
#outliers_removed = [x for x in data if x >= lower and x <= upper]
#print('Non-outlier observations: %d' % len(outliers_removed))

#for column in X_df:
#    plt.figure(figsize=(7, 7))
#    plt.scatter(X_df[column], y)
#plt.show()

#Plots
#fig1, ax1 = plt.subplots()
#ax1.boxplot(X, whis = 0.5)
"""


###################################################################################
#Ransac
# Initializing the model
Ransac = RANSACRegressor()
huber = HuberRegressor()


# training the model
Ransac.fit(X, y)
huber.fit(X, y.ravel())
# inlier mask
inlier_mask = Ransac.inlier_mask_
ransac_outliers = []
huber_outliers = []

mask_huber = huber.outliers_

count = 0
for i in inlier_mask:
    if i == False:
        ransac_outliers.append(count)
    count += 1

count2 = 0
for i in mask_huber:
    if i == True:
        huber_outliers.append(count2)
    count2 += 1

print(ransac_outliers)
ransac_outliers = [15, 18, 24, 29, 30, 33, 36, 47, 48, 62, 63, 65, 71, 72, 83, 88, 93, 95]
#34 e 70


#a_file = open("test.txt", "w")
#np.savetxt(a_file, ransac_outliers)
#a_file.close()
X_inliers = X_df.drop(X_df.index[ransac_outliers])
y_inliers = y_df.drop(y_df.index[ransac_outliers])
X_inliers = X_inliers.reset_index(drop=True)
y_inliers = y_inliers.reset_index(drop=True)
#print(X_inliers)
#print(y_inliers)
X_np = X_inliers.to_numpy()
y_np = y_inliers.to_numpy()
#print(len(X_inliers))
#print(ransac_outliers)
#print(len(huber_outliers))

###################################################################################
# Compare outliers

"""
interception_array0 = []
interception_array1 = []

for i in outliers_key:
    for j in array_outliers[0]:
        if i == j:
            interception_array0.append(i)

for i in interception_array0:
    for j in ransac_outliers:
        if i == j:
            interception_array1.append(i)
print(f"Array of intercepted outliers: {interception_array1}")
"""

###################################################################################
#Predict with inliers

# Ridge
ridge=Ridge()
parameters={'alpha':[-2,-1,-0.02925851703406812,0.000000001,0.001,0.05,0.10,0.11,0.12,0.18,0.19,0.2,0.3,0.4,1,20, 34.9997995991984, 40, 100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(X_np,y_np.ravel())
print("RIDGE")
print(ridge_regressor.best_params_)
#print(ridge_regressor.best_score_)
alphas = np.linspace(0.35, 36, 500)
regressorL = RidgeCV(alphas=alphas, cv=10)
regressorL.fit(X_np, y_np.ravel())
print("Alpha Ridge:" )
print(regressorL.alpha_) 

# Lasso
lasso=Lasso()
parameters={'alpha':[0.000000001,0.00042104208416833663,0.00099,0.001,0.0014,0.002,0.003,0.005,0.05,0.1,0.2,0.3,0.4,1,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=10)
lasso_regressor.fit(X_np,y_np)
print("LASSO")
print(lasso_regressor.best_params_)
#print(lasso_regressor.best_score_)
alphas = np.linspace(0.0001, 0.009, 500)
regressorL = LassoCV(alphas=alphas, cv=10)
regressorL.fit(X_np, y_np.ravel())
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
ridge = Ridge(alpha = 34.9997995991984)
lasso = Lasso(alpha = 0.00042104208416833663)
poli = LinearRegression()


# Step 3 - Split data into data for training and data for tests
# Calculate mse of each iteration of each linear model

for train_index, validate_index in kfold.split(X_np, y_np):
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
print("SSE LineR:", Average(array_mse_linear_regr))
print("SSE Ridge:", Average(array_mse_ridge))
print("SSE Lasso:", Average(array_mse_lasso))
print("SSE Polin:", Average(array_mse_poli))

"""
teste123 = pd.DataFrame(X_np)
teste1234 = pd.DataFrame(y_np)

print(teste123)
print(teste1234)

choosed_linear_model = Ridge(alpha = 0.1)
choosed_linear_model.fit(X_np, y_np)
final_results = choosed_linear_model.predict(x_test)
np.save('y_results.npy', final_results)

resultssss = np.load("y_results.npy")
yyy = pd.DataFrame(resultssss)
print(yyy)
"""

###################################################################################
# Predict 
# Models
"""
linear_regression = LinearRegression()


# Auxiliar functions
def predict_val(model, X_train, X_validation, y_train):
    if(model == huber):
        model.fit(X_train, y_train.ravel())
    else:
        model.fit(X_train, y_train)
    return model.predict(X_validation)

def average(argumento):
    return sum(argumento) / len(argumento)

# Arrays of errors
array_mse_ransac = [];
array_mse_linear_regr = [];
array_mse_huber = [];

# Selection models 
loo = LeaveOneOut() 

# Step 3 - Split data into data for training and data for tests
# Calculate mse of each iteration of each linear model

# Remove Outliers


for train_index, validate_index in loo.split(X, y):
    X_train, X_validation = X[train_index], X[validate_index]
    y_train, y_validation = y[train_index], y[validate_index]
    
    y_predicted_ransac = predict_val(Ransac, X_train, X_validation, y_train)
    y_predicted_LR = predict_val(linear_regression, X_train, X_validation, y_train)
    y_predicted_huber = predict_val(huber, X_train, X_validation, y_train)

    array_mse_ransac.append(mse(y_validation, y_predicted_ransac))
    array_mse_linear_regr.append(mse(y_validation, y_predicted_LR))
    array_mse_huber.append(mse(y_validation, y_predicted_huber))


###################################################################################
# Step 4 - Calculate SSE of each Linear Model
#print("SSE Linear Regression:", average(array_mse_linear_regr))
#print("SSE Ransac:", average(array_mse_ransac))
#print("SSE Huber:", average(array_mse_huber))
"""





#############################################################################

def my_pca(X):
  # returns transformed X, prin components, var explained
  dim = len(X[0])  # n_cols
  means = np.mean(X, axis=0)
  z = X - means  # avoid changing X
  square_m = np.dot(z.T, z)
  (evals, evecs) = np.linalg.eig(square_m)
  trans_x = np.dot(z, evecs[:,0:dim])
  prin_comp = evecs.T
  v = np.var(trans_x, axis=0, ddof=1)  # sample var
  sv = np.sum(v)
  ve = v / sv
  # order everything based on variance explained
  ordering = np.argsort(ve)[::-1]  # sort high to low
  trans_x = trans_x[:,ordering]
  prin_comp = prin_comp[ordering,:]
  ve = ve[ordering]
  return (trans_x, prin_comp, ve)

def reconstructed(X, n_comp, trans_x, p_comp):
  means = np.mean(X, axis=0)
  result = np.dot(trans_x[:,0:n_comp], p_comp[0:n_comp,:])
  result += means
  return result

def recon_error(X, XX):
  diff = X - XX
  diff_sq = diff * diff
  errs = np.sum(diff_sq, axis=1)
  return errs


#print("\nBegin Iris PCA reconstruction error ")
np.set_printoptions(formatter={'float': '{: 0.1f}'.format})



col_divisors = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,1])
X = X / col_divisors
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

(trans_x, p_comp, ve) = my_pca(X)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

XX = reconstructed(X, 4, trans_x, p_comp)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

XX = reconstructed(X, 2, trans_x, p_comp)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

re = recon_error(X, XX)
#print("\nReconstruction errors using two components: ")
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

re
count3 = 0
pca_outliers = []
for i in re:
    if i > 10:
        pca_outliers.append(count3)
    count3 += 1
#print(pca_outliers)
#print(ransac_outliers)
#print(huber_outliers)

#print("\nEnd PCA reconstruction error ")