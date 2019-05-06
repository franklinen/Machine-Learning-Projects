# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:59:30 2019

@author: MAIN
"""

import os

os.chdir(r'C:\Users\MAIN\Desktop\ML\ML Project')
os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.interactive(False)

#import dataset
df= pd.read_csv("insurance.csv")
type(df)
df.describe()
#handle missing values
df.isnull().sum()
df.dropna(axis=0,how='any',inplace=True)

#encode categorical variable using dummy labelling
df.smoker = df.smoker.map({'no':0,'yes':1})
df.sex = df.sex.map({'female':0,'male':1})
df = pd.get_dummies(df,columns=['region'], drop_first=True)

#find statistical description of data(correlation coeficient)
df.describe()
df.corrwith(df.charges)
pd.scatter_matrix(df, figsize=(6,6))
plt.show()

#check correlation with seaborn heatmap
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(40,40))
sns.heatmap(df.corr(), annot=True, cmap='Greens')

#define X,y
X=df.drop(['charges'],axis=1)
y=df.charges

#Normalise the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = (sc.fit_transform(X))
features=X.columns.values

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y.reshape(len(y),1)).ravel()

####################........Linear Regression.........#########################################
#import learning model(Linear Regression)
from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
from sklearn.feature_selection import RFE

#Run RFE and select best subset of X
estimator = LinearRegression() #use regression model for regression problem
list_r2=[]
max_r2 = 0
for i in range(1,len(X.iloc[0])+1):
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(X_scaled, y_scaled)
    adj_r2 = 1 - ((len(X)-1)/(len(X)-i-1))*(1-selector.score(X_scaled, y_scaled))
    list_r2.append(adj_r2)# mse = 
    if max_r2 < adj_r2:
        sel_features = selector.support_
        max_r2 = adj_r2
        
X_sub = X_scaled[:,sel_features]


#statistical summary of the model
import statsmodels.api as sm
X2=sm.add_constant(X_sub)  #investigate
model=sm.OLS(y,X2)
results=model.fit()
print(results.summary())

#split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_scaled, test_size=0.3, random_state=0)

model1 = LinearRegression()
#Train our model
model1.fit(X_train,y_train)
#predict with the model
y_pred = model1.predict(X_test)
b0 =model1.intercept_
b1 =model1.coef_

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('intercept & coefficients: \n', model1.intercept_, model1.coef_)

print('mean squared error (training):', mean_squared_error(y_train, model1.predict(X_train)))
print('mean squared error (testing):', mean_squared_error(y_test, y_pred))

print('R-squared score(training):', r2_score(y_train, model1.predict(X_train)))
print('R-squared score(testing):', r2_score(y_test, y_pred))

#Run K-Fold using R2
from statistics import mean, stdev
from sklearn.model_selection import KFold, cross_val_score

shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model1, X_sub, y_scaled, cv=shuffle)
print(scores)


#####################.......KNearestNeighbor Regression..................###################################
#import Learning model
from sklearn.neighbors import KNeighborsRegressor
Regressor = KNeighborsRegressor(n_neighbors=21, weights='distance', p=1) #by default p=2. tweak n_neighbords to check accuracy

#train classifier
Regressor.fit(X_train,y_train)

#predictions for test
y_pred = Regressor.predict(X_test)

#import performance measure tools
from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
print('R-squared score(training):', r2_score(y_train, Regressor.predict(X_train)))
print('R-squared score(testing):',r2_score(y_test, y_pred))

#split X and y into K-Folds
#Run K-Fold using R2
shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_sub, y_scaled, cv=shuffle)
print(scores)

#####################..........Random Forest..........#####################################################

#import Learning Model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(X_train,y_train)

model.score(X_test,y_test)

mse = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
print('R-squared score(training):', r2_score(y_train, model.predict(X_train)))
print('R-squared score(testing):',r2_score(y_test, y_pred))

#Run K-Fold using R2
shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_sub, y_scaled, cv=shuffle)
print(scores)

#check Gridsearch for model2 = RandomForest
from sklearn.model_selection import GridSearchCV
param_dict= {'n_estimators':range(2,30), 'max_depth':range(1,30)}
model = GridSearchCV(model,param_dict)
model.fit(X_train,y_train)

model.score(X_test,y_test)
model.best_params_

##############............AdaBoost.............#############################################################

#import learning model
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()

model.fit(X_train,y_train)

model.score(X_test,y_test)

mse = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
print('R-squared score(training):', r2_score(y_train, model.predict(X_train)))
print('R-squared score(testing):',r2_score(y_test, y_pred))

#split X and y into K-Folds
#Run K-Fold using R2
shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_sub, y_scaled, cv=shuffle)
print(scores)

#check Gridsearch for model3 = AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
param_dict= {'n_estimators':range(2,30)}
model = GridSearchCV(model,param_dict)
model.fit(X_train,y_train)

model.score(X_test,y_test)
model.best_params_

###############.............SVR...........####################################################################

from sklearn.svm import SVR
model = SVR(kernel='rbf')

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

model.score(X_test,y_test)

mse = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
print('R-squared score(training):', r2_score(y_train, model.predict(X_train)))
print('R-squared score(testing):',r2_score(y_test, y_pred))

#split X and y into K-Folds
#Run K-Fold using R2
shuffle = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_sub, y_scaled, cv=shuffle)
print(scores)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Factors')
plt.ylabel('Insurance Charges')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Factors')
plt.ylabel('Insurance Charges')
plt.legend()
plt.show()