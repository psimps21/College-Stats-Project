#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:42:00 2019

@author: VinayNair
"""

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


Degree_df = pd.read_csv("degrees-that-pay-back.csv")
College_df = pd.read_csv("salaries-by-college-type.csv")
Region_df = pd.read_csv("salaries-by-region.csv")

College_df.columns = ['School Name','Major','StartingMedian','MidCareerMedian','midp10','midp25','midp75','midp90']

dollar_cols = ['StartingMedian','MidCareerMedian','midp10','midp25','midp75','midp90']

for x in dollar_cols:
    College_df[x] = College_df[x].str.replace("$","")
    College_df[x] = College_df[x].str.replace(",","")
    College_df[x] = pd.to_numeric(College_df[x])
    
College_df = pd.get_dummies(College_df, columns = ['Major'])
College_work = College_df.dropna()
y = College_work['midp90']
x = College_work.loc[:, (College_work.columns != 'midp90') & 
                      (College_work.columns != 'midp10') & 
                      (College_work.columns != 'midp25') & 
                      (College_work.columns !=  'School Name')]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test['StartingMedian'],y_test,color='red')
plt.scatter(X_test['StartingMedian'],lin_reg_2.predict(poly_reg.fit_transform(X_test)),color='blue')
plt.title('Polynomial regression Model built to predict Mid-Career 90th Percentile Salary')
plt.xlabel('Starting salary')
plt.ylabel('Salary in Red and Predicted Salary in blue');

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse_val.append(rmse)
print("Polynomial Regression RMSE")
print(rmse)