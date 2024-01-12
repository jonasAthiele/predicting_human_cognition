# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""

from sklearn import linear_model

#Regressing out confounds from a variable
def regress_confounds_fold(score, confounds, train_index, test_index):
    
    regr = linear_model.LinearRegression()
    regr.fit(confounds[train_index, :], score[train_index])
    fit_train = regr.predict(confounds[train_index, :])
    residuals_train = (score[train_index] - fit_train)
    
    mean_res_train = residuals_train.mean()
    std_res_train = residuals_train.std()
    
    residuals_train = (residuals_train - mean_res_train) / std_res_train
     
    fit_test = regr.predict(confounds[test_index,:])
    residuals_test = (score[test_index] - fit_test)
    residuals_test = (residuals_test - mean_res_train) / std_res_train
    
    
    return residuals_train, residuals_test