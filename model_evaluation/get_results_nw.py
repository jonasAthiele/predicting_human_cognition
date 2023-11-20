# -*- coding: utf-8 -*-
"""
Created on Jun 05 2023

@author: Jonas A. Thiele
"""

###Read results of models trained with network-specific links on the main sample (610 subjects) via 5-fold cross-validation


import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os.path
import pickle

#%% Set parameters 

states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']

name_score = 'g_score'
path_models = 'res_models_g'
path_file = 'E:\\Transfer\\Results'
path_complete = os.path.join(path_file, path_models)


#%% Read results models       

#Lists to store results in
corr = []
MSE = []
RMSE = []
MAE = []

cnt_models = np.zeros((len(states),43)) #Number of iterations per state, here should be 10 (10 times with varying startified folds) 

#Get results of models trained with normal order of intelligence scores
#Loop over states
for n_state, state in enumerate(states):
    print(state)
    
    corr_n = []
    MSE_n = []
    RMSE_n = []
    MAE_n = []
    
    
    for n_network in range(43):
        
        #Names of files with results
        name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'real' #real = model trained with normal order of intelligence scores
        files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files)]
        
        #Get all IDs of saved files
        IDs_iteration = [] 
        for f in files_iteration:
            
            ID = f.split('_')[-1]
            IDs_iteration.append(ID)
                
        IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
        
        cnt_models[n_state, n_network] = len(IDs_iteration_unique)
        
        corr_id = []
        MSE_id = []
        RMSE_id = []
        MAE_id = []

        #Loop over file IDs 
        for id_n in IDs_iteration_unique[0:10]:
            
            
            name_file = state + '_' + name_score + '_' + str(n_network) + '_real_' + id_n
            path_file_complete = os.path.join(path_file, path_models, name_file)

            #Get results of file ID 
            with open(path_file_complete, 'rb') as f:
                results = pickle.load(f) #Results of one iteration of 5-fold cross-validation (concatenated results from 5 models, 1 model per fold)

                #Observed scores y_test: results[1]
                #Predicted scores y_pred: results[2]
                
                
            #Normalization
            x = (results[1] - results[1].min()) / (results[1].max() - results[1].min())
            y = (results[2] - results[1].min()) / (results[1].max() - results[1].min())

            corr_id.append(scipy.stats.pearsonr(x, y)[0])
            MSE_id.append(mean_squared_error(x, y, squared=True))
            RMSE_id.append(mean_squared_error(x, y, squared=False))
            MAE_id.append(mean_absolute_error(x, y))


        corr_n.append(corr_id)        
        MSE_n.append(MSE_id)
        RMSE_n.append(RMSE_id)
        MAE_n.append(MAE_id)


    corr.append(corr_n)    
    MSE.append(MSE_n)
    RMSE.append(RMSE_n)
    MAE.append(MAE_n)


corr = np.array(corr)
MSE = np.array(MSE)
RMSE = np.array(RMSE)
MAE = np.array(MAE)



#%% Read results null models (trained with permuted order of intelligence scores)
        
cnt_models_perm = np.zeros((len(states),43)) #Number of permutations, here 100

#Lists for storing results
MSE_perm = []
RMSE_perm = []
MAE_perm = []
corr_perm = []

#Loop over states
for n_state, state in enumerate(states):
    print(state)
    
    MSE_n = []
    RMSE_n = []    
    MAE_n = []    
    corr_n = []
    for n_network in range(43):
        
        #Names of files with results
        name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'perm' #perm = null model = model trained with permuted intelligence scores  
        files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files)]
        
        
        IDs_iteration = []
        for f in files_iteration:
            
            ID = f.split('_')[-1]
            IDs_iteration.append(ID)
                
        IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
        cnt_models_perm[n_state, n_network] = len(IDs_iteration_unique)
        
        MSE_id = []
        RMSE_id = []
        MAE_id = []
        corr_id = []

        for id_n in IDs_iteration_unique[0:100]:
            
            y_pred_test_all = []
            y_test_all = []
            
            name_file = state + '_' + name_score + '_' + str(n_network) + '_perm_' + id_n

                
            path_file_complete = os.path.join(path_file, path_models, name_file)

            
            with open(path_file_complete, 'rb') as f:
                results = pickle.load(f) #Results of one iteration of 5-fold cross-validation (concatenated results from 5 models, 1 model per fold)

                #results[1] = y_test #Observed scores
                #results[2] = y_pred #Predicted scores
            
            x = (results[1] - results[1].min()) / (results[1].max() - results[1].min())
            y = (results[2] - results[1].min()) / (results[1].max() - results[1].min())

            
            MSE_id.append(mean_squared_error(x, y, squared=True))
            RMSE_id.append(mean_squared_error(x, y, squared=False))
            MAE_id.append(mean_absolute_error(x, y))
            corr_id.append(scipy.stats.pearsonr(x, y)[0])

        
        corr_n.append(corr_id)    
        MSE_n.append(MSE_id)
        RMSE_n.append(RMSE_id)
        MAE_n.append(MAE_id)
    
    corr_perm.append(corr_n)    
    MSE_perm.append(MSE_n)
    RMSE_perm.append(RMSE_n)
    MAE_perm.append(MAE_n)


corr_perm = np.array(corr_perm)
MSE_perm = np.array(MSE_perm)
RMSE_perm = np.array(RMSE_perm)
MAE_perm = np.array(MAE_perm)       


#%% Save results

np.save('corr_g_main', corr)
np.save('corr_perm_g_main', corr_perm)

np.save('MSE_g_main', MSE)
np.save('MSE_perm_g_main', MSE_perm)

np.save('RMSE_g_main', RMSE)
np.save('RMSE_perm_g_main', RMSE_perm)

np.save('MAE_g_main', MAE)
np.save('MAE_perm_g_main', MAE_perm)


      