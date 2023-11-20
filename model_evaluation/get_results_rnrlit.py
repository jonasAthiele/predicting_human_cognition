# -*- coding: utf-8 -*-
"""
Created on Jun 2 2023

@author: Jonas A. Thiele
"""

### Get results of prediction models trained in the main sample
### (610 subjects of the HCP) with FC (links) between specific nodes (corresponding to intelligence theories),
### links between randomly chosen nodes, and randomly choosen links

import numpy as np
import os.path
import pickle
import scipy.stats

#%% Set parameters

name_score = 'g_score' #Intelligence component (here only g-score)
path_models = 'E:\\Transfer\\Results' 
name_models = 'res_20nodes_nodes_rand' #Folder where files are 
path_complete = os.path.join(path_models, name_models) #Path of files

#States
states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']


#%% Read results       

corr_all = [] #Lists to store performances of models

cnt_models = np.zeros((len(states),43)) #Number of permutations per state

#Loop over states
for n_state, state in enumerate(states):


    #Get names of files of link selection (files saved in training of models within the main sample)
    n_network = 0  
    name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'real'
    files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files)]
    
    #Get all IDs of saved files
    IDs_iteration = [] 
    for f in files_iteration:
        
        ID = f.split('_')[-1]
        IDs_iteration.append(ID)
            
    IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
    
    cnt_models[n_state, n_network] = len(IDs_iteration_unique)
    
    #Loop over IDs
    corr_id = [] #List for performances of models 
    for id_n in IDs_iteration_unique[0:100]:
        
        #Get path of file with results from ID
        name_file = state + '_' + name_score + '_' + str(n_network) + '_real_' + id_n
        path_file_complete = os.path.join(path_complete, name_file)

        #Get results of file ID 
        with open(path_file_complete, 'rb') as f:
            results = pickle.load(f)

        #Observed scores y_test: results[1]
        #Predicted scores y_pred: results[2]
           
        #Normalization
        x = (results[1] - results[1].min()) / (results[1].max() - results[1].min())
        y = (results[2] - results[1].min()) / (results[1].max() - results[1].min())
        
        corr_id.append(scipy.stats.pearsonr(x, y)[0])

    corr_all.append(corr_id)
       
corr_all = np.array(corr_all)


name_save = 'corr_g_main' + name_models
np.save(name_save, corr_all)


