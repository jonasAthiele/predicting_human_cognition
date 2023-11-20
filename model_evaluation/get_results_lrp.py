# -*- coding: utf-8 -*-
"""
Created on Apr 16 2023

@author: Jonas A. Thiele
"""

### Get results of prediction models trained in the main sample
### (610 subjects of the HCP) with FC of different numbers of most
### relevant links (edges) determined via stepwise LRP

#%% Imports

import numpy as np
import os.path
import pickle

#%% Set parameters

states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']

name_score = 'g_score' #Intelligence component: Choose from ['g_score','gf_score','gc_score']
path_models = 'E:\\Transfer\\Results\\relevance_1000'
name_models = 'res_g_lrp' #Folder with saved models
path_complete = os.path.join(path_models, name_models) #Path of models 


networks =  ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

n_nodes = 100 #Number of brain nodes used in functional connectivity
n_folds = 5 #Number cross-validation folds


#%% Read results of models        


corr_all = [] #List for performances of models
relevance_all = [] #List of relevance of links used for model training 

#Number of iterations per state
cnt_models = np.zeros((len(states),43))

#Loop over states
for n_state, state in enumerate(states):

    #Get names of files of link selection (files saved in training of models within the main sample)
    n_network = 0
    name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'real'
    files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files)]
    
    #Get all IDs of saved files
    IDs_iteration = []
    for f in files_iteration:
        
        ID = f.split('_')[-4]
        IDs_iteration.append(ID)
            
    IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
    
    cnt_models[n_state, n_network] = len(IDs_iteration_unique)
    
    #Loop over IDs
    corr_id = []
    relevance_id = []
    for id_n in IDs_iteration_unique[0:10]:
        
        #Files with predicted intelligence scores
        name_file = state + '_' + name_score + '_' + str(n_network) + '_real_' + id_n + '_relevant_edges_ypred'         
        path_file_complete = os.path.join(path_complete, name_file)
        with open(path_file_complete, 'rb') as f:
            ypred = pickle.load(f)
           
        #Files with observed intelligence scores
        name_file = state + '_' + name_score + '_' + str(n_network) + '_real_' + id_n + '_relevant_edges_ytest'      
        path_file_complete = os.path.join(path_complete, name_file)
        with open(path_file_complete, 'rb') as f:
            ytest = pickle.load(f)
            
        #Files with indexs of relevant links (edges) with which models were trained with
        name_file = state + '_' + name_score + '_' + str(n_network) + '_real_' + id_n + '_relevant_edges_idx'             
        path_file_complete = os.path.join(path_complete, name_file)
        with open(path_file_complete, 'rb') as f:
            relevance = pickle.load(f)
         
        
        #Loop over different numbers of most relevant links (edges) models are trained with
        corr_n_edges = []
        relevance_n_edges = []
        for n_edges in range(len(relevance[0])):
            
            y_test_temp = [] #Observed intelligence scores across folds
            y_pred_temp = [] #Predicted intelligence scores across folds
            idx_temp = np.zeros(4950) #To count number each link (edge) was rated as one of n most relevant links by LRP
            
            #Loop over cross-validation folds
            for k in range(n_folds):
                
                y_pred_temp.append(ypred[k][n_edges]) #Observed intelligence scores
                y_test_temp.append(ytest[k][n_edges]) #Predicted intelligence scores
                
                idx_remain = relevance[k][n_edges]
                idx_temp[idx_remain] += 1
            
            x_temp = np.concatenate(y_test_temp).ravel()
            y_temp = np.concatenate(y_pred_temp).ravel()
            
            #Normalizing
            x = (x_temp - x_temp.min()) / (x_temp.max() - x_temp.min())
            y = (y_temp - x_temp.min()) / (x_temp.max() - x_temp.min())
            
            corr = np.corrcoef(x,y)[0,1]
            corr_n_edges.append(corr)    
            
            relevance_n_edges.append(idx_temp)
                
        corr_id.append(corr_n_edges)
        relevance_id.append(relevance_n_edges)
        
    relevance_id_sum = np.sum(np.array(relevance_id),0)
    relevance_all.append(relevance_id_sum)    
    corr_all.append(np.array(corr_id))

corr_all = np.array(corr_all)           
            

#Create masks with relevance values of links of the n most relevant links
len_idx = [45, 190, 435, 780, 1000] #Number of relevant links in mask (cut off)
masks_relevance_all = []
#Loop over states
for n_state, state in enumerate(states):

    masks_relevance = []
    
    #Loop over diferent numbers of relevant links (edges)
    for cnt, relevance in enumerate(relevance_all[n_state]):
        
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min()) #Normalizing between 0 and 1
        
        idx_sort = np.argsort(relevance)[::-1] #Sort by relevance
        
        #Create array with relevance of n most relevant edges, set others to zero
        idx_relevant = np.zeros(np.triu_indices(n_nodes,1)[0].size)
        idx_relevant[idx_sort[0:len_idx[cnt]]] = relevance[idx_sort[0:len_idx[cnt]]]
        
        #Create mask with relevance of n most relevant edges
        mask = np.zeros((n_nodes,n_nodes))
        idx_x, idx_y = np.triu_indices(n_nodes,1)
        mask[idx_x, idx_y] = idx_relevant
        mask[idx_y, idx_x] = idx_relevant
        
        
        masks_relevance.append(mask)
    masks_relevance_all.append(masks_relevance)
    
# Save results
np.save('mask_g_relevants.npy', masks_relevance_all)
np.save('corr_g_main_lrp.npy', corr_all)
