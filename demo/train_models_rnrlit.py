# -*- coding: utf-8 -*-
"""
Created on Jun 2 2023

@author: Jonas A. Thiele
"""


###Training of prediction models with FC (100 node parcellation) of the main sample (610 subjects, 5-fold cross-validation)
###Training with different selections of brain links: links between random nodes, random links, and links between nodes of intelligence theories

#%% Imports

import pandas as pd
import numpy as np
import scipy.io as spio
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from itertools import product
import os.path
import pickle
import warnings
from lib.regress_confounds_fold import regress_confounds_fold
from lib.create_folds import create_folds
from lib.train_model import train_model
from lib.test_model import test_model
from lib.get_mask_network import get_mask_network
import sys

#Specify device to train pytorch models (cuda = gpu, or cpu, here use gpu if avaiable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def progressBar(count_value, total, suffix=''):
    bar_length = 20
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def train_models_rnrlit(state='WM', score = 'g_score', link_selection = 'links_20'):


    intelligence_score_type = score #Intelligence component: Choose from ['g_score','gf_score','gc_score']
    
    #Cognitive states during which FC was recorded
    states_FC = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
    
    
    #Choose parameters
    n_folds = 5 #Number cross-validation folds 
    n_folds_nested = 3 #Number folds for hyperparameter optimization
    
    #Path of source files
    path_file = os.path.dirname(os.path.abspath("__file__"))
    path_source = os.path.join(path_file, 'source')

    try:
        if 'nodes' in link_selection:
            method_mask = 'nodes'
            n_nodes_mask = int(link_selection.split('_')[1])
            idx_nodes_mask = []
            print('Train models with links between ' + str(n_nodes_mask) + ' randomly selected nodes') 
        elif 'links' in link_selection:
            method_mask = 'connections'
            n_nodes_mask = int(link_selection.split('_')[1])
            idx_nodes_mask = []
            print('Train model with ' + str(int(scipy.special.comb(n_nodes_mask,2))) + ' randomly selected links')
        else:
            method_mask = 'theory'
            theory = link_selection
            n_nodes_mask = []
            print('Train model with links corresponding to ' + theory + '-theory')
            
            #For method_mask = 'theory' load respective file containing nodes
            file_theory_name = 'nodes_' + theory + '.mat'
            path_file = os.path.dirname(os.path.abspath("__file__"))
            file_theory = os.path.join(path_source ,file_theory_name)
            idx_nodes_mask = spio.loadmat(file_theory)['nodes'].ravel().astype(int) - 1 #Nodes of intelligence theory
            
    except: 
        raise ValueError('Wrong link selection. Use e.g. <nodes_20>, <links_40>, <pfit>')
            
       
    
    #Folder to save trained models in
    path_models = 'res_models'
    
    
    #Read behavioral data of subjects
    data_subjects = pd.read_csv(os.path.join(path_source,'families_all_train_cens.csv'))
    
    #Read FC of the chosen state
    idx_state = states_FC.index(state) + 1
    name_FC = 'FC' + str(idx_state) + '_HCP_610_100nodes.mat'
    FC_state = spio.loadmat(os.path.join(path_source,name_FC))['FC']
    
    #Read subject order of subjects in behavioral data
    n_subjects_total = data_subjects["Family_No"].size
    
    #Read intelligence scores and confounds
    intelligence_scores = data_subjects[intelligence_score_type].to_numpy()
    confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T
    
    #Get family-wise intelligence score
    no_families = np.array(data_subjects["Family_No"])
    unique_families, unique_families_index = np.unique(no_families, return_index = True)
    name_intelligence_score_fam = intelligence_score_type + '_family'
    intelligence_scores_families = np.array(data_subjects[name_intelligence_score_fam])[unique_families_index]
    

        
    
    #%%Train models to predict observed intelligence scores
    
    #Configuration of model's hyperparameters to be chosen from (ideal combination of parameters chosen via hyperparameter optimization)
    config = {
        "n_hidden": [10, 50, 100],
        "n_layer": [1,2,3],
        "lr": [1e-2],
    }
    
    hyper_parameter = list(product(config["n_hidden"],config["n_layer"],config["lr"]))
    
        
    
    #Make folds with families stratified for intelligence scores of families
    df_target = pd.DataFrame(np.array([unique_families, intelligence_scores_families]).T,columns=['family','target'])
    fold_assignment_families = create_folds(df = df_target, n_s = n_folds, n_grp = 20)
    
    #Assign subjects to family-based folds (this ensures all subjects of a family to be in the same fold)
    fold_assignment_subjects = np.ones((n_subjects_total,2))*-1
    for fam, fold in zip(fold_assignment_families.family, fold_assignment_families.Fold):
        
        idx = np.where(data_subjects["Family_No"] == fam)[0]
        fold_assignment_subjects[idx,0] = fold
        fold_assignment_subjects[idx,1] = fam
    
    
    #Create mask with selection of specific links (links between random nodes: method_mask=nodes, random links:  
    #method_mask=connections, or links between preselected nodes:  method_mask=theory)
    mask_network, network_list, mask_edges = get_mask_network(n_nodes = n_nodes_mask, method = method_mask, idx_nodes = idx_nodes_mask)


    idx_m,idx_n = np.where(mask_network[0]==1)
    FC_network = FC_state[idx_m, idx_n,:]
    cnt_network = network_list[0]
    
        
    ID_trial = np.random.randint(0,100000) #ID for saving files 
    
        
    y_test_all = []
    y_pred_test_all = []
    y_train_all = []
    
    #Loop over cross-validation folds
    for k in range(n_folds):
        progressBar(k,5)

        #Get indexes of subjects for test and training based on folds
        test_index = np.where(fold_assignment_subjects[:,0]==k)[0]
        train_index = np.where(fold_assignment_subjects[:,0]!=k)[0]
        
        
        #Control for confounds via linear regression
        y_train, y_test = regress_confounds_fold(score = intelligence_scores, confounds = confounds, train_index = train_index, test_index = test_index)
        
        X_train = []
        X_test = []
        for el in range(FC_network.shape[0]):
    
            var_train, var_test = regress_confounds_fold(score = FC_network[el,:], confounds = confounds, train_index = train_index, test_index = test_index)
            X_train.append(var_train)
            X_test.append(var_test)

        X_train = np.array(X_train).T
        X_test = np.array(X_test).T
        
        
        ###Hyperparameter optimization (with training data only!)
        
        #Indexes of families of subjects in training data of this fold for hyperparamter optimization
        idx_familiy_optim = np.where(np.array(fold_assignment_families.Fold) != k)[0] 
        
        #Families of training subjects
        families_optim = np.array(fold_assignment_families.family)[idx_familiy_optim]
        #Families of all training subjects 
        fam_assignment_subjects_optim = fold_assignment_subjects[train_index,1]
        #Mean intelligence scores of families of training subjects
        g_scores_families_optim = np.array(fold_assignment_families.target)[idx_familiy_optim]
        
        df_target_optim =  pd.DataFrame(np.array([families_optim, g_scores_families_optim]).T,columns=['family','target'])
        fold_assignment_families_optim = create_folds(df = df_target_optim, n_s = n_folds_nested, n_grp = 20)
        
        #Assign training subjects to optimization folds
        fold_assignment_subjects_optim = np.ones((train_index.size,2))*-1 #Array for storing fold assignments
        for fam, fold in zip(fold_assignment_families_optim.family, fold_assignment_families_optim.Fold):
            
            idx = np.where(fam_assignment_subjects_optim == fam)[0]
            fold_assignment_subjects_optim[idx,0] = fold
            fold_assignment_subjects_optim[idx,1] = fam
        
        

        y_test_optim_all = []
        y_pred_test_optim_all = []
        #Loop over nested folds
        for v in range(n_folds_nested):
            
            #Indexes of training and test subjects for optimization within the training sample
            test_optim_index = np.where(fold_assignment_subjects_optim[:,0]==v)[0]
            train_optim_index = np.where(fold_assignment_subjects_optim[:,0]!=v)[0]
        
        
            #X,y for hyperparameter optimaziation loop
            y_optim_train = y_train[train_optim_index]
            y_optim_test = y_train[test_optim_index] 
            
            X_optim_train = X_train[train_optim_index, :]
            X_optim_test = X_train[test_optim_index, :]
           
            
        
            y_pred_test_optim_p = []
            y_test_optim_p = []
            #Test different combinations of hyperparameters for optimization fold
            for parameters in hyper_parameter:
                
           
                #Define parameters of model
                n_hidden = parameters[0]
                n_layer = parameters[1]
                lr = parameters[2]
                
                #Train model
                net_v, t = train_model(X = X_optim_train, y = y_optim_train, n_hidden = n_hidden, n_layer = n_layer, lr = lr, prop_valid = 0.3)
                
                #Test model
                y_pred, y = test_model(net = net_v, X_test = X_optim_test, y_test = y_optim_test)
                
                #Store observed and predicted scores of model
                y_pred_test_optim_p.append(y_pred)
                y_test_optim_p.append(y)

                
            #Store observed and predicted scores of models
            y_test_optim_all.append(y_test_optim_p)
            y_pred_test_optim_all.append(y_pred_test_optim_p)
         
        #Store observed and predicted scores of all optimization models    
        y_test_optim_all = np.squeeze(np.concatenate(y_test_optim_all, axis = 1))[0,:]
        y_pred_test_optim_all = np.squeeze(np.concatenate(y_pred_test_optim_all, axis = 1)) 
        
        #Evaluate which model hyperparameters led to best performance
        corr_p = []
        MSE_p = []
        for p in range(y_pred_test_optim_all.shape[0]):
            
            corr_p.append(scipy.stats.pearsonr(y_test_optim_all.ravel(),y_pred_test_optim_all[p,:].ravel())[0])
            MSE_p.append(mean_squared_error(y_test_optim_all.ravel(), y_pred_test_optim_all[p,:].ravel(), squared=True)) 
            
        parameters_train = hyper_parameter[np.argmax(corr_p)]
        
        
        #Hyperparameters choosen from nested loop
        n_hidden_train = parameters_train[0]
        n_layers_train = parameters_train[1]
        lr_train = parameters_train[2]
        
        
        ###Real training and testing with chosen hyperparamters:
        net, t = train_model(X = X_train, y = y_train, n_hidden = n_hidden_train, n_layer = n_layers_train, lr = lr_train, prop_valid = 0.2)
        y_pred, y = test_model(net = net, X_test = X_test, y_test = y_test)

        
        #Save observed and predicted scores of each fold
        y_pred_test_all.append(y_pred)
        y_test_all.append(y)
        y_train_all.append(y_train)
        
        #Save model and model parameters for later evaluation
        name_model = str(state) + '_' + intelligence_score_type + '_' + str(cnt_network) + '_' + str(k) + '_' + 'real' + '_' + str(ID_trial)
        path_complete = os.path.join(path_file, path_models, name_model)
        torch.save(net.state_dict(), path_complete)
        
        list_model_parameters = []
        list_model_parameters.append(n_hidden_train)
        list_model_parameters.append(n_layers_train)
        list_model_parameters.append(lr_train)
        list_model_parameters.append(X_train.shape)
        list_model_parameters.append(test_index)
        list_model_parameters.append(train_index)
        list_model_parameters.append(y_train)
        list_model_parameters.append(y_test)
        list_model_parameters.append(y_pred)
        
        name_list = name_model + '_parameters'
        path_complete = os.path.join(path_file, path_models, name_list)
        with open(path_complete, "wb") as fp:   #Pickling
            pickle.dump(list_model_parameters, fp)
            
            
                    
    #Normalizing observed and predicted intelligence scores across all folds
    y_test_norm = []
    y_pred_norm = []
    for y_train, y_test, y_pred in zip(y_train_all, y_test_all, y_pred_test_all):
        
        y_min = y_train.min()
        y_max = y_train.max()
        
        y_test_norm.append((y_test - y_min) / (y_max - y_min))
        y_pred_norm.append((y_pred - y_min) / (y_max - y_min))
        
        
    #Performance metrics
    y_observed = np.concatenate(y_test_norm).ravel()
    y_predicted = np.concatenate(y_pred_norm).ravel()
    
    
    MSE = mean_squared_error(y_observed, y_predicted, squared=True) #Squared = True for MSE, false for RMSE
    RMSE = mean_squared_error(y_observed, y_predicted, squared=False)
    MAE = mean_absolute_error(y_observed, y_predicted)
    Corr = scipy.stats.pearsonr(y_observed,y_predicted)[0]
    
    
    #plt.figure()
    #ax = sns.regplot(x=y_observed, y=y_predicted, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci = None)
    #ax.set(xlabel='g observed', ylabel='g predicted', title = f'NN Predicition test data: corr = {Corr:.2f}, MSE = {MSE:.2f}, MAE = {MAE:.2f}')
    #plt.show()
    

    trial = str(state) + '_' + intelligence_score_type + '_' + str(cnt_network) + '_' + 'real' + '_' + str(ID_trial)
    list_network = []
    list_network.append(np.concatenate(y_train_all).ravel())
    list_network.append(np.concatenate(y_test_all).ravel())
    list_network.append(np.concatenate(y_pred_test_all).ravel())
    list_network.append(y_observed)
    list_network.append(y_predicted)
    list_network.append(MSE)
    list_network.append(RMSE)
    list_network.append(MAE)
    list_network.append(Corr)
    list_network.append(mask_edges)
                        
    path_complete = os.path.join(path_file, path_models, trial)
    with open(path_complete, "wb") as fp:   
        pickle.dump(list_network, fp)
                     
    
    return Corr