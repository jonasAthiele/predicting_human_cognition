# -*- coding: utf-8 -*-
"""
Created on Nov 4 2022

@author: Jonas A. Thiele
"""
import pandas as pd
import numpy as np
import scipy.io as spio 
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import product
import os.path
import pickle
from lib.regress_confounds_fold import regress_confounds_fold
from lib.create_folds import create_folds
from lib.train_model import train_model
from lib.test_model import test_model
import sys




def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def progressBar(count_value, total, suffix=''):
    bar_length = 20
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

###Cross-validated training of prediction models from FC (100 node parcellation) of main sample (610 subjects)
###Train with different network-specific selections of functional brain links

def train_models_nw(state='WM', score = 'g_score', option='real', link_selection = 'all'):
  
    #Number of brain nodes used in functional connectivity
    n_nodes = 100

    #Folder to save trained models in
    path_models = 'res_models'
    
    #Path of source files
    path_file = os.path.dirname(os.path.abspath("__file__"))
    path_source = os.path.join(path_file, 'source')
    
    #Read behavioral data of subjects
    data_subjects = pd.read_csv(os.path.join(path_source,'families_all_train_cens.csv'))



    #Choose parameters
    n_folds = 5 #Number cross-validation folds 
    n_folds_nested = 3 #Number folds for hyperparameter optimization
    n_yeo_networks = 7 #Number of functional brain networks --> 7 or 17
    network_selection = link_selection #Link selections: Choose from ['all','within_between','allbutone', 'all_combi', 'one']
        
    
    states_FC = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
    
    #Read FC of the chosen state
    idx_state = states_FC.index(state) + 1
    name_FC = 'FC' + str(idx_state) + '_HCP_610_100nodes.mat'
    FC_state = spio.loadmat(os.path.join(path_source,name_FC))['FC']
    
    
    if option == 'real':
        print('train with correct assignment of intelligence scores')
        n_permutations_real = 1 #Number of iterations with observed intelligence scores (varying stratified folds)
        n_permutations_rand = 0 #Number of permutations of null models (permuted intelligence scores)
    
    if option == 'permutation':
        print('train with permuted intelligence scores')
        n_permutations_real = 0 #Number of iterations with observed intelligence scores (varying stratified folds)
        n_permutations_rand = 1 #Number of permutations of null models (permuted intelligence scores)
    
    n_permutations = 1
    
    result_corrs = []
    
    intelligence_score_type = score #Intelligence component: Choose from ['g_score','gf_score','gc_score']
    
   
    #Read subject order of subjects in behavioral 
    n_subjects_total = data_subjects["Family_No"].size
    
    #Read intelligence scores and confounds
    intelligence_scores = data_subjects[intelligence_score_type].to_numpy()
    confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T
    
    #Get family-wise intelligence score
    no_families = np.array(data_subjects["Family_No"])
    unique_families, unique_families_index = np.unique(no_families, return_index = True)
    name_intelligence_score_fam = intelligence_score_type + '_family'
    intelligence_scores_families = np.array(data_subjects[name_intelligence_score_fam])[unique_families_index]
    
    
    
    #%% Compute masks with different brain link selections
    
    #Assign nodes to networks
    node_assignment = pd.read_csv(os.path.join(path_source,'Schaf100Yeo.csv'), header = None).to_numpy()
    
    if n_yeo_networks == 7:
        node_assignment = node_assignment[:,0]
        
    elif n_yeo_networks == 17:
        node_assignment = node_assignment[:,1]
    else:
        print('choose n_yeo_networks to bei either 7 or 17')
    
    
    #Compute masks for links within one network or between two networks
    mask_network_within_between = []
    idx_n1, idx_n2 = np.where(np.triu(np.ones(n_yeo_networks))==1) #All within and between combinations
    
    for n in range(idx_n1.size):
            
            network_1 = idx_n1[n]
            network_2 = idx_n2[n]
    
            if network_1 == network_2:
                #Within
                idx_nodes_n = np.where(node_assignment == network_1+1)[0] # +1 due to indexing
                node_combinations = np.array(list(combinations(idx_nodes_n,2)))
        
            else:
                #Between 
                idx_nodes_n1 = np.where(node_assignment == network_1+1)[0]
                idx_nodes_n2 = np.where(node_assignment == network_2+1)[0]
                
                node_combinations = np.array(np.meshgrid(idx_nodes_n1,idx_nodes_n2)).T.reshape(-1, 2)
                
              
            idx_nodes_x = node_combinations[:,0]
            idx_nodes_y = node_combinations[:,1]
    
            #Make mask with selected links
            mask = np.zeros((n_nodes, n_nodes))
            mask[idx_nodes_x,idx_nodes_y] = 1
            mask[idx_nodes_y,idx_nodes_x] = 1
    
            mask = np.triu(mask,1) #Take half of the mask as mask is symetric
            mask_network_within_between.append(mask)
                
    #Visualize matrices of selections for sanity check            
    fig, axes = plt.subplots(4, 7)
    cnt = 0
    for ax1 in axes:
        for ax2 in ax1:
            if cnt <= 28:
                ax2.imshow(mask_network_within_between[cnt])
                
            cnt += 1
    
    #plt.savefig('masks_within_between.jpg', format='jpg', dpi = 1200)
    plt.figure()
    plt.title('within and between')
    plt.imshow(np.sum(np.array(mask_network_within_between),0))
    
    
    
    
    
    #Network mask: all but within and between links of one specific network
    mask_network_allbutone = []
    for network in range(n_yeo_networks):
            
        idx_nodes_n = np.where(node_assignment == network+1)[0]
        mask = np.ones((n_nodes, n_nodes))
        for node in idx_nodes_n:
            
            mask[idx_nodes_n,:] = 0
            mask[:,idx_nodes_n] = 0
        
        mask = np.triu(mask,1)
    
        mask_network_allbutone.append(mask)
        
    
    #Visualize matrices of selections for sanity check 
    fig, axes = plt.subplots(2, 4)
    cnt = 0
    for ax1 in axes:
        for ax2 in ax1:
            if cnt < 7:
                ax2.imshow(mask_network_allbutone[cnt])
                
            cnt += 1
    
    #plt.savefig('masks_allbutone.jpg', format='jpg', dpi = 1200)
    plt.figure()
    plt.title('all but one')
    plt.imshow(np.sum(np.array(mask_network_allbutone),0))
    
    
    
    #Network mask: all within and between links of one specific network
    mask_network_one = []
    for network in range(n_yeo_networks):
            
        idx_nodes_n = np.where(node_assignment == network+1)[0]
        mask = np.zeros((n_nodes, n_nodes))
        for node in idx_nodes_n:
            
            mask[idx_nodes_n,:] = 1
            mask[:,idx_nodes_n] = 1
        
        mask = np.triu(mask,1)
    
        mask_network_one.append(mask)
        
    
    #Visualize matrices of selections for sanity check 
    fig, axes = plt.subplots(2, 4)
    cnt = 0
    for ax1 in axes:
        for ax2 in ax1:
            if cnt < 7:
                ax2.imshow(mask_network_one[cnt])
                
            cnt += 1
    
    #plt.savefig('masks_one.jpg', format='jpg', dpi = 1200)
    plt.figure()
    plt.title('one')
    plt.imshow(np.sum(np.array(mask_network_one),0))
    
    
    
    
    #Concatenate network selection masks
    if network_selection == 'all':
        mask_network = []
        network_list = list(np.arange(0,1))
        mask_network.append(np.triu(np.ones((n_nodes,n_nodes)),1))
        print('all links')
    elif network_selection == 'within_between':
        network_list = list(np.arange(1,28+1))
        mask_network = mask_network_within_between
        print('within-between links')
    elif network_selection == 'allbutone':
        network_list = list(np.arange(29,35+1))
        mask_network = mask_network_allbutone
        print('links of all but one network')
    elif network_selection == 'one':
        network_list = list(np.arange(36,42+1))
        mask_network = mask_network_one
        print('links of one network')
    elif network_selection == 'all_combi':
        mask_network = []
        mask_network.append(np.triu(np.ones((n_nodes,n_nodes)),1))
        mask_network = mask_network + mask_network_within_between + mask_network_allbutone + mask_network_one
        network_list = list(np.arange(0,43))
        print('use all combi')
    else:
        print('network selection wrong')
    
        
    total_jobs = n_permutations*n_folds*len(network_list)
    cnt_job = 0    
    #%%Train models to predict observed intelligence scores
    
    #Configuration of model's hyperparameters to be chosen from (ideal combination of parameters chosen via hyperparameter optimization)
    config = {
        "n_hidden": [10, 50, 100],
        "n_layer": [1,2,3],
        "lr": [1e-2],
    }
    
    hyper_parameter = list(product(config["n_hidden"],config["n_layer"],config["lr"]))
    
    
    
    for p_real in range(n_permutations_real):
        print('iteration model training: ')
        #print(p_real+1)
        
    
        #Make folds with families stratified for intelligence scores of families
        df_target = pd.DataFrame(np.array([unique_families, intelligence_scores_families]).T,columns=['family','target'])
        fold_assignment_families = create_folds(df = df_target, n_s = n_folds, n_grp = 20)
        
        #Assign subjects to family-based folds (this ensures all subjects of a family to be in the same fold)
        fold_assignment_subjects = np.ones((n_subjects_total,2))*-1
        for fam, fold in zip(fold_assignment_families.family, fold_assignment_families.Fold):
            
            idx = np.where(data_subjects["Family_No"] == fam)[0]
            fold_assignment_subjects[idx,0] = fold
            fold_assignment_subjects[idx,1] = fam
        

        FCs_network = []
        for n in range(len(mask_network)):
            
            idx_m,idx_n = np.where(mask_network[n]==1)
            FCs_network.append(FC_state[idx_m, idx_n,:])
            
        
        
            
        ID_trial = np.random.randint(0,100000) #ID for saving files 
        
        #Loop over link selections
        for cnt_network, FC_network in zip(network_list, FCs_network):
            
            
            y_test_all = []
            y_pred_test_all = []
            y_train_all = []
            
            #Loop over cross-validation folds
            for k in range(n_folds):
                
                progressBar(cnt_job,total_jobs)
                cnt_job += 1
                
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
                
                
                    #X,y for hyperparameter optimization loop
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
            
            
            MSE = mean_squared_error(y_observed, y_predicted, squared=True) #Squared = True for MSE, False for RMSE
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
                                
            path_complete = os.path.join(path_file, path_models, trial)
            with open(path_complete, "wb") as fp:   #Pickling
                pickle.dump(list_network, fp)
            
            #print(Corr)
            result_corrs.append(Corr)
            
    #%% Prediction permutation tests 
    
    for p_rand in range(n_permutations_rand):
        print('iteration model training with permuted scores')
        #print(p_rand+1)
        
        #Shuffle intelligence scores    
        intelligence_scores_perm = np.random.permutation(intelligence_scores) 
        
        #Compute new family averages of permuted intelligence scores
        intelligence_scores_rand_unique_families = []
        for f in range(unique_families.shape[0]):
            
            family_f = unique_families[f]
            
            intelligence_scores_fam_rand = intelligence_scores_perm[np.where(no_families == family_f)[0]] 
            intelligence_scores_rand_unique_families.append([family_f,np.mean(intelligence_scores_fam_rand)])
            
        
        
        df_target_rand = pd.DataFrame(intelligence_scores_rand_unique_families,columns=['family','target'])
        fold_assignment_families = create_folds(df = df_target_rand, n_s = n_folds, n_grp = 20)
        
        #Assign subjects to family-based folds (this ensures all subjects of a family to be in the same fold)
        fold_assignment_subjects = np.ones((n_subjects_total,2))*-1
        
        for fam, fold in zip(fold_assignment_families.family, fold_assignment_families.Fold):
            
            idx = np.where(data_subjects["Family_No"] == fam)[0]
            fold_assignment_subjects[idx,0] = fold
            fold_assignment_subjects[idx,1] = fam    
        
    
        FCs_network = []
        for n in range(len(mask_network)):
            idx_m,idx_n = np.where(mask_network[n]==1)
            FCs_network.append(FC_state[idx_m, idx_n,:])
    
    
    
        ID_trial = np.random.randint(0,100000) #ID for saving files
    
    
        #Loop over link selections
        for cnt_network, FC_network in zip(network_list, FCs_network):
        
            y_test_all = []
            y_pred_test_all = []
            y_train_all = []
            
            #Loop over cross-validation folds
            for k in range(n_folds):
                progressBar(cnt_job,total_jobs)
                cnt_job += 1
                
                #Get indexes of subjects for test and training based on folds
                test_index = np.where(fold_assignment_subjects[:,0]==k)[0]
                train_index = np.where(fold_assignment_subjects[:,0]!=k)[0]
                
                #Controll for confounds via linear regression
                y_train, y_test = regress_confounds_fold(score = intelligence_scores_perm, confounds = confounds, train_index = train_index, test_index = test_index)
                
                X_train = []
                X_test = []
                for el in range(FC_network.shape[0]):
            
                    var_train, var_test = regress_confounds_fold(score = FC_network[el,:], confounds = confounds, train_index = train_index, test_index = test_index)
                    X_train.append(var_train)
                    X_test.append(var_test)
        
                X_train = np.array(X_train).T
                X_test = np.array(X_test).T
                
                
                ###Hyperparameter optimization (with training data only!)
                
                #Indexes of families of subjects in training data of this fold for hyperparameter optimization
                idx_familiy_optim = np.where(np.array(fold_assignment_families.Fold) != k)[0] 
                
                #Families of training subjects
                families_optim = np.array(fold_assignment_families.family)[idx_familiy_optim]
                #Families of all training subjects 
                fam_assignment_subjects_optim = fold_assignment_subjects[train_index,1]
                #Mean intelligence scores of families of training subjects
                g_scores_families_optim = np.array(fold_assignment_families.target)[idx_familiy_optim]
                
                df_target_optim =  pd.DataFrame(np.array([families_optim, g_scores_families_optim]).T,columns=['family','target'])
                fold_assignment_families_optim = create_folds(df = df_target_optim, n_s = n_folds_nested, n_grp = 20)
                
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
                
                
                    #X,y for hyperparameter optimization loop
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
                 
                #Store observed and predicted scores of all models    
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
                name_model = str(state) + '_' + intelligence_score_type + '_' + str(cnt_network) + '_' + str(k) + '_' + 'perm' + '_' + str(ID_trial)
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
            
            trial = str(state) + '_' + intelligence_score_type + '_' + str(cnt_network) + '_' + 'perm' + '_' + str(ID_trial)
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
                                
            path_complete = os.path.join(path_file, path_models, trial)
            with open(path_complete, "wb") as fp:   #Pickling
                pickle.dump(list_network, fp)
                
                
            #print(Corr)
            result_corrs.append(Corr)
                
    
    return result_corrs
                
                   
            
