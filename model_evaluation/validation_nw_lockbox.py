
# -*- coding: utf-8 -*-
"""
Created on Mar 3 2023

@author: Jonas A. Thiele
"""

### Apply models build in main sample (610 subjects of HCP) with network-specific links to
### predict intelligence scores in the lockbox sample (196 subjects of HCP)


import pandas as pd
import numpy as np
import scipy.io as spio
from sklearn import linear_model
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import combinations
import os.path
import pickle
import sys

#Specify device to test pytorch models (cuda = gpu, or cpu, here use gpu if avaiable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Functions

#Regressing out confounds from a variable
def regress_confounds_fold(score_train, confounds_train, score_test, confounds_test):
    
    regr = linear_model.LinearRegression()
    regr.fit(confounds_train, score_train)
    fit_train = regr.predict(confounds_train)
    residuals_train = (score_train - fit_train)
    
    mean_res_train = residuals_train.mean()
    std_res_train = residuals_train.std()
    
    residuals_train = (residuals_train - mean_res_train) / std_res_train
     
    fit_test = regr.predict(confounds_test)
    residuals_test = (score_test - fit_test)
    residuals_test = (residuals_test - mean_res_train) / std_res_train
    
    
    return residuals_train, residuals_test


#Define model
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_layer, n_output):
        super(Net, self).__init__()
        
        
        if n_layer == 1:
            self.hidden = torch.nn.Linear(n_feature, n_hidden)   #Hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)   #Output layer
            
        elif n_layer == 2:
            self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  #Hidden layer
            self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   #Hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)   #Output layer
            
        elif n_layer == 3:
            self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  #Hidden layer
            self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   #Hidden layer
            self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   #Hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)   #Output layer  
        
        self.dropout = torch.nn.Dropout(0.25) #Dropout
        
        self.n_layer = n_layer

    def forward(self, x):
        
        if self.n_layer == 1:
            x = F.relu(self.hidden(x))      #Activation function for hidden layer
            x = self.dropout(x)
            x = self.predict(x)             #Linear output
            
        if self.n_layer == 2:
            x = F.relu(self.hidden1(x))     #Activation function for hidden layer
            x = self.dropout(x)
            x = F.relu(self.hidden2(x))     #Activation function for hidden layer
            x = self.dropout(x)
            x = self.predict(x)             #Linear output
            
        if self.n_layer == 3:
            x = F.relu(self.hidden1(x))     #Activation function for hidden layer
            x = self.dropout(x)
            x = F.relu(self.hidden2(x))     #Activation function for hidden layer
            x = self.dropout(x)
            x = F.relu(self.hidden3(x))     #Activation function for hidden layer
            x = self.dropout(x)
            x = self.predict(x)             #Linear output
            
        return x
    

#%% Define parameters, read data

states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']

#Choose parameters
n_folds = 5 #Number cross-validation folds
n_yeo_networks = 7 #Number of functional brain networks --> 7 or 17
network_selection = 'all_combi' #Link selections: Choose from ['whole','within_between','allbutone', 'all_combi', 'one']

n_nodes = 100 #Number of brain nodes used in functional connectivity


path_models = 'res_models_g'
path_file = 'E:\\Transfer\\Results'
path_complete = os.path.join(path_file, path_models) #Path of models trained on main sample
name_score = 'g_score' #Intelligence component: Choose from ['g_score','gf_score','gc_score'] 

#Read behavioral data of subjects
data_subjects = pd.read_csv('families_all_train.csv') #Main sample
data_subjects_lockbox = pd.read_csv('families_all_test.csv') #Lockbox sample

#Read FCs and corresponding subject IDs
FC = spio.loadmat('FC_HCP_610_100nodes.mat')['FCStatic_combined'] #Main sample
FC_subjects = spio.loadmat('FC_HCP_610_subjects.mat')['subjects'].ravel() #Main sample
FC_lockbox = spio.loadmat('FC_HCP_lockbox_100nodes.mat')['FCStatic_combined'] #Lockbox sample
FC_subjects_lockbox = spio.loadmat('FC_HCP_lockbox_subjects.mat')['subjects'].ravel() #Lockbox sample

#Read order of subjects, intelligence scores, confounds / main sample
subjects = data_subjects['Subject'].to_numpy() 
intelligence_scores = data_subjects[name_score].to_numpy()
confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T

#Read order of subjects, intelligence scores, confounds / lockbox sample
subjects_lockbox = data_subjects_lockbox['Subject'].to_numpy()
intelligence_scores_lockbox = data_subjects_lockbox[name_score].to_numpy()
confounds_lockbox = np.vstack((data_subjects_lockbox.Age, data_subjects_lockbox.Gender, data_subjects_lockbox.Handedness, data_subjects_lockbox.FD_mean, data_subjects_lockbox.Spikes_mean)).T



#Check if FCs are ordered properly
if np.equal(subjects, FC_subjects).all() == False:
    
    sys.exit("Error: Order of FCs not equal to order of subjects")

else:
    
    print('Main sample - Subject order okay')
    
    
if np.equal(subjects_lockbox, FC_subjects_lockbox).all() == False:
    
    sys.exit("Error: Order of FCs not equal to order of subjects")

else:
    
    print('Lockbox sample - Subject order okay')
        
#%% Compute masks with link selections

#Assign nodes to networks
node_assignment = pd.read_csv('Schaf100Yeo.csv', header = None).to_numpy()

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
# =============================================================================
# fig, axes = plt.subplots(4, 7)
# cnt = 0
# for ax1 in axes:
#     for ax2 in ax1:
#         if cnt <= 28:
#             ax2.imshow(mask_network_within_between[cnt])
#             
#         cnt += 1
# plt.savefig('masks_within_between.jpg', format='jpg', dpi = 1200)
# 
# plt.figure()
# plt.title('within and between')
# plt.imshow(np.sum(np.array(mask_network_within_between),0))
# =============================================================================



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
# =============================================================================
# fig, axes = plt.subplots(2, 4)
# 
# cnt = 0
# for ax1 in axes:
#     for ax2 in ax1:
#         if cnt < 7:
#             ax2.imshow(mask_network_allbutone[cnt])
#             
#         cnt += 1
# 
# plt.savefig('masks_allbutone.jpg', format='jpg', dpi = 1200)
# plt.figure()
# plt.title('all but one')
# plt.imshow(np.sum(np.array(mask_network_allbutone),0))
# 
# =============================================================================


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
# =============================================================================
# fig, axes = plt.subplots(2, 4)
# 
# cnt = 0
# for ax1 in axes:
#     for ax2 in ax1:
#         if cnt < 7:
#             ax2.imshow(mask_network_one[cnt])
#             
#         cnt += 1
# 
# plt.savefig('masks_one.jpg', format='jpg', dpi = 1200)
# plt.figure()
# plt.title('one')
# plt.imshow(np.sum(np.array(mask_network_one),0))
# =============================================================================


#Concatenate selection masks
if network_selection == 'whole':
    mask_network = []
    network_list = list(np.arange(0,1))
    mask_network.append(np.triu(np.ones((n_nodes,n_nodes)),1))
    print('use whole')
elif network_selection == 'within_between':
    network_list = list(np.arange(1,28+1))
    mask_network = mask_network_within_between
    print('use within')
elif network_selection == 'allbutone':
    network_list = list(np.arange(29,35+1))
    mask_network = mask_network_allbutone
    print('use all but one')
elif network_selection == 'one':
    network_list = list(np.arange(36,42+1))
    mask_network = mask_network_one
    print('use one')
elif network_selection == 'all_combi':
    mask_network = []
    mask_network.append(np.triu(np.ones((n_nodes,n_nodes)),1))
    mask_network = mask_network + mask_network_within_between + mask_network_allbutone + mask_network_one
    network_list = list(np.arange(0,43))
    print('use all combi')
else:
    print('network selection wrong')



#%% Predicting intelligence scores in lockbox sample with models trained in main sample

#List for performances of models
corr_all = []

cnt_models = np.zeros((len(states),43)) #Number of iterations per state, here should be 10 (10 times with varying startified folds) 

#Loop over states
for n_state, state in enumerate(states):

    idx_state = states.index(state)
    FC_state = FC[:,:,:,idx_state]
    FC_state_lockbox = FC_lockbox[:,:,:,idx_state]
    
    FCs_network = []
    FCs_network_lockbox = []
    for n in range(len(mask_network)):
        idx_m,idx_n = np.where(mask_network[n]==1)
        FCs_network.append(FC_state[idx_m, idx_n,:])
        FCs_network_lockbox.append(FC_state_lockbox[idx_m, idx_n,:])
        


    #Loop over link selections
    corr_n = [] #List for performances of models
    for n_network, FC_network, FC_network_lockbox in zip(network_list, FCs_network, FCs_network_lockbox):

        #Get names of files of link selection (files saved in training of models within the main sample)
        name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'real'
        files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files) and not 'parameters' in filename]
        
        #Get all IDs of saved files
        IDs_iteration = []
        for f in files_iteration:
            
            ID = f.split('_')[-1]
            IDs_iteration.append(ID)
                
        IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))

        cnt_models[n_state, n_network] = len(IDs_iteration_unique)
        
        #Loop over IDs
        corr_id = [] #List for performances of models
        for i in IDs_iteration_unique[0:10]:
            
            #Loop over folds
            corr_k = [] #Store performances of models trained on each fold of main sample in predicting scores of the lockbox sample 
            for k in range(n_folds):
                
                #File name of saved model
                name_model = state + '_' + name_score + '_' + str(n_network) + '_' + str(k) + '_' + 'real' + '_' + i
                #File name of saved parameters of the model
                name_model_parameters = name_model + '_parameters'
                
                #Load model hyperparameters
                path_parameters = os.path.join(path_complete, name_model_parameters)
                with open(path_parameters, "rb") as fp:
                    model_parameter = pickle.load(fp)

                n_hidden = model_parameter[0]
                n_layer = model_parameter[1]
                lr = model_parameter[2]
                
                #Remove confounds from intelligence scores
                y_train, y_test = regress_confounds_fold(score_train = intelligence_scores, confounds_train = confounds, score_test = intelligence_scores_lockbox, confounds_test = confounds_lockbox)

                #Remove confounds from FC
                X_train = []
                X_test = []
                for el in range(FC_network.shape[0]):
            
                    var_train, var_test = regress_confounds_fold(score_train = FC_network[el,:], confounds_train = confounds, score_test = FC_network_lockbox[el,:], confounds_test = confounds_lockbox)
                    X_train.append(var_train)
                    X_test.append(var_test)
        
                X_train = np.array(X_train).T
                X_test = np.array(X_test).T
 

                #Initialize model
                model = Net(n_feature=X_test.shape[1], n_hidden=n_hidden, n_layer = n_layer, n_output=1).to(device)             
                
                #Load saved model parameters
                path_model = os.path.join(path_complete, name_model)
                model.load_state_dict(torch.load(path_model))
                
                #Numpy to torch format
                X_test = torch.from_numpy(X_test.astype(np.float32())).to(device)
                
                #Predict lockbox sample scores from FC with model build on main sample
                model.eval()
                with torch.no_grad():
                    
                   
                    y_pred = model(X_test)
                    y_pred = y_pred.cpu().numpy()
                    


                x = np.array(y_test).ravel() #Observed scores of lockbox sample
                y = np.array(y_pred).ravel() #Predicted scores of lockbox sample
                
                #Compute performance measures
                Corr = scipy.stats.pearsonr(x,y)[0]
                MSE = mean_squared_error(x, y, squared=True) #Squared = True for MSE, false for RMSE
                RMSE = mean_squared_error(x, y, squared=False)
                MAE = mean_absolute_error(x,y)
                
                corr_k.append(Corr)
                
            corr_k_z = np.arctanh(np.array(corr_k)) #Fisher z-transform for averaging correlations
            corr_mean_z = np.mean(corr_k_z) #Mean over folds    
            corr_id.append(np.tanh(corr_mean_z)) #Back-transforming to correlations

            
        corr_n.append(corr_id)
    
    corr_all.append(corr_n)
       
corr_all = np.array(corr_all)

#%% Predicting intelligence scores in lockbox sample with null models trained in main sample

corr_all_perm = [] #List for performances of models

cnt_models_perm = np.zeros((len(states),43)) #Number of permutations per state, here should be 100 

#Loop over states
for n_state, state in enumerate(states):
    
    idx_state = states.index(state)
    FC_state = FC[:,:,:,idx_state]
    FC_state_lockbox = FC_lockbox[:,:,:,idx_state]
    
    FCs_network = []
    FCs_network_lockbox = []
    for n in range(len(mask_network)):
        idx_m,idx_n = np.where(mask_network[n]==1)
        FCs_network.append(FC_state[idx_m, idx_n,:])
        FCs_network_lockbox.append(FC_state_lockbox[idx_m, idx_n,:])
         
    #Loop over link selections   
    corr_n = [] #List for performances of models
    for n_network, FC_network, FC_network_lockbox in zip(network_list, FCs_network, FCs_network_lockbox):
        
    
        #Get names of files of selection
        name_files = state + '_' + name_score + '_' + str(n_network) + '_' + 'perm'        
        files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files) and not 'parameters' in filename]
        
        #Get all IDs of saved files
        IDs_iteration = []
        for f in files_iteration:
            
            ID = f.split('_')[-1]
            IDs_iteration.append(ID)
            
        IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
        
        cnt_models_perm[n_state, n_network] = len(IDs_iteration_unique)
        
        #Loop over IDs
        corr_id = [] #List for performances of models
        for i in IDs_iteration_unique[0:100]:
           
            #Loop over folds
            corr_k = [] #List for performances of models
            for k in range(n_folds):
                
                #Filename of saved model
                name_model = state + '_' + name_score + '_' + str(n_network) + '_' + str(k) + '_' + 'perm' + '_' + i
                #File name of saved parameters of the model
                name_model_parameters = name_model + '_parameters'
                
                #Load model hyperparameters
                path_parameters = os.path.join(path_complete, name_model_parameters)
                with open(path_parameters, "rb") as fp:   # Unpickling
                    model_parameter = pickle.load(fp)

                n_hidden = model_parameter[0]
                n_layer = model_parameter[1]
                lr = model_parameter[2]
                
                #Remove confounds from intelligence scores
                y_train, y_test = regress_confounds_fold(score_train = intelligence_scores, confounds_train = confounds, score_test = intelligence_scores_lockbox, confounds_test = confounds_lockbox)
                
                #Remove confounds from FC
                X_train = []
                X_test = []
                for el in range(FC_network.shape[0]):
            
                    var_train, var_test = regress_confounds_fold(score_train = FC_network[el,:], confounds_train = confounds, score_test = FC_network_lockbox[el,:], confounds_test = confounds_lockbox)
                    X_train.append(var_train)
                    X_test.append(var_test)
        
                X_train = np.array(X_train).T
                X_test = np.array(X_test).T
 

                #Initialize model
                model = Net(n_feature=X_test.shape[1], n_hidden=n_hidden, n_layer = n_layer, n_output=1).to(device)             
                
                #Load model parameters
                path_model = os.path.join(path_complete, name_model)
                model.load_state_dict(torch.load(path_model))
                
                #Numpy to torch format
                X_test = torch.from_numpy(X_test.astype(np.float32())).to(device)
                
                #Predict lockbox sample scores from FC with null model build on main sample
                model.eval()
                with torch.no_grad():
                    
                   
                    y_pred = model(X_test)
                    y_pred = y_pred.cpu().numpy()
                    


                x = np.array(y_test).ravel() #Observed scores of lockbox sample
                y = np.array(y_pred).ravel() #Predicted scores of lockbox sample
                
                #Compute performance measures
                Corr = scipy.stats.pearsonr(x,y)[0]
                MSE = mean_squared_error(x, y, squared=True) #Squared = True for MSE, false for RMSE
                RMSE = mean_squared_error(x, y, squared=False)
                MAE = mean_absolute_error(x,y)

                corr_k.append(Corr)
            
            corr_k_z = np.arctanh(np.array(corr_k)) #Fisher z-transform for averaging correlations
            corr_mean_z = np.mean(corr_k_z) #Mean over folds    
            corr_id.append(np.tanh(corr_mean_z)) #Back-transforming to correlations

        corr_n.append(corr_id)
    
    corr_all_perm.append(corr_n)
       
corr_all_perm = np.array(corr_all_perm)


np.save('corr_g_lockbox', corr_all)
np.save('corr_perm_g_lockbox',corr_all_perm)
