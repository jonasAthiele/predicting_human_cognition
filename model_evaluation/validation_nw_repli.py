
# -*- coding: utf-8 -*-
"""
Created on Mar 13 2023

@author: Jonas A. Thiele
"""

### Apply models build in the HCP sample (806 subjects of HCP) with network-specific links to
### predict intelligence scores in the replication samples (PIOP1 and PIOP2 combined)


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


#Define prediction model
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


states_hcp = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
states_common = ['rest','WM','emotion','latent_states','latent_task']

states_piop1 = {'rest','WM','anticipation','emotion','faces','gstroop', 'latent_states','latent_task'};
states_piop2 = {'rest','WM','emotion','stopsignal', 'latent_states','latent_task'};

idx_states_hcp = np.array([0,1,7,8,9])
states_models = []
for i in range(len(idx_states_hcp)):
    states_models.append(states_hcp[idx_states_hcp[i]])

idx_states_piop1 = np.array([0,1,3,6,7])
idx_states_piop2 = np.array([0,1,2,4,5])



#Choose parameters
n_folds = 5 #Number cross-validation folds
n_yeo_networks = 7 #Number of functional brain networks --> 7 or 17
network_selection = 'all_combi' #Network link selections: Choose from ['whole','within_between','allbutone', 'all_combi', 'one']


n_nodes = 100 #Number of brain nodes used in functional connectivity

path_models = 'res_models_g_806'
path_file = 'E:\\Transfer\\Results'
path_complete = os.path.join(path_file, path_models)
name_score = 'g_score' #Intelligence component: Choose from ['g_score','gf_score']



#Data HCP sample
data_subjects1 = pd.read_csv('families_all_train.csv')
data_subjects2 = pd.read_csv('families_all_test.csv')
frames = [data_subjects1, data_subjects2]
data_subjects = pd.concat(frames) #Behavioral data HCP

FC1 = spio.loadmat('FC_HCP_610_100nodes.mat')['FCStatic_combined'] 
FC2 = spio.loadmat('FC_HCP_lockbox_100nodes.mat')['FCStatic_combined'] 
FC = np.concatenate((FC1,FC2), axis = 2)
FC = FC[:,:,:,idx_states_hcp] #FC of HCP for states common to both PIOP samples

FC_subjects1 = spio.loadmat('FC_HCP_610_subjects.mat')['subjects'].ravel()
FC_subjects2 = spio.loadmat('FC_HCP_lockbox_subjects.mat')['subjects'].ravel()
FC_subjects = np.concatenate((FC_subjects1,FC_subjects2)) #Subject order of FCs

subjects = data_subjects['Subject'].to_numpy() #Subjects HCP 
intelligence_scores = data_subjects[name_score].to_numpy() #Intelligence scores HCP
confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T #Confounds HCP


#Data replication sample
data_subjects_repli1 = pd.read_csv('data_beh_sel_subjects_PIOP1.csv')
data_subjects_repli2 = pd.read_csv('data_beh_sel_subjects_PIOP2.csv')
frames = [data_subjects_repli1, data_subjects_repli2]
data_subjects_repli = pd.concat(frames) #Behavioral data AOMIC


FC_repli1 = spio.loadmat('FC_PIOP1_100nodes.mat')['FCStatic_combined']
FC_repli1 = FC_repli1[:,:,:,idx_states_piop1]
FC_repli2 = spio.loadmat('FC_PIOP2_100nodes.mat')['FCStatic_combined']
FC_repli2 = FC_repli2[:,:,:,idx_states_piop2]
FC_repli = np.concatenate((FC_repli1,FC_repli2), axis = 2) #FC of both PIOP samples for states common to both PIOP samples


FC_subjects_repli1 = spio.loadmat('FC_PIOP1_subjects.mat')['subjects'].ravel()
FC_subjects_repli2 = spio.loadmat('FC_PIOP2_subjects.mat')['subjects'].ravel()
FC_subjects_repli = np.concatenate((FC_subjects_repli1,FC_subjects_repli2)) #Subject order of FCs


subjects_repli = data_subjects_repli['Subject'].to_numpy() #Subjects PIOP1 + PIOP2
intelligence_scores_repli = data_subjects_repli['raven_score'].to_numpy() #Intelligence scores of PIOP1 + PIOP2
intelligence_scores_repli = (intelligence_scores_repli - intelligence_scores_repli.mean()) / intelligence_scores_repli.std() #Normalizing intelligence scores
confounds_repli = np.vstack((data_subjects_repli.age, data_subjects_repli.sex, data_subjects_repli.handedness, data_subjects_repli.meanFD, data_subjects_repli.perc_small_spikes)).T #Confounds of PIOP1 + PIOP2


#Check if FCs are ordered properly
if np.equal(subjects, FC_subjects).all() == False:
    
    sys.exit("Error: Order of FCs not equal to order of subjects")

else:
    
    print('Main sample - Subject order okay')
    
    
if np.equal(subjects_repli, FC_subjects_repli).all() == False:
    
    sys.exit("Error: Order of FCs not equal to order of subjects")

else:
    
    print('Replication sample - Subject order okay')
        
        
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


#%% Predicting intelligence scores in replication sample with models trained in main sample


#List for performances of models
corr_all = []   

cnt_models = np.zeros((len(states_models),43)) #Number of iterations per state, here should be 10 (10 times with varying startified folds) 

#Loop over states    
for n_state, state in enumerate(states_models):

    FC_state = FC[:,:,:,n_state]
    FC_state_repli = FC_repli[:,:,:,n_state]
    
    FCs_network = []
    FCs_network_repli = []
    for n in range(len(mask_network)):
        idx_m,idx_n = np.where(mask_network[n]==1)
        FCs_network.append(FC_state[idx_m, idx_n,:])
        FCs_network_repli.append(FC_state_repli[idx_m, idx_n,:])
        
    
    #Loop over link selections
    corr_n = [] #List for performances of models
    for n_network, FC_network, FC_network_repli in zip(network_list, FCs_network, FCs_network_repli):
        
        #Get names of files of network link selection (files saved in training of models within the HCP sample)
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
           
            
            #File name of saved model
            name_model = state + '_' + name_score + '_' + str(n_network) + '_' + 'real' + '_' + i
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
            y_train, y_test = regress_confounds_fold(score_train = intelligence_scores, confounds_train = confounds, score_test = intelligence_scores_repli, confounds_test = confounds_repli)

            #Remove confounds from FC
            X_train = []
            X_test = []
            for el in range(FC_network.shape[0]):
        
                var_train, var_test = regress_confounds_fold(score_train = FC_network[el,:], confounds_train = confounds, score_test = FC_network_repli[el,:], confounds_test = confounds_repli)
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
            
            #Predict replication sample scores from FC with model build on HCP sample
            model.eval()
            with torch.no_grad():
                            
                y_pred = model(X_test)
                y_pred = y_pred.cpu().numpy()
                

            
            x = np.array(y_test).ravel() #Observed scores of replication sample
            y = np.array(y_pred).ravel() #Predicted scores of replication sample
            
            #Compute performance measures
            Corr = scipy.stats.pearsonr(x,y)[0]
            MSE = mean_squared_error(x, y, squared=True) #Squared = True for MSE, false for RMSE
            RMSE = mean_squared_error(x, y, squared=False) #Squared = True for MSE, false for RMSE
            MAE = mean_absolute_error(x,y)
            
 
            corr_id.append(Corr)
    
        corr_n.append(corr_id)
    
    corr_all.append(corr_n)
       
corr_all = np.array(corr_all)


#%% Predicting intelligence scores in replication sample with null models trained in main sample

corr_all_perm = [] #List for performances of models   
cnt_models_perm = np.zeros((len(states_models),43)) #Number of permutations per state, here should be 100 
 
#Loop over states
for n_state, state in enumerate(states_models):

    FC_state = FC[:,:,:,n_state]
    FC_state_repli = FC_repli[:,:,:,n_state]
    
    FCs_network = []
    FCs_network_repli = []
    for n in range(len(mask_network)):
        idx_m,idx_n = np.where(mask_network[n]==1)
        FCs_network.append(FC_state[idx_m, idx_n,:])
        FCs_network_repli.append(FC_state_repli[idx_m, idx_n,:])
        
    
    #Loop over link selections 
    corr_n = [] #List for performances of models
    for n_network, FC_network, FC_network_repli in zip(network_list, FCs_network, FCs_network_repli):
        
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

            #Filename of saved model
            name_model = state + '_' + name_score + '_' + str(n_network) + '_' + 'perm' + '_' + i
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
            y_train, y_test = regress_confounds_fold(score_train = intelligence_scores, confounds_train = confounds, score_test = intelligence_scores_repli, confounds_test = confounds_repli)

            #Remove confounds from FC
            X_train = []
            X_test = []
            for el in range(FC_network.shape[0]):
        
                var_train, var_test = regress_confounds_fold(score_train = FC_network[el,:], confounds_train = confounds, score_test = FC_network_repli[el,:], confounds_test = confounds_repli)
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
            
            #Predict replication sample scores from FC with null model build on HCP sample
            model.eval()
            with torch.no_grad():
                               
                y_pred = model(X_test)
                y_pred = y_pred.cpu().numpy()
                

            
            x = np.array(y_test).ravel() #Observed scores of replication sample
            y = np.array(y_pred).ravel() #Predicted scores of replication sample  
            
            #Compute performance measures
            Corr = scipy.stats.pearsonr(x,y)[0]            
            MSE = mean_squared_error(x, y, squared=True) #Squared = True for MSE, false for RMSE
            RMSE = mean_squared_error(x, y, squared=False) #Squared = True for MSE, false for RMSE
            MAE = mean_absolute_error(x,y)

            corr_id.append(Corr)

        corr_n.append(corr_id)
    
    corr_all_perm.append(corr_n)
       
corr_all_perm = np.array(corr_all_perm)


      
np.save('corr_g_repli', corr_all)
np.save('corr_perm_g_repli',corr_all_perm)

