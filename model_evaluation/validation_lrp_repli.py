
# -*- coding: utf-8 -*-
"""
Created on Jun 22 2023

@author: Jonas A. Thiele
"""

### Apply models build in the HCP sample to
### predict intelligence scores in the replication sample (PIOP1 and PIOP2 combined)
### for models that were trained with different numbers of most relevant links (determined by stepwise LRP)

import pandas as pd
import numpy as np
import scipy.io as spio #Read matlab files (.mat)
from sklearn import linear_model
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn.functional as F
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
states_common = ['rest','WM','emotion','latent_states','latent_task'] #States common to both PIOP samples

states_piop1 = {'rest','WM','anticipation','emotion','faces','gstroop', 'latent_states','latent_task'};
states_piop2 = {'rest','WM','emotion','stopsignal', 'latent_states','latent_task'};

idx_states_hcp = np.array([0,1,7,8,9]) #States common to both PIOP samples

states_models = []
for i in range(len(idx_states_hcp)):
    states_models.append(states_hcp[idx_states_hcp[i]])

idx_states_piop1 = np.array([0,1,3,6,7]) #States common to both PIOP samples
idx_states_piop2 = np.array([0,1,2,4,5]) #States common to both PIOP samples


#Choose parameters
n_folds = 5 #Number cross-validation folds
n_nodes = 100 #Number of brain nodes used in functional connectivity

path_models = 'res_g_lrp' #Folder of saved models
name_nodes = path_models
path_file = 'E:\\Transfer\\Results\\relevance_1000'
path_complete = os.path.join(path_file, path_models)
name_score = 'g_score' #Intelligence component: Choose from ['g_score','gf_score','gc_score']

n_edges_relevant = [45,190,435,780,1000] #Numbers of most relevant links (edges) used for prediction


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


#%% Predicting intelligence scores in replication sample with models trained in main sample

#List for performances of models
corr_all = []  

cnt_models = np.zeros(len(states_models)) #Number of iterations per state
   
for n_state, state in enumerate(states_models):


    FC_state = FC[:,:,:,n_state]
    FC_state_repli = FC_repli[:,:,:,n_state]

    #Get names of files of link selection (files saved in training of models within the HCP sample)
    n_network = 0  
    name_files = state + '_' + name_score + '_' + str(n_network)  
    files_iteration = [filename for filename in os.listdir(path_complete) if filename.startswith(name_files) and not 'parameters' in filename
                  and not 'corr' in filename and not 'idx' in filename and not 'y_pred' in filename and not 'y_test' in filename and not 'ypred' in filename and not 'ytest' in filename]
    
    
    #Get all IDs of saved files
    IDs_iteration = []
    for f in files_iteration:
        
        ID = f.split('_')[-1]
        IDs_iteration.append(ID)
            
    IDs_iteration_unique = list(np.unique(np.array(IDs_iteration)))
    
    cnt_models[n_state] = len(IDs_iteration_unique)
    
    #Loop over IDs
    corr_id = [] #List for performances of models
    for i in IDs_iteration_unique[0:10]:
        
        #Loop over models trained with different numbers of relevant links (edges)
        corr_n = [] #List for performances of models trained with different numbers of relevant edges
        for n in n_edges_relevant:
            
            #Loop over folds
            corr_k = [] #Store performances of models trained on each fold of HCP sample in predicting scores of the replication sample
            for k in range(n_folds):
                
                #File name of saved model
                name_model = state + '_' + name_score + '_' + str(n_network) + '_' + str(n) + '_' + str(k) + '_' + 'real' + '_' + i
                #File name of saved parameters of the model
                name_model_parameters = name_model + '_parameters'
                
                #Load model hyperparameters
                path_parameters = os.path.join(path_complete, name_model_parameters)
                with open(path_parameters, "rb") as fp:
                    model_parameter = pickle.load(fp)
        
                n_hidden = model_parameter[0]
                n_layer = model_parameter[1]
                lr = model_parameter[2]
            
                #Get edges used for training the model
                relevant_edges = model_parameter[-1]
                
                #Get FC values of these edges
                idx_m,idx_n = np.triu_indices(n_nodes,1)
                FC_network = FC_state[idx_m, idx_n,:]
                FC_network_repli = FC_state_repli[idx_m, idx_n,:]
                
                FC_network = FC_network[relevant_edges,:] 
                FC_network_repli = FC_network_repli[relevant_edges,:] 
                
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
                
                #Predict replication sample scores from FC with model build on main sample
                model.eval()
                with torch.no_grad():
                    
                   
                    y_pred = model(X_test)
                    y_pred = y_pred.cpu().numpy()
                    
   

                x = np.array(y_test).ravel() #Observed scores of replication sample
                y = np.array(y_pred).ravel() #Predicted scores of replication sample

                #Compute performance measures
                Corr = scipy.stats.pearsonr(x,y)[0]
                MSE = mean_squared_error(x, y, squared=True) #Squared = True for MSE, false for RMSE
                RMSE = mean_squared_error(x, y, squared=False)
                MAE = mean_absolute_error(x,y)
                
                
                corr_k.append(Corr)

            corr_k_z = np.arctanh(np.array(corr_k)) #Fisher z-transform for averaging correlations
            corr_mean_z = np.mean(corr_k_z) #Mean over folds    
            corr_n.append(np.tanh(corr_mean_z)) #Back-transforming to correlations
            
        corr_id.append(corr_n)
        
    corr_all.append(corr_id)

       
corr_all = np.array(corr_all)

name_save = 'corr_repli' + name_nodes
np.save(name_save, corr_all)



