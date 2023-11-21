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
from sklearn import linear_model
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from itertools import combinations
from itertools import product
import os.path
import pickle
import warnings
import copy

#Specify device to train pytorch models (cuda = gpu, or cpu, here use gpu if avaiable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Functions

#Compute masks with specific selection of links
def get_mask_network(n_nodes, method, idx_nodes):
    
    
    # n: number of links
    # methods: 'connections', 'nodes', 'theory'
    # idx_nodes: idx of preselected nodes (intelligence theories) 
    
    if method == 'connections':
        
        n_connections = scipy.special.comb(n_nodes, 2)
        
        edges = []
        
        while len(edges) < n_connections:
            
            node1 = np.random.randint(0,100,1)
            node2 = np.random.randint(0,100,1)
            
            
            if node1 != node2:
                
                node_combi = np.array([node1, node2]).ravel()
                
                test = 0
                for n in edges:
                    
                    nT = np.ones(n.size)
                    nT.astype(int)
                    nT[0] = n[1]
                    nT[1] = n[0]
                    
                    if (node_combi == n).all():
                        test = 1
                    
                    if (node_combi == nT).all():
                        test = 1
                    
                if test == 0:
                
                    edges.append(np.array([node1, node2]).ravel())
        
        
        
        mask = np.zeros((100, 100))
        
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1))
        
        
    if method == 'nodes':
        
        nodes = []
        
        while len(nodes) < n_nodes:
            
            node = np.random.randint(0,100,1)
            test = 0
            
            if np.isin(node, np.array(nodes))[0]:
                
                test = 1
            
        
            if test == 0:
            
                nodes.append(node)
          
        
        nodes = np.array(nodes).ravel()        
        edges = np.array(list(combinations(nodes,2)))
        
        
        mask = np.zeros((100, 100))
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1)) 
        
    
    
    if method == 'theory':
        
        edges = np.array(list(combinations(idx_nodes,2)))
        
        mask = np.zeros((100, 100))
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1)) 
            
    
    return mask_network, network_list, edges


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

#Create stratified folds
def create_folds(df, n_s, n_grp):
    
    df['Fold'] = -1
    
    skf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=None)
    df['grp'] = pd.cut(df.target, n_grp, labels=False)
    target = df.grp
    
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no
    return df

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


#Train model with training data
def train_model(X, y, n_hidden, n_layer, lr, prop_valid):
    
    #Dividing into train and evaluation data with prior shuffling
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=prop_valid)
    
    
        
    #Reshape for torch format
    y_train = y_train.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)

    #Numpy to torch format
    X_train = torch.from_numpy(X_train.astype(np.float32())).to(device) 
    y_train = torch.from_numpy(y_train.astype(np.float32())).to(device) 

    X_eval = torch.from_numpy(X_eval.astype(np.float32())).to(device) 
    y_eval = torch.from_numpy(y_eval.astype(np.float32())).to(device) 
    


    n_samples, n_features = X_train.shape

    #Define a network with chosen number of hidden layers and number of neurons per hidden layer
    net = Net(n_feature=n_features, n_hidden=n_hidden, n_layer = n_layer, n_output=1).to(device)
    
    #net = torch.nn.DataParallel(net).to(device) #Use for parallel processing
    
    #print(net)  #Net architecture 
    
    #Define optimizer 
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    #Define loss
    loss_func = torch.nn.MSELoss()  #This is for regression mean squared loss

    #Initial parameters prior to training
    net_best = copy.deepcopy(net) #Best performing net
    loss_eval_best = 100 #Initial loss
    loss_eval_best_intervall = 100 #Initial loss
    early_stop = 0 #Bool if training should be stopped due to decrease in performance
    
    #Define maximal number of training epochs
    n_epochs_max = 20000
    
    #Train the network
    for t in range(n_epochs_max):
        
        net.train()     
        
        prediction = net(X_train)     #Input x and predict based on x

        loss = loss_func(prediction, y_train)     #Must be (1. nn output, 2. target)


        loss.backward()         #Backpropagation, compute gradients
        optimizer.step()        #Apply gradients
        optimizer.zero_grad()   #Clear gradients for next training epoch
        
        #Evaluate trained model on validation data
        net.eval()
        with torch.no_grad():
            prediction_eval = net(X_eval)
            loss_eval = loss_func(prediction_eval, y_eval)
            
        #Check if validation loss decreased in last training epoch
        #if so save new best loss and best model
        if loss_eval < loss_eval_best:
            
            net_best = copy.deepcopy(net)
            loss_eval_best = loss_eval
        
        #Check if best validation loss of the last n epochs is smaller than
        #the best validation loss of n epochs before
        #if not, stop training (early stop)
        if (t%100)==0:
           
            
            if loss_eval_best < loss_eval_best_intervall:
                
                loss_eval_best_intervall = loss_eval_best
            
            else:
                
                early_stop = 1
                
        if early_stop:
            #print('--------------stop no progress -----------------------')
            break        
              
    #print(f'loss = {loss}')  
         
    return net_best, t


#Test trained model on test data
def test_model(net, X_test, y_test):
    
    #Reshape for torch format
    y_test = y_test.reshape(-1, 1)
    
    #Numpy to torch format
    y_test = torch.from_numpy(y_test.astype(np.float32())).to(device) 
    X_test = torch.from_numpy(X_test.astype(np.float32())).to(device) 
    
    #Test net
    net.eval()
    with torch.no_grad():
       
        y_pred = net(X_test)
        y_pred = y_pred.cpu().numpy()
        y = y_test.cpu().numpy()

    return y_pred, y



#%% Define parameters, read data

#Cognitive states for which models are trained
states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']


#Choose parameters
n_folds = 5 #Number cross-validation folds 
n_folds_nested = 3 #Number folds for hyperparameter optimization
n_permutations_real = 10 #Number of permutations with observed intelligence scores (varying stratified folds)
n_permutations_rand = 0 #Number of permutations of null models (permuted intelligence scores)
intelligence_score_type = 'g_score' #Intelligence component: Choose from ['g_score','gf_score','gc_score']


n_nodes_mask = 40 #Number of nodes, important for method_mask = 'connections' or 'nodes'
method_mask = 'nodes' #Method for selection of links: Choose from 'connections', 'nodes', 'theory'

#Name of theory for method_mask = 'theory' 
theory = 'pfc_extended' #Choose from: 'pfit', 'md_assem_core', 'md_assem_extended', 'md_diacheck', 'md_duncan', 'pfc_core', 'pfc_extend', 'cole'

#For method_mask = 'theory' load respective file containing nodes
file_theory_name = 'nodes_' + theory + '.mat'
path_file = os.path.dirname(os.path.abspath("__file__"))
file_theory = os.path.join(path_file, 'intelligence_theories_nodes' ,file_theory_name)
idx_nodes_mask = spio.loadmat(file_theory)['nodes'].ravel().astype(int) - 1 #Nodes of intelligence theory


#Folder to save trained models in
path_models = 'res_40nodes_nodes_rand'

#Read behavioral data of subjects
data_subjects = pd.read_csv('families_all_train.csv')

#Read FCs and corresponding subject IDs
FC = spio.loadmat('FC_HCP_610_100nodes.mat')['FCStatic_combined']
FC_subjects = spio.loadmat('FC_HCP_610_subjects.mat')['subjects'].ravel() #Subjects assigned to FC data

#Read subject order of subjects in behavioral data
subjects = data_subjects['Subject'].to_numpy()

#Read intelligence scores and confounds
intelligence_scores = data_subjects[intelligence_score_type].to_numpy()
confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T

#Get family-wise intelligence score
no_families = np.array(data_subjects["Family_No"])
unique_families, unique_families_index = np.unique(no_families, return_index = True)
name_intelligence_score_fam = intelligence_score_type + '_family'
intelligence_scores_families = np.array(data_subjects[name_intelligence_score_fam])[unique_families_index]

#Check if FCs are ordered properly, otherwise sort FCs to match subject order of behavioral data
if np.equal(subjects, FC_subjects).all() == False:
    
    warnings.warn("Order of FCs not equal to order of subjects")
    idx_subjects = []
    for s in list(subjects):
            idx_subjects.append(np.where(FC_subjects == s)[0])

    
    idx_subjects = np.array(idx_subjects).flatten()
    
    FC = FC[:,:,idx_subjects,:]
    
else:
    
    
    print('Subject order okay')
        
    

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
    print(p_real)
    

    #Make folds with families stratified for intelligence scores of families
    df_target = pd.DataFrame(np.array([unique_families, intelligence_scores_families]).T,columns=['family','target'])
    fold_assignment_families = create_folds(df = df_target, n_s = n_folds, n_grp = 20)
    
    #Assign subjects to family-based folds (this ensures all subjects of a family to be in the same fold)
    fold_assignment_subjects = np.ones((subjects.size,2))*-1
    for fam, fold in zip(fold_assignment_families.family, fold_assignment_families.Fold):
        
        idx = np.where(data_subjects["Family_No"] == fam)[0]
        fold_assignment_subjects[idx,0] = fold
        fold_assignment_subjects[idx,1] = fam
    
    
    #Create mask with selection of specific links (links between random nodes: method_mask=nodes, random links:  
    #method_mask=connections, or links between preselected nodes:  method_mask=theory)
    mask_network, network_list, mask_edges = get_mask_network(n_nodes = n_nodes_mask, method = method_mask, idx_nodes = idx_nodes_mask)
    
    #Loop over states
    for state in states:
        
        idx_state = states.index(state)
        FC_state = FC[:,:,:,idx_state]

            
        idx_m,idx_n = np.where(mask_network[0]==1)
        FC_network = FC_state[idx_m, idx_n,:]
        cnt_network = network_list[0]
        
            
        ID_trial = np.random.randint(0,100000) #ID for saving files 
        
            
        y_test_all = []
        y_pred_test_all = []
        y_train_all = []
        
        #Loop over cross-validation folds
        for k in range(n_folds):
            

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
                 
