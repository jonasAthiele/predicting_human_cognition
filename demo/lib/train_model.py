# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy


#Specify device to train pytorch models (cuda = gpu, or cpu, here use gpu if avaiable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    
    # print(net)  #Net architecture 
    
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