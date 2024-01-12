# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""

import torch
import numpy as np

#Specify device to train pytorch models (cuda = gpu, or cpu, here use gpu if avaiable)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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