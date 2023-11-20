# -*- coding: utf-8 -*-
"""
Created on Jul 28 2023

@author: Jonas A. Thiele
"""

### Test if relevant links/edges (identified via stepwise LRP in training of main sample) are more related to confounds, more reliable than random edges, and if
### included nodes have higher/lower participation coefficient and within-module degree z-score

import pandas as pd
import numpy as np
import scipy.io as spio
from sklearn import linear_model

#%% Functions

#Removing confounds via linear regression
def regress_confounds(score, confounds):
    
    regr = linear_model.LinearRegression()
    regr.fit(confounds, score)
    fit_train = regr.predict(confounds)
    residuals = (score - fit_train)
    
    mean_res = residuals.mean()
    std_res = residuals.std()
    
    residuals = (residuals - mean_res) / std_res
     
    return residuals


#%% Set parameters

#Load mask of relevant links (edges) for g, gc, or gf
mask_relevants = np.load('mask_g_relevants.npy') # shape: state x iteration (different number of relevant links) x nodes x nodes

#Intelligence score ('g_score', 'gc_score','gf_score')
intelligence_score_type = 'g_score' #Choose from ['g_score','gf_score','gc_score']

#Cognitive States
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO']

#Assign nodes to networks
n_yeo_networks = 7
node_assignment = pd.read_csv('Schaf100Yeo.csv', header = None).to_numpy()

if n_yeo_networks == 7:
    node_assignment = node_assignment[:,0]
    
elif n_yeo_networks == 17:
    node_assignment = node_assignment[:,1]
else:
    print('choose n_yeo_networks to bei either 7 or 17')

n_permutations = 1000  #Number of permutations for testing if relevant edges do differ from random edges (permutation test)

#%% Load data


#FCs used for prediction
FC = spio.loadmat('FC_HCP_610_100nodes.mat')['FCStatic_combined']

#Behavioral data of subjects
data_subjects = pd.read_csv('families_all_train.csv') 
subjects = data_subjects['Subject'].to_numpy()
intelligence_scores = data_subjects[intelligence_score_type].to_numpy()
confounds = np.vstack((data_subjects.Age, data_subjects.Gender, data_subjects.Handedness, data_subjects.FD_mean, data_subjects.Spikes_mean)).T

#ICC of edges for resting-state and 7 tasks 
icc = spio.loadmat('icc_HCP_610_100nodes.mat')['ICC_vals']

#Participation coefficient of nodes
Ppos = spio.loadmat('Parti_HCP_610_100nodes.mat')['Ppos']
Pneg = spio.loadmat('Parti_HCP_610_100nodes.mat')['Pneg']

#Within-module degree z-score of nodes
Z = spio.loadmat('ModuleZ_HCP_610_100nodes.mat')['Z']

#%% Test properties of relevant links (edges) 

n_states = len(state_names) #Number states
n_iterations = mask_relevants.shape[1] #Number iterations with different numbers of most relevant links
n_nodes = FC.shape[0] #Number nodes
n_subjects = subjects.size #Number subjects

##Initialize arrays for storing results


#Relation to confounds
age_relevants = np.zeros((n_states,n_iterations))
sex_relevants = np.zeros((n_states,n_iterations))
hand_relevants = np.zeros((n_states,n_iterations))
fd_relevants = np.zeros((n_states,n_iterations))
spikes_relevants = np.zeros((n_states,n_iterations))

age_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
sex_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
hand_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
fd_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
spikes_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))

p_stronger_age = np.zeros((n_states,n_iterations))
p_stronger_sex = np.zeros((n_states,n_iterations))
p_stronger_hand = np.zeros((n_states,n_iterations))
p_stronger_motion_fd = np.zeros((n_states,n_iterations))
p_stronger_motion_spike = np.zeros((n_states,n_iterations))


#Relation to intelligence
intelligence_relevants = np.zeros((n_states,n_iterations))
intelligence_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
p_stronger_intelligence = np.zeros((n_states,n_iterations))


#Test-retest reliability
icc_relevants = np.zeros((n_states,n_iterations))
icc_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
p_stronger_icc = np.zeros((n_states,n_iterations))


#Participation coefficient
Ppos_relevants = np.zeros((n_states,n_iterations))
Ppos_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
p_stronger_Ppos = np.zeros((n_states,n_iterations))

Pneg_relevants = np.zeros((n_states,n_iterations))
Pneg_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
p_stronger_Pneg = np.zeros((n_states,n_iterations))

#Within-module degree z-score
Z_relevants = np.zeros((n_states,n_iterations))
Z_relevants_perm = np.zeros((n_states,n_iterations,n_permutations))
p_stronger_Z = np.zeros((n_states,n_iterations))


#Count number of each node's occurences in edges
relevance_nodes_all = np.zeros((n_states,n_iterations,100))

#Confounds
all_confounds = np.hstack((confounds, np.reshape(intelligence_scores, (-1,1)))) #Confounds + intelligence scores

age = np.reshape(confounds[:,0],(-1,1))
sex = np.reshape(confounds[:,1],(-1,1))
hand = np.reshape(confounds[:,2],(-1,1))
fd = np.reshape(confounds[:,3],(-1,1))
spike = np.reshape(confounds[:,4],(-1,1))
intelligence = np.reshape(intelligence_scores, (-1,1))
                                 

for sc in range(n_states): #Loop over states
    
    print(sc)
    
    FC_mean = np.mean(FC[:, :, :, sc],2) #Mean FC across all subjects
    
    #Participation coefficient and within-module degree z-score of state
    Ppos_sc = Ppos[:,:,sc]
    Pneg_sc = Pneg[:,:,sc]
    Z_sc = Z[:,:,sc]
    
    #ICC of state
    icc_sc = icc[:,:,sc]
        
    
    #Initalize matrices to store FCs corrected for specific confounds
    FC_age = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but age
    FC_sex = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but sex
    FC_hand = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but handedness
    FC_fd = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but motion (FD & spikes)
    FC_spike = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but motion (FD & spikes)  
    FC_intelligence = np.zeros((n_nodes,n_nodes,n_subjects)) #FC corrected for all confounds but intelligence
    
    #Indices for confounds to be corrected for 0-age, 1-sex, 2-handedness, 3-FD, 4-number of spikes, 5-intelligence
    idx_age_confounds = [1,2,3,4,5]; 
    idx_sex_confounds = [0,2,3,4,5];
    idx_hand_confounds = [0,1,3,4,5];
    idx_fd_confounds = [0,1,2,5];
    idx_spike_confounds = [0,1,2,5];
    idx_intelligence_confounds = [0,1,2,3,4];
    
    age_confounds = all_confounds[:, idx_age_confounds] #Potential confounds of age
    sex_confounds = all_confounds[:, idx_sex_confounds] #Potential confounds of sex
    hand_confounds = all_confounds[:, idx_hand_confounds] #Potential confounds of handedness
    fd_confounds = all_confounds[:, idx_fd_confounds] #Potential confounds of FD
    spike_confounds = all_confounds[:, idx_spike_confounds] #Potential confounds of spikes
    intelligence_confounds = all_confounds[:, idx_intelligence_confounds] #Potential confounds of intelligence scores

    #Regressing out confounds from FC
    idx1, idx2 = np.triu_indices(100,1) #Upper triangle
    for i1,i2 in zip(idx1,idx2):

        FC_age[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = age_confounds) 
        FC_sex[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = sex_confounds) 
        FC_hand[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = hand_confounds) 
        FC_fd[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = fd_confounds) 
        FC_spike[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = spike_confounds) 
        FC_intelligence[i1,i2,:] = regress_confounds(score = FC[i1,i2,:,sc], confounds = intelligence_confounds) 
    
    
    for n in range(n_iterations): #Loop over iterations with different numbers of relevant edges
        
        print(n)
        
        mask_relevants_it = mask_relevants[sc,n,:,:]
        mask_relevants_it = np.triu(mask_relevants_it,1)
        idx1, idx2 = np.where(mask_relevants_it > 0) #Indices of relevant edges in upper triangle of FC
        strength_relevance_it = mask_relevants_it[idx1,idx2]
        
        #Node relevance
        #Weighted relvance nodes
        relevance_nodes = np.sum(mask_relevants_it + mask_relevants_it.T,0)
        relevance_nodes = np.reshape(np.array(relevance_nodes),(-1,1))
        
        relevance_nodes_all[sc,n,:] = relevance_nodes.ravel()
        
        
        #Extract relevant edges 
        FC_mean_it = FC_mean[idx1,idx2]
        FC_age_it = FC_age[idx1,idx2,:]
        FC_sex_it = FC_sex[idx1,idx2,:]
        FC_hand_it = FC_hand[idx1,idx2,:]
        FC_fd_it = FC_fd[idx1,idx2,:]
        FC_spike_it = FC_spike[idx1,idx2,:]
        FC_intelligence_it = FC_intelligence[idx1,idx2,:]


        
        #Correlation of relevant edges with confounds
        r_age = np.corrcoef(np.vstack((FC_age_it, age.T)))[-1,:-1]
        r_sex = np.corrcoef(np.vstack((FC_sex_it, sex.T)))[-1,:-1]
        r_hand = np.corrcoef(np.vstack((FC_hand_it, hand.T)))[-1,:-1]
        r_fd = np.corrcoef(np.vstack((FC_fd_it, fd.T)))[-1,:-1]
        r_spike = np.corrcoef(np.vstack((FC_spike_it, spike.T)))[-1,:-1]
        r_intelligence = np.corrcoef(np.vstack((FC_intelligence_it, intelligence.T)))[-1,:-1]
        
        #Mean of absolut correlations to confounds of relevant edges
        age_relevants[sc,n] = abs(r_age).mean()
        sex_relevants[sc,n] = abs(r_sex).mean()
        hand_relevants[sc,n] = abs(r_hand).mean()
        fd_relevants[sc,n] = abs(r_fd).mean()
        spikes_relevants[sc,n] = abs(r_spike).mean()
        intelligence_relevants[sc,n] = abs(r_intelligence).mean()
         
        
        #Mean ICC of relevant edges
        icc_it = icc_sc[idx1,idx2]
        icc_relevants[sc,n] = np.multiply(icc_it, strength_relevance_it).mean()
        
        #Mean participation coefficient of nodes of relevant edges
        Ppos_relevants[sc,n] = np.multiply(Ppos_sc, relevance_nodes).mean()
        Pneg_relevants[sc,n] = np.multiply(Pneg_sc, relevance_nodes).mean()
        
        #Mean within-module degree z-score of nodes of relevant edges
        Z_relevants[sc,n] = np.multiply(Z_sc, relevance_nodes).mean()    
         
        
        #Properties of randomly selected edges, same es above but with permutated edge relevances
        for p in range(n_permutations):
           
            mask_relevants_elements = mask_relevants_it[np.triu_indices(100,1)]
                        
            mask_relevants_elements_perm = np.random.permutation(mask_relevants_elements) #Permute relevance scores 
            mask_relevants_it_perm = np.zeros((100,100))
            mask_relevants_it_perm[np.triu_indices(100,1)] = mask_relevants_elements_perm
            
            #Extract edges with high randomized relevance
            idx1, idx2 = np.where(mask_relevants_it_perm > 0)
            
            strength_relevance_it_perm = mask_relevants_it_perm[idx1,idx2]
            
            #Weighted node relevances              
            relevance_nodes = np.sum(mask_relevants_it_perm + mask_relevants_it_perm.T,0)
            relevance_nodes = np.reshape(np.array(relevance_nodes),(-1,1))

            
            #Extract randomized relevant edges
            FC_mean_it_perm = FC_mean[idx1,idx2]
            FC_age_it_perm = FC_age[idx1,idx2,:]
            FC_sex_it_perm = FC_sex[idx1,idx2,:]
            FC_hand_it_perm = FC_hand[idx1,idx2,:]
            FC_fd_it_perm = FC_fd[idx1,idx2,:]
            FC_spike_it_perm = FC_spike[idx1,idx2,:]
            FC_intelligence_it_perm = FC_intelligence[idx1,idx2,:]
            
                                   
    
            #Correlation of random edges with confounds
            r_age = np.corrcoef(np.vstack((FC_age_it_perm, age.T)))[-1,:-1]
            r_sex = np.corrcoef(np.vstack((FC_sex_it_perm, sex.T)))[-1,:-1]
            r_hand = np.corrcoef(np.vstack((FC_hand_it_perm, hand.T)))[-1,:-1]
            r_fd = np.corrcoef(np.vstack((FC_fd_it_perm, fd.T)))[-1,:-1]
            r_spike = np.corrcoef(np.vstack((FC_spike_it_perm, spike.T)))[-1,:-1]
            r_intelligence = np.corrcoef(np.vstack((FC_intelligence_it_perm, intelligence.T)))[-1,:-1]
            
            #Mean of absolut correlations to confounds of random edges
            age_relevants_perm[sc,n,p] = abs(r_age).mean()
            sex_relevants_perm[sc,n,p] = abs(r_sex).mean()
            hand_relevants_perm[sc,n,p] = abs(r_hand).mean()
            fd_relevants_perm[sc,n,p] = abs(r_fd).mean()
            spikes_relevants_perm[sc,n,p] = abs(r_spike).mean()
            intelligence_relevants_perm[sc,n,p] = abs(r_intelligence).mean()
    
            
            #Mean ICC of random edges
            icc_it_perm = icc_sc[idx1,idx2]
            icc_relevants_perm[sc,n,p] = np.multiply(icc_it_perm, strength_relevance_it_perm).mean()
                
            #Mean participation coefficient of nodes of random edges
            Ppos_relevants_perm[sc,n,p] = np.multiply(Ppos_sc, relevance_nodes).mean()
            Pneg_relevants_perm[sc,n,p] = np.multiply(Pneg_sc, relevance_nodes).mean()
            
            #Mean within-module degree z-score of nodes of random edges
            Z_relevants_perm[sc,n,p] = np.multiply(Z_sc, relevance_nodes).mean()  

        
        
        #Are more relevant edges more related to confounds?
        p_stronger_age[sc,n] = np.sum(age_relevants_perm[sc,n,:] > age_relevants[sc,n])/n_permutations
        p_stronger_sex[sc,n] = np.sum(sex_relevants_perm[sc,n,:] > sex_relevants[sc,n])/n_permutations
        p_stronger_hand[sc,n] = np.sum(hand_relevants_perm[sc,n,:] > hand_relevants[sc,n])/n_permutations
        p_stronger_motion_fd[sc,n] = np.sum(fd_relevants_perm[sc,n,:] > fd_relevants[sc,n])/n_permutations
        p_stronger_motion_spike[sc,n] = np.sum(spikes_relevants_perm[sc,n,:] > spikes_relevants[sc,n])/n_permutations
        
        #Are more relevant edges more related to intelligence?
        p_stronger_intelligence[sc,n] = np.sum(intelligence_relevants_perm[sc,n,:] > intelligence_relevants[sc,n])/n_permutations
        

        #Have nodes of more relevant edges higher participation coefficient?
        p_stronger_Ppos[sc,n] = np.sum(Ppos_relevants_perm[sc,n,:] > Ppos_relevants[sc,n])/n_permutations
        p_stronger_Pneg[sc,n] = np.sum(Pneg_relevants_perm[sc,n,:] > Pneg_relevants[sc,n])/n_permutations
        
        #Have nodes of more relevant edges higher within-module degree z-score?
        p_stronger_Z[sc,n] = np.sum(Z_relevants_perm[sc,n,:] > Z_relevants[sc,n])/n_permutations
        
        #Have more relevant edges stronger ICC? 
        p_stronger_icc[sc,n] = np.sum(icc_relevants_perm[sc,n,:] > icc_relevants[sc,n])/n_permutations

