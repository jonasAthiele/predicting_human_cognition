# -*- coding: utf-8 -*-
"""
Created on Jun 21 2023

@author: Jonas A. Thiele
"""


###Visualizing differences between results from: models trained with functional links (functional connections) between random nodes vs. random links,
###models trained with random links vs. most relevant links,
###models trained with most relevant links vs. all links


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os.path


#%% Parameters

number_nodes = [10, 20, 30, 40] 
number_connections = [45, 190, 435, 780]
# 10 nodes = 45 connections (links)
# 20 nodes = 190 connections (links)
# 30 nodes = 435 connections (links)
# 40 nodes = 780 connections (links)


name_score = 'g' #Intelligence component 
name_sample = 'main' #Sample: 'main', 'lockbox', 'repli'
name_measure = 'corr' #Correlation between observed an predicted intelligence scores


path_file = 'E:\\Transfer\\Results\\corrs\\' #Folder of files with performance scores (here correlation between observed an predicted intelligence scores)

p_thresh = 0.05 #Significance threshold
      
#%% Read data

#Read results of models trained with links between random nodes and random links

#Lists to store performance scores of models trained with links between random nodes and random links
measures_nodes = []
measures_nodes_mean = []
measures_connections = []
measures_connections_mean = []


#Loop over different numbers of random nodes or random links 
for n in number_nodes:
 
    #Read results from models trained with links between randomly selected nodes
    name_file = name_measure + '_' + name_score + '_' + name_sample + '_' + str(n) + 'nodes_nodes_rand' + '.npy'
    path_complete = os.path.join(path_file, name_file)
    measure = np.load(path_complete)
    measures_nodes.append(measure) #Shape measure: state x iterations
    measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
    measure_mean_z = np.mean(measure_z) #Mean over iterations with different random nodes and over all states
    measures_nodes_mean.append(np.tanh(measure_mean_z)) #Transform back to correlation 
    
    
    #Read results from models trained with randomly selected links
    name_file = name_measure + '_' + name_score + '_' + name_sample + '_' + str(n) + 'nodes_connection_rand' + '.npy'
    path_complete = os.path.join(path_file, name_file)
    measure = np.load(path_complete) 
    measures_connections.append(measure) #Shape measure: state x iterations
    measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
    measure_mean_z = np.mean(measure_z) #Mean over iterations with different random connections and over all states
    measures_connections_mean.append(np.tanh(measure_mean_z)) #Transform back to correlation 
    


#Read results from models trained with different numbers of most relevant links
name_file = name_measure + '_' + name_score + '_' + name_sample + '_lrp' + '.npy' 
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)
measures_relevance = measure #Shape measure: state x iterations x number_connections
measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations  
measure_mean_z = np.mean(measure_z, axis = 1) #Mean over iterations with different stratified sample divisions
measure_mean_z = np.mean(measure_mean_z, axis = 0)
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation 
measures_relevance_mean = measure_mean[0:-1] # 0-45 connections, 1-190 connections, 2-435 connections, 3-780 connections, 4-1000 connections
  


#Read results from models trained with all links
name_file = name_measure + '_' + name_score + '_' + name_sample + '.npy'
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)  
measure_all_edges = measure[:,0,:] #Results of models trained with all links (edges, connections)
measure_z = np.arctanh(measure_all_edges) #Fisher z-transform for averaging correlations
measure_mean_z = np.mean(measure_z) #Mean over iterations with different stratified sample divisions and states 
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation
measures_all_edges_mean = measure_mean #Mean over iterations and states



#Test for significant differences between results from models trained with links between random nodes vs. random links,
#random links vs most relevant links,
#most relevant links vs all links
#Significance of differences of performances averaged over all states

p_all_relevant = [] #P-value of difference between results of models trained with all links vs. most relevant links
p_relevant_conn = [] #P-value difference most relevant links vs. random links
p_conn_node = [] #P-value random links vs random nodes

#Loop over different numbers of nodes/links
for n in range(4):
    
    measure_all_edges_n = np.arctanh(measure_all_edges) 
    measure_relevance_n = np.arctanh(measures_relevance[:,:,n])  
    
    
    measure_connection_n = np.arctanh(measures_connections[n])
    measure_nodes_n = np.arctanh(measures_nodes[n])
    
    #Take mean of each of 10 iterations of results of random links to fit number of iterations of results from relevant links
    rand_idx = np.random.permutation(np.arange(100))
    rand_idx = np.reshape(rand_idx,(10,10))
    n_states = measure_connection_n.shape[0] #Number states
    measure_connection_n_short = np.zeros((n_states,10))

    for p in range(10):
        measure_connection_n_short[:,p] = np.mean(measure_connection_n[:,rand_idx[:,p]],1)


    #Ttests
    p_all_relevant.append(scipy.stats.ttest_rel(measure_all_edges_n.ravel(), measure_relevance_n.ravel())[1])
    p_relevant_conn.append(scipy.stats.ttest_rel(measure_relevance_n.ravel(), measure_connection_n_short.ravel())[1])
    p_conn_node.append(scipy.stats.ttest_rel(measure_connection_n.ravel(), measure_nodes_n.ravel())[1])
    

#Find best performing intelligence theory
#Read performances of models trained with links between nodes proposed by intelligence theories
name_file = name_measure + '_' + name_score + '_' + name_sample + '_cole.npy'
path_complete = os.path.join(path_file, name_file)
measure_cole = np.load(path_complete)
measure_cole_z = np.arctanh(measure_cole)
measure_cole_mean = np.tanh(measure_cole_z.mean())

name_file = name_measure + '_' + name_score + '_' + name_sample + '_md_diachek.npy'
path_complete = os.path.join(path_file, name_file)
measure_diachek = np.load(path_complete)
measure_diachek_z = np.arctanh(measure_diachek)
measure_diachek_mean = np.tanh(measure_diachek_z.mean())

name_file = name_measure + '_' + name_score + '_' + name_sample + '_md_duncan.npy'
path_complete = os.path.join(path_file, name_file)
measure_duncan = np.load(path_complete)
measure_duncan_z = np.arctanh(measure_duncan)
measure_duncan_mean = np.tanh(measure_duncan_z.mean())

name_file = name_measure + '_' + name_score + '_' + name_sample + '_pfc_extended.npy'
path_complete = os.path.join(path_file, name_file)
measure_pfc_extended = np.load(path_complete)
measure_pfc_extended_z = np.arctanh(measure_pfc_extended)
measure_pfc_extended_mean = np.tanh(measure_pfc_extended_z.mean())

name_file = name_measure + '_' + name_score + '_' + name_sample + '_pfit.npy'
path_complete = os.path.join(path_file, name_file)
measure_pfit = np.load(path_complete)
measure_pfit_z = np.arctanh(measure_pfit)
measure_pfit_mean = np.tanh(measure_pfit_z.mean())

name_theories = ['PFC-Cole','MD-Diachek','MD-Duncan','PFC-extended','P-Fit-revised']
measure_theories = np.array([measure_cole_mean, measure_diachek_mean, measure_duncan_mean, measure_pfc_extended_mean, measure_pfit_mean])
best_theory = name_theories[np.argmax(measure_theories)]
print('Best performing theory: ' + best_theory)
measure_best_theory = np.max(measure_theories)

#%% Visualize differences between results from models trained with links between random nodes vs. random links,
#random links vs. most relevant links,
#most relevant links vs all links

  
plt.figure()
ax = plt.subplot()

color = 'whitesmoke'
color = [240/255,240/255,240/255]
data1 = measures_nodes_mean
ax.plot(np.array(number_connections), data1, marker ='o', linestyle='dashed', linewidth=0, markersize=10, markerfacecolor = color, markeredgecolor = 'k', alpha = 0.8)
   
color = 'gray'
color = [240/255,110/255,110/255]
data2 = measures_connections_mean
ax.plot(np.array(number_connections), data2, marker ='o', linestyle='dashed', linewidth=0, markersize=10, markerfacecolor = color, markeredgecolor = 'k', alpha = 0.8)

color = 'black'
color = [220/255,20/255,20/255]
data3 = measures_relevance_mean
ax.plot(np.array(number_connections), data3, marker ='o', linestyle='dashed', linewidth=0, markersize=10, markerfacecolor = color, markeredgecolor = 'k', alpha = 0.8)      

mean_all_edges = measures_all_edges_mean
ax.hlines(mean_all_edges, 30, 795, 'black', '--')

#Mean over states of most predictive intelligence theory
if name_sample == 'repli':
    ax.hlines(measure_diachek_mean, 30, 795, 'dimgrey', '--') #To match Figure in manuscript: 
else:        
    ax.hlines(measure_best_theory, 30, 795, 'dimgrey', '--')

vmax = 0.35 #Limit y-axis
if name_sample == 'repli':
    vmax = 0.15


#Mark significant differences with asterisks
for e in range(len(p_conn_node)):
    
    x1 = number_connections[e]+30 #Positions of lines for marking significant differences
    x2 = number_connections[e]+50
    x3 = number_connections[e]-30
    

    if p_conn_node[e] < p_thresh:
        
        ax.axvline(x=x1, ymin=data1[e]/vmax, ymax=data2[e]/vmax, color = 'k', linewidth = 2) #Plot line
        y = data2[e] - (data2[e] - data1[e])/2 #Center of line - y-position asterisk
        ax.plot(x1+25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2) #Plot asterisk
        
    if p_relevant_conn[e] < p_thresh:    
        ax.axvline(x=x2, ymin=data3[e]/vmax, ymax=data2[e]/vmax, color = 'k', linewidth = 2)
        y = data3[e] - (data3[e] - data2[e])/2
        ax.plot(x2+25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)
        
    if p_all_relevant[e] < p_thresh:    
        ax.axvline(x=x3, ymin=data3[e]/vmax, ymax=mean_all_edges/vmax, color = 'k', linewidth = 2)
        y = mean_all_edges - (mean_all_edges - data3[e])/2
        ax.plot(x3-25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)
        
plt.grid(color='gray', linestyle='dashed', linewidth=0.5) #Add grid
       

ax.set_xticks(number_connections, ['45', '190', '435', '780'], rotation = 0, fontname = 'Arial', fontsize = 22)
ax.set_ylim(0, vmax)

if name_sample == 'repli':
    ax.set_yticks([0, 0.05, 0.1, 0.15], ['0','.05','.10','.15'], fontname = 'Arial', fontsize = 22)
else:
    ax.set_yticks([0, 0.1, 0.2, 0.3], ['0','.1','.2','.3'], fontname = 'Arial', fontsize = 22)

ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savename = 'results_nodes_' + name_sample + '.jpg'
plt.savefig(savename, format='jpg', dpi = 1200)
