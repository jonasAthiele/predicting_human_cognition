# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 2023

@author: Jonas A. Thiele
"""


### Visualizing differences between results from models trained with most relevant links vs. models trained with all links
### Determining overlap of most relevant links between states and intelligence components

import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import argparse


#%% Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str, default = 'main')

args = parser.parse_args()
name_sample = args.sample #Sample: 'main', 'lockbox', 'repli'


states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

state_repli = ['rest','WM', 'emotion', 'latent_states', 'latent_task']


number_connections = [45, 190, 435, 780, 1000] 
name_measure = 'corr' #Correlation between observed an predicted intelligence scores

    
p_thresh = 0.05 #Significance threshold
      
path_output = os.path.join(os.getcwd(),'figures')

#%% Read data
path_current = os.path.dirname(os.path.abspath("__file__"))
path_file = os.path.join(path_current, 'results')
#Read results of models trained with different numbers of most relevant links

#Results of predicting g
name_file = name_measure + '_' + 'g' + '_' + name_sample + '_lrp' + '.npy' 
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)
measures_relevance_g = measure
measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
measure_mean_z = np.mean(measure_z, axis = 1) #Mean over iterations with varying stratified sample divisions
measure_mean_z = np.mean(measure_mean_z, axis = 0) #Mean over states
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation  
measures_relevance_g_mean = measure_mean #Mean over iterations and states

#Results of predicting gc
if name_sample != 'repli': #gc models not used for prediction in replication sample
    name_file = name_measure + '_' + 'gc' + '_' + name_sample + '_lrp' + '.npy'
    path_complete = os.path.join(path_file, name_file)
    measure = np.load(path_complete)
    measures_relevance_gc = measure
    measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
    measure_mean_z = np.mean(measure_z, axis = 1) #Mean over iterations with varying stratified sample divisions
    measure_mean_z = np.mean(measure_mean_z, axis = 0) #Mean over states
    measure_mean = np.tanh(measure_mean_z) #Transform back to correlation  
    measures_relevance_gc_mean = measure_mean #Mean over iterations and states

#Results of predicting gf
name_file = name_measure + '_' + 'gf' + '_' + name_sample + '_lrp' + '.npy'
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)
measures_relevance_gf = measure
measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
measure_mean_z = np.mean(measure_z, axis = 1) #Mean over iterations with varying stratified sample divisions
measure_mean_z = np.mean(measure_mean_z, axis = 0) #Mean over states
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation  
measures_relevance_gf_mean = measure_mean #Mean over iterations and states


#Read results of models trained with all links
#Results of predicting g
name_file = name_measure + '_' + 'g' + '_' + name_sample + '.npy'
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)  
measure_all_edges_g = measure[:,0,:] #Results of models trained with all links (edges)
measure_z = np.arctanh(measure_all_edges_g) #Fisher z-transform for averaging correlations
measure_mean_z = np.mean(measure_z) #Mean over iterations with different stratified sample divisions and states 
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation
measures_all_edges_mean_g = measure_mean #Mean over iterations and states

#Results of predicting gc
if name_sample != 'repli':
    name_file = name_measure + '_' + 'gc' + '_' + name_sample + '.npy'
    path_complete = os.path.join(path_file, name_file)
    measure = np.load(path_complete)  
    measure_all_edges_gc = measure[:,0,:] #Results of models trained with all links (edges)
    measure_z = np.arctanh(measure_all_edges_gc) #Fisher z-transform for averaging correlations
    measure_mean_z = np.mean(measure_z) #Mean over iterations with different stratified sample divisions and states 
    measure_mean = np.tanh(measure_mean_z) #Transform back to correlation
    measures_all_edges_mean_gc = measure_mean #Mean over iterations and states

#Results of predicting gf
name_file = name_measure + '_' + 'gf' + '_' + name_sample + '.npy'
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete)  
measure_all_edges_gf = measure[:,0,:] #Results of models trained with all links (edges)
measure_z = np.arctanh(measure_all_edges_gf) #Fisher z-transform for averaging correlations
measure_mean_z = np.mean(measure_z) #Mean over iterations with different stratified sample divisions and states 
measure_mean = np.tanh(measure_mean_z) #Transform back to correlation
measures_all_edges_mean_gf = measure_mean #Mean over iterations and states




#Test for significant differences between results from models trained with 
#most relevant links vs. all links
#Significance of differences of performances averaged over all states

#P-values of difference between results of models trained with all links vs. most relevant links
p_all_relevance_g = []
p_all_relevance_gc = []
p_all_relevance_gf = []

for n in range(5):

    measure_all_edges_g_n = np.arctanh(measure_all_edges_g).ravel() 
    measure_g_n = np.arctanh(measures_relevance_g[:,:,n]).ravel() 
    p_all_relevance_g.append(scipy.stats.ttest_rel(measure_all_edges_g_n, measure_g_n)[1]) #Ttest
    
    if name_sample != 'repli':
        measure_all_edges_gc_n = np.arctanh(measure_all_edges_gc).ravel()  
        measure_gc_n = np.arctanh(measures_relevance_gc[:,:,n]).ravel() 
        p_all_relevance_gc.append(scipy.stats.ttest_rel(measure_all_edges_gc_n, measure_gc_n)[1]) #Ttest
    
    measure_all_edges_gf_n = np.arctanh(measure_all_edges_gf).ravel() 
    measure_gf_n = np.arctanh(measures_relevance_gf[:,:,n]).ravel()  
    p_all_relevance_gf.append(scipy.stats.ttest_rel(measure_all_edges_gf_n, measure_gf_n)[1]) #Ttest
    


#%% Visualize differences between results from models trained with 
### most relevant links vs. all links


customPalette = sns.color_palette('Set2')

plt.figure()
ax = plt.subplot()


data1 = measures_relevance_g_mean
color = customPalette[1]
ax.plot(np.array(number_connections), data1, marker ='o', markerfacecolor = color, linestyle='dashed', linewidth=0, markersize=8, markeredgecolor = 'k', alpha = 0.8)      
plt.hlines(measures_all_edges_mean_g, 20, 1000, colors = color, linestyles = '--')

if name_sample != 'repli':
    data2 = measures_relevance_gc_mean
    color = customPalette[2]
    ax.plot(np.array(number_connections), data2, marker ='o', markerfacecolor = color, linestyle='dashed', linewidth=0, markersize=8, markeredgecolor = 'k', alpha = 0.8)      
    ax.hlines(measures_all_edges_mean_gc, 20, 1000, colors = color, linestyles = '--')

data3 = measures_relevance_gf_mean
color = customPalette[0]
ax.plot(np.array(number_connections), data3, marker ='o', markerfacecolor = color, linestyle='dashed', linewidth=0, markersize=8, markeredgecolor = 'k', alpha = 0.8)      
ax.hlines(measures_all_edges_mean_gf, 20, 1000, colors = color, linestyles = '--')


vmax = 0.35 #Limit y-axis
if name_sample == 'repli':
    vmax = 0.15

#Mark significant differences with asterisks
for e in range(len(p_all_relevance_g)):
    
    x1 = number_connections[e]-17
    x2 = number_connections[e]+30
    x3 = number_connections[e]-30
    
    if p_all_relevance_g[e] < p_thresh:
        
        ax.axvline(x=x1, ymin=data1[e]/vmax, ymax=measures_all_edges_mean_g/vmax, color = 'k', linewidth = 2)
        y = measures_all_edges_mean_g - (measures_all_edges_mean_g - data1[e])/2
        if name_sample == 'repli':
            ax.plot(x1+25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)
        else:
            ax.plot(x1-25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)
            
    if name_sample != 'repli':
        if p_all_relevance_gc[e] < p_thresh:    
            ax.axvline(x=x2, ymin=measures_all_edges_mean_gc/vmax, ymax=data2[e]/vmax, color = 'k', linewidth = 2)
            y = measures_all_edges_mean_gc - (measures_all_edges_mean_gc - data2[e])/2
            ax.plot(x2+25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)
        
    if p_all_relevance_gf[e] < p_thresh:    
        ax.axvline(x=x3, ymin=data3[e]/vmax, ymax=measures_all_edges_mean_gf/vmax, color = 'k', linewidth = 2)
        y = measures_all_edges_mean_gf - (measures_all_edges_mean_gf - data3[e])/2
        ax.plot(x3-25,y, marker = (5, 2, 0), color = 'k', ms = 11, mew=2)



plt.xticks(number_connections, ['45', '190', '435', '780', '1000'], rotation = 0, fontname = 'Arial', fontsize = 20)
plt.ylim(0, vmax)
ax.spines[['top', 'right']].set_visible(False)

if name_sample == 'repli':
    ax.set_yticks([0, 0.05, 0.1, 0.15], ['0','.05','.10','.15'], fontname = 'Arial', fontsize = 22)
else:
    ax.set_yticks([0, 0.1, 0.2, 0.3], ['0','.1','.2','.3'], fontname = 'Arial', fontsize = 22)

plt.grid(color='gray', linestyle='dashed', linewidth=0.5)
plt.tight_layout()

name_save = 'results_relevance_' + name_sample + '.jpg'
path_save = os.path.join(path_output, name_save)
plt.savefig(path_save, format='jpg', dpi = 1200)


#%% Compute overlap of relevant links between states and intelligence components
### Only for main sample (relevant edegs of main sample are also used for predicting intelligence in the lockbox and replication sample)

if name_sample == 'main':
    
    n = 4 # 0-45 connections (links), 1-190 connections, 2-435 connections, 3-780 connections, 4-990 connections, 5-1225 connections
    
    #Load masks with relevant edges
    path_complete = os.path.join(path_file, 'mask_g_relevants.npy')
    mask_g = np.load(path_complete)
    mask_g = mask_g[:,n,:,:] 
    
    path_complete = os.path.join(path_file, 'mask_gc_relevants.npy')
    mask_gc = np.load(path_complete)
    mask_gc = mask_gc[:,n,:,:] 
    
    path_complete = os.path.join(path_file, 'mask_gf_relevants.npy')
    mask_gf = np.load(path_complete)
    mask_gf = mask_gf[:,n,:,:] 
    
    
    #Overlap of relevant links between states within an intelligence component
    #g
    mask = mask_g
    overlap_g = np.zeros((10,10))
    overlap_g[:] = np.nan
    idx_state1, idx_state2 = np.triu_indices(10,1) #All combinations of states
    for s1, s2 in zip(idx_state1, idx_state2): #Loop over state combinations
        
        m1 = mask[s1,:,:].ravel()
        m2 = mask[s2,:,:].ravel()
        
        overlap_g[s1,s2] = ( (m1 > 0) & (m2 > 0) ).sum() / (m1>0).sum()
        
    #gc  
    mask = mask_gc
    overlap_gc = np.zeros((10,10))
    overlap_gc[:] = np.nan
    for s1, s2 in zip(idx_state1, idx_state2):
        
        m1 = mask[s1,:,:].ravel()
        m2 = mask[s2,:,:].ravel()
        
        overlap_gc[s1,s2] = ( (m1 > 0) & (m2 > 0) ).sum() / (m1>0).sum()
    
        
    #gf
    mask = mask_gf
    overlap_gf = np.zeros((10,10))
    overlap_gf[:] = np.nan
    for s1, s2 in zip(idx_state1, idx_state2):
        
        m1 = mask[s1,:,:].ravel()
        m2 = mask[s2,:,:].ravel()
        
        overlap_gf[s1,s2] = ( (m1 > 0) & (m2 > 0) ).sum() / (m1>0).sum()    
        
    
    
    
    #State-wise overlap of relevant links between intelligence components
    overlap_g_gc = np.zeros((10))
    overlap_g_gf = np.zeros((10))
    overlap_gc_gf = np.zeros((10))
    
    for sc in range(10):
        
        m_g = mask_g[sc,:,:].ravel()
        m_gc = mask_gc[sc,:,:].ravel()
        m_gf = mask_gf[sc,:,:].ravel()
        
        g_gc = ( (m_g > 0) & (m_gc > 0) ).sum() / (m_g>0).sum()
        g_gf = ( (m_g > 0) & (m_gf > 0) ).sum() / (m_g>0).sum()
        gc_gf = ( (m_gc > 0) & (m_gf > 0) ).sum() / (m_g>0).sum()
        
        overlap_g_gc[sc] = g_gc
        overlap_g_gf[sc] = g_gf
        overlap_gc_gf[sc] = gc_gf


    ###Overlap values for manuscript
    
    ##Within intelligence components
    
    #Between tasks
    min_o_g_task = np.nanmin(overlap_g[0:8,0:8]) 
    max_o_g_task = np.nanmax(overlap_g[0:8,0:8]) 
    min_o_gc_task = np.nanmin(overlap_gc[0:8,0:8]) 
    max_o_gc_task = np.nanmax(overlap_gc[0:8,0:8]) 
    min_o_gf_task = np.nanmin(overlap_gf[0:8,0:8]) 
    max_o_gf_task = np.nanmax(overlap_gf[0:8,0:8]) 
    
    #Between task and latent
    min_o_g_task_latent = np.nanmin(overlap_g[0:-2,8::]) 
    max_o_g_task_latent = np.nanmax(overlap_g[0:-2,8::]) 
    min_o_gc_task_latent = np.nanmin(overlap_gc[0:-2,8::]) 
    max_o_gc_task_latent = np.nanmax(overlap_gc[0:-2,8::]) 
    min_o_gf_task_latent = np.nanmin(overlap_gf[0:-2,8::]) 
    max_o_gf_task_latent = np.nanmax(overlap_gf[0:-2,8::]) 

    #Between latent
    o_g_latent = overlap_g[8,-1]
    o_gc_latent = overlap_gc[8,-1] 
    o_gf_latent = overlap_gf[8,-1] 
    
    
    ##Between intelligence components (mean over states)
    o_g_gc_mean = overlap_g_gc.mean()
    o_g_gf_mean = overlap_g_gf.mean()
    o_gf_gc_mean = overlap_gc_gf.mean()
    
    