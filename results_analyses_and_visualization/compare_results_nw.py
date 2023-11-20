# -*- coding: utf-8 -*-
"""
Created on Mar 9 2023

@author: Jonas A. Thiele
"""

### Comparisons between results of network-specific link selections in the main sample

#%% Imports

import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import colors  
import matplotlib.pyplot as plt
import os.path
from matplotlib import ticker
from matplotlib import cm


#%% Parameters

#States
states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

n_yeo_networks = 7 #Number of Yeo networks
networks =  ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN'] #Names of Yeo networks


path_file = 'E:\\Transfer\\Results\\corrs'

p_thresh = 0.05

name_sample = 'main' #Sample
name_measure = 'corr' #Performance measure: Correlation between observed and predicted intelligence scores

idx_all_edges = 0
idx_within_between = np.arange(1,29)
idx_allbutone = np.arange(29,36)
idx_one = np.arange(36,43)

#%% Read results of prediciting g, gc, gf 

#General intelligence g
name_score = 'g' 
name_file = name_measure + '_' + name_score + '_' + name_sample + '.npy' #Normal models
path = os.path.join(path_file, name_file)
measure_g = np.load(path)
measure_g = np.arctanh(measure_g) #Fisher z-transformed correlation
measure_g_mean = np.mean(measure_g, axis = 2) #Fisher z-transformed mean correlation (across iterations with different stratified fold divisions)

name_file = name_measure + '_perm_' + name_score + '_' + name_sample + '.npy' #Null models ('perm')
path = os.path.join(path_file, name_file)
measure_perm_g = np.load(path)
measure_perm_g = np.arctanh(measure_perm_g)


#Crystallized intelligence gc
name_score = 'gc' 
name_file = name_measure + '_' + name_score + '_' + name_sample + '.npy' #Normal models
path = os.path.join(path_file, name_file)
measure_gc = np.load(path)
measure_gc = np.arctanh(measure_gc)
measure_gc_mean = np.mean(measure_gc, axis = 2)

name_file = name_measure + '_perm_' + name_score + '_' + name_sample + '.npy' #Null models ('perm')
path = os.path.join(path_file, name_file)
measure_perm_gc = np.load(path)
measure_perm_gc = np.arctanh(measure_perm_gc)

#Fluid intelligence gf
name_score = 'gf'
name_file = name_measure + '_' + name_score + '_' + name_sample + '.npy' #Normal models
path = os.path.join(path_file, name_file)
measure_gf = np.load(path)
measure_gf = np.arctanh(measure_gf)
measure_gf_mean = np.mean(measure_gf, axis = 2)

name_file = name_measure + '_perm_' + name_score + '_' + name_sample + '.npy' #Null models ('perm')
path = os.path.join(path_file, name_file)
measure_perm_gf = np.load(path)
measure_perm_gf = np.arctanh(measure_perm_gf)

#%% Results of models trained with all links: Means across states (and iterations) compared to null models (for g, gf, gc separately)


print('Means across states (and iterations) compared to null models')

mean_g = np.mean(measure_g[:,idx_all_edges,:].ravel())
mean_g_perm = np.mean(measure_perm_g[:,idx_all_edges,:], axis = 0)
print('g')
print('rho: ' +  str(np.round(np.tanh(mean_g),2)) + '   ' + 'p: ' + str(np.sum(mean_g_perm>mean_g)))
print('')

mean_gc = np.mean(measure_gc[:,idx_all_edges,:].ravel())
mean_gc_perm = np.mean(measure_perm_gc[:,idx_all_edges,:], axis = 0)
print('gc')
print('rho: ' +  str(np.round(np.tanh(mean_gc),2)) + '   ' + 'p: ' + str(np.sum(mean_gc_perm>mean_gc)))
print('')


mean_gf = np.mean(measure_gf[:,idx_all_edges,:].ravel())
mean_gf_perm = np.mean(measure_perm_gf[:,idx_all_edges,:], axis = 0)
print('gf')
print('rho: ' +  str(np.round(np.tanh(mean_gf),2)) + '   ' + 'p: ' + str(np.sum(mean_gf_perm>mean_gf)))


#%% Results of models trained with all links: Paired t-test for comparing state-wise predictions between g, gc, gf

#Ttests
[t_g_gc, p_g_gc] = scipy.stats.ttest_rel(measure_g_mean[:,idx_all_edges].ravel(), measure_gc_mean[:,idx_all_edges].ravel())
[t_g_gf, p_g_gf] = scipy.stats.ttest_rel(measure_g_mean[:,idx_all_edges].ravel(), measure_gf_mean[:,idx_all_edges].ravel())
[t_gc_gf, p_gc_gf] = scipy.stats.ttest_rel(measure_gc_mean[:,idx_all_edges].ravel(), measure_gf_mean[:,idx_all_edges].ravel())

print('Paired t-test for comparing state-wise predictions between g, gc, gf')
print('g - gc')
print('t: ' + str(np.round(t_g_gc,2)) + '   ' + 'p: ' + str(np.round(p_g_gc,3)))
print('')
print('g - gf')
print('t: ' + str(np.round(t_g_gf,2)) + '   ' + 'p: ' + str(np.round(p_g_gf,3)))
print('')
print('gc - gf')
print('t: ' + str(np.round(t_gc_gf,2)) + '   ' + 'p: ' + str(np.round(p_gc_gf,3)))
print('')

#Boxplot
data = pd.DataFrame(np.array([np.tanh(measure_g_mean[:,idx_all_edges].ravel()), np.tanh(measure_gc_mean[:,idx_all_edges].ravel()), np.tanh(measure_gf_mean[:,idx_all_edges].ravel())]).T,columns = ['g', 'gc', 'gf'])
x1 = 0.05
x2 = 0.95
x3 = 1.05
x4 = 1.95
y11 = 0.32
y12 = 0.28
y13 = 0.21
y21 =  0.6
y22 = 0.5

plt.figure()
ax = sns.boxplot(data=data, orient="v", showmeans=True, medianprops={'color': 'black', 'ls': ':', 'lw': 1.5}, meanprops={"marker":"_", "markeredgecolor":"black", "markerfacecolor":"black", "markersize":"50"}, palette="Blues")
plt.plot([x1,x1, x2, x2], [y11, y21, y21, y12], linewidth=1, color='black')
plt.plot(x1 + (x2-x1)/2, y21+0.05, marker = (5, 2, 0), color = 'k', ms = 8, mew=2)
plt.plot([x3,x3, x4, x4], [y12, y22, y22, y13], linewidth=1, color='black')
plt.plot(x3 + (x4-x3)/2,y22+0.05, marker = (5, 2, 0), color = 'k', ms = 8, mew=2)
plt.ylim((0,0.72))
#plt.savefig('ttest_means_fullFC.png', dpi=1200)


#%% Results of models trained with all links: Comparing performances of each state using model comparisons for each intelligence component separately

## g
means_states_g = measure_g_mean[:,idx_all_edges] #Mean of state performances (across model iterations)
idx_state1, idx_state2 = np.triu_indices(10, 1) #All state combinations
p_g = np.ones((10,10)) #Store p-values
for state1, state2 in zip(idx_state1, idx_state2):
    
    measure_state1 = means_states_g[state1]
    measure_state2 = means_states_g[state2]

    g_perm_state1 = measure_perm_g[state1,idx_all_edges,:]
    g_perm_state2 = measure_perm_g[state2,idx_all_edges,:]
    
    g_diff = abs(measure_state1 - measure_state2)
    g_perm_diff = abs(g_perm_state1 - g_perm_state2)
    
    p = np.sum(g_perm_diff > g_diff)/100
    
    p_g[state1, state2] = p  

#Plot mask with highlighted significant differences
p_g[p_g >= p_thresh]= np.nan
p_g[p_g < p_thresh] = 1
plt.figure()
plt.imshow(p_g, cmap = 'bwr', vmin = -1, vmax = 1)   
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20)      
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)



## gc
means_states_gc = measure_gc_mean[:,idx_all_edges]
p_gc = np.ones((10,10))
for state1, state2 in zip(idx_state1, idx_state2):
    
    measure_state1 = means_states_gc[state1]
    measure_state2 = means_states_gc[state2]

    gc_perm_state1 = measure_perm_gc[state1,idx_all_edges,:]
    gc_perm_state2 = measure_perm_gc[state2,idx_all_edges,:]
    
    gc_diff = abs(measure_state1 - measure_state2)
    gc_perm_diff = abs(gc_perm_state1 - gc_perm_state2)
    
    p = np.sum(gc_perm_diff > gc_diff)/100
    
    p_gc[state1, state2] = p 

#Plot mask with highlighted significant differences    
p_gc[p_gc >= p_thresh]= np.nan
p_gc[p_gc < p_thresh] = 1
plt.figure()
plt.imshow(p_gc, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20) 
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)




## gf
means_states_gf = measure_gf_mean[:,idx_all_edges]
p_gf = np.ones((10,10))
for state1, state2 in zip(idx_state1, idx_state2):
    
    measure_state1 = means_states_gf[state1]
    measure_state2 = means_states_gf[state2]

    gf_perm_state1 = measure_perm_gf[state1,idx_all_edges,:]
    gf_perm_state2 = measure_perm_gf[state2,idx_all_edges,:]
    
    gf_diff = abs(measure_state1 - measure_state2)
    gf_perm_diff = abs(gf_perm_state1 - gf_perm_state2)
    
    p = np.sum(gf_perm_diff > gf_diff)/100
    
    p_gf[state1, state2] = p 

#Plot mask with highlighted significant differences    
p_gf[p_gf >= p_thresh]= np.nan
p_gf[p_gf < p_thresh] = 1
plt.figure()
plt.imshow(p_gf, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20) 
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)


#%% Results of models trained with links within- and between networks: Comparison between intelligence components

#Corellations between patterns of results of the different intelligence components
[rho_g_gc, p_g_gc] = scipy.stats.pearsonr(measure_g_mean[:,idx_within_between].ravel(), measure_gc_mean[:,idx_within_between].ravel())
[rho_g_gf, p_g_gf] = scipy.stats.pearsonr(measure_g_mean[:,idx_within_between].ravel(), measure_gf_mean[:,idx_within_between].ravel())
[rho_gc_gf, p_gc_gf] = scipy.stats.pearsonr(measure_gc_mean[:,idx_within_between].ravel(), measure_gf_mean[:,idx_within_between].ravel())

print('Results of models trained with edges within- and between networks: Comparison between result patterns of intelligence components')
print('g - gc')
print('rho: ' + str(np.round(rho_g_gc,2)) + '   ' + 'p: ' + str(np.round(p_g_gc,3)))
print('')
print('g - gf')
print('rho: ' + str(np.round(rho_g_gf,2)) + '   ' + 'p: ' + str(np.round(p_g_gf,3)))
print('')
print('gc - gf')
print('rho: ' + str(np.round(rho_gc_gf,2)) + '   ' + 'p: ' + str(np.round(p_gc_gf,3)))
print('')



#Ttest for testing significance of differences in performances between the components
[t_g_gc, p_g_gc] = scipy.stats.ttest_rel(measure_g_mean[:,idx_within_between].ravel(), measure_gc_mean[:,idx_within_between].ravel())
[t_g_gf, p_g_gf] = scipy.stats.ttest_rel(measure_g_mean[:,idx_within_between].ravel(), measure_gf_mean[:,idx_within_between].ravel())
[t_gc_gf, p_gc_gf] = scipy.stats.ttest_rel(measure_gc_mean[:,idx_within_between].ravel(), measure_gf_mean[:,idx_within_between].ravel())


print('Results of models trained with edges within- and between networks: Comparison between performance strengths of intelligence components')
print('g - gc')
print('t: ' + str(np.round(t_g_gc,2)) + '   ' + 'p: ' + str(np.round(p_g_gc,3)))
print('')
print('g - gf')
print('t: ' + str(np.round(t_g_gf,2)) + '   ' + 'p: ' + str(np.round(p_g_gf,3)))
print('')
print('gc - gf')
print('t: ' + str(np.round(t_gc_gf,2)) + '   ' + 'p: ' + str(np.round(p_gc_gf,3)))
print('')



data =pd.DataFrame(np.array([np.tanh(measure_g_mean[:,idx_within_between].ravel()), np.tanh(measure_gc_mean[:,idx_within_between].ravel()), np.tanh(measure_gf_mean[:,idx_within_between].ravel())]).T,columns = ['g', 'gc', 'gf'])
x1 = 0.05
x2 = 0.95
x3 = 1.05
x4 = 1.95
y11 = 0.14
y12 = 0.12
y13 = 0.09
y21 =  0.5
y22 = 0.45

plt.figure()
ax = sns.boxplot(data=data, orient="v", showmeans=True, medianprops={'color': 'black', 'ls': ':', 'lw': 1.5}, meanprops={"marker":"_", "markeredgecolor":"black", "markerfacecolor":"black", "markersize":"50"}, palette="Blues")
plt.plot([x1,x1, x2, x2], [y11, y21, y21, y12], linewidth=1, color='black')
plt.plot(x1 + (x2-x1)/2, y21+0.05, marker = (5, 2, 0), color = 'k', ms = 8, mew=2)
plt.plot([x3,x3, x4, x4], [y12, y22, y22, y13], linewidth=1, color='black')
plt.plot(x3 + (x4-x3)/2,y22+0.05, marker = (5, 2, 0), color = 'k', ms = 8, mew=2)
plt.ylim((-0.1,0.62))
#plt.savefig('ttest_means_withinbetweenFC.png', dpi=1200)

#%% #Results of models trained with links within- and between networks: 
    #Comparison between intelligence constructs -
    #State-wise differences in ranks of prediction performances of network-specific link selections between g, gc, gf


idx_within_between = np.arange(1,29)


#Ranks g
data = measure_g_mean[:,idx_within_between]
ranks_g = np.zeros(data.shape)
for n_state in range(data.shape[0]):

    sorted_ranks = np.argsort(data[n_state,:])
    for n_network in range(data.shape[1]):
        
        rank = np.where(sorted_ranks == n_network)[0]
        ranks_g[n_state, n_network] = rank

#Ranks gc
data = measure_gc_mean[:,idx_within_between]
ranks_gc = np.zeros(data.shape)
for n_state in range(data.shape[0]):

    sorted_ranks = np.argsort(data[n_state,:])
    for n_network in range(data.shape[1]):
        
        rank = np.where(sorted_ranks == n_network)[0]
        ranks_gc[n_state, n_network] = rank


#Ranks gf
data = measure_gf_mean[:,idx_within_between]
ranks_gf = np.zeros(data.shape)
for n_state in range(data.shape[0]):

    sorted_ranks = np.argsort(data[n_state,:])
    for n_network in range(data.shape[1]):
        
        rank = np.where(sorted_ranks == n_network)[0]
        ranks_gf[n_state, n_network] = rank



#Plot rank differences
vmin = 0  
vmax = 10
#g-gc
for n_state in range(data.shape[0]):
    
    mask_g_gc = np.zeros((7,7))
    mask_g_gc[np.triu_indices(7)] = abs(ranks_g[n_state,:] - ranks_gc[n_state,:])
    plt.figure()
    cmap = cm.get_cmap("Greys").copy()
    im = plt.imshow(mask_g_gc, cmap=cmap, vmin=vmin, vmax=vmax)

    ax = plt.gca();
    
    #Major ticks
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))
    
    #Labels for major ticks
    ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
    ax.set_yticklabels(networks, fontname='Arial', fontsize = 20)
    
    #Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
    
    #Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    plt.title(state_names[n_state], fontname = 'Arial', fontsize = 22, pad=10)
    plt.tight_layout()
    name_save = 'rank_difference_g_gc' + '_' + name_sample + '_' + name_measure + '_'+ states[n_state] + '.jpg'
    plt.savefig(name_save, format='jpg', dpi = 1200)
    
#g-gf
for n_state in range(data.shape[0]):
    mask_g_gf = np.zeros((7,7))
    mask_g_gf[np.triu_indices(7)] = abs(ranks_g[n_state,:] - ranks_gf[n_state,:])
    plt.figure()
    cmap = cm.get_cmap("Greys").copy()
    im = plt.imshow(mask_g_gf, cmap=cmap, vmin=vmin, vmax=vmax)

    ax = plt.gca();
    
    #Major ticks
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))
    
    #Labels for major ticks
    ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
    ax.set_yticklabels(networks, fontname='Arial', fontsize = 20)
    
    #Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
    
    #Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    plt.title(state_names[n_state], fontname = 'Arial', fontsize = 22, pad=10)   
    plt.tight_layout()
    name_save = 'rank_difference_g_gf' + '_' + name_sample + '_' + '_' + name_measure + '_'+ states[n_state] + '.jpg'
    plt.savefig(name_save, format='jpg', dpi = 1200)    



#gc-gf    
for n_state in range(data.shape[0]):
    mask_gc_gf = np.zeros((7,7))
    mask_gc_gf[np.triu_indices(7)] = abs(ranks_gc[n_state,:] - ranks_gf[n_state,:])
    plt.figure()
    cmap = cm.get_cmap("Greys").copy()
    im = plt.imshow(mask_gc_gf, cmap=cmap, vmin=vmin, vmax=vmax)

    ax = plt.gca();
    
    #Major ticks
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))
    
    #Labels for major ticks
    ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
    ax.set_yticklabels(networks, fontname='Arial', fontsize = 20)
    
    #Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
    
    #Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    plt.title(state_names[n_state], fontname = 'Arial', fontsize = 22, pad=10)
    plt.tight_layout()
    name_save = 'rank_difference_gc_gf' + '_' + name_sample + '_' + name_measure + '_'+ states[n_state] + '.jpg'
    plt.savefig(name_save, format='jpg', dpi = 1200)    
    
#Plot colorbar    
fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="horizontal", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=6)
cb.locator = tick_locator
for t in cb.ax.get_xticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_rankk_differences' + '_' + name_sample + '_' + name_measure + '_' + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)   


#%% Results of models trained with links within- and between networks, links of all but one network or links of one network:
    # Compare predicition between states for each intelligence construct separatly

##Choose within-between, all but one, or one:
idx_plot = idx_within_between #Choose from: idx_within_between, idx_allbutone, idx_one


#g
idx_state1, idx_state2 = np.triu_indices(10, 1) #All state combinations
p_g = np.ones((10,10))
for state1, state2 in zip(idx_state1, idx_state2):

    [t, p_g[state1,state2]] = scipy.stats.ttest_rel(measure_g_mean[state1,idx_plot].ravel(), measure_g_mean[state2,idx_plot].ravel())
    

#Plot mask with highlighted significant differences    
p_g[p_g >= p_thresh]= np.nan
p_g[p_g < p_thresh] = 1
plt.figure()
plt.imshow(p_g, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

#Labels for major ticks
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network-specific ttest g')


#Plot order of states sorted by mean predictive performance 
plt.figure()
state_means = np.tanh(np.mean(measure_g_mean[:,idx_plot], axis = 1))
plt.bar(np.arange(10),state_means)
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)



#gc
p_gc = np.ones((10,10))
for state1, state2 in zip(idx_state1, idx_state2):

    [t, p_gc[state1,state2]] = scipy.stats.ttest_rel(measure_gc_mean[state1,idx_plot].ravel(), measure_gc_mean[state2,idx_plot].ravel())
    

#Plot mask with highlighted significant differences    
p_gc[p_gc >= p_thresh]= np.nan
p_gc[p_gc < p_thresh] = 1
plt.figure()
plt.imshow(p_gc, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

#Labels for major ticks
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network-specific ttest gc')


#Plot order of states by mean predictive performance
plt.figure()
state_means = np.tanh(np.mean(measure_gc_mean[:,idx_plot], axis = 1))
plt.bar(np.arange(10),state_means)
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)


#gf
p_gf = np.ones((10,10))
for state1, state2 in zip(idx_state1, idx_state2):

    [t, p_gf[state1,state2]] = scipy.stats.ttest_rel(measure_gf_mean[state1,idx_plot].ravel(), measure_gf_mean[state2,idx_plot].ravel())
    

#Plot mask with highlighted significant differences    
p_gf[p_gf >= p_thresh]= np.nan
p_gf[p_gf < p_thresh] = 1
plt.figure()
plt.imshow(p_gf, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();


#Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

#Labels for major ticks
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network-specific ttest gf')


#Plot order of states by mean predictive performance
plt.figure()
state_means = np.tanh(np.mean(measure_gf_mean[:,idx_plot], axis = 1))
plt.bar(np.arange(10),state_means)
ax = plt.gca();
ax.set_xticks(np.arange(0, 10, 1))
ax.set_xticklabels(state_names, fontname='Arial', fontsize = 20, rotation = 90)


        

#%% Results of models trained with links of all but one network
### and links of one network only

#Test for differences between g, gc, gf
#Links of all but one network

[t_g_gc, p_g_gc] = scipy.stats.ttest_rel(measure_g_mean[:,idx_allbutone].ravel(), measure_gc_mean[:,idx_allbutone].ravel())
[t_g_gf, p_g_gf] = scipy.stats.ttest_rel(measure_g_mean[:,idx_allbutone].ravel(), measure_gf_mean[:,idx_allbutone].ravel())
[t_gc_gf, p_gc_gf] = scipy.stats.ttest_rel(measure_gc_mean[:,idx_allbutone].ravel(), measure_gf_mean[:,idx_allbutone].ravel())

print('Results of models trained with edges of all but one network: Ttest for results of intelligence constructs')
print('g - gc')
print('rho: ' + str(np.round(t_g_gc,2)) + '   ' + 'p: ' + str(np.round(p_g_gc,3)))
print('')
print('g - gf')
print('rho: ' + str(np.round(t_g_gf,2)) + '   ' + 'p: ' + str(np.round(p_g_gf,3)))
print('')
print('gc - gf')
print('rho: ' + str(np.round(t_gc_gf,2)) + '   ' + 'p: ' + str(np.round(p_gc_gf,3)))
print('')

#Links of one network

[t_g_gc, p_g_gc] = scipy.stats.ttest_rel(measure_g_mean[:,idx_one].ravel(), measure_gc_mean[:,idx_one].ravel())
[t_g_gf, p_g_gf] = scipy.stats.ttest_rel(measure_g_mean[:,idx_one].ravel(), measure_gf_mean[:,idx_one].ravel())
[t_gc_gf, p_gc_gf] = scipy.stats.ttest_rel(measure_gc_mean[:,idx_one].ravel(), measure_gf_mean[:,idx_one].ravel())

print('Results of models trained with edges of one network: Ttest for results of intelligence constructs')
print('g - gc')
print('t: ' + str(np.round(t_g_gc,2)) + '   ' + 'p: ' + str(np.round(p_g_gc,3)))
print('')
print('g - gf')
print('t: ' + str(np.round(t_g_gf,2)) + '   ' + 'p: ' + str(np.round(p_g_gf,3)))
print('')
print('gc - gf')
print('t: ' + str(np.round(t_gc_gf,2)) + '   ' + 'p: ' + str(np.round(p_gc_gf,3)))
print('')

#%% Test differences between results of models trained with all links and results of models 
#   trained with links of all but one network and links of one network only - via model difference permutation tests

#Choose respective intelligence component 
measure_mean = measure_g_mean #measure_g_mean, measure_gc_mean, measure_gf_mean
measure_perm = measure_perm_g #measure_perm_g, measure_perm_gc, measure_perm_gf

#Links of all but one network
p_allbutone = np.ones((10,7)) #P-value of differences
perf_diff_allbutone = np.ones((10,7)) #Differences between all links and links of all but one network
for n_state in range(10):
    for n_network in range(7):
    
        measure_all = measure_mean[n_state, idx_all_edges]
        measure_allbutone = measure_mean[n_state, 29+n_network]
    
        perm_all = measure_perm_g[n_state,idx_all_edges,:]
        perm_allbutone = measure_perm[n_state,29+n_network,:]
        
        diff = abs(measure_all - measure_allbutone)
        perm_diff = abs(perm_all - perm_allbutone)
        
        p = np.sum(perm_diff > diff)/100
        
        p_allbutone[n_state, n_network] = p  
        perf_diff_allbutone[n_state, n_network] = measure_all - measure_allbutone
        
#Plot differences in performances (correlations between observed and predicted intelligence scores)
plt.figure()
vmin = -0.1  
vmax = 0.1
cmap = cm.get_cmap("bwr").copy()
divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
im = plt.imshow(np.tanh(perf_diff_allbutone), cmap=cmap, norm=divnorm, aspect='equal')
nbins = 6   
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 10, 1))

#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20)

#Minor ticks
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('allbutone network edges compared to all edges', fontname = 'Arial', fontsize = 22, pad=10)

sig_x, sig_y = np.where(p_allbutone<p_thresh) #Mark significant differences with asteriks
for x,y in zip(sig_x, sig_y):
    plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 12, mew=2)
plt.colorbar()
plt.tight_layout()



#Links of one network
p_one = np.ones((10,7))
perf_diff_one = np.ones((10,7))
for n_state in range(10):
    for n_network in range(7):
    
        measure_all = measure_mean[n_state, idx_all_edges]
        measure_one = measure_mean[n_state, 36+n_network]
    
        perm_all = measure_perm[n_state,idx_all_edges,:]
        perm_one = measure_perm[n_state,36+n_network,:]
        
        diff = abs(measure_all - measure_one)
        perm_diff = abs(perm_all - perm_one)
        
        p = np.sum(perm_diff > diff)/100
        
        p_one[n_state, n_network] = p  
        perf_diff_one[n_state, n_network] = measure_all - measure_one
        
        
plt.figure()
vmin = -0.1  
vmax = 0.1
cmap = cm.get_cmap("bwr").copy()
divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
im = plt.imshow(np.tanh(perf_diff_one), cmap=cmap, norm=divnorm, aspect='equal')
nbins = 6   
ax = plt.gca();
#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 10, 1))

#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(state_names, fontname='Arial', fontsize = 20)

#Minor ticks
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 10.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('one network edges compared to all edges', fontname = 'Arial', fontsize = 18, pad=10)

sig_x, sig_y = np.where(p_one<p_thresh) #Mark significant differences with asteriks
for x,y in zip(sig_x, sig_y):
    plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 12, mew=2)
plt.colorbar()
plt.tight_layout()

#%% Test importance of functional brain networks in predicting intelligence

#Choose measure of specific intelligence component
measure_mean = measure_g_mean #measure_g_mean, measure_gc_mean, measure_gf_mean

measure_mean_wb = measure_mean[:, idx_within_between]
measure_mean_allbutone = measure_mean[:, idx_allbutone]
measure_mean_one = measure_mean[:, idx_one]



##Within- between links
idx_n1_wb, idx_n2_wb = np.where(np.triu(np.ones(n_yeo_networks))==1) #Network combinations in idx_within_between
p_wb = np.ones((7,7))
n1, n2 = np.triu_indices(7,1) #All combinations of two networks
for net1, net2 in zip(n1, n2):
    
    idx_n1 = (idx_n1_wb == net1) | (idx_n2_wb == net1)
    idx_n2 = (idx_n1_wb == net2) | (idx_n2_wb == net2)
    
    [t, p_wb[net1,net2]] = scipy.stats.ttest_rel(measure_mean_wb[:,idx_n1].ravel(), measure_mean_wb[:,idx_n2].ravel())
    

#Plot mask with highlighted significant differences    
p_wb[p_wb >= p_thresh]= np.nan
p_wb[p_wb < p_thresh] = 1
plt.figure()
plt.imshow(p_wb, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 7, 1))

#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(networks, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network differences within-between')


#Mean performance of networks across all within- and between network combinations and states
g_n = []
for n in range(n_yeo_networks):
    
    idx_n = (idx_n1_wb == n) | (idx_n2_wb == n)
    g_n.append(np.mean(measure_mean_wb[:, idx_n]))

g_n = np.tanh(np.array(g_n))
plt.figure()

plt.bar(np.arange(7),g_n)
ax = plt.gca();
ax.set_xticks(np.arange(0, 7, 1))
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
    



##Links of all but one network

#Performance drops when removing a specific network = network was important

p_abo = np.ones((7,7))
n1, n2 = np.triu_indices(7,1)
for net1, net2 in zip(n1, n2):
    
    idx_n1 = net1
    idx_n2 = net2
    
    [t, p_abo[net1,net2]] = scipy.stats.ttest_rel(measure_mean_allbutone[:,idx_n1].ravel(), measure_mean_allbutone[:,idx_n2].ravel())


#Plot mask with highlighted significant differences    
p_abo[p_abo >= p_thresh]= np.nan
p_abo[p_abo < p_thresh] = 1
plt.figure()
plt.imshow(p_abo, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 7, 1))

#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(networks, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network differences all but one')

baseline = np.mean(measure_mean[:,idx_all_edges]) #Mean prediction (across states) with all edges
g_n = []
for n in range(n_yeo_networks):
    
    idx_n = n
    g_n.append(baseline-np.mean(measure_mean_allbutone[:, idx_n])) #Drop in performance

g_n = np.tanh(np.array(g_n)) 
plt.figure()
plt.bar(np.arange(7),g_n)
ax = plt.gca();
ax.set_xticks(np.arange(0, 7, 1))
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)    




##Links of one network 
p_o = np.ones((7,7))
n1, n2 = np.triu_indices(7,1)
for net1, net2 in zip(n1, n2):
    
    idx_n1 = net1
    idx_n2 = net2
    
    [t, p_o[net1,net2]] = scipy.stats.ttest_rel(measure_mean_one[:,idx_n1].ravel(), measure_mean_one[:,idx_n2].ravel())
    

#Plot mask with highlighted significant differences    
p_o[p_o >= p_thresh]= np.nan
p_o[p_o < p_thresh] = 1
plt.figure()
plt.imshow(p_o, cmap = 'bwr', vmin = -1, vmax = 1)  
ax = plt.gca();

#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 7, 1))

#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
ax.set_yticklabels(networks, fontname='Arial', fontsize = 20) 

#Minor ticks
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)

#Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
plt.title('network differences one network')

g_n = []
for n in range(n_yeo_networks):
    idx_n = n
    g_n.append(np.mean(measure_mean_one[:, idx_n]))

g_n = np.tanh(np.array(g_n))
plt.figure()
plt.bar(np.arange(7),g_n)
ax = plt.gca();
#Major ticks
ax.set_xticks(np.arange(0, 7, 1))
#Labels for major ticks
ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
   