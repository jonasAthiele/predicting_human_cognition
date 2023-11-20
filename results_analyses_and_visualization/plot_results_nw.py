# -*- coding: utf-8 -*-
"""
Created on Jun 5 2023

@author: Jonas A. Thiele
"""

###Plot results of network-specific link selections for main sample, lockbox sample, and replication sample

#%% Imports

import numpy as np
import seaborn as sns
from matplotlib import colors  
import matplotlib.pyplot as plt
import os.path
from matplotlib import ticker
from matplotlib import cm

      
#%% Define what to plot

name_score = 'g' # 'g', 'gc', 'gf' 
name_sample = 'main' # 'main', 'lockbox', 'repli'
name_measure = 'corr' # 'corr', 'MSE', 'RMSE', 'MAE'


#%% Parameters

states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

idx_aomic = [0,1,7,8,9]
networks =  ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']
p_thresh = 0.05

if name_sample == 'repli':
    states = [states[index] for index in idx_aomic]
    state_names = [state_names[index] for index in idx_aomic]


#Load measure to plot
path_file = 'E:\\Transfer\\Results\\corrs'
name_file = name_measure + '_' + name_score + '_' + name_sample + '.npy'
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete) #Measure from real models

name_file = name_measure + '_perm_' + name_score + '_' + name_sample + '.npy'
path_complete = os.path.join(path_file, name_file)
measure_perm = np.load(path_complete) #Measure from null models


#Determine p-values (does real model predict better than null models?)
if name_measure == 'corr':
    measure_z = np.arctanh(measure) #Fisher z-transform for averaging correlations
    measure_mean_z = np.mean(measure_z, axis = 2) #Mean over iterations
    measure_mean = np.tanh(measure_mean_z) #Back-transforming to correlations
    
    measure_p_perm = np.ones((len(states), 43)) #43 different selection of links
    for n_state, state in enumerate(states):
        for n_network in range(43):
            
            measure_perm_z = np.arctanh(measure_perm)
            #CI95 = np.percentile(measure_perm_z[n_state, n_network,:].ravel(), 95)
            sum_bad = np.sum(measure_perm_z[n_state, n_network,:] > measure_mean_z[n_state, n_network])
            measure_p_perm[n_state, n_network] = sum_bad/measure_perm.shape[2]
    
    
else:
    
    measure_mean = np.mean(measure, axis = 2)
    
    measure_p_perm = np.ones((len(states), 43))
    for n_state, state in enumerate(states):
        for n_network in range(43):
            #CI95 = np.percentile(measure_perm[n_state, n_network,:].ravel(), 95)
            sum_bad = np.sum(measure_perm[n_state, n_network,:] < measure_mean[n_state, n_network])
            measure_p_perm[n_state, n_network] = sum_bad/measure_perm.shape[2]
    
    

#%% Full FC
customPalette = sns.color_palette('Set2')


if name_score == 'g':
    color_sns = customPalette[1]
elif name_score == 'gc':
    color_sns = customPalette[2]
elif name_score == 'gf':
    color_sns = customPalette[0]

if name_sample == 'repli':
    color_sns = customPalette[3]
        

idx_all = 0
data = measure[:,idx_all,:].T
p_all = measure_p_perm[:,idx_all]

if name_measure == 'corr':
    ymin = 0
    ymax = 0.50
elif name_measure == 'MSE':
    ymin = 0.02
    ymax = 0.04
elif name_measure == 'RMSE':
    ymin = 0.14
    ymax = 0.2
elif name_measure == 'MAE':
    ymin = 0.11
    ymax = 0.16
    
y_range = ymax-ymin


fig,a = plt.subplots(figsize = (5.5,4.5))

if name_measure == 'corr':
    measure_mean_all_states = np.tanh(np.mean(np.arctanh(measure_mean[:,idx_all]))) #Average correlation over states
    a.hlines(measure_mean_all_states, 0 - 0.5, len(states)-0.5, zorder=2, color = color_sns, linewidth=3, linestyle = '--', alpha = 1)

p = sns.stripplot(data=data, size=7, color=color_sns, zorder=0.1, edgecolor = '0.3', linewidth = 1)
plt.grid(color='gray', linestyle='dashed', linewidth=0.5)
plt.xticks(np.arange(len(states)), state_names, rotation = 90, fontname = 'Arial', fontsize = 24)
plt.ylim((ymin, ymax))
if name_measure == 'corr':
    plt.yticks(plt.yticks()[0], ['0', '.1', '.2', '.3', '.4', '.5'], fontname = 'Arial', fontsize = 24)
plt.yticks(fontname = 'Arial', fontsize = 24)


a.spines['right'].set_visible(False)
a.spines['top'].set_visible(False)
df_mean = measure_mean[:,idx_all]
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color = 'k', linewidth=3) for i, y in enumerate(list(df_mean))]

plt.locator_params(axis='y', nbins=6)


sig_x = np.where(p_all<p_thresh)[0]

if name_measure == 'corr':
    for x,y in zip(sig_x, df_mean[sig_x]):
        plt.plot(x+0.38,y+y_range/25, marker = (5, 2, 0), color = 'k', ms = 12, mew=2)
        #plt.plot(x, y_range-0.02, marker = (5, 2, 0), color = 'k', ms = 12, mew=2)

plt.tight_layout()
name_save = 'FC_full' + '_' + name_sample + '_' + name_score + '_' + name_measure + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)


#%% Whithin and between networks 

idx_withinbetween = np.arange(1,28+1) #Indexes of within and between network link selections


measure_mean_n = measure_mean[:,idx_withinbetween]
measure_p_n = measure_p_perm[:,idx_withinbetween]

idx_n1, idx_n2 = np.where(np.triu(np.ones(7))==1)

measure_n_plot = []
p_n_plot = []
for n_state in range(len(states)):
    
    measure_n = np.zeros((7,7))
    p_n = np.ones((7,7))
    for i in range(idx_n1.shape[0]):

        measure_n[idx_n1[i], idx_n2[i]] = measure_mean_n[n_state, i]
        p_n[idx_n1[i], idx_n2[i]] = measure_p_n[n_state, i]

          
    measure_n_plot.append(measure_n) 
    p_n_plot.append(p_n)



for n_state in range(len(states)):

    plt.figure()
    if name_measure == 'corr':
        vmin = -0.33   
        vmax = 0.33
        cmap = cm.get_cmap("bwr").copy()
        divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = plt.imshow(measure_n_plot[n_state], cmap=cmap, norm=divnorm, aspect='equal')
        nbins = 6
    else:
        cmap = cm.get_cmap("Reds").copy()
        cmap.set_bad(color='white')
        data = measure_n_plot[n_state].copy()
        data[np.tril_indices(7, k=-1)] = np.nan
        if name_measure == 'MSE':
            #vmin = 0.025
            #vmax = 0.036
            vmin = np.around(measure_mean_n.min(),3) 
            vmax = np.around(measure_mean_n.max(),3)
            im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
            nbins = 5
        elif name_measure == 'RMSE':
            #vmin = 0.15
            #vmax = 0.2
            vmin = np.around(measure_mean_n.min(),3)
            vmax = np.around(measure_mean_n.max(),3)
            im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
            nbins = 4
        elif name_measure == 'MAE':
            #vmin = 0.12
            #vmax = 0.16
            vmin = np.around(measure_mean_n.min(),3)
            vmax = np.around(measure_mean_n.max(),3)
            im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
            nbins = 4
    
    ax = plt.gca();
    
    # Major ticks
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))
    
    # Labels for major ticks
    ax.set_xticklabels(networks, fontname='Arial', fontsize = 20, rotation = 90)
    ax.set_yticklabels(networks, fontname='Arial', fontsize = 20)
    
    # Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
    
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    #cb = plt.colorbar()
    #tick_locator = ticker.MaxNLocator(nbins=nbins)
    #cb.locator = tick_locator
    #cb.update_ticks()
    plt.title(state_names[n_state], fontname = 'Arial', fontsize = 22, pad=10)
    
    if name_measure == 'corr':
        sig_x, sig_y = np.where(p_n_plot[n_state]<p_thresh)
        for x,y in zip(sig_x, sig_y):
            plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 12, mew=2)
    
    plt.tight_layout()
    name_save = 'FC_within_between' + '_' + name_sample + '_' + name_score + '_' + name_measure + '_'+ states[n_state] + '.jpg'
    plt.savefig(name_save, format='jpg', dpi = 1200)
    
    
# plot colorbar    
fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="horizontal", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cb.locator = tick_locator
for t in cb.ax.get_xticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_FC_within_between' + '_' + name_sample + '_' + name_score + '_' + name_measure + '_' + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)


#%% All but one network

idx_allbutone = np.arange(29,35+1)
measure_allbutone = measure_mean[:,idx_allbutone]
p_allbutone = measure_p_perm[:,idx_allbutone]


plt.figure()
if name_measure == 'corr':
    vmin = -0.5
    vmax = 0.5
    cmap = cm.get_cmap("bwr").copy()
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(measure_allbutone, cmap="bwr", norm=divnorm, aspect='equal')
    nbins = 6
else:
    cmap = cm.get_cmap("Reds").copy()
    data = measure_allbutone.copy()
    if name_measure == 'MSE':
        vmin = 0.024
        vmax = 0.04
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 4
    elif name_measure == 'RMSE':
        vmin = 0.15
        vmax = 0.19
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 5
    elif name_measure == 'MAE':
        vmin = 0.115
        vmax = 0.155
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 4


ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, len(states), 1))

if name_sample != 'repli':
    fontsize = 16
else: 
    fontsize = 24

ax.set_yticklabels(state_names, fontname = 'Arial', fontsize = fontsize)
ax.set_xticklabels(networks, fontname = 'Arial', fontsize = fontsize, rotation = 90)

# Minor ticks
ax.set_yticks(np.arange(0.5, len(states) +0.5, 1), minor=True)
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

#plt.title(state_names[i], fontname = 'Arial', fontsize = 22, pad=10)
if name_measure == 'corr':
    sig_x, sig_y = np.where(p_allbutone<p_thresh)
    for x,y in zip(sig_x, sig_y):
        if name_sample != 'repli':
            plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 10, mew=1.5)
        else:
            plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 14, mew=2)

plt.tight_layout()
name_save = 'FC_allbutone' + '_' + name_sample + '_' + name_score + '_' + name_measure + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)


fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="vertical", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cb.locator = tick_locator
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_FC_allbutone' + '_' + name_sample + '_' + name_score + '_' + name_measure + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)



#%% One network

idx_one = np.arange(36,43)
measure_one = measure_mean[:,idx_one]
p_one = measure_p_perm[:,idx_one]

plt.figure()
if name_measure == 'corr':
    vmin = -0.5
    vmax = 0.5
    divnorm=colors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = plt.imshow(measure_one, cmap="bwr", norm=divnorm, aspect='equal')
    nbins = 6
else:
    cmap = cm.get_cmap("Reds").copy()
    data = measure_one.copy()
    if name_measure == 'MSE':
        vmin = 0.024
        vmax = 0.04
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 4
    elif name_measure == 'RMSE':
        vmin = 0.15
        vmax = 0.19
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 5
    elif name_measure == 'MAE':
        vmin = 0.115
        vmax = 0.155
        im = plt.imshow(data, cmap=cmap, vmin = vmin, vmax = vmax, aspect='equal')
        nbins = 4


ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, len(states), 1))

# Labels for major ticks
if name_sample != 'repli':
    fontsize = 16
else: 
    fontsize = 24

ax.set_yticklabels(state_names, fontname = 'Arial', fontsize = fontsize)
ax.set_xticklabels(networks, fontname = 'Arial', fontsize = fontsize, rotation = 90)

# Minor ticks
ax.set_yticks(np.arange(0.5, len(states)+0.5, 1), minor=True)
ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
#plt.title(state_names[i], fontname = 'Arial', fontsize = 22, pad=10)

if name_measure == 'corr':
    sig_x, sig_y = np.where(p_one<p_thresh)
    for x,y in zip(sig_x, sig_y):
        if name_sample != 'repli':
            plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 10, mew=1.5)
        else:
            plt.plot(y,x, marker = (5, 2, 0), color = 'k', ms = 14, mew=2)

plt.tight_layout()
name_save = 'FC_one' + '_' + name_sample + '_' + name_score + '_' + name_measure + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)

        
fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="vertical", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cb.locator = tick_locator
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_FC_one' + '_' + name_sample + '_' + name_score + '_' + name_measure + '.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)
        
        

        