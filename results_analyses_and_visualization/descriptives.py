# -*- coding: utf-8 -*-
"""
Created on Feb 3 2023

@author: Jonas A. Thiele
"""

### Compute and visualize descriptives of the HCP and AOMIC (PIOP) samples


import pandas as pd
import numpy as np
import scipy.io as spio
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm

#Function to normalize data within a specified range
def normalize(x, newRange=(0, 1)): #X is an array. Default range is between zero and one
    xmin, xmax = np.nanmin(x), np.nanmax(x) #Get max and min from input array
    norm = (x - xmin)/(xmax - xmin) #Scale between zero and one
    
    if newRange == (0, 1):
        return(norm) #Wanted range is the same as norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0] #Scale to a different range.    


### HCP sample

#Names of cognitive states
states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']

#Yeo parcellation - Assignment of Nodes to networks
yeo_parcel = np.array(pd.read_csv('Schaf100Yeo.csv', header = None))[:,0]

#Read data from HCP main and lockbox sample
data_subjects1 = pd.read_csv('families_all_train.csv') #Main sample
data_subjects2 = pd.read_csv('families_all_test.csv') #Lockbox sample

#Concatenate data from both samples
frames = [data_subjects1, data_subjects2]
data_subjects = pd.concat(frames)

#Read functional connectivity (FC) of HCP main and lockbox sample and concatenate data of both samples
FC1 = spio.loadmat('FC_HCP_610_100nodes.mat')['FCStatic_combined']
FC2 = spio.loadmat('FC_HCP_lockbox_100nodes.mat')['FCStatic_combined']
FC = np.concatenate((FC1,FC2), axis = 2)

#Get intelligence scores (g, gc, gf) of both samples and concatenate data of both samples
g_scores1 = data_subjects1['g_score'].to_numpy()
g_scores2 = data_subjects2['g_score'].to_numpy()
g_scores = np.concatenate((g_scores1, g_scores2))

gc_scores1 = data_subjects1['gc_score'].to_numpy()
gc_scores2 = data_subjects2['gc_score'].to_numpy()
gc_scores = np.concatenate((gc_scores1, g_scores2))

gf_scores1 = data_subjects1['gf_score'].to_numpy()
gf_scores2 = data_subjects2['gf_score'].to_numpy()
gf_scores = np.concatenate((gf_scores1, gf_scores2))

#Read confound data of both samples
confounds1 = np.vstack((data_subjects1.Age, data_subjects1.Gender, data_subjects1.Handedness, data_subjects1.FD_mean, data_subjects1.Spikes_mean)).T
confounds2 = np.vstack((data_subjects2.Age, data_subjects2.Gender, data_subjects2.Handedness, data_subjects2.FD_mean, data_subjects2.Spikes_mean)).T
confounds = np.concatenate((confounds1, confounds2))

#Correlation between intelligence scores of 806 subjects of the HCP (= main and lockbox sample combined)
scipy.stats.pearsonr(g_scores, gc_scores)
scipy.stats.pearsonr(g_scores, gf_scores)
scipy.stats.pearsonr(gf_scores, gc_scores)

#Visualize frequency distributions of intelligence scores (Supplementary Figure S1)
customPalette = sns.color_palette('Set2')
alpha = 0.7
bins = 30
plt.figure()
ax=sns.histplot(g_scores, bins=bins, color = customPalette[1], alpha = alpha)
plt.ylabel('Frequency', fontsize = '16') 
plt.xlabel('General intelligence', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
plt.xlim(-3,3)
plt.ylim(0,80)
plt.locator_params(axis='y', nbins=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('g_score_distribution.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

plt.figure()
ax=sns.histplot(gc_scores,bins=bins, color = customPalette[2], alpha = alpha)
plt.ylabel('Frequency', fontsize = '16') 
plt.xlabel('Crystallized intelligence', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
plt.xlim(-3,3)
plt.ylim(0,80)
plt.locator_params(axis='y', nbins=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('gc_score_distribution.jpg', format='jpg', dpi = 1200, bbox_inches='tight')

plt.figure()
ax=sns.histplot(gf_scores,bins=bins, color = customPalette[0], alpha = alpha)
plt.ylabel('Frequency', fontsize = '16') 
plt.xlabel('Fluid intelligence', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
plt.xlim(-3,3)
plt.ylim(0,80)
plt.locator_params(axis='y', nbins=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('gf_score_distribution.jpg', format='jpg', dpi = 1200, bbox_inches='tight')



#Visualize group mean FCs for every state (Supplementary Figure S2)
cmap = cm.get_cmap("Reds").copy()
cmap.set_bad(color='white')
pos_lines = np.where(np.diff(yeo_parcel.ravel()) != 0)[0] + 0.5
for sc in range(10):

    
    
    FC_i = FC[:,:,:,sc]
    FC_i = np.mean(FC_i,2)
    np.fill_diagonal(FC_i, np.nan, wrap=False)
    FC_i = normalize(FC_i, newRange=(0, 1))
    
    np.fill_diagonal(FC_i, 0, wrap=False)
    
    fig,ax = plt.subplots()
    plt.imshow(FC_i, cmap=cmap, vmin = 0, vmax = 1)

    for pos in pos_lines:
    
        x1 = 0
        x2 = 99
        y1 = pos
        y2 = pos
         
        plt.hlines(pos, x1, x2, color = 'black', linewidth = 0.5)
        plt.vlines(pos, x1, x2, color = 'black', linewidth = 0.5)
    
    
    ax.spines[['right', 'top']].set_visible(False) 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticks(pos_lines)
    ax.set_yticks(pos_lines)
    #plt.colorbar()
    name = 'M_' + states[sc] + '.jpg'
    plt.savefig(name, format='jpg', dpi = 1200)


#Visualize group SD of FCs for every state (Supplementary Figure S2)
for sc in range(10):

    
    
    FC_i = FC[:,:,:,sc]
    FC_i = np.std(FC_i,2)
    np.fill_diagonal(FC_i, np.nan, wrap=False)
    FC_i = normalize(FC_i, newRange=(0, 1))
    np.fill_diagonal(FC_i, 0, wrap=False)


    vmin = 0
    vmax = 1
    
    
    fig,ax = plt.subplots()
    
    plt.imshow(FC_i, cmap = cmap, vmin = vmin, vmax = vmax)
    for pos in pos_lines:
    
        x1 = 0
        x2 = 99
        y1 = pos
        y2 = pos
         
        plt.hlines(pos, x1, x2, color = 'black', linewidth = 0.5)
        plt.vlines(pos, x1, x2, color = 'black', linewidth = 0.5)
    
    
    ax.spines[['right', 'top']].set_visible(False) 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticks(pos_lines)
    ax.set_yticks(pos_lines)
    #plt.colorbar()
    
    name = 'SD_' + states[sc] + '.jpg'
    plt.savefig(name, format='jpg', dpi = 1200)
    
nbins = 5    
#Plot colorbar    
fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="horizontal", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cb.locator = tick_locator
for t in cb.ax.get_xticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_mean_sd_fc_hcp_806.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)
    


    
    
### PIOP sample   

#Name of all states PIOP1
#   states_piop1 = {'rest','WM','anticipation','emotion','faces','gstroop', 'latent_states','latent_task'};
#Names of all states of PIOP2
#   scans_piop2 = {'rest','WM','emotion','stopsignal', 'latent_states','latent_task'};

#States used in analyses 
states = ['rest','WM','emotion','latent_states','latent_task']

idx_scans_piop1 = np.array([0,1,3,6,7]) #Indices of states used in analyses 
idx_scans_piop2 = np.array([0,1,2,4,5]) #Indices of states used in analyses 

#Read behavioral data of the replication sample and combine date of both PIOP1 and PIOP2
data_subjects_repli1 = pd.read_csv('data_beh_sel_subjects_PIOP1.csv')
data_subjects_repli2 = pd.read_csv('data_beh_sel_subjects_PIOP2.csv')
frames = [data_subjects_repli1, data_subjects_repli2]
data_subjects_repli = pd.concat(frames)

#Read functional connectivity (FC) matrices of the PIOP1 and PIOP2 sample and combine both samples
FC_repli1 = spio.loadmat('FC_PIOP1_100nodes.mat')['FCStatic_combined']
FC_repli1 = FC_repli1[:,:,:,idx_scans_piop1]
FC_repli2 = spio.loadmat('FC_PIOP2_100nodes.mat')['FCStatic_combined']
FC_repli2 = FC_repli2[:,:,:,idx_scans_piop2]
FC_repli = np.concatenate((FC_repli1,FC_repli2), axis = 2)

#Get intelligence scores of replication sample
intelligence_scores_lockbox = data_subjects_repli['raven_score'].to_numpy()

#Get confounds of replication sample
confounds_lockbox = np.vstack((data_subjects_repli.age, data_subjects_repli.sex, data_subjects_repli.handedness, data_subjects_repli.meanFD, data_subjects_repli.perc_small_spikes)).T

#Visualize distribution of RAPM scores of replication sample (Supplementary Figure S3)
alpha = 0.7
bins = 20
plt.figure()
ax=sns.histplot(intelligence_scores_lockbox, bins=bins, color = customPalette[3], alpha = alpha)
plt.ylabel('Frequency', fontsize = '16') 
plt.xlabel('RAPM scores', fontsize = '16') 
plt.yticks(fontsize = '16')
plt.xticks(fontsize = '16')
plt.ylim(0,50)
plt.locator_params(axis='y', nbins=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('Raven_Piop_distribution.jpg', format='jpg', dpi = 1200, bbox_inches='tight')


#Visualize group mean FCs for every state (Supplementary Figure S3)
for sc in range(5):

    
    FC_i = FC_repli[:,:,:,sc]
    FC_i = np.mean(FC_i,2)
    np.fill_diagonal(FC_i, np.nan, wrap=False)
    FC_i = normalize(FC_i, newRange=(0, 1))
    
    np.fill_diagonal(FC_i, 0, wrap=False)
    
    fig,ax = plt.subplots()
    plt.imshow(FC_i, cmap=cmap, vmin = 0, vmax = 1)

    for pos in pos_lines:
    
        x1 = 0
        x2 = 99
        y1 = pos
        y2 = pos
         
        plt.hlines(pos, x1, x2, color = 'black', linewidth = 0.5)
        plt.vlines(pos, x1, x2, color = 'black', linewidth = 0.5)
    
    
    ax.spines[['right', 'top']].set_visible(False) 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticks(pos_lines)
    ax.set_yticks(pos_lines)
    name = 'AOMIC_' + 'M_' + states[sc] + '.jpg'
    plt.savefig(name, format='jpg', dpi = 1200)


#Visualize group SD FCs for every state (Supplementary Figure S3)
for sc in range(5):
    
    FC_i = FC_repli[:,:,:,sc]
    FC_i = np.std(FC_i,2)
    np.fill_diagonal(FC_i, np.nan, wrap=False)
    FC_i = normalize(FC_i, newRange=(0, 1))
    np.fill_diagonal(FC_i, 0, wrap=False)


    vmin = 0
    vmax = 1
    
    
    fig,ax = plt.subplots()
    
    plt.imshow(FC_i, cmap = cmap, vmin = vmin, vmax = vmax)
    for pos in pos_lines:
    
        x1 = 0
        x2 = 99
        y1 = pos
        y2 = pos
         
        plt.hlines(pos, x1, x2, color = 'black', linewidth = 0.5)
        plt.vlines(pos, x1, x2, color = 'black', linewidth = 0.5)
    
    
    ax.spines[['right', 'top']].set_visible(False) 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticks(pos_lines)
    ax.set_yticks(pos_lines)
    
    name = 'AOMIC_' + 'SD_' + states[sc] + '.jpg'
    plt.savefig(name, format='jpg', dpi = 1200)
    

nbins = 5    
#Plot colorbar    
fig, ax = plt.subplots()
im = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
ax.set_visible(False)
cb = plt.colorbar(im, ax=ax,orientation="horizontal", fraction=0.5)
tick_locator = ticker.MaxNLocator(nbins=nbins)
cb.locator = tick_locator
for t in cb.ax.get_xticklabels():
    t.set_fontsize(16)
cb.update_ticks()
name_save = 'colorbar_mean_sd_fc_AOMIC.jpg'
plt.savefig(name_save, format='jpg', dpi = 1200)
