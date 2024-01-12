# -*- coding: utf-8 -*-
"""
Created on May 4 2023

@author: Jonas A. Thiele
"""

### Plot relevant links (edges) identified via stepwise LRP during model training in main sample


import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib import colors 
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
import argparse
import os.path
from os.path import dirname as up

#%% Data

#Cognitive States
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

parser = argparse.ArgumentParser()
parser.add_argument("--score", type=str, default = 'g')

args = parser.parse_args()
name_score = args.score #Sample: 'main', 'lockbox', 'repli'

path_current = os.path.dirname(os.path.abspath("__file__"))
path_file = os.path.join(path_current, 'results')


#Load mask of relevant links (edges) for g, gc, or gf
if name_score =='g':
    name_file = 'mask_g_relevants.npy'
elif name_score =='gc':
    name_file = 'mask_gc_relevants.npy'
elif name_score =='gf':
    name_file = 'mask_gf_relevants.npy'
else:
    raise ValueError('Choose valid score: <g>, <gc>, or <gf>')    
    
path_complete = os.path.join(path_file, name_file)
mask_relevants = np.load(path_complete) #Shape: state x iteration (different number of relevant edges) x nodes x nodes

#FCs used for prediction
demo_folder = up(path_current)
path_fc = os.path.join(demo_folder, 'source', 'FC_HCP_610_100nodes.mat')
FC = spio.loadmat(path_fc)['FCStatic_combined']

#Lines between networks for plotting
path_yeo = os.path.join(demo_folder, 'source', 'Schaf100Yeo.csv')
yeo_parcel = np.array(pd.read_csv(path_yeo, header = None))[:,0]
pos_lines = np.where(np.diff(yeo_parcel.ravel()) != 0)[0] + 0.5

n_states = len(state_names)
n_nodes = 100

path_output = os.path.join(os.getcwd(),'figures')
#%% Plot most relevant links (edges) in matrix form with sign of mean edge weigth (red or blue) 
#   and strength of relevance (intensity of color)

n = 4 #Iteration with different numbers of most relevant edges (n=0 -> n_edges=45, n=1 -> n_edges=190, n=2 -> n_edges=435, n=3 -> n_edges=780, n=4 -> n_edges=1000)

#Loop over states
for sc in range(n_states):
    
    #Binarize FC
    FC_mean_bin = np.mean(FC[:, :, :, sc],2)
    FC_mean_bin[FC_mean_bin>0] = 1
    FC_mean_bin[FC_mean_bin<0] = -1
    
    
    idx1, idx2 = np.where(mask_relevants[sc,n,:,:] > 0) #Indices of most relevant edges
    
    mask = np.zeros((n_nodes,n_nodes))
    mask[idx1, idx2] = np.mean(FC[idx1, idx2, :, sc],1)
    
    mask = np.multiply(FC_mean_bin, mask_relevants[sc,n,:,:])
    
    #plt.figure()
    divnorm=colors.TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.7)
    
    fig,ax = plt.subplots();
    
    plt.imshow(mask, cmap = 'bwr', norm = divnorm)

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
    name_save = state_names[sc] + '_' + name_score + '_' + 'relevant_FC'
    path_save = os.path.join(path_output, name_save)
    fig.savefig(path_save + '.jpg', format='jpg', dpi = 1200)
    plt.close()
    

#%% Connectograms of most relevant links (edges)

n_edges_relevant = 100 #Number of edges to plot
n = 4 #Iteration with different numbers of most relevant edges (n=0 -> n_edges=45, n=1 -> n_edges=190, n=2 -> n_edges=435, n=3 -> n_edges=780, n=4 -> n_edges=1000)

yeo_parcel = pd.read_csv(path_yeo, header = None).to_numpy()[:,0]-1 #Assignment nodes to networks

#Borders between networks
borders = [0,7,13,20,27,30,37,50,62,71,73,80,86,94]

#Colors of each network
colors = ['purple','steelblue','seagreen','pink','lemonchiffon','darkorange','indianred']
node_colors = list(np.array(colors)[yeo_parcel])

#Names of nodes
label_names_short = list(np.arange(100).astype(str)) #Number of node as name

#Arrangement of each node in connectogram
node_angles = circular_layout(label_names_short, label_names_short[0:50] + label_names_short[50::][::-1] , start_pos=90, group_boundaries=borders)

#Plot
for sc in range(n_states):
            
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    mask = mask_relevants[sc,n,:,:] 
    plot_connectivity_circle(mask, label_names_short, n_lines=n_edges_relevant, node_angles=node_angles, node_colors=node_colors, padding=1.0, title = state_names[sc], facecolor = 'white', colorbar = False, colorbar_size=0.5, colormap='Reds', linewidth=1, node_linewidth=1, ax = ax, show=False)          
    plt.tight_layout()
    name_save = state_names[sc] + '_' + name_score + '_' + 'connectogram' + '.jpg'
    path_save = os.path.join(path_output, name_save)
    fig.savefig(path_save, format='jpg', dpi = 1200)