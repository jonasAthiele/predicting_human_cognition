# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from matplotlib import cm


networks = ['VIS', 'SOM', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']


path_output = os.path.join(os.getcwd(),'results_run_demo')




def plot_within_between(corr, state, score, option):
    
    
    idx_n1, idx_n2 = np.where(np.triu(np.ones(7))==1)


        
    measure_n_plot = np.zeros((7,7))
    for i in range(idx_n1.shape[0]):

        measure_n_plot[idx_n1[i], idx_n2[i]] = corr[i]

    
    plt.figure()
    
    vmin = -0.33   
    vmax = 0.33
    cmap = cm.get_cmap("bwr").copy()
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(measure_n_plot, cmap=cmap, norm=divnorm, aspect='equal')
    
    
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
    plt.title(state, fontname = 'Arial', fontsize = 22, pad=10)
    plt.colorbar(im,orientation="vertical")
    
    plt.tight_layout()
    name_save = 'FC_within_between' + '_' + 'mainsample' + '_' + state + '_' + score + '_' + option + '_' + 'corr' + '.jpg'
    path_save = os.path.join(path_output, name_save)
    plt.savefig(path_save, format='jpg', dpi = 1200)


              
def plot_allbutone(corr, state, score, option):


    measure_allbutone = np.reshape(np.array(corr),[1,-1])

    
    plt.figure()
    
    vmin = -0.5
    vmax = 0.5
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(measure_allbutone, cmap="bwr", norm=divnorm, aspect='equal')

    
    
    ax = plt.gca();
    
    # Major ticks
    ax.set_xticks(np.arange(0, 7, 1))

    

    fontsize = 24
    
    ax.set_yticklabels(state, fontname = 'Arial', fontsize = fontsize)
    ax.set_xticklabels(networks, fontname = 'Arial', fontsize = fontsize, rotation = 90)
    
    # Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.colorbar(im,orientation="vertical")
    #plt.title(state_names[i], fontname = 'Arial', fontsize = 22, pad=10)
    plt.tight_layout()
    name_save = 'FC_allbutone' + '_' + 'mainsample' + '_' + state + '_' + score + '_' + option + '_' + 'corr' + '.jpg'
    path_save = os.path.join(path_output, name_save)
    plt.savefig(path_save, format='jpg', dpi = 1200)
    
    
def plot_one(corr, state, score, option):


    measure_allbutone = np.reshape(np.array(corr),[1,-1])

    
    plt.figure()
    
    vmin = -0.5
    vmax = 0.5
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(measure_allbutone, cmap="bwr", norm=divnorm, aspect='equal')

    
    
    ax = plt.gca();
    
    # Major ticks
    ax.set_xticks(np.arange(0, 7, 1))

    

    fontsize = 24
    
    ax.set_yticklabels(state, fontname = 'Arial', fontsize = fontsize)
    ax.set_xticklabels(networks, fontname = 'Arial', fontsize = fontsize, rotation = 90)
    
    # Minor ticks
    ax.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
    
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.colorbar(im,orientation="vertical")
    #plt.title(state_names[i], fontname = 'Arial', fontsize = 22, pad=10)
    plt.tight_layout()
    name_save = 'FC_one' + '_' + 'mainsample' + '_' + state + '_' + score + '_' + option + '_' + 'corr' + '.jpg'
    path_save = os.path.join(path_output, name_save)
    plt.savefig(path_save, format='jpg', dpi = 1200)
    
    
