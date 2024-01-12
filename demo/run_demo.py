# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""

import numpy as np
import argparse
import pandas as pd
from train_models_nw import train_models_nw
from train_models_rnrlit import train_models_rnrlit
from lib.plot_single_iteration import plot_within_between, plot_allbutone, plot_one
import os.path

## Run demo

## Includes minimal examples of intelligence prediction within the main sample (610 subjects, 5-fold cross-validation)
# Note that only one iteration of model training is performed (either with correctly assigned or permutated intelligence scores) and the output is the result of this specific iteration
# For results of the manuscript 10 iterations with correctly assigned scores and 100 permutations with permuted scores were performed


# One iteration respectively either with right ordered intelligence scores (option='real') or with permutated scores (option='permutation')
# Training with FC of one state only (choose from: 'rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task')




##To run without parser
# =============================================================================
# link_selection = 'cole'
# score='g_score'
# state='WM'
# option = 'real'
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--state", type=str, default = 'WM')
parser.add_argument("--score", type=str, default = 'g_score')
parser.add_argument("--option", type=str, default = 'real')
parser.add_argument("--link_selection", type=str, default = 'all')

args = parser.parse_args()

state = args.state
score = args.score
option = args.option
link_selection = args.link_selection

name_list_rnrlit = ['nodes', 'links', 'pfc_extended', 'pfit', 'md_diachek', 'md_duncan', 'cole']
rnrlit = 0
for name in name_list_rnrlit:

    if name in link_selection:
        
        rnrlit = 1




#Names of functional networks
networks = ['VIS', 'SOM', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']



#Perform training and output results dependend on the selection of functional brain links used for model training
if link_selection == 'all':
    
    # Train models with all brain links
    corr_all = train_models_nw(state=state, score=score, option=option, link_selection=link_selection) 
    
    res = np.round(np.array(corr_all),2)
    print('Result all links for ' + state +' with ' + option + ' intelligence scores: ' + score)
    print('Correlation between observed and predicted intelligence scores:')
    df_res = pd.DataFrame(columns = ['all_links'], data = [res])
    df_res.index = ['r']
    print(df_res)
    name_save = 'corr_allLinks' + '_' + '_' + state + '_' + score + '_' + option + '.csv'
    path_save = os.path.join('results_run_demo', name_save)
    df_res.to_csv(path_save, index=False)


elif link_selection == 'within_between':
    
    #Train models with links within a network or between two networks
    corr_all = train_models_nw(state = state, score = score, option= option, link_selection = link_selection) 
    
    res = np.round(np.array(corr_all),2)
    print('Result within and between network links for ' + state +'with ' + option + ' intelligence scores: ' + score)
    print('Correlation between observed and predicted intelligence scores:')
    plot_within_between(corr = res, state=state, score=score, option=option)
    
    #Make df
    idx_n1, idx_n2 = np.where(np.triu(np.ones(7))==1) 
    measure_n = np.zeros((7,7))
    for i in range(idx_n1.shape[0]):

        measure_n[idx_n1[i], idx_n2[i]] = res[i]
    
    measure_n[measure_n==0] = np.nan
    df_res = pd.DataFrame(columns = networks, data = measure_n)
    df_res.index = networks
    print(df_res)
    name_save = 'corr_within-between' + '_' + '_' + state + '_' + score + '_' + option + '.csv'
    path_save = os.path.join('results_run_demo', name_save)
    df_res.to_csv(path_save)


elif link_selection == 'allbutone':
    
    #Train models with links of all but one network
    corr_all = train_models_nw(state = state, score = score, option = option, link_selection = link_selection) 
    
    res = np.round(np.array(corr_all),2)
    print('Result links of all but one networks links for ' + state +'with ' + option + ' intelligence scores: ' + score)
    print('Correlation between observed and predicted intelligence scores:')
    df_res = pd.DataFrame(columns = networks, data = [res])
    df_res.index = [state]
    print(df_res)
    plot_allbutone(corr = res, state=state, score=score, option=option)
    name_save = 'corr_allbutone' + '_' + '_' + state + '_' + score + '_' + option + '.csv'
    path_save = os.path.join('results_run_demo', name_save)
    df_res.to_csv(path_save)
    


elif link_selection == 'one':
    
    #Train models with within-and between links of one network
    corr_all = train_models_nw(state = state, score = score, option = option, link_selection = link_selection)
    
    res = np.round(np.array(corr_all),2)
    print('Result links of one network ' + state +'with ' + option + ' intelligence scores: ' + score)
    print('Correlation between observed and predicted intelligence scores:')
    df_res = pd.DataFrame(columns = networks, data = [res])
    df_res.index = [state]
    print(df_res)
    plot_one(corr = res, state=state, score=score, option=option)
    name_save = 'corr_one' + '_' + '_' + state + '_' + score + '_' + option + '.csv'
    path_save = os.path.join('results_run_demo', name_save)
    df_res.to_csv(path_save)



elif rnrlit:
    
    #Train models with links between randomly selected nodes, randomly selected links, and links of intelligence theories
    corr_all = train_models_rnrlit(state=state, score=score, link_selection=link_selection)
    
    res = np.round(np.array(corr_all),2)
    print('Result for ' + state +' with ' + 'real' + ' intelligence scores: ' + score)
    print(link_selection)
    print('Correlation between observed and predicted intelligence scores:')
    df_res = pd.DataFrame(columns = ['link selection'], data = [res])
    df_res.index = ['r']
    print(df_res)
    name_save = 'corr_' + link_selection + '_' + '_' + state + '_' + score + '.csv'
    path_save = os.path.join('results_run_demo', name_save)
    df_res.to_csv(path_save, index=False)
    
    
else:
    
    raise ValueError('Wrong parameters - check with Readme')

