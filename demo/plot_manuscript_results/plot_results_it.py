# -*- coding: utf-8 -*-
"""
Created on Apr 20 2023

@author: Jonas A. Thiele
"""

###Plot performances of prediction models trained with FC links between nodes of intelligence theories vs. models
###trained with FC links between the same number of random nodes



import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import argparse


states = ['rest','WM','gambling', 'motor', 'language', 'social', 'realtional', 'emotion', 'latent_states', 'latent_task']
state_names = ['RES', 'WM', 'GAM', 'MOT', 'LAN', 'SOC', 'REL', 'EMO', 'LAT', 'LAT-T']

#Names of theories and corresponding ramdom nodes 
# 'pfc_extended' - '14nodes_nodes_rand'
# 'pfit' - '8nodes_nodes_rand'
# 'md_diachek' - '26nodes_nodes_rand'
# 'md_duncan' - '13nodes_nodes_rand'
# 'cole' - '4nodes_nodes_rand'



parser = argparse.ArgumentParser()
parser.add_argument("--theory", type=str, default = 'md_diachek')

args = parser.parse_args()
name_theory = args.theory

name_sample = 'main' #main sample
name_score = 'g' #Intelligence components
name_measure = 'corr' #Correlation between observed and predicted intelligence scores



if name_theory == 'pfc_extended':

    name_rand_nodes = '14nodes_nodes_rand' #Choose random nodes (number nodes as per list above)
    
elif name_theory == 'pfit' :
    
    name_rand_nodes = '8nodes_nodes_rand'
    
elif name_theory == 'md_diachek' :
    
    name_rand_nodes = '26nodes_nodes_rand'
    
elif name_theory == 'md_duncan' :
    
    name_rand_nodes = '13nodes_nodes_rand'

elif name_theory == 'cole' :
    
    name_rand_nodes = '4nodes_nodes_rand'

else:
    raise ValueError('Choose valid name of intelligence theory: <pfc_extended>, <pfit>, <md_diachek>, <md_duncan>, or <cole>')

p_thresh = 0.05
path_output = os.path.join(os.getcwd(),'figures')
      
#Read data
path_current = os.path.dirname(os.path.abspath("__file__"))
path_file = os.path.join(path_current, 'results')
name_file = name_measure + '_' + name_score + '_' + name_sample + '_' + name_theory + '.npy' 
path_complete = os.path.join(path_file, name_file)
measure = np.load(path_complete) #Performance intelligence theory


name_file = name_measure + '_' + name_score + '_' + name_sample + '_' + name_rand_nodes + '.npy'
path_complete = os.path.join(path_file, name_file)
measure_perm = np.load(path_complete) #Performance of random nodes



#Performance intelligence theory
measure_z = np.arctanh(measure)#Fisher z-transformed correlation between predicted and observed intelligence scores 
measure_mean_z = np.mean(measure_z, axis = 1) #Mean across iterations
measure_mean = np.tanh(measure_mean_z) #Back tranform to correlation
print('Across state performance')
print(name_theory)
print(np.tanh(measure_z.mean()))

#Performance random nodes
measure_perm_z = np.arctanh(measure_perm) #Fisher z-transformed correlation between predicted and observed intelligence scores 

#Ttest for testing significance of difference between means of performances in predicting g-scores from FC between nodes 
#of intelligence theories vs. FC betweem random nodes
p_ttest = [] #Store p-values
theory_better = [] #Store if mean performance of theory is better   

#Loop over states
for n_state, state in enumerate(states):
        
    measure_theory_ttest = measure_z[n_state,:] #Performances theory
    measure_perm_ttest = measure_perm_z[n_state,:] #Performances random nodes
    
    p_ttest.append(scipy.stats.ttest_ind(measure_theory_ttest, measure_perm_ttest)[1])
    theory_better.append(measure_theory_ttest.mean() > measure_perm_ttest.mean())
    

    
    

#%% Visualize as stripplot

#Define color
customPalette = sns.color_palette('Set2')
color_sns = customPalette[1]

#Performances and mean performances of theory and random nodes
data_perm = measure_perm.T
data_theory = measure[0:len(states),:].T
data_theory_mean = measure_mean
data_perm_mean = np.tanh(np.mean(np.arctanh(measure_perm), axis = 1))


#Define range of y-axis
ymin = 0
ymax = 0.5   
y_range = ymax-ymin

#Plot all performance values as stripplots
fig,a = plt.subplots(figsize = (5.5,4.5))
sns.stripplot(data=data_perm, size=7, color='lightgrey', alpha = 0.2, zorder=0.1, edgecolor = '0.4', linewidth = 1)
p = sns.stripplot(data=data_theory, size=7, color=color_sns, zorder=0.1, edgecolor = '0.4', linewidth = 1)
plt.grid(color='gray', linestyle='dashed', linewidth=0.5)
plt.xticks(np.arange(len(states)), state_names, rotation = 90, fontname = 'Arial', fontsize = 24)
plt.ylim((ymin, ymax))

plt.yticks(plt.yticks()[0], ['0', '.1', '.2', '.3', '.4', '.5'], fontname = 'Arial', fontsize = 24)
plt.yticks(fontname = 'Arial', fontsize = 24)
a.spines['right'].set_visible(False)
a.spines['top'].set_visible(False)

#Add means of performances
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color = 'dimgrey', linewidth=3, alpha=0.8) for i, y in enumerate(list(data_perm_mean))]
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color = 'k', linewidth=3, alpha =0.8) for i, y in enumerate(list(data_theory_mean))]

plt.locator_params(axis='y', nbins=6)

#Highlight significant differences 
sig_x = np.where(np.array(p_ttest)<p_thresh)[0]
sig_color = np.array(theory_better)[sig_x]
  
for x,y1,y2, color in zip(sig_x, data_theory_mean[sig_x], data_perm_mean[sig_x], sig_color):
    if color:
        plt.plot(x+0.33,y1+y_range/20, marker = (5, 2, 0), color = 'k', ms = 12, mew=2.5)
    else:
        plt.plot(x+0.33,y2+y_range/20, marker = (5, 2, 0), color = 'dimgrey', ms = 12, mew=2.5)

plt.tight_layout()
name_save = 'theory_vs_random' + '_' + name_sample + '_' + name_score + '_' + name_measure + '_' + name_theory + '.jpg'
path_save = os.path.join(path_output, name_save)
plt.savefig(path_save, format='jpg', dpi = 1200)











        