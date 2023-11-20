# -*- coding: utf-8 -*-
"""
Created on Aug 8 2022

@author: Jonas A. Thiele
"""


import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from random import randint
import matplotlib.pyplot as plt


#%% Functions

#Compute total number of subjects
def get_sample_size(families_all, sample_array):
    
    sample_size = 0
    for f in sample_array:
        indexes_family_members = np.where(families_all == f)[0]
        sample_size = sample_size + indexes_family_members.size
    
    return sample_size

#%% Read data

behavioral_data_analysis = pd.read_csv("behavioral_data_analysis.csv") 

restricted_data = pd.read_csv("tables_HCP/RESTRICTED_jonasthiele_11_1_2020_8_46_1.csv") 

indexes_subjects_analysis = restricted_data['Subject'].isin(behavioral_data_analysis['Subject'])
restricted_data_analysis = restricted_data[indexes_subjects_analysis]
restricted_data_analysis = restricted_data_analysis.reset_index(drop=True)

#Mother and father IDs of all selected subjects
ID_vec_family = restricted_data_analysis.loc[:,['Subject','Family_ID']]
ID_vec_family = ID_vec_family.to_numpy()

subject_IDs_left = ID_vec_family[:,0]
family_IDs_left = ID_vec_family[:,1]

#%% Get main (train) and lockbox (test) sample

#Collect subjects belonging to same families
families_all = []
cnt_families = 0
while subject_IDs_left.size != 0:
    
    
    indexes_family_members = np.where(family_IDs_left==family_IDs_left[0])[0]
    
    new_family_subject_IDs = subject_IDs_left[indexes_family_members]
    new_family_family_ID = family_IDs_left[indexes_family_members]
    new_family_cnt = cnt_families*np.ones((indexes_family_members.size,), dtype=int)
   
    family_new = np.transpose(np.array((new_family_subject_IDs, new_family_family_ID, new_family_cnt)))
    
    families_all.append(family_new)
    
    subject_IDs_left = np.delete(subject_IDs_left, indexes_family_members, axis=0)
    family_IDs_left = np.delete(family_IDs_left, indexes_family_members, axis=0)
    
    cnt_families = cnt_families + 1


families_all = np.concatenate(families_all)


#Find corresponding intelligence scores for our subjects
intelligence_scores = pd.read_csv('intelligence_factors_1186Subjects.csv')
g_scores_all = intelligence_scores.g.to_numpy()
IDs_g_scores = intelligence_scores.Subject.to_numpy()
gf_scores_all = intelligence_scores.gf.to_numpy()
gc_scores_all = intelligence_scores.gc.to_numpy()

g_scores_analysis = []
gf_scores_analysis = []
gc_scores_analysis = []
for s in families_all[:,0]:
    idx_s = np.where(IDs_g_scores==s)[0]
    g_score_subject = g_scores_all[idx_s]
    gf_score_subject = gf_scores_all[idx_s]
    gc_score_subject = gc_scores_all[idx_s]
    g_scores_analysis.append(g_score_subject)
    gf_scores_analysis.append(gf_score_subject)
    gc_scores_analysis.append(gc_score_subject)
    

families_all = np.hstack((families_all, g_scores_analysis, gf_scores_analysis, gc_scores_analysis))

#Average intelligence scores of each family
g_scores_family_average_all = []
gf_scores_family_average_all = []
gc_scores_family_average_all = []
summarized_g_scores = []
family_number = []
for f in np.unique(families_all[:,2]):
    indexes_family = np.where(families_all[:,2]==f)[0]
    g_scores_family_average = np.mean(families_all[indexes_family,3])
    gf_scores_family_average = np.mean(families_all[indexes_family,4])
    gc_scores_family_average = np.mean(families_all[indexes_family,5])
    
    summarized_g_scores.append(g_scores_family_average)
    family_number.append(f)
    
    g_scores_family_average = np.full((indexes_family.size,), g_scores_family_average)
    gf_scores_family_average = np.full((indexes_family.size,), gf_scores_family_average)
    gc_scores_family_average = np.full((indexes_family.size,), gc_scores_family_average)
    
    g_scores_family_average_all.append(g_scores_family_average)
    gf_scores_family_average_all.append(gf_scores_family_average)
    gc_scores_family_average_all.append(gc_scores_family_average)
    

g_scores_family_average_all = np.concatenate(g_scores_family_average_all) 
g_scores_family_average_all = np.reshape(g_scores_family_average_all,(-1,1))
gf_scores_family_average_all = np.concatenate(gf_scores_family_average_all) 
gf_scores_family_average_all = np.reshape(gf_scores_family_average_all,(-1,1))
gc_scores_family_average_all = np.concatenate(gc_scores_family_average_all) 
gc_scores_family_average_all = np.reshape(gc_scores_family_average_all,(-1,1))

families_all = np.hstack((families_all, g_scores_family_average_all, gf_scores_family_average_all, gc_scores_family_average_all))


#Add confounds data to subjects
small_spikes = pd.read_csv("dataSmallSpikes_allfMRI_sets_941subjects.csv").to_numpy() 
FD = pd.read_csv("dataMeanFD_allfMRI_sets_941subjects.csv").to_numpy() 

small_spikes_mean = np.mean(small_spikes[:,1::], axis = 1) #Average over all scan conditions
FD_mean = np.mean(FD[:,1::], axis = 1) #Average over all scan conditions

confounds = []
for subject in families_all[:,0]:
      
    index_subject_behavioral_data = np.where(behavioral_data_analysis['Subject'] == subject)[0]
    gender_subject_string = np.array(behavioral_data_analysis['Gender'][index_subject_behavioral_data])
    
    if gender_subject_string == 'M':
        gender_subject = 1
    elif gender_subject_string == 'F':
        gender_subject = 0
    else:
        raise ValueError('Problem with sex value')
            
    gender_subject = np.array([gender_subject])    
    index_subject_restricted_data = np.where(restricted_data_analysis['Subject'] == subject)[0]
    age_subject = restricted_data_analysis['Age_in_Yrs'][index_subject_restricted_data].to_numpy()
    handedness_subject = restricted_data_analysis['Handedness'][index_subject_restricted_data].to_numpy()
    
    index_subject_FD_data = np.where(FD[:,0]== subject)[0]
    FD_mean_subject = FD_mean[index_subject_FD_data]
    
    index_subject_spikes_data = np.where(small_spikes[:,0]== subject)[0]
    spikes_mean_subject = small_spikes_mean[index_subject_spikes_data]
    
    
    confounds.append((age_subject, gender_subject, handedness_subject, FD_mean_subject, spikes_mean_subject))
    
confounds = np.concatenate(confounds)

confounds = np.reshape(confounds, (families_all.shape[0], -1))    

families_all = np.hstack((families_all, confounds))    


#Sort families according to g-score
families_summarized = np.array((family_number,summarized_g_scores))
indexes_sorted_g_scores = np.argsort(summarized_g_scores)
families_sorted = families_summarized[:,indexes_sorted_g_scores]


#Randomly choose one family from each of 4 consecutive families (sorted according to IQ)
#to split data --> 75 % for analysis (main sample), 25 % for later testing (lockbox sample)
#Only recommended for percentage_test = 2,5,10,20,25,50 %
percentage_train = 75
percentage_test = 25
size_family_group = int(100/percentage_test)

number_family_groups = int(families_sorted.shape[1]/size_family_group) #Number of families per group 
number_families_rest = families_sorted.shape[1]%size_family_group #Number of families remaining after group assignment
number_families_group_total = number_family_groups*size_family_group #Total number of families in all groups

family_groups_split = np.split(families_sorted[0,0:number_families_group_total], number_family_groups)

#Get random families from groups to assign to lockbox sample, rest to main sample
test_sample = []
train_sample = []
for f in family_groups_split:
    random_int = randint(0,size_family_group-1) 
    test_sample.append(f[random_int])
    f_rest=np.delete(f, random_int)
    train_sample.append(f_rest)


train_sample = list(np.concatenate(train_sample))

#Compute ratio between main (training) and lockbox (test) sample
size_test_sample = get_sample_size(families_all[:,2],  np.array(test_sample))
size_train_sample = get_sample_size(families_all[:,2], np.array(train_sample))
percentage_test_temp = size_test_sample/(size_train_sample + size_test_sample)


#Assign remaining families to test or train sample depending on train-test ratio
if number_families_rest != 0:
    families_rest = families_sorted[0,-number_families_rest:]
    
    #Assign remaining families to test sample if test ratio is too low
    while (families_rest.size != 0): 
        
        #Add to test data randomly as long as not exeeding given percentage
        random_int = randint(0,families_rest.size-1) 
        
        
        if (percentage_test_temp < percentage_test/100):
        #Assign remaining families to test sample if test ratio is too low
            test_sample.append(families_rest[random_int])
        else:
        #Assign remaining families to train sample if train ratio is too low    
            train_sample.append(families_rest[random_int])
                
        families_rest = np.delete(families_rest,random_int)
        
        size_test_sample = get_sample_size(families_all[:,2],  np.array(test_sample))
        size_train_sample = get_sample_size(families_all[:,2], np.array(train_sample))
        percentage_test_temp = size_test_sample/(size_train_sample + size_test_sample)
  
print('Actual precentage test = ', percentage_test_temp)



#Assign all subjects to train or test samples 
indexes_train = []
for f in train_sample:
    indexes_family_members = np.where(families_all[:,2] == int(f))[0]
    indexes_train.append(indexes_family_members)


families_all_train = families_all[np.concatenate(indexes_train),:]


indexes_test = []
for f in test_sample:
    indexes_family_members = np.where(families_all[:,2] == int(f))[0]
    indexes_test.append(indexes_family_members)


families_all_test = families_all[np.concatenate(indexes_test),:]

column_names = ["Subject","Family_ID","Family_No","g_score", "gf_score", "gc_score", "g_score_family", "gf_score_family","gc_score_family" ,"Age","Gender","Handedness",'FD_mean','Spikes_mean'] 

families_all_test_df = pd.DataFrame(data=families_all_test, columns=column_names)
families_all_train_df = pd.DataFrame(data=families_all_train, columns=column_names)

families_all_test_df.to_csv('families_all_test.csv', index=False)
families_all_train_df.to_csv('families_all_train.csv', index=False)


#Visualizing g-score distributions of test and training sample
variables_to_visualize = ['g_score']
for var in variables_to_visualize:

    x = np.array(families_all_test_df[var])
    mean = np.mean(x)
    std = np.std(x)
    median = np.median(x)
    
    n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(var + '_test')
    plt.ylabel('Frequency')
    title = 'Histogram, mean = {}, std = {}, median = {}'.format(round(mean,2), round(std,2), round(median,2))
    plt.title(title)
    maxfreq = n.max()
    # Set a clean upper y-axis limit
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    
    print( 'Excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x) ))
    print( 'Skewness of normal distribution (should be 0): {}'.format( skew(x) ))
    
for var in variables_to_visualize:

    x = np.array(families_all_train_df[var])
    mean = np.mean(x)
    std = np.std(x)
    median = np.median(x)
    
    n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(var + '_train')
    plt.ylabel('Frequency')
    title = 'Histogram, mean = {}, std = {}, median = {}'.format(round(mean,2), round(std,2), round(median,2))
    plt.title(title)
    maxfreq = n.max()
    # Set a clean upper y-axis limit
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    
    print( 'Excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x) ))
    print( 'Skewness of normal distribution (should be 0): {}'.format( skew(x) ))
