# -*- coding: utf-8 -*-
"""
Created on Aug 23 2022

@author: Jonas A. Thiele
"""

import pandas as pd 
import numpy as np
from scipy import stats


data_behavioral=pd.read_csv("tables_HCP/unrestricted_jonasthiele_11_1_2020_8_39_10.csv") 


#Exclude subjects without all cognitive scores
data_behavioral_analysis=data_behavioral[(data_behavioral['SCPT_Compl'] == True) &
                (data_behavioral['IWRD_Compl'] == True) &
                (data_behavioral['VSPLOT_Compl'] == True & 
                (data_behavioral['PMAT_Compl'] == True)) &
                (data_behavioral['PicVocab_Unadj'].notna()) &
                (data_behavioral['ReadEng_Unadj'].notna()) &
                (data_behavioral['PicSeq_Unadj'].notna()) &
                (data_behavioral['Flanker_Unadj'].notna()) &
                (data_behavioral['CardSort_Unadj'].notna()) &
                (data_behavioral['ProcSpeed_Unadj'].notna()) &
                (data_behavioral['PMAT24_A_CR'].notna()) &
                (data_behavioral['VSPLOT_TC'].notna()) &
                (data_behavioral['IWRD_TOT'].notna()) &
                (data_behavioral['ListSort_Unadj'].notna()) &
                (data_behavioral['DDisc_AUC_200'].notna()) &
                (data_behavioral['SCPT_TP'].notna())
                ]

#Exclude subjects MMSE below threshold
data_behavioral_analysis=data_behavioral_analysis[(data_behavioral_analysis['MMSE_Compl']==True) & (data_behavioral_analysis['MMSE_Score']>=26)]
print(data_behavioral_analysis.shape)


#Calculation of extra cognitive measures from given tests
scores_DDisc_SCPT = data_behavioral_analysis.loc[:, ['SCPT_TP', 'SCPT_TN', 'SCPT_FP','SCPT_FN','SCPT_TPRT','DDisc_AUC_200','DDisc_AUC_40K']]


normalize = lambda x: (x-x.min()) / (x.max()-x.min()) + 1  #Normalize range [1,2] (not [0,1] to avoid division by 0)
scores_DDisc_SCPT['SCPT_TPRT'] = scores_DDisc_SCPT['SCPT_TPRT'].pipe(normalize)

normalize = lambda x: (x-x.min()) / (x.max()-x.min()) #Normalize in range [0, 1]
scores_DDisc_SCPT['SCPT_TP'] = scores_DDisc_SCPT['SCPT_TP'].pipe(normalize)
scores_DDisc_SCPT['SCPT_TN'] = scores_DDisc_SCPT['SCPT_TN'].pipe(normalize)
scores_DDisc_SCPT['SCPT_FP'] = scores_DDisc_SCPT['SCPT_FP'].pipe(normalize)
scores_DDisc_SCPT['SCPT_FN'] = scores_DDisc_SCPT['SCPT_FN'].pipe(normalize)
scores_DDisc_SCPT['DDisc_AUC_200'] = scores_DDisc_SCPT['DDisc_AUC_200'].pipe(normalize)
scores_DDisc_SCPT['DDisc_AUC_40K'] = scores_DDisc_SCPT['DDisc_AUC_40K'].pipe(normalize)


cognitive_scores_analysis = data_behavioral_analysis.loc[:, ['PicVocab_Unadj','ReadEng_Unadj','PicSeq_Unadj','Flanker_Unadj',
                             'CardSort_Unadj','ProcSpeed_Unadj','PMAT24_A_CR','VSPLOT_TC','IWRD_TOT',
                             'ListSort_Unadj']]
cognitive_scores_analysis['SCPT_Eff'] = (scores_DDisc_SCPT['SCPT_TP'] + scores_DDisc_SCPT['SCPT_TN'])/((scores_DDisc_SCPT['SCPT_TP'] + scores_DDisc_SCPT['SCPT_TN'] + scores_DDisc_SCPT['SCPT_FN'] + scores_DDisc_SCPT['SCPT_FP'])*scores_DDisc_SCPT['SCPT_TPRT'])
cognitive_scores_analysis['DDisc'] = scores_DDisc_SCPT['DDisc_AUC_200'] + scores_DDisc_SCPT['DDisc_AUC_40K']


standardize = lambda x: (x-x.mean()) / x.std() 
cognitive_scores_analysis = cognitive_scores_analysis.pipe(standardize)
cognitive_scores_analysis = cognitive_scores_analysis.reset_index(drop=True)
cognitive_scores_analysis.to_csv('CogScores_12_1186subjects.csv', index=False)
cognitive_scores_analysis.insert(0, "Subject", data_behavioral_analysis['Subject'].reset_index(drop=True))
cognitive_scores_analysis.to_csv('CogScores_12_1186subjects_index.csv', index=False)

APM_scores = data_behavioral_analysis.loc[:,['Subject','PMAT24_A_CR']]
gf_scores = data_behavioral_analysis.loc[:,['Subject','CogFluidComp_Unadj']]
gc_scores = data_behavioral_analysis.loc[:,['Subject','CogCrystalComp_Unadj']]

APM_scores.to_csv('APM_1186subjects_index.csv', index=False)
gf_scores.to_csv('gf_hcp_1186subjects_index.csv', index=False)
gc_scores.to_csv('gc_hcp_1186subjects_index.csv', index=False)




#Exclude subjects with missing performance scores
data_behavioral_analysis=data_behavioral_analysis[
                (data_behavioral_analysis['Emotion_Task_Acc'].notna()) &
                (data_behavioral_analysis['Emotion_Task_Median_RT'].notna()) &
                (( (data_behavioral_analysis['Gambling_Task_Median_RT_Larger'].notna()) | (data_behavioral_analysis['Gambling_Task_Perc_Larger'] == 0))) &
                (( (data_behavioral_analysis['Gambling_Task_Median_RT_Smaller'].notna()) | (data_behavioral_analysis['Gambling_Task_Perc_Smaller'] == 0))) &
                (data_behavioral_analysis['Language_Task_Acc'].notna()) &
                (data_behavioral_analysis['Language_Task_Median_RT'].notna()) &
                (data_behavioral_analysis['Relational_Task_Acc'].notna()) &
                (data_behavioral_analysis['Relational_Task_Median_RT'].notna()) &
                (( (data_behavioral_analysis['Social_Task_Median_RT_Random'].notna()) | (data_behavioral_analysis['Social_Task_Perc_Random'] == 0)) ) &
                (( (data_behavioral_analysis['Social_Task_Median_RT_TOM'].notna()) | (data_behavioral_analysis['Social_Task_Perc_TOM'] == 0)) ) &
                (data_behavioral_analysis['WM_Task_2bk_Acc'].notna()) &
                (data_behavioral_analysis['WM_Task_2bk_Median_RT'].notna()) &
                (data_behavioral_analysis['WM_Task_0bk_Acc'].notna()) &
                (data_behavioral_analysis['WM_Task_0bk_Median_RT'].notna())
                ]



#Compute performance scores as per Greene et al 2020 https://doi.org/10.1016/j.celrep.2020.108066
#acc = accuracy, RT = Reaction time 
emotion_acc = data_behavioral_analysis['Emotion_Task_Acc'].reset_index(drop=True)

gambling_RT_larger = data_behavioral_analysis['Gambling_Task_Median_RT_Larger'].reset_index(drop=True)
gambling_RT_smaller = data_behavioral_analysis['Gambling_Task_Median_RT_Smaller'].reset_index(drop=True)
gambling_mean = np.nanmean(np.array([gambling_RT_larger.to_numpy(), gambling_RT_smaller.to_numpy()]), axis=0)

normalize = lambda x: (x-x.min()) / (x.max()-x.min())
language_story_acc = data_behavioral_analysis['Language_Task_Story_Acc'].reset_index(drop=True).pipe(normalize)
language_math_acc = data_behavioral_analysis['Language_Task_Math_Acc'].reset_index(drop=True).pipe(normalize)
language_story_difficulty = data_behavioral_analysis['Language_Task_Story_Avg_Difficulty_Level'].reset_index(drop=True).pipe(normalize)
language_math_difficulty = data_behavioral_analysis['Language_Task_Math_Avg_Difficulty_Level'].reset_index(drop=True).pipe(normalize)
language_mean = np.nanmean(np.array([language_story_acc.to_numpy(), language_math_acc.to_numpy(), language_story_difficulty.to_numpy(), language_math_difficulty.to_numpy()]), axis=0)

relational_acc = data_behavioral_analysis['Relational_Task_Acc'].reset_index(drop=True)

social_random_perc = data_behavioral_analysis['Social_Task_Random_Perc_Random'].reset_index(drop=True)
social_tom_perc = data_behavioral_analysis['Social_Task_TOM_Perc_TOM'].reset_index(drop=True)
social_perc_mean =  np.nanmean(np.array([social_random_perc.to_numpy(), social_tom_perc.to_numpy()]), axis = 0)

wm_acc = data_behavioral_analysis['WM_Task_Acc'].reset_index(drop=True)

performance_scores_analysis = np.array([emotion_acc, gambling_mean, language_mean, relational_acc, social_perc_mean, wm_acc])
performance_scores_analysis = stats.zscore(performance_scores_analysis, axis = 1)
performance_scores_analysis = np.transpose(performance_scores_analysis)

column_names_performance_scores = ['Emotion', 'Gambling', 'Language','Relational','Social','WM']
performance_scores_analysis_df = pd.DataFrame(performance_scores_analysis, columns = column_names_performance_scores)
performance_scores_analysis_df.insert(0, "Subject", data_behavioral_analysis['Subject'].reset_index(drop=True))

performance_scores_analysis_df.to_csv('PerfScores_6_1011subjects_index.csv', index=False)
cognitive_scores_analysis_df = cognitive_scores_analysis[cognitive_scores_analysis['Subject'].isin(performance_scores_analysis_df['Subject'])]
cognitive_scores_analysis_df.to_csv('CogScores_12_1011subjects_index.csv', index=False)
    
#Exclude subjects with missing  fMRI data
data_behavioral_analysis=data_behavioral_analysis[(data_behavioral_analysis['3T_Full_Task_fMRI']==True) &
                (data_behavioral_analysis['3T_RS-fMRI_PctCompl']==100.0)
                ] #Only keep subjects with full fMRI data

data_behavioral_analysis=data_behavioral_analysis[data_behavioral_analysis['Subject']!=668361]  #tfRMI_WM_RL missing
 
              
#Definition of fMRI_sets that we want to analyze
fMRI_sets=["rfMRI_REST1_RL","rfMRI_REST1_LR","rfMRI_REST2_RL","rfMRI_REST2_LR", "tfMRI_WM_RL","tfMRI_WM_LR",
          "tfMRI_GAMBLING_RL","tfMRI_GAMBLING_LR", "tfMRI_MOTOR_RL","tfMRI_MOTOR_LR", 
          "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_SOCIAL_RL","tfMRI_SOCIAL_LR",
          "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR", "tfMRI_EMOTION_RL","tfMRI_EMOTION_LR"
          ]

#Thresholds for high motion
thresh_small_spike = 0.25
thresh_mean_FD = 0.2
thresh_large_spike = 5
thresh_percent = 0.2

#Path of motion data
path = "Movement_RelativeRMS_files"

columns = int(len(fMRI_sets*3))
data_RMS_all_subjects = np.array([], dtype=np.int64).reshape(0,columns+1)
high_motion_all_subjects = np.array([], dtype=np.int64).reshape(0,columns)
for subject in data_behavioral_analysis['Subject']:
    
    data_RMS_subject=np.array([])
    high_motion_subject=np.array([])
    for task in fMRI_sets:
        
        #Path of the RMS file for the specific subject for the specific task
        path_RMS_data=path+"/{}".format(subject)+"/MNINonLinear/Results"+"/{}".format(task)
        
        #Open, read the .txt file and save it in an array 
        file = open(path_RMS_data+'/Movement_RelativeRMS.txt')
        relative_RMS_text = file.read()
        relative_RMS = relative_RMS_text.splitlines()
        relative_RMS = np.array(relative_RMS).astype(np.float64)
        
        #Compute relevant RMS measures
        numberTimeSteps = len(relative_RMS)
        mean_FD = np.mean(relative_RMS)
        number_small_spikes = (relative_RMS>thresh_small_spike).sum()
        number_large_spikes = (relative_RMS>thresh_large_spike).sum()
        perc_small_spikes = number_small_spikes/numberTimeSteps
        
        #Save RMS values extracted from RMS file in data_RMS_subject array
        data_RMS_subject = np.append(data_RMS_subject, np.array((mean_FD,number_small_spikes,number_large_spikes)))
        
        #Check if values are above threshold and save results in high_motion_subject array
        high_motion_subject = np.append(high_motion_subject, np.array(mean_FD>=thresh_mean_FD))
        high_motion_subject = np.append(high_motion_subject, np.array(perc_small_spikes>=thresh_percent))
        high_motion_subject = np.append(high_motion_subject, np.array(number_large_spikes>0))
        
        
    
    data_RMS_subject = np.insert(data_RMS_subject,0,subject) #Add ID of subject into array
    data_RMS_all_subjects = np.vstack([data_RMS_all_subjects,data_RMS_subject])
    high_motion_all_subjects = np.vstack([high_motion_all_subjects,high_motion_subject])  
   
indexes_mean_FD = np.append(0,np.arange(1,np.shape(data_RMS_all_subjects)[1]-1,3))
mean_FD_all_subjects = data_RMS_all_subjects[:, indexes_mean_FD]
namesColumns = fMRI_sets.copy()
namesColumns.insert(0,'Subjects')
mean_FD_all_subjects=pd.DataFrame(mean_FD_all_subjects, columns = namesColumns)
mean_FD_all_subjects.to_csv('dataMeanFD_allfMRI_sets_941subjects.csv', index=False)

indexes_small_spikes = np.append(0,np.arange(2,np.shape(data_RMS_all_subjects)[1]-1,3))
small_spikes_all_subjects = data_RMS_all_subjects[:, indexes_small_spikes]
small_spikes_all_subjects = pd.DataFrame(small_spikes_all_subjects, columns = namesColumns)
small_spikes_all_subjects.to_csv('dataSmallSpikes_allfMRI_sets_941subjects.csv', index=False)
    
#High motion exclusion    
data_behavioral_analysis = data_behavioral_analysis[((high_motion_all_subjects==1).any(axis=1)) == False]

number_subjects_analysis = data_behavioral_analysis.shape[0]
print('\nNumber of Subjects: {}\n\n'.format(number_subjects_analysis))

#Save behavioral data of remaining sample
data_behavioral_analysis.to_csv('behavioral_data_analysis.csv', index=False)


#Info: Subjects [104012 146129 287248 512835 660951 662551 825048] WM performance scores not complete
               