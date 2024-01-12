# Predicting Human Cognition

## 1. Scope
The repository contains scripts for the analyses used in the paper **"Can machine learning-based predictive modelling improve our understanding of human cognition?"** coauthored by Jonas A. Thiele, Joshua Faskowitz, Olaf Sporns, and Kirsten Hilger. Herein, intelligence is predicted from functional brain connectivity with different selections of brain links. If you have questions or trouble with the scripts, feel free to contact me: jonas.thiele@uni-wuerzburg.de
## 2. Data
We used data provided by the Human Connectome Project HCP (1), funded by the National Institute of Health for our main sample analysis. Data from the Amsterdam Open MRI Collection (2) were used for the replication analyses (PIOP1 and PIOP2 sample).
All data used in the current study can be accessed online under: https://www.humanconnectome.org/study/hcp-young-adult (HCP), https://doi.org/10.18112/openneuro.ds002785.v2.0.0 (AOMIC-PIOP1), and https://doi.org/10.18112/openneuro.ds002790.v2.0.0 (AOMIC-PIOP2).
## 3. Preprocessing
We used the minimally preprocessed HCP fMRI data (3) and implemented further preprocessing comprising a nuisance regression strategy with 24 head motion parameters, eight mean signals from white matter and cerebrospinal fluid, and four global signals (4). For task data, basis-set task regressors (5) were used with the nuisance regressors to remove mean task-evoked activations.
Code for the further preprocessing steps is available here: https://github.com/faskowit/app-fmri-2-mat.
For the replication, the data of the Amsterdam Open MRI Collection was downloaded in the minimally preprocessed (using fMRIPrep version 1.4.1, ref. 6) form and all further preprocessing followed the same regression steps as specified for the main sample. For all data, timeseries of neural activation were extracted from 100 nodes covering the entire cortex (7) that were assigned to the seven Yeo canonical systems (8).
## 4. Structure and script description
### Preparation 
`preparation`

1.	`exclude_subjects` - Excludes subjects with missing cognitive scores, performance scores, fMRI data and excessive head motion; extracts cognitive performance scores and motion data
  
  
2.	`get_intelligence_factors` - Computes general (g), crystallized (gC), and fluid intelligence (gF) scores from cognitive scores
  
  
3.	`split_sample` - Splits HCP sample in a main and lockbox sample
  
 
4.	`get_FC_100nodes` - Computes functional connectivity (FC) matrices from resting state, tasks and latent FCs for HCP sample


5.	`get_FC_100nodes_repli` - Computes functional connectivity (FC) matrices from resting state, tasks and latent FCs for replication samples

  
6.	`get_nodes` - Gets nodes related to intelligence theories; necessary input files "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm" (retrieved from: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI), and "Multiple demand functional masks" (retrieved from: https://osf.io/pdtk9/) can be found in the folder


Subject IDs for each sample can be found in the subfolder `subject_ids`

### Model training
`model_training`

1.	`train_models_nw` - Training of prediction models in the main sample (610 subjects of the HCP) with **n**et**w**ork-specific link selections for internal and lockbox validation 


2.	`train_models_nw_806`  - Training of prediction models in the HCP sample (**806** subjects) with **n**et**w**ork-specific link selections for later application to the replication sample 


3.	`train_models_rnrlit` - Training of prediction models in the main sample (610 subjects of the HCP) with links between **r**andom **n**odes, **r**andom **l**inks, or links between nodes of **i**ntelligence **t**heories


4.	`train_models_lrp` - Training of prediction models in the main sample (610 subjects of the HCP) with the most relevant links (relevance estimated by **l**ayerwise **r**elevance **p**ropagation - LRP)


### Model evaluation
`model_evaluation`

1.	`get_results_nw` - Get performances and error measures from models trained in the main sample with **n**et**w**ork-specific links

  
2. `validation_nw_lockbox` - Apply models trained in the main sample with **n**et**w**ork-specific links to the lockbox sample


3. `validation_nw_repli` - Apply models trained in the main sample with **n**et**w**ork-specific links to the replication sample


4. `get_results_rnrlit` - Get performances from models trained in the main sample with links between **r**andom **n**odes, **r**andom **l**inks, or links between nodes of **i**ntelligence **t**heories.
  
  
5. `validation_rnrlit_lockbox` - Apply models trained in the main sample with links between **r**andom **n**odes, **r**andom **l**inks, or links between nodes of **i**ntelligence **t**heories to the lockbox sample

   
6.	`validation_rnrlit_repli` - Apply models trained in the main sample with links between **r**andom **n**odes, **r**andom **l**inks, or links between nodes of **i**ntelligence **t**heories to the replication sample


7.	`get_results_lrp` - Get performances from models trained in the main sample with the most relevant links (estimated by **LRP**) and masks with the most relevant links


8.	`validation_lrp_lockbox` - Apply models trained in the main sample with the most relevant links (estimated by **LRP**) to the lockbox sample


9.	`validation_lrp_repli` - Apply models trained in the main sample with the most relevant links (estimated by **LRP**) to the replication sample

### Results analyses and visualization
`results_analyses_and_visualization`

1.	`descriptives` - Compute descriptives of samples

  
2.	`plot_results_nw` - Visualize results of models trained with **n**et**w**ork-specific links

   
4.	`compare_results_nw` - Compare results of models trained with **n**et**w**ork-specific links between intelligence components, cognitive states and functional brain networks

   
6.	`plot_results_rnrl` - Visualize results of models trained with links between **r**andom **n**odes vs. with **r**andom **l**inks vs. with the most relevant links vs. with all links

   
8.	`plot_results_it` - Visualize results of models trained with links between nodes of **i**ntelligence **t**heories

   
10.	`plot_results_lrp` - Visualize results of models trained with the most relevant links (estimated by **LRP**)

    
12.	`plot_connectograms_lrp` - Visualize the most relevant links (estimated by **LRP**)

### Post hoc
`post_hoc`

1.	`get_edge_ICC` - Compute link-wise (edge-wise) test-retest reliability

   
2.	`get_partiCoeff_moduleZscore` - Compute participation coefficients and within-module degree z-scores

   
3.	`get_properties_relevant_edges` - Test differences between the most relevant links and random links


### External functions 

External functions used in the scripts can be found in the `external_functions` folder. The functions can be found elsewhere but are included here for convenience: For example, the functions `agreement`, `community_louvain`, `consensus_und`, `module_degree_zscore`, and `participation_coef_sign` are part of the Brain Connectivity Toolbox (9) retrieved from: https://sites.google.com/site/bctnet/. Comments on the authorship and licenses of other functions are provided within the folder.

### Demo
The `demo` folder includes minimal examples for running intelligence prediction with different brain link selections, and scripts for visualizing results found in the manuscript. See `ReadMe` within the folder for details on how to setup and how to run the scripts. 

## 5. Software requirements
-	Matlab version 2021a
-	R version 4.0.2
- Python version 3.8.18, see `python_requirements.txt` file and `ReadMe` within the `demo` folder (setup time ~ 10 minutes)

## References
1.	D. C. Van Essen, et al., The WU-Minn Human Connectome Project: An overview. Neuroimage 80, 62–79 (2013).
2.	L. Snoek, et al., The Amsterdam Open MRI Collection, a set of multimodal MRI datasets for individual difference analyses. Sci. Data 8, 85 (2021).
3.	M. F. Glasser, et al., The minimal preprocessing pipelines for the Human Connectome Project. Neuroimage 80, 105–124 (2013).
4.	L. Parkes, et al., An evaluation of the efficacy, reliability, and sensitivity of motion correction strategies for resting-state functional MRI. Neuroimage 171, 415–436 (2018).
5.	M. W. Cole, et al., Task activations produce spurious but systematic inflation of task functional connectivity estimates. Neuroimage 189, 1–18 (2019).
6.	O. Esteban, et al., fMRIPrep: a robust preprocessing pipeline for functional MRI. Nat. Methods 16, 111–116 (2019).
7.	A. Schaefer, et al., Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cereb. Cortex 28, 3095–3114 (2018).
8.  T. B. T. Yeo, et al., The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J. Neurophysiol. 106, 1125–1165 (2011).
9.  M. Rubinov et al., Complex network measures of brain connectivity: Uses and interpretations. Neuroimage 52, 1059-1069 (2010). 

## Copyright
Copyright (cc) 2023 by Jonas A. Thiele

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Files of `predicting_human_cognition`</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/jonasAthiele/predicting_human_cognition" property="cc:attributionName" rel="cc:attributionURL">Jonas A. Thiele</a> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

Note that external functions may have other licenses.
