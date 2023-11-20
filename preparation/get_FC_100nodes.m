
%%% Scope:  Reading preprocessed BOLD-signals, calculating FCs
%%%         (functional connectivity)
%%% Author: Jonas A. Thiele
%%% Date:   24.06.2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Read behavioral data of subjects
behavioralData = readtable('families_all_train.csv');
subjects = behavioralData.Subject; %Subject IDs
folderTask = 'taskregress_36pNS_schaefer100-yeo17';
folderRest = 'regress_36pNS_schaefer100';
filename_nii = 'schaefer100-yeo17.ptseries.nii';
nNodes = 100;
nSubjects = length(subjects);

scans = {'rfMRI_REST1_RL','rfMRI_REST1_LR','rfMRI_REST2_RL','rfMRI_REST2_LR','tfMRI_WM_RL','tfMRI_WM_LR',...
   'tfMRI_GAMBLING_RL', 'tfMRI_GAMBLING_LR', 'tfMRI_MOTOR_RL', 'tfMRI_MOTOR_LR', 'tfMRI_LANGUAGE_RL',...
   'tfMRI_LANGUAGE_LR', 'tfMRI_SOCIAL_RL', 'tfMRI_SOCIAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_RELATIONAL_LR',...
   'tfMRI_EMOTION_RL', 'tFMRI_EMOTION_LR'};

%Mask for joining scans 
mask_states = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]; %Which tasks belong together 4x rest, 2 per task.
nStates = max(mask_states); %Number of scan conditions (fMRI states) after joining

inds_rest = 1:4; %Indices of resting state

for sc = 1:length(scans) %Loop over all scans
    
    
    %Choose folder in which files are stored - here different for rest and
    %task data
    if ismember(sc, inds_rest)
       folderData = folderRest;
    else 
       folderData = folderTask; 
    end
 
    for s=1:nSubjects %Loop over subjects

        idSubject = subjects(s);
      

        filepath = fullfile(folderData, num2str(idSubject), string(scans(sc)), filename_nii);

        file = niftiread(filepath);
        bold = squeeze(file); %Neural activity time x region

        bold_b=bold(:,1:nNodes); %Select cortical nodes only

        FCi = corr(bold_b); %Corellating time series of nodes

        FCi = (FCi+FCi')./2; %Symmetrize matrix
        FCi(1:size(FCi,1)+1:end) = 0; %Set diagonal elements to zero
        FCi = fisherZTransform(FCi); %Fisher z-transform all correlations

        FCStatic(:,:,sc,s) = FCi;
            

    end
    
end

for sc = 1:nStates
      
    FCStatic_combined(:,:,:,sc) = squeeze(mean(FCStatic(:,:,mask_states==sc,:),3));
                 
end


%Latent FC all states
FC_latent_all_states = zeros(nNodes,nNodes,nSubjects);
lambdas = zeros(nNodes,nNodes,nStates);
for i=1:nNodes
   for j = 1:nNodes
      
      if i ~= j
        
        FC = squeeze(FCStatic_combined(i,j,:,:));
        %FC = rescale(FC,'InputMin',min(FC),'InputMax',max(FC));
        [lambda,psi,T,stats,factor] = factoran(double(FC),1,'Rotate','none'); 
        FC_latent_all_states(i,j,:) = sum(lambda.*double(FC)');
        lambdas(i,j,:) = lambda;

      end
  
    
   end
end

FCStatic_combined(:,:,:,nStates+1) = FC_latent_all_states;

%Latent FC task states
FC_latent_task = zeros(nNodes,nNodes,nSubjects);
lambdas = zeros(nNodes,nNodes,nStates-1);
FCStatic_combined_tasks = FCStatic_combined(:,:,:,2:nStates);
for i=1:nNodes
   for j = 1:nNodes
      
      if i ~= j
        
        FC = squeeze(FCStatic_combined_tasks(i,j,:,:));
        %FC = rescale(FC,'InputMin',min(FC),'InputMax',max(FC));
        [lambda,psi,T,stats,factor] = factoran(double(FC),1,'Rotate','none'); 
        FC_latent_task(i,j,:) = sum(lambda.*double(FC)');
        lambdas(i,j,:) = lambda;

      end
  
    
   end
end

FCStatic_combined(:,:,:,nStates+2) = FC_latent_task;

save('FC_HCP_100nodes.mat', 'FCStatic_combined');
save('FC_HCP_subjects.mat', 'subjects');