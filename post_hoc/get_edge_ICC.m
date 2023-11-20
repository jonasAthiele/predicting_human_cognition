
%%% Scope:  Reading preprocessed BOLD-signals, calculating FCs,
%%%         calculating ICC between RL-phase and LR-phase scans
%%% Author: Jonas A. Thiele
%%% Date:   18.07.2023

%%% Uses function:  Arash Salarian (2023). 
%%%                 Intraclass Correlation Coefficient (ICC) 
%%%                 (https://www.mathworks.com/matlabcentral/fileexchange/22099-intraclass-correlation-coefficient-icc),
%%%                 MATLAB Central File Exchange. Retrieved July 18, 2023. 
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


nEdges = sum(sum((triu(ones(nNodes),1)==1))); %Number edges in upper triangle
ICC_vals = zeros(nNodes,nNodes,nStates); %Store ICC values
for sc = 1:nStates
    
    ind_state = find(mask_states == sc); %Indices of FCs belonging to respective state
    FC_state = FCStatic(:,:,ind_state,:); %FCs of all scans of respective state
    
    %Get upper triangle of FCs of respective states
    M_ICC_alledges = zeros(nSubjects,length(ind_state),nEdges);
    for n =1:length(ind_state)   
        for s = 1:nSubjects
            
            FC_s = squeeze(FC_state(:,:,n,s));
            FC_s_triu = FC_s(triu(ones(100,100),1)==1);
            M_ICC_alledges(s,n,:) = FC_s_triu;
            
        end     
    end
    
    
    ICC_r = zeros(nEdges,1);
    for e = 1:nEdges %Loop over edges
        
        M_ICC = squeeze(M_ICC_alledges(:,:,e)); %FC value of an edge of all subjects and scans of a state
        [r, LB, UB, F, df1, df2, p] = ICC(M_ICC, 'C-1'); %Compute ICC of edge
        ICC_r(e) = r; %ICC of an edge
        
    end
    
    
    %Bring ICC values back to matrix form
    ICC_sc = zeros(nNodes);
    ICC_sc(triu(ones(100,100),1)==1) = ICC_r;
    ICC_vals(:,:,sc) = ICC_sc; %Store ICC values of state
     
end

%Save ICC values
save('ICC_HCP_610_100nodes.mat', 'ICC_vals');