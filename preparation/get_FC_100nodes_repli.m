
%%% Scope:  Reading preprocessed BOLD-signals, calculating FCs
%%%         (functional connectivity), joining FCs
%%% Author: Jonas A. Thiele
%%% Date:   24.06.2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Choose sample
name_sample = 'PIOP2'; % 'PIOP1' , 'PIOP2'

if strcmp(name_sample,'PIOP1') 
    
    %Read behavioral data of subjects
    behavioralData = readtable('data_beh_sel_subjects_PIOP1.csv');
    
    %List of scans to be considered
    scans = {'task-restingstate_acq-mb3','task-workingmemory_acq-seq',...
            'task-anticipation_acq-seq','task-emomatching_acq-seq',...
            'task-faces_acq-mb3','task-gstroop_acq-seq'};
        
elseif strcmp(name_sample,'PIOP2') 
    
    %Read behavioral data of subjects
    behavioralData = readtable('data_beh_sel_subjects_PIOP2.csv');
    
    %List of scans to be considered
    scans = {'task-restingstate_acq-seq','task-workingmemory_acq-seq',...
            'task-emomatching_acq-seq','task-stopsignal_acq-seq'};
    
end
    

subjects = behavioralData.Subject; %Subject IDs
nNodes = 100;
background_regions = [1,52];



filename_fMRI = 'out_schaefer100-yeo17_timeseries.hdf5';

nStates = length(scans); %Number of scan conditions (fMRI states) after joining


nSubjects = length(subjects);


for sc = 1:nStates
    
    for s=1:nSubjects %Loop over subjects

        idSubject = cell2mat(subjects(s));
        
        %Read fMRI data
        if strcmp(name_sample,'PIOP1')
            
            filepath = fullfile('fmrip2mat_36pNS_PIOP1', idSubject, append(cell2mat(scans(sc)),'_out'),'output_makemat',filename_fMRI); 
            
        elseif strcmp(name_sample,'PIOP2')

            filepath = fullfile('fmrip2mat_36pNS_PIOP2', idSubject, append(cell2mat(scans(sc)),'_out'),'output_makemat',filename_fMRI); 
             
            
        end
            
        file = h5read(filepath, '/timeseries');
      
        bold = squeeze(file); %Region x neural activity

        %Remove background nodes 
        bold(background_regions,:)=[];
        bold=bold(1:nNodes,:); %Take cortical nodes only

        FCi = corr(bold'); %Correlating node signals

        FCi = (FCi+FCi')./2; %Symmetrize matrix
        FCi(1:size(FCi,1)+1:end) = 0; %Set diagonal elements to zero
        FCi = fisherZTransform(FCi); %Fisher Z transform all correlations

        FCStatic_combined(:,:,s,sc) = FCi; %Named as combined for applicability on codes from HCP sample

    end
    
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

if strcmp(name_sample,'PIOP1')
    save('FC_PIOP1_100nodes.mat', 'FCStatic_combined');
    save('FC_PIOP1_subjects.mat', 'subjects');
elseif strcmp(name_sample,'PIOP2')
    save('FC_PIOP2_100nodes.mat', 'FCStatic_combined');
    save('FC_PIOP2_subjects.mat', 'subjects');
end