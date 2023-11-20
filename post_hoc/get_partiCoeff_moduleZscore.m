 

%%% Scope:  Calculating participation coefficient and within-module degree z-score 
%%% Author: Jonas A. Thiele
%%% Date:   18.07.2023

%%% Uses functions from the Brain Connectivity Toolbox from:
%%% Complex network measures of brain connectivity: Uses and interpretations
%%% Rubinov M, Sporns O (2010) NeuroImage 52:1059-69
%%% Retrieved from: https://sites.google.com/site/bctnet/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load FC_HCP_610_100nodes %Load FC

FC = FCStatic_combined; %Rename FC
clear FCStatic_combined


[N,~,S,n_states] = size(FC); %N-number nodes, S-number subjects, n_states-number states

R = 100; %Runs of Louvain

%Parameters for calculating consensus between runs of Louvain
tau = 0.1; %Threshold which controls the resolution of the reclustering
reps = 10; %Number of times that the clustering algorithm is reapplied

gam = 3; G = length(gam); %Gamma influences number of clusters/communities (higher gamma = more clusters)

%Loop over states
for state =1:n_states

    parfor s=1:S %Loop over subjects
        
        disp(num2str(state)); %Display state
        disp(num2str(s)); %Display actual subject working on 

            FCi = squeeze(FC(:,:,s,state)); %FC of subject and state 
            
            for g=1:G %Loop over range of gamma

                gamma = gam(g); %Resolution parameter gamma

                
                qmax = -10; %Modularity
                ciall = zeros(N,R); %Community affiliation vectors
                for r=1:R %Loop over runs (of Louvain)
                    
                    %Compute community affiliation and modularity
                    [ci, q] = community_louvain(FCi,gamma,[],'negative_asym');
                    
                    %Save community affiliation CI vector
                    %that maximizes modularity Q
                    if(q>qmax)
                        qmax = q;
                        CI(:,s,g) = ci; %CI of max Q
                        Q(s,g,state) = q; %Max Q
                    end
                    
                    ciall(:,r) = ci; %CIs of all runs

                end
                
                %Consensus community affiliation over all Louvain runs
                CI2(:,s,g,state) = consensus_und(agreement(ciall),tau,reps); 
                

            end

    end
    
    
end

%Save modularity and consensus community affiliation
save('Modularity_HCP_610_100nodes.mat', 'Q','CI2');



%Participation coefficient
for state =1:n_states %Loop over states
    parfor s=1:S %Loop over subjects
       
        W = squeeze(FC(:,:,s,state)); %FC
        C = squeeze(CI2(:,s,:,state)); %Community affiliation
        
        %Participation coefficient from positive and negative weights separately 
        [Ppos(:,s,state), Pneg(:,s,state)] = participation_coef_sign(W,C);
        
    end
end


%Save participation coefficients
save('Parti_HCP_610_100nodes.mat', 'Ppos','Pneg');


%Within-module degree z-score
for state =1:n_states %Loop over states
    parfor s=1:S %Loop over subjects 
       
        W = squeeze(FC(:,:,s,state)) %FC
        
        %Normalize FC
        W_min = min(min(W))
        W_max = max(max(W))
        W_nrm = (W - W_min) / (W_max - W_min)
        W_nrm = W_nrm.*~eye(size(W_nrm)) %Set diagonale zero
        
        C = squeeze(CI2(:,s,:,state)); %Community affiliation
        
        %Within-module degree z-score
        Z(:,s,state) = module_degree_zscore(W_nrm,C);
        
    end
end

%Save within-module degree z-score
save('ModuleZ_HCP_610_100nodes.mat', 'Z');