%% Nodes intelligence theories

% Scope: Determining nodes corresponding to intelligence theories
% Date: 10.03.2023
% Author: Jonas A. Thiele

%% Multidemand (MD) Diacheck

schaefer_atlas = niftiread('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
multidemand_atlas = niftiread('Multiple demand functional masks.nii'); %Mask from Diachek et al.: https://doi.org/10.1523/JNEUROSCI.2036-19.2020

ind_x = find(multidemand_atlas(:) ~= 0); %Find voxels of mask
schaefer_flat = schaefer_atlas(:);

%Find nodes of Schaefer in MD mask   
nodes_sch_md = schaefer_flat(ind_x); %Schaefer nodes within MD mask
node_multidemand = unique(nodes_sch_md); %Unique Schaefer nodes within MD mask

[GC_schaefer,GR_schaefer] = groupcounts(schaefer_flat); %Total number voxels of each node
[GC_multi,GR_multi] = groupcounts(nodes_sch_md); %Number of voxels of each node within MD mask

%Determine the proportion of voxels of each Schaefer node located within MD
%mask
for n = 1:length(node_multidemand)
    
    node = node_multidemand(n);
    
    idx = find(GR_schaefer == node);
    region_percent(n) =  GC_multi(n)/GC_schaefer(idx);

end

idx_region = find(region_percent>0.4); %Only take nodes that have more than 40% of their voxels within the MD mask 
nodes = node_multidemand(idx_region);
nodes = [nodes; 72; 23]'; %Add insula manually as it would not be included otherwise

save('nodes_md_diacheck', 'nodes')

%Take Schaefer mask and set voxels of nodes that are not in MD mask to zero
schaefer_multidemand = schaefer_atlas; 
for el = 1:100
    
    if ~ismember(el,nodes)
        
        schaefer_multidemand(schaefer_multidemand == el) = 0;
        
    end
end

%Save MD mask mapped to Schaefer
info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_multidemand,'schaefer_multidemand_diachek.nii',info)


%Evaluate overlap (voxel-level) between MD mask and the mapping to Schaefer
overlap = sum( (schaefer_multidemand(:) > 0) & (multidemand_atlas(:) > 0));

at_schaf = schaefer_multidemand(:);
at_schaf(at_schaf>0) = 1;
at_md = multidemand_atlas(:);
at_md(at_md>0) = 1;

nr_hit_voxels = sum((at_md == at_schaf) & (at_md == 1));
nr_missed_voxels = sum((at_md ~= at_schaf) & (at_md == 1));
nr_additional_voxels = sum((at_md ~= at_schaf) & (at_schaf == 1));


%% Multidemand Duncan
schaefer_atlas = niftiread('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');

%Peak coordinates MD Duncan 2010
IPS_right = [37, -56, 41]; 
IFS_right = [41, 23, 29]; 
AIFO_right = [35, 18, 2]; 
PFC_right = [21, 43, -10];  
ACC_right = [3, 31, 24]; %x coordinate changed to be slighlty positive
pre_SMA_right = [3, 18, 50]; %x coordinate changed to be slighlty positive


coords_MD = [IPS_right; IFS_right; AIFO_right;...
    PFC_right; ACC_right; pre_SMA_right];

%Matrix for transforming coordinates  
T = [-2,  0,  0,   90;...
      0,  2,  0, -126;...
      0,  0,  2,  -72;...
      0,  0,  0,    1];

%Transforming coordinates to match Schaefer in .nii file and find
%corresponding nodes
shell = 1; 
shell_arr = -shell:1:shell;
[x,y,z] = ndgrid(shell_arr,shell_arr,shell_arr);
combs = [x(:),y(:),z(:)]; %Searchgrid
for n = 1:length(coords_MD)
    
    x_mni = coords_MD(n,1); 
    y_mni = coords_MD(n,2);
    z_mni = coords_MD(n,3); 
    
    n_coords = 1;
    
    coordinate = [x_mni y_mni z_mni ones(n_coords ,1)]*(inv(T))';
    coordinate(:,4) = [];
    coordinate = round(coordinate);
    
    
    x_voxel = coordinate(1);
    y_voxel = coordinate(2);
    z_voxel = coordinate(3);
    
    %Add a searchgrid to find all nodes close to coordinates
    for c = 1:length(combs)
        
        x_off = combs(c,1);
        y_off = combs(c,2);
        z_off = combs(c,3);
        nodes_MD(n,c) = schaefer_atlas(x_voxel + x_off, y_voxel+y_off, z_voxel+z_off);
        %Collect all nodes close to coordinate
    
    end
    
    
end

potential_nodes = unique(nodes_MD);

%Select nodes manually and add corresponding left hemispheric regions
nodes_MD = [80, 31, 32, 81, 72, 23, 86, 46, 47, 77, 27, 74, 25];

% 80 - 17Networks_RH_ContA_IPS_1
% 31 - 17Networks_LH_ContA_IPS_1
% 32 - 17Networks_LH_ContA_PFCl_1
% 81 - 17Networks_RH_ContA_PFCl_1
% 72 - 17Networks_RH_SalVentAttnA_Ins_1
% 23 - 17Networks_LH_SalVentAttnA_Ins_2
% 86 - 17Networks_RH_ContB_PFClv_1
% 46 - 17Networks_LH_DefaultB_PFCv_1
% 47 - 17Networks_LH_DefaultB_PFCv_2
% 77 - 17Networks_RH_SalVentAttnB_PFCmp_1
% 27 - 17Networks_LH_SalVentAttnB_PFCmp_1
% 74 - 17Networks_RH_SalVentAttnA_FrMed_1
% 25 - 17Networks_LH_SalVentAttnA_FrMed_1

nodes = nodes_MD;
save('nodes_md_duncan', 'nodes')

schaefer_multidemand = schaefer_atlas;

for el = 1:100
    
    if sum(el == nodes_MD) == 0
        
        schaefer_multidemand(schaefer_multidemand == el) = 0;
        
    end
end


info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_multidemand,'schaefer_multidemand_duncan.nii',info)


%% PFIT

schaefer_atlas = niftiread('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');

%Coordinates from Basten et al. 2015
x = [50, -42,  24, 2,  -10, -26,  10,  50];
y = [24,  12 , 6,  18,   6, -60, -64, -70];
z = [28,  28,  54, 50,  46,  54,  54,   4];

coords_PFIT = [x',y',z'];

%Transformation matrix 
T = [-2,  0,  0,   90;...
      0,  2,  0, -126;...
      0,  0,  2,  -72;...
      0,  0,  0,    1];
 

%Transforming coordinates to match Schaefer in .nii file and find
%corresponding nodes
for n = 1:length(coords_PFIT)
    
    x_mni = coords_PFIT(n,1); 
    y_mni = coords_PFIT(n,2);
    z_mni = coords_PFIT(n,3); 
    
    n_coords = 1;
    
    coordinate = [x_mni y_mni z_mni ones(n_coords ,1)]*(inv(T))';
    coordinate(:,4) = [];
    coordinate = round(coordinate);
    x_voxel = coordinate(1);
    y_voxel = coordinate(2);
    z_voxel = coordinate(3);
    
    nodes_PFIT(n) = schaefer_atlas(x_voxel, y_voxel, z_voxel);
    
end



nodes_PFIT = unique(nodes_PFIT);
nodes_PFIT = nodes_PFIT(nodes_PFIT~=0);

schaefer_pfit = schaefer_atlas;

for el = 1:100
    
    if sum(el == nodes_PFIT) == 0
        
        schaefer_pfit(schaefer_pfit == el) = 0;
        
    end
end

info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_pfit,'schaefer_pfit.nii',info)


nodes = nodes_PFIT;
save('nodes_pfit', 'nodes')

%% Lateral PFC

schaefer_atlas = niftiread('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');

%LPFC nodes of Schaefer 100 node parcellation
nodes_PFC_core = [26, 32, 76, 81]; % to centered: 92,44, 93, % to frontal: 34, 86
nodes_PFC_extended = [ 33, 38, 45, 46, 47, 82, 85, 90, 94, 95];  
nodes_PFC_extended = [nodes_PFC_core, nodes_PFC_extended];

schaefer_PFC_core = schaefer_atlas;
for el = 1:100
    
    if sum(el == nodes_PFC_core) == 0
        
        schaefer_PFC_core(schaefer_PFC_core == el) = 0;
        
    end
end

info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_PFC_core,'schaefer_PFC_core.nii',info)

nodes = nodes_PFC_core;
save('nodes_PFC_core', 'nodes')


schaefer_PFC_extended = schaefer_atlas;

for el = 1:100
    
    if sum(el == nodes_PFC_extended) == 0
        
        schaefer_PFC_extended(schaefer_PFC_extended == el) = 0;
        
    end
end

info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_PFC_extended,'schaefer_PFC_extended.nii',info)

nodes = nodes_PFC_extended;
save('nodes_PFC_extended', 'nodes')

%% PFC Cole

schaefer_atlas = niftiread('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');

%Coordinates from PFC from Cole 2012: Global Connectivity of Prefrontal Cortex Predicts Cognitive
%Control and Intelligence https://doi.org/10.1523/JNEUROSCI.0536-12.2012 

%Coordinates from Paper
coords_Cole_original = [ -43.4 13.6 29.4;...
                -46.3 14.9 28.5;...
                -44.2 13.7 29.8];

%Add right hemisphere            
coords_Cole_extend = [ 43.4 13.6 29.4;...
                46.3 14.9 28.5;...
                44.2 13.7 29.8];
            
coords_Cole = [coords_Cole_original ; coords_Cole_extend];           
for n = 1:length(coords_Cole)
    
    x_mni = coords_Cole(n,1); 
    y_mni = coords_Cole(n,2);
    z_mni = coords_Cole(n,3); 
    
    n_coords = 1;
    
    coordinate = [x_mni y_mni z_mni ones(n_coords ,1)]*(inv(T))';
    coordinate(:,4) = [];
    coordinate = round(coordinate);
    x_voxel = coordinate(1);
    y_voxel = coordinate(2);
    z_voxel = coordinate(3);
    
    nodes_Cole(n) = schaefer_atlas(x_voxel, y_voxel, z_voxel);
    
    
end

nodes_Cole = [nodes_Cole, 81]; %Add one region of right cortex manually to match left side
nodes_Cole = unique(nodes_Cole);
nodes_Cole = nodes_Cole(nodes_Cole~=0);

schaefer_Cole = schaefer_atlas;

for el = 1:100
    
    if sum(el == nodes_Cole) == 0
        
        schaefer_Cole(schaefer_Cole == el) = 0;
        
    end
end



info = niftiinfo('Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
niftiwrite(schaefer_Cole,'schaefer_Cole.nii',info)

nodes = nodes_Cole;
save('nodes_Cole', 'nodes')