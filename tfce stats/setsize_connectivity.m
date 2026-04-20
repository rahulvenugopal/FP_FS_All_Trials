%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Centre for Consciousness Studies - NIMHANS
% Purpose: From the connectivity matrices, get averaged setsize matrices
% Script returns setsize * band * chan * chan * subj
% Date of Creation: 17th July 2023
% Author: Rahul Venugopal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the mean connectivity mat file
load 'results/FULL_connectivity_matrix_mean.mat';
load 'results/FULL_connectivity_matrix_max.mat';

% The dimensions of the conn matrix is 300*4*55*55*72
% We have entries for the first dimension (trials) only for correct and clean trials

% Load the setsizes list as well
load 'results/export_setsize_all.mat';

% Setsize based conn matrix
% Robust averaging across trials of same setsize
setsize_conn_mean_GT = nan(7,4,55,55,72);
setsize_conn_max_GT = nan(7,4,55,55,72);

% Loop through subject, band and setsize
for subs = 1:size(connectivity_matrix_mean_GT,5)
    setsizes = double(export_setsize_all{1,subs});
    for bands = 1:size(connectivity_matrix_mean_GT,2)
        % Get the connectivity matrix for all correct trials for subj,band
        tempdata_mean = squeeze(connectivity_matrix_mean_GT(1:length(setsizes),bands,:,:,subs));
        tempdata_max = squeeze(connectivity_matrix_max_GT(1:length(setsizes),bands,:,:,subs));
        
        % create empty cells of 7 to accomodate the indices of each setizes
        indices = cell(1,7);
        for i = 2:8
            indices{i-1} = find(setsizes == i);
        end
        for setsize = 1:7
            setsize_conn_mean_GT(setsize,bands,:,:,subs) = nanmean(tempdata_mean(indices{1,setsize},:,:), 1);
            setsize_conn_max_GT(setsize,bands,:,:,subs) = nanmean(tempdata_max(indices{1,setsize},:,:), 1);
        end
    end
end

%% Saving the FC matrices across setsizes, bands and all subjects
save('results/FC_mean_across_trials.mat', 'setsize_conn_mean_GT');
save('results/FC_max_across_trials.mat', 'setsize_conn_max_GT');

%% Get a subject level average for prewm
FC_mean_prewm = nanmean(setsize_conn_mean_GT(:,:,:,:,2:2:72),5);
FC_max_prewm = nanmean(setsize_conn_max_GT(:,:,:,:,2:2:72),5);