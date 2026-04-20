% Get the connectivity matrices and calculate a mean matrix for Vizaj viz
% 1. Load the setsize level connectivity matrix and opt setsize mat files
% 2. Select only the prewm files and respective Opt level indices
% 3. Get the mean of below, above and just Opt for all subjects
% 4. Combine the three matrices to a single one

% author @ Rahul Venugopal on 16th of August 2023

load results\FC_mean_across_trials.mat
% Picking only prewm files
setsize_conn_mean_GT = setsize_conn_mean_GT(:,:,:,:,2:2:72);

load results\opt_setsize_all.mat
Opt_array_all = Opt_array_all(2:2:72);
% Get the index of opt level
Opt_array_all = Opt_array_all -1;

% Convert the setsize level dim (first one) to Below, Opt and Above
% Create empty arrays to hold above, below, and at groups
below = nan(4,55,55,36);
at = nan(4,55,55,36);
above = nan(4,55,55,36);

% Loop through each subject and get their Below|Opt|Above
for subjs=1:size(setsize_conn_mean_GT,5)
    % Get the optimal setsize index for that subj
    opt_value = Opt_array_all(subjs);

    % Group into three levels
    below(:,:,:,subjs) = squeeze(nanmean(setsize_conn_mean_GT(1:opt_value-1, :,:,:,subjs),1));
    at(:,:,:,subjs) = squeeze(setsize_conn_mean_GT(opt_value, :,:,:,subjs));
    above(:,:,:,subjs) = squeeze(nanmean(setsize_conn_mean_GT(opt_value+1:7, :,:,:,subjs),1));
end

% Combine the levels into a single matrix
mean_conn_matrix_prewm = cat(5, below, at, above);

% Do an average across subjects and get 12 matrices out for Vizaj
theta_below = squeeze(nanmean(mean_conn_matrix_prewm(1,:,:,:,1),4));
alpha_below = squeeze(nanmean(mean_conn_matrix_prewm(2,:,:,:,1),4));
beta_below = squeeze(nanmean(mean_conn_matrix_prewm(3,:,:,:,1),4));
gamma_below = squeeze(nanmean(mean_conn_matrix_prewm(4,:,:,:,1),4));

theta_at = squeeze(nanmean(mean_conn_matrix_prewm(1,:,:,:,2),4));
alpha_at = squeeze(nanmean(mean_conn_matrix_prewm(2,:,:,:,2),4));
beta_at = squeeze(nanmean(mean_conn_matrix_prewm(3,:,:,:,2),4));
gamma_at = squeeze(nanmean(mean_conn_matrix_prewm(4,:,:,:,2),4));

theta_above = squeeze(nanmean(mean_conn_matrix_prewm(1,:,:,:,3),4));
alpha_above = squeeze(nanmean(mean_conn_matrix_prewm(2,:,:,:,3),4));
beta_above = squeeze(nanmean(mean_conn_matrix_prewm(3,:,:,:,3),4));
gamma_above = squeeze(nanmean(mean_conn_matrix_prewm(4,:,:,:,3),4));

% Get the csv s out
writematrix(theta_below, 'results/matrices_vizaj/mean_conn_matricestheta_below.csv');
writematrix(alpha_below, 'results/matrices_vizaj/mean_conn_matricesalpha_below.csv');
writematrix(beta_below, 'results/matrices_vizaj/mean_conn_matricesbeta_below.csv');
writematrix(gamma_below, 'results/matrices_vizaj/mean_conn_matricesgamma_below.csv');

writematrix(theta_at, 'results/matrices_vizaj/mean_conn_matricestheta_at.csv');
writematrix(alpha_at, 'results/matrices_vizaj/mean_conn_matricesalpha_at.csv');
writematrix(beta_at, 'results/matrices_vizaj/mean_conn_matricesbeta_at.csv');
writematrix(gamma_at, 'results/matrices_vizaj/mean_conn_matricesgamma_at.csv');

writematrix(theta_above, 'results/matrices_vizaj/mean_conn_matricestheta_above.csv');
writematrix(alpha_above, 'results/matrices_vizaj/mean_conn_matricesalpha_above.csv');
writematrix(beta_above, 'results/matrices_vizaj/mean_conn_matricesbeta_above.csv');
writematrix(gamma_above, 'results/matrices_vizaj/mean_conn_matricesgamma_above.csv');