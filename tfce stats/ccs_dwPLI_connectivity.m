%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Centre for Consciousness Studies - NIMHANS
% Purpose: Calculate dwPLI connectivity matrices
% Script returns a full connectivity symmetrical matrix
% Date of Creation: 10th July 2023
% Author: Rahul Venugopal

% Noteworthy points
% Switching to Dr. Arun's accs_phaseconnectivitytime function to compute
% the phase connectivity measure (this can return PLI, wPLI, dwPLI & ISPC)

% Loads the cleaned data, epoch and throws away incorrect trials
% We pass chan*delaytime*trials
% We have to compute dwPLI for each channel pair and frequency for each trial of each subject

% The script outputs two giant mat files, one with mean connectivity values
% and another with max connectivity values
% These mat files go to ccs_graphmeasures.m and then to Gather_graph.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select .set files for analysis
[fullfile_List, filepath, ~] = uigetfile({'*.set'}, ...
        'Select .set file(s) to be analysed', ...
        'MultiSelect', 'on');
    
% Take care of single file selection
if ischar(fullfile_List) == 1
    fullfile_List = {fullfile_List};
end
cd(filepath);

% Fire up EEGLAB silently | silent fire!
eeglab nogui;

% Initialise frequency bands
freq_list = [4 8;
    8 12;
    13 30;
    30 60];

% dwPLI for all frequencies
freqlist = [4:0.25:60];

subject_name = 'Subject ';

% Initialise the ALL connectivity matrix
% Trials is first dimension
% EEG bands are second dimension
% EEG channels are third and fourth dimension
% Last dimension is the number of subjects
connectivity_matrix_mean_GT = NaN([300 4 55 55 72]);
connectivity_matrix_max_GT = NaN([300 4 55 55 72]);

% Load the correct indices mat file
load '../FC_GT_ACDMT/results/correct_indices_all.mat';

%% Let the loop begin

% get the file name
for file_no = 1:length(fullfile_List)
    filename = fullfile_List{file_no};
    
    % Loading the dataset and epoching
    EEG = pop_loadset('filename',filename);

    % Epoch the data based on markers
    [EEG] = pop_epoch( EEG, {  'S 21'  'S 29'  'S 31'  'S 39'  'S 41'  'S 49'  'S 51'  'S 59'  'S 61'  'S 69'  'S 71'  'S 79'  'S 81'  'S 89'  }, ...
        [-2.5 0.6], ...
        'newname', 'eeg_data_epochs', 'epochinfo', 'yes');

    % Throw away incorrect trials
    EEG = pop_select(EEG, 'trial', correct_indices_all{1,file_no});
    
    % Get the channel pairs
    chanpairs   = fliplr(fullfact([EEG.nbchan,EEG.nbchan]));
    
    % Pre-allocating connectivity matrix (chanpairs * trials)
    conndata = NaN(EEG.trials, length(chanpairs), length(freqlist));  

    % get trials * chan * time data
    data = double(EEG.data(:,1601:2500,:));
    
    for trial = 1:size(data,3)
        % Looping through frequencies | Deploying parfor
        parfor frq_no = 1:length(freqlist)
            connvalues  = accs_phaseconnectivitytime(squeeze(data(:,:,trial)),...
                EEG.srate, freqlist(frq_no));
            conndata(trial,:,frq_no)   = connvalues.dwpli;
        end
    end

    %% At this point in code life, we have conndata having trials * chanpairs * freq computed
    % We can average OR pick the frequency which has maximum average (trimmean) connectivity
    % Both of the above operation happens at band level

    % Band averaged connectivity
    % We have defined the canonical EEG bands in freq_list

    for trials = 1: size(conndata,1)

        for frequency_band = 1:size(freq_list,1)
    
            % the freq_list may not have the exact frequency
            % if we use FFT or other non wavelet based decompositions which set freq resolution
            % borders (say 0.5 Hz or 8 Hz, it would be 8.15 based on frequency resolution)
    
            [~, fstart] = min(abs(freqlist-freq_list(frequency_band,1))); % Scans and find starting freq of EEG band
            [~, fend]   = min(abs(freqlist-freq_list(frequency_band,2))); % Scans and find ending freq of EEG band
    
            % Average across frequencies in the band
            conndata_mean_trial_band = squeeze(nanmean(conndata(trials,:,fstart:fend), 3));
    
            % Visualise the histogram of band level connectivity mean values
            % histogram(conndata_mean_trial_band(:,1))
    
            % Pre-allocating a sqaure matrix for connectivity
            connectivity_matrix = NaN(EEG.nbchan,EEG.nbchan);
    
            diag_removed = tril(reshape(conndata_mean_trial_band, [EEG.nbchan, EEG.nbchan]));
        
            % Removing diagonal element inplace to diag_removed
            for rows = 1:length(diag_removed)
                for columns = 1:length(diag_removed)
                    if rows == columns
                         diag_removed(rows,columns) = 0;
                    end
                end
            
                 % Adding the connectivity matrix trial wise
                 connectivity_matrix(:,:) = diag_removed;
            end
            
            % Copying the lower triangular matrix to upper triangular
            connectivity_matrix = tril(connectivity_matrix,1)+tril(connectivity_matrix,1)';
              
            % Gathering connectivity matrices per frequency band
            connectivity_matrix_mean_GT(trials,frequency_band,:,:,file_no) = connectivity_matrix;
            
            % Using trimmed means of all pairwise connectivity values, identify the peak of average connectivity
            % Trimmed mean is across the channel pairs
    
            [~,max_peak_freq] = max(trimmean(conndata(trials,:,fstart:fend),10,2));
    
            % Getting those max frequency's connpairs
            conndata_max_trial_band = conndata(trials,:, fstart + max_peak_freq -1);
    
            % Pre-allocating a sqaure matrix for connectivity
            connectivity_matrix_max = NaN(EEG.nbchan,EEG.nbchan);
    
            diag_removed_max = tril(reshape(conndata_max_trial_band, [EEG.nbchan, EEG.nbchan]));
        
            % Removing diagonal element inplace to diag_removed
            for rows = 1:length(diag_removed_max)
                for columns = 1:length(diag_removed_max)
                    if rows == columns
                         diag_removed_max(rows,columns) = 0;
                    end
                end
            
                 % Adding the connectivity matrix trial wise
                 connectivity_matrix_max(:,:) = diag_removed_max;
            end
    
            % Copying the lower triangular matrix to upper triangular
            connectivity_matrix_max = tril(connectivity_matrix_max,1)+tril(connectivity_matrix_max,1)';
    
            % Gathering the band max connectivity matrix per frequency band
            connectivity_matrix_max_GT(trials,frequency_band,:,:,file_no) = connectivity_matrix_max;
                               
        end
    end

    % Update that one subject is done
	fprintf('\n -------------------\n');
    fprintf('\n %s %d Done\n',...
        subject_name,file_no);
    fprintf('\n -------------------\n');
end

% Saving the FULL connectivity matrix as a single file
save('../FC_GT_ACDMT/results/FULL_connectivity_matrix_mean.mat','connectivity_matrix_mean_GT', '-v7.3');
save('../FC_GT_ACDMT/results/FULL_connectivity_matrix_max.mat','connectivity_matrix_max_GT', '-v7.3');