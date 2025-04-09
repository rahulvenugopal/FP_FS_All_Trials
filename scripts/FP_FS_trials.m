% Analyse the working memory data at trial level for theta and alpha power

% Load the data
% Pick the capacity values of each trial as well
% Compute the power in two bands as well as instantaneous frequencies

% author @ Rahul Venugopal on April 7th 2025

%% Select .set files for analysis
[fullfile_List, filepath, ~] = uigetfile({'*.set'}, ...
        'Select .set file(s) to be analysed', ...
        'MultiSelect', 'on');
    
% Take care of single file selection
if ischar(fullfile_List) == 1
    fullfile_List = {fullfile_List};
end
cd(filepath);

eeglab nogui;

% Initialise variables
master_slider_data = {};
master_power_data = {};

% Initialising a cell array to store the preprocessed trial indices
capacities = {};

% Parameters config
% Setting up the frequency slider
times2save = -2200:1:299;

% define boundaries for frequency bands
freq_bands = {
    [4 8];
    [8 12]
    };

num_freq_bands = size(freq_bands,1);

n_order = 10;

% Load the files
parfor file_no = 1:length(fullfile_List)
    filename = fullfile_List{file_no};
    
    % get fileparts and create a filename
    name_parts = strsplit(filename, '.');
    save_name = name_parts{1};

    % Load the data
    EEG = pop_loadset(filename);

    eventtypes = cellfun(@(x,y) x(cell2mat(y)==0),{EEG.epoch.eventtype},{EEG.epoch.eventlatency},'UniformOutput',false);
    
    % Get capacity values
    capacityvals = cellfun(@(x) str2double(x{:}(end-1)),eventtypes)';

    % setup time indexing
    times2saveidx = dsearchn(EEG.times',times2save');

    % convert orders from ms to timepoints
    orders = linspace(10,400,n_order)/2;
    orders = round(orders/(1000/EEG.srate));

    [tfhz,tfpw]  = deal(zeros(EEG.nbchan,num_freq_bands,length(times2save), EEG.trials));

    % loop through frequency bands
    for fi=1:num_freq_bands
        
        % filter data
        
        % apply a band-pass filter with 15% transition zones.
        
        trans_width    = .15;
        idealresponse  = [ 0 0 1 1 0 0 ];
        filtfreqbounds = [ 0 (1-trans_width)*freq_bands{fi,1}(1) freq_bands{fi,1}(1) freq_bands{fi,1}(2) freq_bands{fi,1}(2)*(1+trans_width) EEG.srate/2 ]/(EEG.srate/2);
        filt_order     = round(3*(EEG.srate/freq_bands{fi,1}(1)));
        filterweights  = firls(filt_order,filtfreqbounds,idealresponse);
        
        % this part does the actual filtering
        filterdata = zeros(size(EEG.data));
        for chani=1:EEG.nbchan
            filterdata(chani,:,:) = reshape( filtfilt(filterweights,1,double(reshape(EEG.data(chani,:,:),1,EEG.pnts*EEG.trials))) ,EEG.pnts,EEG.trials);
        end
        
        % compute pre-filtered frequency sliding       
        freqslide_prefilt = zeros(EEG.nbchan,EEG.pnts,EEG.trials);
        temppow = zeros(EEG.nbchan,length(times2save),EEG.trials);
        
        % loop over trials
        for triali=1:EEG.trials
            
            % get analytic signal via Hilbert transform
            temphilbert = hilbert(squeeze(filterdata(:,:,triali))')';
            
            % compute frequency sliding (note that this signal may be noisy;
            % median filtering is recommended before interpretation)
            freqslide_prefilt(:,1:end-1,triali) = diff(EEG.srate*unwrap(angle(temphilbert'),[],2)',1,2)/(2*pi);

            % get power for each time point and trial for the specific band under loop
            temppow(:,:,triali) = abs(temphilbert(:,times2saveidx)).^2;

        end
        
        % Adding power for the band to tfpw matrix
        tfpw(:,fi,:,:) = squeeze(temppow);

        % apply median filter    
        % temporary matrix involve in filtering
        phasedmed = zeros(EEG.nbchan,length(orders),length(times2save),EEG.trials);    
        
        % median filter
        for oi=1:n_order
            for ti=1:length(times2save)
                
                % using compiled fast_median
                for triali=1:EEG.trials

                    % Commenting out the normal median
%                     temp = sort(freqslide_prefilt(:,max(times2saveidx(ti)-orders(oi),1):min(times2saveidx(ti)+orders(oi),EEG.pnts-1),:),2);
%                     phasedmed(:,oi,ti,:) = temp(:,floor(size(temp,2)/2)+1,:);
                    phasedmed(:,oi,ti,triali) = fast_median(freqslide_prefilt(:,max(times2saveidx(ti)-orders(oi),1):min(times2saveidx(ti)+orders(oi),EEG.pnts-1),triali)');
                end            
            end
        end
        
        % the final step is to take the mean of medians
        tfhz(:,fi,:,:) = median(phasedmed,2);
        
    end % end frequency band loop

    % Save the tfhz file which has chan * freq * time * trials
    master_slider_data{file_no} = tfhz;

    % save the tfpw file chan * freq * time * trials
    master_power_data{file_no} = tfpw;

    % Save the indices of the trials
    capacities{file_no} = capacityvals;

    fprintf('\nFrequencies slided for %s file\n',save_name);
    fprintf('Files %d of %d completed\n', file_no, length(fullfile_List));  

end

%% Save the master_slider_data which contains all subejcts data
% This needs to be saved with the file list as well for matching

filenames_table = cell2table(fullfile_List');
filenames_table = renamevars(filenames_table,["Var1"], ...
                 ["filenames"]);
writetable(filenames_table,'files.csv');

% I came across this '-v7.3' flag for saving files beyond 2 GB
% Should I round off to save space
save('tfhz_sliders.mat','master_slider_data', '-v7.3');
save('wmload_sequence.mat', 'capacities');

save('tfpw_powers.mat','master_power_data', '-v7.3');

%% Rounded matrix
rounded_power = cellfun(@(x) round(x, 4), master_power_data, 'UniformOutput', false);
rounded_sliding = cellfun(@(x) round(x, 4), master_slider_data, 'UniformOutput', false);

save('tfhz_sliders_rounded.mat','rounded_sliding', '-v7.3');
save('tfpw_powers_rounded.mat','rounded_power', '-v7.3');

%% Calculate trimmed means across third dimensions (time series)

% Example: 10% trimming from both tails
trim_percent = 20;  % Total percent to trim (e.g., 15% from each end)

trimmed_means_sliding = cellfun(@(x) squeeze(trimmean(x, trim_percent, 3)), master_slider_data, 'UniformOutput', false);
trimmed_means_power = cellfun(@(x) squeeze(trimmean(x, trim_percent, 3)), master_power_data, 'UniformOutput', false);

% Round off to 4 decimal places
rounded_power = cellfun(@(x) round(x, 4), trimmed_means_power, 'UniformOutput', false);
rounded_sliding = cellfun(@(x) round(x, 4), trimmed_means_sliding, 'UniformOutput', false);

% Save them
save('tfhz_sliders_rounded.mat','rounded_sliding', '-v7.3');
save('tfpw_powers_rounded.mat','rounded_power', '-v7.3');