% Analyse the working memory data at trial level for theta and alpha power
% modulations with working memory capacity using FOOOF

% Load the data
% Pick the capacity values of each trial as well
% Compute the time frequency graph and get PSD for each trial
% Pass it to fooof and get the pure oscillatory part

% author @ Rahul Venugopal on April 18th 2026

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
master_power_data = {};

% Initialising a cell array to store the preprocessed trial indices
capacities = {};

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

    [tfpw]  = deal(zeros(EEG.nbchan,num_freq_bands,length(times2save), EEG.trials));
    tic
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
    toc

end

%% Save the master_slider_data which contains all subejcts data
% This needs to be saved with the file list as well for matching

filenames_table = cell2table(fullfile_List');
filenames_table = renamevars(filenames_table,["Var1"], ...
                 ["filenames"]);
writetable(filenames_table,'files.csv');

% I came across this '-v7.3' flag for saving files beyond 2 GB
save('wmload_sequence.mat', 'capacities');
save('tfpw_powers.mat','master_power_data', '-v7.3');