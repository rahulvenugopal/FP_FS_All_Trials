% Author Rahul Venugopal on 22.11.2020
% Modified on 18th July 2023 for ACDMT data
% Gather the graph.mat file and write its contents to a csv file
load graphdata_mean.mat

%% Creating a header with all variable names in row 1
header = {graphdata{:,1}};
header{1,length(header)+1} = "setsize";
header{1,length(header)+1} = 'Frequency';
header{1,length(header)+1} = 'Subject_no';
header{1,length(header)+1} = 'Threshold';

%% Filling the mastersheet

% We have the data to be filled in graphdata.mat, second column
% This is a 4D data (setsize*freq*subjects*threshold)
% Each row is one paramters
% Prefill the last four columns based on dimensions of graphdata second
% column which are filled with data USE ANY command to find empty data
% structures

% Number of setsizes
no_of_setsizes = 7;

% Frequency bands, thresholds and total variables
no_of_freq = 4;
no_of_thresholds = 37;
total_no_of_variables = 8;
no_of_subj = 72;

% Let the data fill in
for variable_no = 1:total_no_of_variables
    for set = 1:no_of_setsizes        
        for subj = 1:no_of_subj
            for fband = 1:no_of_freq
                for thresholds = 1:no_of_thresholds
                    locator = (set-1)*72*4*37 + (subj-1)*4*37 + (fband -1)*37 + 1;
                    header{locator+thresholds,variable_no} = graphdata{variable_no,2}(set,fband,subj,fband,thresholds);
                    header{locator+thresholds, 9} = set;
                    header{locator+thresholds, 10} = fband;
                    header{locator+thresholds, 11} = subj;
                    header{locator+thresholds, 12} = thresholds;
                end
            end
        end
    end
end

% Converting cell to tables to write it to .xlsx format without headers
Table = cell2table(header);
filename = 'graphtheory_mastersheet_mean.csv';
writetable(Table,filename,'WriteVariableNames',false)