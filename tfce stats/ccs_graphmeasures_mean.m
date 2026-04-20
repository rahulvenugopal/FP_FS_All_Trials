%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load connectivity matrix data of 5 dimensions as below
% setsize,frequency_band,chan,chan,subjects
% Author Rahul Venugopal on 08.08.2022
% Modified from Dr. Srivas codebase
% Modified on 11th January 2023 to check if any NaN is there
% Modified on 18th July 2023 for ACDMT paradigm

% Noteworthy points
% The connectivity matrix has to be a full matrix
% Previously, I passed a lower triangular matrix and it was giving wrong parameters!

% Copyright (C) 2018 Srivas Chennu, University of Kent and University of Cambrige,
% srivas@gmail.com 
% 
% Binarise connectivity matrices and calculate graph theoretic metrics
% capturing micro-, meso- and macro-scale properties of the matrices
% modelled as networks. Accepts optional input argument, heuristic,
% specifiying the number of times to calculate and average over heuristic
% graph-theory metrics like modularity and derivatives like participation
% coefficient. Default value of heuristic is 50.
%
% For more, see [1,2]
% 
% [1] Chennu S, Annen J, Wannez S, Thibaut A, Chatelle C, Cassol H, et al.
% Brain networks predict metabolism, diagnosis and prognosis at the bedside
% in disorders of consciousness. Brain. 2017;140(8):2120-32.

% [2] Chennu S, Finoia P, Kamau E, Allanson J, Williams GB, Monti MM, et al. 
% Spectral signatures of reorganised brain networks in disorders of consciousness. 
% PLOS Computational Biology. 2014;10(10):e1003887.
%
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

% Notes
% The level of thresholding is limited by average degree of nodes.
% The lower limit should ensue that the average degree should not be smaller
% than 2∗log(N) where N is the number of electrodes
% The lower boundary should guarantee that the resulting networks were estimable

% Ref: Complex network measures of brain connectivity: Uses and
% interpretations, Neuroimage, 2010

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load connectivity matrix
load 'results/FC_mean_across_trials.mat';
size(setsize_conn_mean_GT)

% Load channel locs and chandist
load 'results/chandist.mat';
load 'results/chanlocs_55.mat';

% Parameters setup

% Modularity and community structure calcualted by heurostic Louvain algorithm
% all measures derived therefrom, were averaged over 50 repetitions
param.heuristic = 50;

% Normalising channel distance between 0 and 1
chandist = chandist / max(chandist(:));

% Listing header for graph theory measures
graphdata{1,1} = 'clustering';
graphdata{2,1} = 'characteristic path length';
graphdata{3,1} = 'global efficiency';
graphdata{4,1} = 'modularity';
graphdata{5,1} = 'centrality';
graphdata{6,1} = 'modular span';
graphdata{7,1} = 'participation coefficient';
graphdata{8,1} = 'degree';

% proportional network density thresholds at which to threshold, binarise and
% calculate graph theory metrics. Starts at 1 = keep all network edges, and
% steps in decrements .025 to 0.1 = keep only 10% of strongest edges
% The lowest threshold of 10% ensured that the average degree was not
% smaller than 2*log(N), where N is the number of nodes in the network

tvals = 1:-0.025:0.1;

for setsize = 1:size(setsize_conn_mean_GT,1)
       for subject_no = 1:size(setsize_conn_mean_GT,5)
           fprintf('Subject no %d\n',subject_no);
           check_valid_conn = squeeze(setsize_conn_mean_GT(setsize,:,:,:,subject_no));

           % Checking for any NaN, skip the graph theory analysis if NaNs are found (even a single NaN)
           % This is a crude way to handle unequal subject number in three groups
           if sum(isnan(check_valid_conn(:))) == 0
               
               for band = 1:size(setsize_conn_mean_GT,2)
                   fprintf('Frequency band %d\n',band);
                   connmat = squeeze(setsize_conn_mean_GT(setsize,band,:,:,subject_no));
                   
                   for thresh = 1:length(tvals)
                       bin_connmat = double(threshold_proportional(connmat,tvals(thresh)) ~= 0);
                       allcc(thresh,:) = clustering_coef_bu(bin_connmat);
                       % The clustering coeff can suffer from the degree
                       % Consider transitivity which is normalised for degree
                       % allcc(thresh,:) = transitivity_bu(bin_connmat);
                       allcp(thresh) = charpath(distance_bin(bin_connmat),0,0);
                       % uses Kintali's algorithm
                       alleff(thresh) = efficiency_bin(bin_connmat);
                       allbet(thresh,:) = betweenness_bin(bin_connmat);
                       alldeg(thresh,:) = degrees_und(bin_connmat);
                       
                       for runs = 1:param.heuristic
                           [Ci, allQ(thresh,runs)] = community_louvain(bin_connmat);
                           modspan = zeros(1,max(Ci));
                           
                           for m = 1:max(Ci)
                               if sum(Ci == m) > 1
                                   distmat = chandist(Ci == m,Ci == m) .* bin_connmat(Ci == m,Ci == m);
                                   distmat = nonzeros(triu(distmat,1));
                                   modspan(m) = sum(distmat)/sum(Ci == m);
                               end
                           end
                           allms(thresh,runs) = max(nonzeros(modspan));
                           allpc(thresh,runs,:) = participation_coef(bin_connmat,Ci);
                       end
                       
                       % clustering coeffcient
                       graphdata{1,2}(setsize,band,subject_no,thresh) = mean(allcc(thresh,:),2);
                       
                       % characteristic path length
                       graphdata{2,2}(setsize,band,subject_no,thresh) = allcp(thresh);
                       
                       % global efficiency
                       graphdata{3,2}(setsize,band,subject_no,thresh) = alleff(thresh);
                        
                       % modularity
                       graphdata{4,2}(setsize,band,subject_no,thresh) = mean(allQ(thresh,:));
                        
                       % betweenness centrality
                       graphdata{5,2}(setsize,band,subject_no,thresh) = mean(allbet(thresh,:),2);
                        
                       % modular span (for definition, see [2])
                       graphdata{6,2}(setsize,band,subject_no,thresh) = mean(allms(thresh,:));
                        
                       % participation coefficient
                       graphdata{7,2}(setsize,band,subject_no,thresh) = mean(mean(squeeze(allpc(thresh,:,:))),2);
                        
                       % degree
                       graphdata{8,2}(setsize,band,subject_no,thresh) = mean(alldeg(thresh,:),2);
                   end
               end
           end
       end
end

% Save the GT measures | 8 of them
save('graphdata_mean','graphdata');