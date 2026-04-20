% Find the optimal value for finding neighbouring electrodes for LIMO
% The catch is the distance change based on cap manufacturer
% https://www.fieldtriptoolbox.org/faq/how_does_ft_prepare_neighbours_work/

% author@ Rahul Venugopal on 13th July 2023

% We need to know the channel locations in X,Y and Z and the labels
sens.label  = {chanlocs_55.labels}';
sens.pnt    = [chanlocs_55.X;chanlocs_55.Y;chanlocs_55.Z]';

nsensors = length(sens.label);

% compute the distance between all sensors
dist = zeros(nsensors,nsensors);
for i=1:nsensors
  dist(i,:) = sqrt(sum((sens.pnt(1:nsensors,:) - repmat(sens.pnt(i,:), nsensors, 1)).^2,2))';
end

% Try out a distance value
neighbourdist = 0.5;

% find the neighbouring electrodes based on distance
% later we have to restrict the neighbouring electrodes to those actually selected in the dataset
channeighbstructmat = (dist<neighbourdist);

% electrode istelf is not a neighbour
channeighbstructmat = (channeighbstructmat .* ~eye(nsensors));

% construct a structured cell array with all neighbours
neighbours=cell(1,nsensors);
for i=1:nsensors
  neighbours{i}.label       = sens.label{i};
  neighbours{i}.neighblabel = sens.label(find(channeighbstructmat(i,:)));
end

% Manually, match the neighbours.neighlabel with a topomap from cap vendor
% Try out different distance values until it makes sense
