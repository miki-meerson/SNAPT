function [spikeT,nspike] = spikefind2(intens,inter,guide,thresh)
% function [spikeT,nspike] = spikefind2(intens,inter,guide,thresh);
%
% INPUTS
% intens: the fluorescence time trace
% inter: the minimum number of allowed points between spikes
% guide: a guess for the threshold intensity to help guide the eye
% thresh: the threshold to use for finding spikes
% 
% OUTPUTS
% spikeT: the indices of the spike peaks
% nspike: the number of spikes
%
% Thresh is an optional argument giving the initial threshold value to use.
% If Thresh is not specified then the function prompts the user to select
% a threshold by clicking on a plot of intens.  A click <1 indicates no spikes.
% DRH 09/26/13
% modified Kit 11/22/13

figure(888); clf;
plot(intens)
hold on;
plot([1 length(intens)], [guide guide], 'g-');
hold off;
if nargin == 3;
    title({'Right-click to indicate threshold';'A click <1 indicates no spikes'})
    [x, y] = getpts(gca);
    thresh = y(end);
end;
hold all;
plot([1 length(intens)], [thresh thresh], 'r-');

if thresh <= 1;
    spikeT = [];
    nspike = 0;
    return
else
    spikeT = spikefind3(intens, thresh, inter);  % The second argument is the threshold above which each spike must go.
    nspike = size(spikeT,2);
end
    
% Plot the spikes overlaid on the data.  Each spike should be marked by a
% red star.
plot(spikeT, intens(spikeT), 'r*')