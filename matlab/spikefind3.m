% Copyright (c) 2012, Adam Cohen
% All rights reserved.
% 
% Redistribution in source or binary forms, with or without modification,
% is not permitted
%        
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function out = spikefind3(dat, thresh, inter);
% Function out = spikefind3(dat, thresh);
% for each time that dat goes above thresh, returns the index of the local
% maximum in dat
% DRH and AEC 24 Feb. 2011

spikeon = find(dat(2:end) > thresh & dat(1:end-1) < thresh);
spikeoff = find(dat(2:end) < thresh & dat(1:end-1) > thresh);

if isempty(spikeon) | isempty (spikeoff)
    out = [];
    return
end;

if spikeoff(1) < spikeon(1);
    spikeoff(1) = [];
end;
if spikeon(end) > spikeoff(end);
    spikeon(end) = [];
end;

nspike0 = length(spikeon); % preliminary guess for number of spikes, includes spikes too close together
spikeT = zeros(nspike0, 1);
for j = 1:nspike0;
    [y(j), indx] = max(dat(spikeon(j):spikeoff(j)));  % find the maximum between each spike onset and offset.
    spikeT(j) = indx + spikeon(j)-1;
end;

if isempty(spikeon) | isempty (spikeoff)  % no spikes
    out = [];
    return
end;

if spikeoff(1) < spikeon(1);    % falling edge of a spike at the beginning of the trace
    spikeoff(1) = [];
end;
if spikeon(end) > spikeoff(end);  % rising edge of a spike at the end of the trace
    spikeon(end) = [];
end;
    
% Get rid of spikes that are too close together.  Find the maximum among clusters of spikes.    
dT = diff(spikeT);
clusters = find(dT < inter);
while ~isempty(clusters)
    badspike = zeros(size(spikeT));
    ncluster = length(clusters);
    for j = 1:ncluster;
       [~,idx] =  min([dat(spikeT(clusters(j))), dat(spikeT(clusters(j)+1))]);
       badspike(clusters(j) + idx - 1) = 1;
    end;
    spikeT(find(badspike)) = [];
    dT = diff(spikeT);
    clusters = find(dT < inter);
end;

out = spikeT;



