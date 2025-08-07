% Copyright (c) 2012, Adam Cohen, Daniel Hochbaum
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

function out = shiftkern(beta, kernel);
% function out = shiftkern(beta, kernel);
% Translates and scales a kernel so out = a*k(c*(t-dt)) + b,
% where a = beta(1); b = beta(2); c = beta(3); dt = beta(4);
%
% DRH and AEC 20 Oct. 2012

a = beta(1); 
b = beta(2); 
c = beta(3); 
dt = beta(4);

kernel = a*kernel + b;
kernel = [kernel(1); kernel; kernel(end)]; % Replicate the values as the end for extrapolation.
[~, t0] = max(kernel);
L = length(kernel);

x1 = (1:L)-t0;
x2 = (x1(2:end-1) - dt)*c;

% diff1 = kernel(end-1)-kernel(end);
% diff2 = kernel(2)-kernel(1);
% avgval = (kernel(end)+kernel(1))/2 - (diff1+diff2)/2;

out = interp1(x1, kernel, x2, 'linear', 'extrap')';
% out = interp1(x1, kernel, x2, 'linear', (kernel(1)+kernel(end))/2)';
