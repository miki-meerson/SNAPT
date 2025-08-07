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

function rsquare_map(rsquare_matrix, thresh)
% function rsquare_map(rsquare_matrix, thresh)
%
% Displays a color image of an r^2 map, indicating pixel-by-pixel quality of 
% a SNAPT fit.  The colormap is designed to emphasize distinctions between
% r^2 values near the top of the range.  The 'thresh' parameter sets where
% the colormap will start to cut off values of r^2.

Rthresh = thresh;
colors = 128;
colormap(hsv(colors))
cmap = colormap('cool');
cmap2(1:colors,:) = cmap;
extra = round(1/(1-Rthresh)*colors);
cmap2((length(cmap)+1):extra,:) = linspace(1,0,round(extra-colors)).^2'*cmap2(colors,:);
cmap2 = flipud(cmap2);
cmap2 = cmap2(1:10:end,:);

imagesc(rsquare_matrix, [0, 1]);
axis('off')
daspect([1 1 1])
colormap(cmap2)
colorbar
