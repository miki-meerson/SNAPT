%%
clc;clearvars;close all

movname = 'Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\AceM-neon\AcAx3\28-10-2024-acax3-s4-awake\fov1\vol_001\vol.tif';
info = imfinfo(movname);
nframes = numel(info);
mov = zeros(info(1).Height, info(1).Width, nframes, 'uint16');
tic
for k = 1:nframes
    mov(:,:,k) = imread(movname, 'Index', k, 'Info', info);
end
toc
%%
movR = -double(mov);

%%
% check the noise:
intens = squeeze(mean(mean(movR)));
t = 1:length(intens);
figure(1); clf
plot(t, intens, 'k-');

%%
% get rid of the bad frames;
intensHP = intens - smoothdata(intens, 'sgolay', 10);
figure(1);
plot(intensHP)
badFrames = find(intensHP < -3.5);
plot(t, intens, 'k-', badFrames, intens(badFrames), 'r*')
movR(:,:,badFrames) = [];
title("Bad Frames")

% start again
avgImg = mean(movR, 3);
avgImg = avgImg - prctile(avgImg(:), 20);

% check the noise:
intens = squeeze(mean(mean(movR)));
t = 1:length(intens);
figure(2);
plot(t, intens, 'k-');
title("Clean")

%%
% check the power spectrum of the noise
intensPSD = fft(intens).*conj(fft(intens));
freq = (0:length(intensPSD)-1)*1000/length(intensPSD);
figure(2); clf
semilogy(freq, intensPSD)
% the systematic noise is mostly at > 150 Hz
smoothT = 15;  %ms. filter at ~60 Hz
intensHP = intens - smooth(intens, smoothT, 'sgolay', 2);
plot(t, intensHP)
noisePSD = fft(intensHP).*conj(fft(intensHP));
semilogy(freq, intensPSD, freq, noisePSD)

% Get rid of the first few frames where the filter doesn't work:
movR2 = movR(:,:,10:end);
intensHP2 = intensHP(10:end);
[~,~,nframe2]=size(movR2);
t2 = 1:nframe2;

%% project out the high freq noise and any linear or quadratic drift
[nrow, ncol, ~] = size(movR2);
regressors = [intensHP2'; t2 - mean(t2); (t2 - mean(t2)).^2];
regressors = regressors';  % [T x R]
regressors = regressors - mean(regressors);  % center regressors

% Precompute pseudoinverse for efficiency
X = regressors;
pinvX = pinv(X);  % [R x T]

movR2_clean = zeros(size(movR2));
scaleImgs = zeros(nrow, ncol, size(regressors, 2));

for y = 1:nrow
    for x = 1:ncol
        ytrace = squeeze(movR2(y, x, :));  % [T x 1]
        beta = pinvX * ytrace;             % [R x 1]
        fit = X * beta;                    % [T x 1]
        movR2_clean(y, x, :) = ytrace - fit;
        scaleImgs(y, x, :) = beta;
    end
end

movR2 = movR2_clean;
for j = 1:3
    subplot(2, 2, j)
    imagesc(scaleImgs(:, :, j)); 
    axis image off; 
    colormap(gray); 
    colorbar;
    title(['Regressor ' num2str(j)]);
end

%%
figure(3); clf
intens2 = squeeze(mean(mean(movR2)));
plot(intens, DisplayName="Original"); hold all;
plot(intens2, DisplayName="Clean")

%%
[ROI, Intens]=clicky3(movR2_clean, avgImg);  % Start at soma.

% convert to units of dF/F
f0 = apply_clicky(ROI, avgImg);
Intens = Intens./repmat(f0, [length(Intens), 1]);

plot(Intens)

%% Find Spikes
% All the spikes are basically in the soma only.
Intens=intens2; 
figure(4); clf
plot(Intens)
allMov = movR2_clean;
nframe=length(Intens);
nRoi=size(Intens,2);
Pulse=50;  % time to look before and after each spike
t=1:length(Intens);
f=mat2gray(Intens(:,1));  % Use the first clicky ROI for spike finding
fhp=f-smoothdata(f,'movmean',40);
spikeT = spikefind2(fhp+1,10,1.05);pause(1);close
nSpike = length(spikeT);
spikes = zeros(2*Pulse+1, nRoi, nSpike);
spikeMov = zeros(nrow, ncol, 2*Pulse + 1);

c = 1; % spike counter
for j=1:nSpike  % Calculate the average spike waveform in each ROI
    if spikeT(j)>Pulse && spikeT(j)<nframe-Pulse
        fs=Intens((spikeT(j)-Pulse):(spikeT(j)+Pulse),:);
        f0=mean(fs(1:10,:),1);
        spikes(:,:,j)=(fs-repmat(f0, [2*Pulse+1 1]));
        spikeMov = spikeMov + allMov(:,:,(spikeT(j)-Pulse):(spikeT(j)+Pulse));
        c = c + 1;
    
    else
        spikes(:,:,j)=NaN(2*Pulse+1, nRoi);
    end
end
spikeMov = spikeMov/c;
%%
playmov(spikeMov(:,:,(Pulse-10):(Pulse+10)), .1, 1)
[~, tmp] = clicky3(spikeMov);
plot(mat2gray(tmp(:,1))); hold all
plot(mat2gray(tmp(:,2)));
plot(mat2gray(tmp(:,3))); 

%% 
%Select a time just around the peak to use for the SNAPT fit
kernel = squeeze(mean(mean(spikeMov)));  % Flat average isn't as good as...
[~, kernel] = clicky(spikeMov, avgImg);  % clicking on the soma and proximal dendrite.
%%
kernel = kernel((Pulse-13):(Pulse+17));
kernel = mat2gray(kernel);  % Set the kernel to range from 0 to 1
%%
spikeMovFit = spikeMov(:,:,(Pulse-13):(Pulse+17));
spikeMovFit = imfilter(spikeMovFit, fspecial('Gaussian', [4 4], 1.5), 'replicate');  % filter a bit to smooth noise
spikeMovFit = (spikeMovFit - min(spikeMovFit(:)))/(max(spikeMovFit(:)) - min(spikeMovFit(:)));  % Set the max value of the spike movie to 1
[Vout, corrimg, weightimg, offsetimg] = extractV(spikeMovFit, kernel);
subplot(2,2,1); plot(Vout); hold all; plot(kernel);
subplot(2,2,2); imshow(corrimg, []); title('Corr')
subplot(2,2,3); imshow(weightimg, []); title('Weight')
subplot(2,2,4); imshow(offsetimg, []); title('Offset')

%% try SNAPT!
% Set up the matrices to store the results
xsize = ncol;
ysize = nrow;
betamat2 = zeros(ysize, xsize, 4);  % Matrix of fit parameters for each pixel
cimat = zeros(ysize, xsize, 4,2); % Matrix of confidence intervals for each fit parameter at each pixel
rsquaremat = zeros(ysize, xsize);  % Matrix of r^2, indicating goodness of fit at each pixel
goodfit = ones(ysize, xsize);  % matrix indicating pixels where the fit succeeded

% Perform the fit!  This is the heart of the SNAPT algorithm.

for x = 1:xsize
    for y = 1:ysize
        dat = squeeze(spikeMovFit(y,x,:));  % Take the AP waveform at pixel y,x
        lastwarn('');
        % Perform four-parameter fit.  Input is k(t).  Output is a*k(c*(t-dt)) + b,
        % where beta = [a,b,c,dt] is the vector of fitting parameters.
        % This will generate warnings.  Don't worry.
        [betamat2(y,x, :), r, J, COVB, mse] = nlinfit(kernel, dat, @shiftkern, [1 0 1 0], statset('Robust', 'off'));
        if ~isempty(lastwarn)  % Flag pixels where fit failed
            goodfit(y,x)=0;
        end
        rsquaremat(y,x) = 1 - var(r)/var(dat);
        cimat(y,x,:,:) = nlparci(squeeze(betamat2(y,x,:)), r, 'covar', COVB, 'alpha', .318);  % these are the 68% confidence intervals
    end
    ['Completed column ' num2str(x)];
end
ampimg = betamat2(:,:,1);
baseimg = betamat2(:,:,2);
widthimg = 1./betamat2(:,:,3);
dtimg = betamat2(:,:,4);

% See where the fit converged
figure(13); clf;
imshow(goodfit,[], 'InitialMagnification', 'fit');
title('Dark pixels indicate failed fit')

% Check the raw amplitudes from the fit
figure(14); clf;
imshow(ampimg ,[prctile(ampimg(:),1), prctile(ampimg(:),99)], 'InitialMagnification', 'fit');
title('Raw fit results for AP amplitude');

% Set constraints on the fitting parameters.  Adjust these values as
% appropriate.
min_a = 0;  % Amplitude
min_c = .2;  % Time scale factor
max_c = 4;
min_dt = -4; % Time shift (in frames)
max_dt = 4;

goodpix =   (ampimg > min_a).*...
    (widthimg > min_c).*...
    (widthimg < max_c).*...
    (dtimg > min_dt).*...
    (dtimg < max_dt);
figure(13);clf;
imshow(goodpix,[], 'InitialMagnification', 'fit');
title('Bright pixels indicate that the fit satistfied constraints')

% Set the amplitude to zero where the fit parameters lay outside the constraints;
ampimg = ampimg.*goodpix;

% constrain the time scaling to be within the boundaries
widthimg(widthimg < min_c) = min_c;
widthimg(widthimg > max_c) = max_c;

% Constrain the time shift to be within the boundaries
dtimg(dtimg < min_dt) = min_dt;
dtimg(dtimg > max_dt) = max_dt;

% Look at the results
figure(17); clf; colormap('gray');
subplot(2,3,1); imshow(ampimg, [], 'InitialMagnification', 'fit'); title('Amplitude image'); freezeColors;
subplot(2,3,2); imshow(baseimg, [], 'InitialMagnification', 'fit'); title('Offset image'); freezeColors;
subplot(2,3,4); imshow(widthimg, [], 'InitialMagnification', 'fit'); title('width image'); freezeColors;
subplot(2,3,5); imshow(dtimg, [], 'InitialMagnification', 'fit'); title('Delay image'); freezeColors;
subplot(2,3,6); rsquare_map(rsquaremat, 0.8); title('R^2'); % This function uses a nice colormap to make results easier to interpret
unfreezeColors

% Make a pretty color image of the delay
figure(18); clf
dtColImg = grs2rgb(dtimg, colormap('jet'), 0, 1);
dtColImg = dtColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(dtColImg)
colorbar
title('Delay map (ms)')
saveas(gca, '.\snapt_results\matlab\delay map.fig')
saveas(gca, '.\snapt_results\matlab\delay map.png')

% Make a pretty color image of the spike width
figure(19); clf
widthColImg = grs2rgb(widthimg, colormap('jet'), .6, 1.4);
widthColImg = widthColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(widthColImg)
title('width map')
colorbar('Ticks',0:.25:1,...
    'TickLabels',{'0.6', '0.8', '1.0', '1.2', '1.4'})
saveas(gca, '.\snapt_results\matlab\width map.fig')
saveas(gca, '.\snapt_results\matlab\width map.png')

%% PCA filter spikeMovFit

dSpikeMov = spikeMov - repmat(mean(spikeMov(:,:,:), 3), [1 1 size(spikeMov(:,:,:),3)]);
% dSpikeMov = imfilter(dSpikeMov, fspecial('Gaussian', [4 4], 1.5), 'replicate');
% dSpikeMov = spikeMovFit - repmat(mean(spikeMovFit, 3), [1 1 size(spikeMovFit,3)]);

spikeMovV = reshape(dSpikeMov, [], size(dSpikeMov, 3));
covMat = spikeMovV'*spikeMovV;
[VSpikes, ~] = eig(covMat);
VSpikes = VSpikes(:,end:-1:1);
VSpikes(:,1) = -VSpikes(:,1);  %manually set the signs of some of the terms to be more intuitive
VSpikes(:,3) = -VSpikes(:,3);
figure(1); clf
stackplot(VSpikes(:,1:20))

vec = spikeMovV*VSpikes(:,1:9);
eigImgsSpikes = reshape(vec, nrow, ncol, size(vec, 2));

figure(2); clf
for j = 1:9
    subplot(3,3,j)
    imshow(eigImgsSpikes(:,:,j), [])
end

spikesFiltPCA = eigImgsSpikes(:,:,1:3);
spikesFiltPCAVec = reshape(spikesFiltPCA, [], size(spikesFiltPCA, 3)) * VSpikes(:,1:3)';
spikeMovPCA = reshape(spikesFiltPCAVec, nrow, ncol, size(spikesFiltPCAVec, 2));  % reconstruct the PCA filtered spike movie

% clicky(spikeMovPCA, avgImg);
% saveas(gca, 'C:\Users\Adam\Desktop\update mtgs\In vivo\Short_Fig2\clicky spikeMovPCA.fig');
% saveas(gca, 'C:\Users\Adam\Desktop\update mtgs\In vivo\Short_Fig2\clicky spikeMovPCA.png');

% Make a nice movie of the PCA-filtered response.
nF = size(spikeMovPCA, 3);
spikeMovPCA = spikeMovPCA - repmat(spikeMovPCA(:,:,1), [1 1 nF]);
spikeMovPCA = spikeMovPCA/max(spikeMovPCA(:));
playmov(spikeMovPCA, .1, 1)  % this looks great!
%%
ampImg = mat2gray(std(spikeMovPCA, [], 3));
colormov = repmat(0.5 * mat2gray(ampImg), [1 1 3 nF]);
for j = 1:nF
    colormov(:,:,:,j) = colormov(:,:,:,j) + 3*ampImg.*grs2rgb(spikeMovPCA(:,:,j), colormap('jet'), 0, .4);
end
for j = 1:6
    subplot(2,3,j);
    imshow(colormov(:,30:110,:,j+48))
end
saveas(gca, '.\snapt_results\matlab\montage.fig')
saveas(gca, '.\snapt_results\matlab\montage.png')

for j = 1:nF
    imshow(colormov(:,:,:,j));
    text(5, 5, [num2str(j) ' ms'], 'Color', 'White', 'FontSize', 18);
    %     M(j) = getframe(gca);
    pause(.1)
end

%% Try SNAPT again, on the PCA filtered movie:
spikeMovFit = spikeMovPCA(:,:,(Pulse-13):(Pulse+17));
playmov(spikeMovFit, .1, 1)

%Select a time just around the peak to use for the SNAPT fit
kernel = squeeze(mean(mean(spikeMov)));  % Flat average isn't as good as...
[~, kernel] = clicky(spikeMov, avgImg);  % clicking on the soma and proximal dendrite.
kernel = kernel((Pulse-13):(Pulse+17));
kernel = mat2gray(kernel);  % Set the kernel to range from 0 to 1

spikeMovFit = spikeMov(:,:,(Pulse-13):(Pulse+17));
spikeMovFit = imfilter(spikeMovFit, fspecial('Gaussian', [4 4], 1.5), 'replicate');  % filter a bit to smooth noise
spikeMovFit = (spikeMovFit - min(spikeMovFit(:)))/(max(spikeMovFit(:)) - min(spikeMovFit(:)));  % Set the max value of the spike movie to 1
[Vout, corrimg, weightimg, offsetimg] = extractV(spikeMovFit, kernel);
subplot(2,2,1); plot(Vout); hold all; plot(kernel);
subplot(2,2,2); imshow(corrimg, []); title('Corr')
subplot(2,2,3); imshow(weightimg, []); title('Weight')
subplot(2,2,4); imshow(offsetimg, []); title('Offset')

% Set up the matrices to store the results
xsize = ncol;
ysize = nrow;
betamat2 = zeros(ysize, xsize, 4);  % Matrix of fit parameters for each pixel
cimat = zeros(ysize, xsize, 4,2); % Matrix of confidence intervals for each fit parameter at each pixel
rsquaremat = zeros(ysize, xsize);  % Matrix of r^2, indicating goodness of fit at each pixel
goodfit = ones(ysize, xsize);  % matrix indicating pixels where the fit succeeded

% Perform the fit!  This is the heart of the SNAPT algorithm.
for x = 1:xsize
    for y = 1:ysize
        dat = squeeze(spikeMovFit(y,x,:));  % Take the AP waveform at pixel y,x
        lastwarn('');
        % Perform four-parameter fit.  Input is k(t).  Output is a*k(c*(t-dt)) + b,
        % where beta = [a,b,c,dt] is the vector of fitting parameters.
        % This will generate warnings.  Don't worry.
        [betamat2(y,x, :), r, J, COVB, mse] = nlinfit(kernel, dat, @shiftkern, [1 0 1 0], statset('Robust', 'off'));
        if ~isempty(lastwarn)  % Flag pixels where fit failed
            goodfit(y,x)=0;
        end
        rsquaremat(y,x) = 1 - var(r)/var(dat);
        cimat(y,x,:,:) = nlparci(squeeze(betamat2(y,x,:)), r, 'covar', COVB, 'alpha', .318);  % these are the 68% confidence intervals
    end
    ['Completed column ' num2str(x)];
end
ampimg = betamat2(:,:,1);
baseimg = betamat2(:,:,2);
widthimg = 1./betamat2(:,:,3);
dtimg = betamat2(:,:,4);

% See where the fit converged
figure(13); clf;
imshow(goodfit,[], 'InitialMagnification', 'fit');
title('Dark pixels indicate failed fit')

% Check the raw amplitudes from the fit
figure(14); clf;
imshow(ampimg ,[prctile(ampimg(:),1), prctile(ampimg(:),99)], 'InitialMagnification', 'fit');
title('Raw fit results for AP amplitude');

% Set constraints on the fitting parameters.  Adjust these values as
% appropriate.
min_a = 0;  % Amplitude
min_c = .2;  % Time scale factor
max_c = 4;
min_dt = -4; % Time shift (in frames)
max_dt = 4;

goodpix =   (ampimg > min_a).*...
    (widthimg > min_c).*...
    (widthimg < max_c).*...
    (dtimg > min_dt).*...
    (dtimg < max_dt);
figure(13);clf;
imshow(goodpix,[], 'InitialMagnification', 'fit');
title('Bright pixels indicate that the fit satistfied constraints')

% Set the amplitude to zero where the fit parameters lay outside the constraints;
ampimg = ampimg.*goodpix;

% constrain the time scaling to be within the boundaries
widthimg(widthimg < min_c) = min_c;
widthimg(widthimg > max_c) = max_c;

% Constrain the time shift to be within the boundaries
dtimg(dtimg < min_dt) = min_dt;
dtimg(dtimg > max_dt) = max_dt;

% Look at the results
figure(17); clf; colormap('gray');
subplot(2,3,1); imshow(ampimg, [], 'InitialMagnification', 'fit'); title('Amplitude image'); freezeColors;
subplot(2,3,2); imshow(baseimg, [], 'InitialMagnification', 'fit'); title('Offset image'); freezeColors;
subplot(2,3,4); imshow(widthimg, [], 'InitialMagnification', 'fit'); title('width image'); freezeColors;
subplot(2,3,5); imshow(dtimg, [], 'InitialMagnification', 'fit'); title('Delay image'); freezeColors;
subplot(2,3,6); rsquare_map(rsquaremat, 0.8); title('R^2'); % This function uses a nice colormap to make results easier to interpret
unfreezeColors

% Make a pretty color image of the delay
figure(18); clf
dtColImg = grs2rgb(dtimg, colormap('jet'), 0, 1);
dtColImg = dtColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(dtColImg)
colorbar
title('Delay map (ms)')
saveas(gca, '.\snapt_results\matlab\delay map PCA.fig')
saveas(gca, '.\snapt_results\matlab\delay map PCA.png')

% Make a pretty color image of the spike width
figure(19); clf
widthColImg = grs2rgb(widthimg, colormap('jet'), .6, 1.4);
widthColImg = widthColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(widthColImg)
title('width map')
colorbar('Ticks',0:.25:1,...
    'TickLabels',{'0.6', '0.8', '1.0', '1.2', '1.4'})
saveas(gca, '.\snapt_results\matlab\width map PCA.fig')
saveas(gca, '.\snapt_results\matlab\width map PCA.png')

%%

% make an image of dF/F:
eigImgs = eigImgsSpikes;

avgImgS = imfilter(avgImg, fspecial('Gaussian', [4 4], 1.5), 'replicate');
dffImg = eigImgs(:,:,1)./avgImgS;
figure(5); clf
imshow(dffImg, [])
dffColImg = grs2rgb(dffImg, colormap('jet'), 0, 10);
dffColImg = dffColImg.*repmat(mat2gray(avgImg), [1 1 3]);
% this doesn't look great.
imshow(dffColImg)

%%
% try plotting spike height rel to pre-spike depolarization.
[~, tmp] = clicky(spikeMov, avgImg);
figure(21); clf
for j = 1:size(tmp,2);
    sig = tmp(:,j) - mean(tmp(1:20,j));
    sig = sig/max(sig);
    plot(sig); hold all
end; hold off
spikeAmpImg = (mean(spikeMovPCA(:,:,(Pulse + 1):(Pulse + 2)),3) - mean(spikeMovPCA(:,:,(Pulse-3):Pulse),3))./...
    (mean(spikeMovPCA(:,:,(Pulse-3):Pulse),3) - mean(spikeMovPCA(:,:,1:(Pulse/2)),3));
spikeAmpColImg = grs2rgb(spikeAmpImg, colormap('jet'), 0.0, 1.6);
spikeAmpColImg = spikeAmpColImg.*repmat(mat2gray(avgImg), [1 1 3]);
% This looks ok--it shows highest spike amplitude at the AIS?  Unclear how
% to interpret.
figure; clf
imshow2(spikeAmpColImg)

% Look at the ratio of the second to first PC amplitude.  Not clear what
% this measures.
figure(3); clf
for j = -2:.2:2;
    plot(V(:,1) + j*V(:,2)+ 3*j); hold all
end;
hold off

% Look at the ratio of the third to first PC amplitude.  Not clear what
% this measures.
for j = -.5:.05:.3;
    plot(V(:,1) + j*V(:,3)+ 3*j); hold all
end;
hold off

figure(5); clf
spikeShiftImg = -eigImgs(:,:,2)./eigImgs(:,:,1);
spikeShiftColImg = grs2rgb(spikeShiftImg, colormap('jet'), -.5, .5);
spikeShiftColImg = spikeShiftColImg.*repmat(mat2gray(eigImgs(:,:,1))*1.3, [1 1 3]);
subplot(2,2,1)
plot(-Pulse:Pulse, V(:,1:2)')
xlim([-50 50])
legend('PC1', 'PC2')
subplot(2,2,3)
imshow2(spikeShiftColImg)
title('PC2/PC1')

% Look at the spike width
spikeWidthImg = -eigImgs(:,:,3)./eigImgs(:,:,1);
spikeWidthColImg = grs2rgb(spikeWidthImg, colormap('jet'), -.5, .5);
spikeWidthColImg = spikeWidthColImg.*repmat(mat2gray(eigImgs(:,:,1))*1.3, [1 1 3]);
subplot(2,2,2)
plot(-Pulse:Pulse, V(:,[1 3])')
xlim([-50 50])
legend('PC1', 'PC3')
subplot(2,2,4)
imshow2(spikeWidthColImg)
title('PC3/PC1')

%% Now analyze the sub-threshold dynamics, ignoring the spikes.
% plot(squeeze(mean(mean(movBon))));  % the different epochs seem to have different subthreshold dynamics; check each one separately
movBonHP = movBon - imfilter(movBon, ones(1,1,200)/200, 'replicate');  % high pass filter at 5 Hz
plot(squeeze(mean(mean(movBonHP))));
%
% figure(28); clf
% for j = 1:6;
%     img = abs(mean(movBonHP(:,:,(j-1)*1800+2:j*1800).*movBonHP(:,:,(j-1)*1800+1:j*1800-1),3)).^.5;
%     subplot(2,3,j)
%     imshow2(img, []);
% end;

lpDt = 5; % high pass cutoff window.
movBonLP = imfilter(movBon, ones(1,1,lpDt)/lpDt, 'replicate');% low-pass filter the movie at 200 Hz.
movBonLP = movBonLP(:,:,1:lpDt:end);
movBonLP = movBonLP - imfilter(movBonLP, ones(1,1,200)/200, 'replicate');  % high pass filter at 1 Hz
% movBonLP = imfilter(movBonLP, fspecial('Gaussian', [4 4], 1.5), 'replicate');  % low pass filter a little in space
% clicky(movBonLP, avgImg)
dMovBonLP = movBonLP - repmat(mean(movBonLP, 3), [1 1 size(movBonLP, 3)]);
%
% figure(28); clf
% for j = 1:6;
%     img = abs(mean(dMovBonLP(:,:,(j-1)*1800/lpDt+2:j*1800/lpDt).*dMovBonLP(:,:,(j-1)*1800/lpDt+1:j*1800/lpDt-1),3)).^.5;
%     subplot(2,3,j)
%     imshow2(img, []);
% end;
%
% % Manually project out background noise
% [~, intensLP] = clicky(dMovBonLP, avgImg); % clicky whole cell first, then background
% covMat = intensLP'*intensLP;  % this is just a 2x2 matrix.  Use PCA to unmix cell signal and bkg signal
% [V, D] = eig(covMat);
% V = V(:,end:-1:1);
% eigTraces = intensLP*V;
% figure(16); clf
% plot(eigTraces)
% [out, scaleImgs] = SeeResiduals(dMovBonLP, eigTraces(:,:), 1); % check that the images correspond to cell and bkg.
% figure(17); clf
% for j = 1:3;
%     subplot(1,3,j);
%     imshow2(scaleImgs(:,:,j), []);
% end;
%
% % look at a variance image:
varImg1a = abs(mean(dMovBonLP(:,:,1:end).*dMovBonLP(:,:,1:end),3)).^.5;  % the original movie
varImg1b = abs(mean(dMovBonLP(:,:,2:end).*dMovBonLP(:,:,1:end-1),3)).^.5;  % the original movie
varImg1c = abs(mean(movBonHP(:,:,1:end).*movBonHP(:,:,1:end),3)).^.5;  % the original movie
varImg1d = abs(mean(movBonHP(:,:,3:end).*movBonHP(:,:,1:end-2),3)).^.5;  % the original movie

% % project out the nominal bkg noise:
% [out, scaleImgs] = SeeResiduals(dMovBonLP, eigTraces(:,1), 1);
% varImg2a = abs(mean(out(:,:,1:end).*out(:,:,1:end),3)).^.5;
% varImg2b = abs(mean(out(:,:,2:end).*out(:,:,1:end-1),3)).^.5;
%
% figure(18); clf
% subplot(2,2,1);
% imshow2(varImg1a, []);
% subplot(2,2,2);
% imshow2(varImg1b, []);
% subplot(2,2,3);
% imshow2(varImg2a, []);
% subplot(2,2,4);
% imshow2(varImg2b, []);
%
% figure(19); clf
% imshow2(std(spikeMovBon, [], 3), [])
% imshow2(sum(spikeMovBon(:,:,Pulse:Pulse+3), 3), [])
%
% figure(19); clf
% imshow2(std(spikeMovBon, [], 3)./varImg1a, [])

movBonLPV = tovec(dMovBonLP);
covMat = movBonLPV'*movBonLPV;
[V, D] = eig(covMat);
V = V(:,end:-1:1);
figure(1); clf
stackplot(V(:,1:20))
eigImgs = toimg(movBonLPV*V(:,1:20), nrow, ncol);
figure(22); clf
for j = 1:20;
    subplot(4,5,j)
    imshow2(eigImgs(:,:,j), [])
    title(num2str(j))
end;
plot(V(:,3)); hold all
plot(V(:,4)); hold off

% eigKeep = [2 3 5 8 11]; % the rest are motion or noise
eigKeep = [2 5:20]; % This keeps most of the structure, including blood flow
eigKeep = [2 5 8 11 15]  % this tries to get rid of blood flow.

nEigKeep = length(eigKeep);
figure(1);
for j = 1:nEigKeep;
    plot(V(:,eigKeep(j)) + j/4); hold all
    plot(round(spikeT(spikeT<10800)/lpDt), V(round(spikeT(spikeT<10800)/lpDt),eigKeep(j)) + j/4, 'r*');
    [mean(V(round(spikeT(spikeT<10800)/lpDt),eigKeep(j))) ...% average value during a spike
        std(V(round(spikeT(spikeT<10800)/lpDt),eigKeep(j)))/sqrt(sum(spikeT<10800)) ...  % SEM of voltage during a spike
        std(V(:,eigKeep(j)))]
end;
hold off

% resynthesize the PCA filtered movie
subThreshMovPCA = toimg(tovec(eigImgs(:,:,eigKeep))*(V(:,eigKeep)'), nrow, ncol);
playmov(subThreshMovPCA, .03, 1)
% There are clearly blood flow artifacts in parts of the movie.  Try to
% remove

% try plotting spike height rel to subthreshold amplitude.
% clicky(spikeMovPCA)
% subthreshImg = imfilter(eigImgs(:,:,2), fspecial('Gaussian', [4 4], .5), 'replicate');
intensHP = squeeze(mean(mean(movBonHP)));
plot(intensHP)
plot(-length(intensHP)+1:length(intensHP)-1, xcov(intensHP))
nStart = 1801;
nStop = 7200;


lpDt = 1; % high pass cutoff window.
movBonLP = imfilter(movBon(:,:,nStart:nStop), ones(1,1,lpDt)/lpDt, 'replicate');% low-pass filter the movie at 200 Hz.
movBonLP = movBonLP(:,:,1:lpDt:end);

% subthreshImg = mean(movBonHP(:,:,nStart + 10:nStop).*movBonHP(:,:,nStart:nStop-10),3);
subthreshImg = mean(movBonLP(:,:,1:end-1).*movBonLP(:,:,2:end),3);
subthreshImg = imfilter(abs(subthreshImg).^.5, fspecial('Gaussian', [4 4], 1), 'replicate');
% subthreshImg = (subthreshImg).^.5;
plot(squeeze(mean(mean(spikeMovPCA))))
spikeImg = mean((spikeMovPCA(:,:,Pulse-10:Pulse+5)), 3);
spikeImgS = imfilter(spikeImg, fspecial('Gaussian', [4 4], 1), 'replicate');
figure(23); clf
subplot(1,2,1); imshow2(spikeImgS, []);
subplot(1,2,2); imshow2(subthreshImg, []);
spikeAmpImg = spikeImgS./subthreshImg;
figure(24); clf
imshow2(spikeAmpImg, [prctile(spikeAmpImg(:), .2) prctile(spikeAmpImg(:), 99.9)])
% hist(spikeAmpImg(:), 100)
% spikeAmpImg = eigImgsSpikes(:,:,1)./std(subThreshMovPCA, [],3);
spikeAmpColImg = grs2rgb(spikeAmpImg, colormap('jet'), 0, prctile(spikeAmpImg(:), 99.95));
prctile(spikeAmpImg(:), 99.95); % 0.79
spikeAmpColImg = spikeAmpColImg.*repmat(mat2gray(spikeImgS)*2.5-.1, [1 1 3]);
figure(19); clf
imshow2(spikeAmpColImg)
cbh = colorbar;
set(cbh,'XTickLabel',{'0','0.16','0.32','0.48','0.64','0.8'})

saveas(gca, 'X:\Lab\Papers\In vivo\Short_Fig2\spikeAmpRelToSubthresh.fig')
saveas(gca, 'X:\Lab\Papers\In vivo\Short_Fig2\spikeAmpRelToSubthresh.png')

[~, intensHP] = clicky(movBonHP, avgImg);

%%
% Alternatively, use clicky to define ROIs, and ask which pixels are
% correlated with each ROI
movBonHP = movBon - imfilter(movBon, ones(1,1,300)/300, 'replicate');  % high pass filter at 3 Hz
[~, intensSub] = clicky(movBonHP, avgImg);
[out, scaleImgs] = SeeResiduals(movBonHP, intensSub, 1);
figure(25); clf
for j = 1:3;
    subplot(1,3,j)
    imshow2(scaleImgs(:,:,j), [])
end


covImg1 = mean(movBonHP.*repmat(reshape(intensSub(:,1), [1 1 length(intensSub)]), [nrow, ncol, 1]),3);
covImg2 = mean(movBonHP.*repmat(reshape(intensSub(:,2), [1 1 length(intensSub)]), [nrow, ncol, 1]),3);
covImg3 = mean(movBonHP.*repmat(reshape(intensSub(:,3), [1 1 length(intensSub)]), [nrow, ncol, 1]),3);
figure(20); clf
subplot(2,3,1); imshow2(covImg1, [])
subplot(2,3,2); imshow2(covImg2, [])
subplot(2,3,3); imshow2(covImg3, [])
subplot(2,3,4); imshow2(covImg2-covImg1, [])
subplot(2,3,5); imshow2(covImg3-covImg1, [])
subplot(2,3,6); imshow2(covImg3-covImg2, [])

corrImg1 = mean(movBonHP.*repmat(reshape(intensSub(:,1), [1 1 length(intensSub)]), [nrow, ncol, 1]),3)./(std(dMovBonLP, [], 3)*std(intensSub(:,1)));
corrImg2 = mean(movBonHP.*repmat(reshape(intensSub(:,2), [1 1 length(intensSub)]), [nrow, ncol, 1]),3)./(std(dMovBonLP, [], 3)*std(intensSub(:,2)));
corrImg3 = mean(movBonHP.*repmat(reshape(intensSub(:,3), [1 1 length(intensSub)]), [nrow, ncol, 1]),3)./(std(dMovBonLP, [], 3)*std(intensSub(:,3)));
figure(20); clf
subplot(2,3,1); imshow2(corrImg1, [])
subplot(2,3,2); imshow2(corrImg2, [])
subplot(2,3,3); imshow2(corrImg3, [])
subplot(2,3,4); imshow2(corrImg2-corrImg1, [])
subplot(2,3,5); imshow2(corrImg3-corrImg1, [])
subplot(2,3,6); imshow2(corrImg3-corrImg2, [])
% Looks mostly like noise.

% Look for bAP failures.
allMovHP = allMov - imfilter(allMov, ones(1, 1, 30)/30, 'replicate');
allSpikesMov = zeros(nrow, ncol);
for j = 1:nSpike;
    allSpikesMov = cat(3, allSpikesMov, allMovHP(:,:,(spikeT(j)-10):(spikeT(j)+10)));
end;
allSpikesMov(:,:,1) = [];
[~, spikeIntens] = clicky(allSpikesMov, avgImg);

somaTrace = reshape(spikeIntens(:,1), 21, nSpike);
subplot(1,3,1); plot(somaTrace, 'r-'); hold all
plot(mean(somaTrace,2), 'k-'); hold off
d1Trace = reshape(spikeIntens(:,2), 21, nSpike);
subplot(1,3,2); plot(d2Trace, 'r-'); hold all
plot(mean(d2Trace,2), 'k-'); hold off
d2Trace = reshape(spikeIntens(:,3), 21, nSpike);
subplot(1,3,3); plot(d2Trace, 'r-'); hold all
plot(mean(d2Trace,2), 'k-'); hold off
% compare the blue on vs blue off periods.  Same waveforms
plot(-10:10, mean(somaTrace, 2), -10:10, mean(d1Trace, 2), -10:10, mean(d2Trace, 2)); hold all
plot(-10:10, mean(somaTrace(:,1:46), 2), -10:10, mean(d1Trace(:,1:46), 2), -10:10, mean(d2Trace(:,1:46), 2));
plot(-10:10, mean(somaTrace(:,47:end), 2), -10:10, mean(d1Trace(:,47:end), 2), -10:10, mean(d2Trace(:,47:end), 2));
hold off

% Find the amplitude of each spike:
somaAmp = somaTrace(11,:) - mean(somaTrace(5:10,:));
d1Amp = d1Trace(12,:) - mean(d1Trace(6:11,:));
d2Amp = d2Trace(12,:) - mean(d2Trace(6:11,:));
figure(1); clf
plot(1:nSpike, somaAmp, 1:nSpike, d1Amp, 1:nSpike, d2Amp)

pIndx = find(d2Amp > prctile(d2Amp, 50));
fIndx = find(d2Amp < prctile(d2Amp, 50));
plot(-10:10, mean(somaTrace(:,pIndx), 2), -10:10, mean(d1Trace(:,pIndx), 2), -10:10, mean(d2Trace(:,pIndx), 2)); hold all;
plot(-10:10, mean(somaTrace(:,fIndx), 2), -10:10, mean(d1Trace(:,fIndx), 2), -10:10, mean(d2Trace(:,fIndx), 2)); hold off;

allSpikesMov = reshape(allSpikesMov, [nrow, ncol, 21, nSpike]);
playmov(mean(allSpikesMov(:,:,:,pIndx),4) - mean(allSpikesMov(:,:,:,fIndx),4), .1, 1)
figure(21); clf
imshow2(mean(allSpikesMov(:,:,12,pIndx),4) - mean(allSpikesMov(:,:,12,fIndx),4), [])

% Look at the amplitude of the subthreshold dynamics:
dMovBon = movBon - repmat(mean(movBon,3), [1 1 length(movBon)]);
vAmpImg = abs(mean(dMovBon(:,:,2:end).*dMovBon(:,:,1:end-1),3)).^.5;
imshow2(vAmpImg, [])
% Low pass filter in time


nROI = size(spikeIntensN,2);
cmap = generateColorSpecLocal(nROI);
for j = 1:nROI;
    subplot(2,1,1)
    plot((spikeIntensN(:,j)), 'Color', cmap(j,:)); hold all
    subplot(2,1,2)
    plot(cumsum(spikeIntensN(:,j)), 'Color', cmap(j,:)); hold all
end;
hold off
plot(nanmean(spikes,3))

spikeMovN = spikeMov - repmat(mean(spikeMov(:,:,95:100),3), [1 1 2*Pulse + 1]);
spikeMovN = spikeMovN./repmat(mean(spikeMov(:,:,95:100),3) - mean(spikeMov(:,:,1:20),3), [1 1 2*Pulse + 1]);
clicky(spikeMovN, avgImg);

tmax = 4;
tau = repmat(reshape(0:tmax, [1 1 tmax+1]), [nrow, ncol, 1]);
cmImg = sum(tau.*spikeMovN(:,:,100:(100+tmax)),3)./sum(spikeMovN(:,:,100:(100+tmax)),3);
cmImg(cmImg(:) > tmax) = tmax;
cmImg(cmImg(:) < 0) = 0;

hist(cmImg
imshow2(cmImg, [0 tmax])

playmov(spikeMovN(:,:,95:110), .1, 1)

spikeHtImg =


R=1;
Spikes1=cell(1,nRoi);
for i=1:nRoi;
    for j=1:size(spikeT{R},1);
        if spikeT{R}(j)>Pulse && spikeT{R}(j)<nframe-Pulse
            fs=Intens(spikeT{R}(j)-Pulse:spikeT{R}(j)+Pulse,i);
            f0=fs(1);
            Spikes1{i}(:,j)=(fs-f0)/f0;
        else
            Spikes1{i}(:,j)=NaN(1,Pulse*2+1);
        end
    end
end

%% Plot average spike
figure(1);clf;hold on
%Colors={'r','g','b','m','k'}
subplot(1,3,1);hold on
imshow2(RefIm,[]);
for i=1:nRoi;plot(ROI{i}(:,1),ROI{i}(:,2));end
for i=1:nRoi
    subplot(1,3,2);hold on
    plot(nanmean(Spikes1{i},2));
    ylabel('dF/F')
    title('0=soma')
    set(gca,'Xtick',[1:10:101],'XTickLabel',[-50:10:50])
    axis tight;
    %axis([35 65 -0.02 0.06])
    %xlim([35 65]);
    
    subplot(1,3,3);hold on
    plot(nanmean(Spikes{i},2));
    title('0=compartment')
    set(gca,'Xtick',[1:5:101],'XTickLabel',[-50:5:50])
    axis tight
    %axis([35 65 -0.02 0.06])
    %xlim([35 65]);
end

% Plot average spike for figure
figure(2);clf;hold on
set(gcf,'position',[sz(3)/10, sz(4)/5, sz(3)/4, sz(4)/5]);
imshow2(RefIm,[]);
for i=1:nRoi;
    plot(ROI{i}(:,1),ROI{i}(:,2));
end
axis tight
axis([20 112 1 80]);
plot([25 25+round(10/0.6)],[72 72],'w','LineWidth',2)

figure(3);clf;hold on
set(gcf,'position',[sz(3)/10, sz(4)/5, sz(3)/5, sz(4)/4]);
for i=1:nRoi
    dff=nanmean(Spikes1{i},2);
    plot(dff,'LineWidth',1.5);
    plot([49 51],[0 0],'k','LineWidth',2.5)
    plot([41 41],[0.01 0.04],'k','LineWidth',2.5)
    set(gca,'Xtick',[1:5:31],'XTickLabel',[-15:5:15])
    axis tight off
    xlim([41 62]);
end

% Single trial figure
figure(4);clf;hold on
set(gcf,'position',[sz(3)/10, sz(4)/5, sz(3)/5, sz(4)/4]);
T=1376;
% T=5771;
% T=9442;
Pulse=15;
for i=1:nRoi
    f1=Intens(T-Pulse:T+Pulse,i);
    f10=nanmean(f1(1:5));
    dff=(f1-f10)/f10;
    plot(mat2gray(dff),'LineWidth',1.5);
    plot([14 16],[-0.01 -0.01],'k','LineWidth',2.5)
    plot([9 9],[0.5 0.8],'k','LineWidth',2.5)
    set(gca,'Xtick',[1:5:31],'XTickLabel',[-15:5:15])
    axis tight off
    xlim([9 25])
end


% Stackplot
figure(5);clf;hold on
d=0.8;
for i=1:nRoi
    F2=mat2gray(Intens(1:end,i));
    %F2=mat2gray(F(1:end,i));
    plot(F2+d*i)
    % plot(SpikeT{i},F2(SpikeT{i})+d*i,'r*')
end


%% Find spike failures
Pulse=50;
figure(21);clf
figure(22);clf
figure(23);clf
figure(24);clf
for i=1:4;
    figure(21);hold on
    subplot(2,2,i);hold on
    plot(Spikes1{i})
    plot(nanmean(Spikes1{i},2),'k','Linewidth',2)
    axis tight
    Peak{i}=max(Spikes1{i}(49:54,:))-min(Spikes1{i});
    figure(22);hold on
    subplot(2,2,i);hold on
    hist(Peak{i},20)
    axis([-0.05 0.15 0 12]);
    SpikeTF{i}=spikeT{1}(Peak{i}<prctile(Peak{i},20));
    SpikeTG{i}=spikeT{1}(Peak{i}>prctile(Peak{i},80));
    for j=1:size(SpikeTF{i},1);
        fs=Intens(SpikeTF{i}(j)-Pulse:SpikeTF{i}(j)+Pulse,i);
        f0=fs(1);
        SpikesF{i}(:,j)=(fs-f0)/f0;
    end
    for j=1:size(SpikeTG{i},1);
        fs=Intens(SpikeTG{i}(j)-Pulse:SpikeTG{i}(j)+Pulse,i);
        f0=fs(1);
        SpikesG{i}(:,j)=(fs-f0)/f0;
    end
    figure(23);hold on
    subplot(2,2,i);hold on
    plot(nanmean(SpikesF{i},2),'b','Linewidth',2)
    plot(nanmean(SpikesG{i},2),'r','Linewidth',2)
    axis tight
    axis([0 101 -0.02 0.07])
    suptitle('Average per compartment')
    %xlim([35 65])
end

for i=1:nRoi
    for j=1:size(SpikeTF{i},1);
        fs=Intens(SpikeTF{4}(j)-Pulse:SpikeTF{4}(j)+Pulse,i);
        f0=fs(1);
        SpikesF4{i}(:,j)=(fs-f0)/f0;
    end
    for j=1:size(SpikeTG{i},1);
        fs=Intens(SpikeTG{4}(j)-Pulse:SpikeTG{4}(j)+Pulse,i);
        f0=fs(1);
        SpikesG4{i}(:,j)=(fs-f0)/f0;
    end
    figure(24);hold on
    subplot(2,2,i);hold on
    plot(nanmean(SpikesF4{i},2),'b','Linewidth',2)
    plot(nanmean(SpikesG4{i},2),'r','Linewidth',2)
    axis tight
    axis([0 101 -0.02 0.07])
    suptitle('Averaged based on compartment 4')
    %xlim([35 65])
end

% Check the Failed and good spikes
% Stackplot
figure(5);clf;hold on
d=1;
for i=1:nRoi
    F2=mat2gray(Intens(1:end,i));
    plot(F2+d*i)
    plot(SpikeTF{i},F2(SpikeTF{i})+d*i,'b*')
    plot(SpikeTG{i},F2(SpikeTG{i})+d*i,'r*')
end



%% Movie
smov=[];
Dsmov=[];
smovF=[];
smovF2=[];

T=1376;
N=10;
cmap1=jet(1024);
cmap2=gray(1024);
for i=1:N;
    %smov(:,:,i)=nanmean(mov1S(:,:,SpikeT{1}-3+i),3);
    smov(:,:,i)=nanmean(mov1S(:,:,SpikeTF{4}-3+i),3);
    %smov(:,:,i)=movS(:,:,T-3+i);
    smovF(:,:,i) = imfilter(smov(:,:,i), fspecial('Gaussian', [7, 7], 3), 'replicate');
    smovF2(:,:,i)= smovF(:,:,i).*RefIm;
end

m=smovF2;
m0=repmat(m(:,:,1),[1 1 N]);
Dsmov=mat2gray(m-m0);
for i=1:N
    RGB{i}=ind2rgb(uint16(Dsmov(:,:,i)*1024),cmap1);
end
RefIm2=ind2rgb(uint16(mat2gray(RefIm)*1024),cmap2);
figure(14);clf;hold on
set(gcf,'position',[sz(3)/10, sz(4)/5, sz(3)/3, sz(4)/3]);
for i=1:6
    subplot(2,3,i);hold on
    Im=mat2gray(mat2gray(RGB{i}-RGB{1})+mat2gray(RefIm2));
    Im(Im<0.04)=0;
    imshow2(Im,[]);
    %imagesc(flipud(Dsmov(:,:,i)),[0.2 1]);colormap('jet')
    axis image off
    axis([20 112 1 80]);
    %title(num2str(i-3))
    %     colormap(cmap1)
    %     colorbar
    %     if i==1;
    %         for j=1:Nroi;
    %             plot(ROI{j}(:,1),ROI{j}(:,2));
    %         end
    %     end
    if i==6;
        plot([25 25+round(10/0.6)],[72 72],'w','LineWidth',2)
        %colorbar
    end
end



