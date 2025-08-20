%%
clc;clearvars;close all
%%
movname = '..\vol.tif';
info = imfinfo(movname);
nframes = numel(info);
mov = zeros(info(1).Height, info(1).Width, nframes, 'uint16');
tic
for k = 1:nframes
    mov(:,:,k) = imread(movname, 'Index', k, 'Info', info);
end
toc

%%
filename = '../efrat_video.raw';
width = 236;       % image width in pixels
height = 146;      % image height in pixels
nframes = 15000;     % number of frames
datatype = 'uint16';  % match your data type (commonly 'uint16' or 'uint8')

% Open and read the raw binary file
fid = fopen(filename, 'r');
raw_data = fread(fid, width * height * nframes, datatype);
fclose(fid);

% Reshape into 3D movie [Height x Width x Time]
mov = reshape(raw_data, [width, height, nframes]);
mov = permute(mov, [2 1 3]);  % to get [height x width x nframes]

%% Preprocessing
mov = -double(mov);

% check the noise:
intens = squeeze(mean(mean(mov)));
t = 1:length(intens);
figure
plot(t, intens, 'k-');
title("Average intensity over time");

% get rid of the bad frames;
intensHP = intens - smoothdata(intens, 'sgolay', 10);
figure;
plot(intensHP)
badFrames = find(intensHP < -3.5);
plot(t, intens, 'k-', badFrames, intens(badFrames), 'r*')
mov(:,:,badFrames) = [];
title("Bad Frames")

% start again
avgImg = mean(mov, 3);
avgImg = avgImg - prctile(avgImg(:), 20);

% check the noise:
intens = squeeze(mean(mean(mov)));
t = 1:length(intens);
figure;
plot(t, intens, 'k-');
title("Clean")

% check the power spectrum of the noise
intensPSD = fft(intens).*conj(fft(intens));
freq = (0:length(intensPSD)-1)*1000/length(intensPSD);

% the systematic noise is mostly at > 150 Hz
smoothT = 15;  %ms. filter at ~60 Hz
intensHP = intens - smooth(intens, smoothT, 'sgolay', 2);
figure;
plot(t, intensHP)
title("Average intensity highpass")

figure;
intensHP_PSD = fft(intensHP).*conj(fft(intensHP));
semilogy(freq, intensPSD, freq, intensHP_PSD);
legend('Raw PSD', 'High-pass (residual) PSD');
xlabel('Frequency (Hz)');
ylabel('Power');
title("original vs. highpass intensity PSDs");

% Get rid of the first few frames where the filter doesn't work:
mov_temp = mov(:,:,10:end);
intensHP2 = intensHP(10:end);
[nrow, ncol, nframe2]=size(mov_temp);
t2 = 1:nframe2;

% project out the high freq noise and any linear or quadratic drift
regressors = [intensHP2'; t2 - mean(t2); (t2 - mean(t2)).^2];
regressors = regressors';  % [T x R]
regressors = regressors - mean(regressors);  % center regressors

% Precompute pseudoinverse for efficiency
X = regressors;
pinvX = pinv(X);  % [R x T]

movie_clean = zeros(size(mov_temp));
scaleImgs = zeros(nrow, ncol, size(regressors, 2));

for y = 1:nrow
    for x = 1:ncol
        ytrace = squeeze(mov_temp(y, x, :));  % [T x 1]
        beta = pinvX * ytrace;             % [R x 1]
        fit = X * beta;                    % [T x 1]
        movie_clean(y, x, :) = ytrace - fit;
        scaleImgs(y, x, :) = beta;
    end
end

for j = 1:3
    subplot(2, 2, j)
    imagesc(scaleImgs(:, :, j)); 
    axis image off; 
    colormap(gray); 
    colorbar;
    title(['Regressor ' num2str(j)]);
end

figure;
intens_clean = squeeze(mean(mean(movie_clean)));
plot(intens, DisplayName="Original"); hold on;
plot(intens_clean, DisplayName="Clean")
legend;
title("Intensity before and after cleaning");

%% ROI analysis

[ROI, roi_intens]=clicky3(movie_clean, avgImg);  % Start at soma

% convert to units of dF/F
f0 = apply_clicky(ROI, avgImg);
roi_intens = roi_intens./repmat(f0, [length(roi_intens), 1]);

plot(roi_intens)

%% Find Spikes
% All the spikes are basically in the soma only.
nframe=length(roi_intens);
nRoi=size(roi_intens,2);

Pulse=50;  % time to look before and after each spike
t=1:nframe;
f=mat2gray(roi_intens(:,1));  % Use the first clicky ROI for spike finding
fhp=f-smoothdata(f,'movmean',40);

spike_frames = spikefind2(fhp+1,10,1.05); pause(1); close
nSpike = length(spike_frames);
spikes = zeros(2*Pulse+1, nRoi, nSpike);
spikeMov = zeros(nrow, ncol, 2*Pulse + 1);
c = 0; % spike counter

for j=1:nSpike  % Calculate the average spike waveform in each ROI
    if spike_frames(j)>Pulse && spike_frames(j)<nframe-Pulse
        fs=roi_intens((spike_frames(j)-Pulse):(spike_frames(j)+Pulse),:);
        f0=mean(fs(1:10,:),1);
        spikes(:,:,j) = fs - f0;

        chunk = movie_clean(:,:,(spike_frames(j)-Pulse):(spike_frames(j)+Pulse));
        baseline = mean(chunk(:,:,1:10), 3);  
        chunk = chunk - baseline;              % subtract baseline per pixel
        spikeMov = spikeMov + chunk;           % accumulate peri-spike movie
        c = c + 1;
    else
        spikes(:,:,j)=NaN(2*Pulse+1, nRoi);
    end
end
spikeMov = spikeMov/c;

playmov(spikeMov(:,:,(Pulse-10):(Pulse+10)), .1, 1)
title("Spike Movie");

%%
[~, spike_traces] = clicky3(spikeMov);
figure;
plot(mat2gray(spike_traces(:,1))); hold on;
plot(mat2gray(spike_traces(:,2))); hold on;
plot(mat2gray(spike_traces(:,3))); 
title("STA in each ROI");

%% 
%Select a time just around the peak to use for the SNAPT fit
[~, kernel] = clicky(spikeMov, avgImg);  % clicking on the soma and proximal dendrite.

kernel = kernel((Pulse-13):(Pulse+17));
kernel = mat2gray(kernel);  % Set the kernel to range from 0 to 1

spikeMovFit = spikeMov(:,:,(Pulse-13):(Pulse+17));
spikeMovFit = imfilter(spikeMovFit, fspecial('Gaussian', [4 4], 1.5), 'replicate');  % filter a bit to smooth noise
spikeMovFit = (spikeMovFit - min(spikeMovFit(:)))/(max(spikeMovFit(:)) - min(spikeMovFit(:)));  % Set the max value of the spike movie to 1
[Vout, corrimg, weightimg, offsetimg] = extractV(spikeMovFit, kernel);
subplot(2,2,1); 
plot(Vout, DisplayName="Vout"); hold on; 
plot(kernel, DisplayName="Kernel");
legend();

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
figure;
imshow(goodfit,[], 'InitialMagnification', 'fit');
title('Dark pixels indicate failed fit')

% Check the raw amplitudes from the fit
figure;
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
figure;
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
figure;
colormap('gray');
subplot(2,3,1); imshow(ampimg, [], 'InitialMagnification', 'fit'); title('Amplitude image'); freezeColors;
subplot(2,3,2); imshow(baseimg, [], 'InitialMagnification', 'fit'); title('Offset image'); freezeColors;
subplot(2,3,4); imshow(widthimg, [], 'InitialMagnification', 'fit'); title('width image'); freezeColors;
subplot(2,3,5); imshow(dtimg, [], 'InitialMagnification', 'fit'); title('Delay image'); freezeColors;
subplot(2,3,6); rsquare_map(rsquaremat, 0.8); title('R^2'); % This function uses a nice colormap to make results easier to interpret
unfreezeColors

% Make a pretty color image of the delay
figure;
dtColImg = grs2rgb(dtimg, colormap('jet'), 0, 1);
dtColImg = dtColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(dtColImg)
colorbar
title('Delay map (ms)')
saveas(gcf, '..\snapt_results\matlab\delay map.fig')
saveas(gcf, '..\snapt_results\matlab\delay map.png')

% Make a pretty color image of the spike width
figure;
widthColImg = grs2rgb(widthimg, colormap('jet'), .6, 1.4);
widthColImg = widthColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(widthColImg)
title('width map')
colorbar('Ticks',0:.25:1,...
    'TickLabels',{'0.6', '0.8', '1.0', '1.2', '1.4'})
saveas(gcf, '..\snapt_results\matlab\width map.fig')
saveas(gcf, '..\snapt_results\matlab\width map.png')

%% PCA filter spikeMovFit

dSpikeMov = spikeMov - repmat(mean(spikeMov(:,:,:), 3), [1 1 size(spikeMov(:,:,:),3)]);
spikeMovV = reshape(dSpikeMov, [], size(dSpikeMov,3));
covMat = spikeMovV'*spikeMovV;

[VSpikes, ~] = eig(covMat);
VSpikes = VSpikes(:,end:-1:1);
VSpikes(:,1) = -VSpikes(:,1);  %manually set the signs of some of the terms to be more intuitive
VSpikes(:,3) = -VSpikes(:,3);
figure;
stackplot(VSpikes(:,1:20))
title('Top 20 Temporal Eigenvectors (Spike Movie)')

projSpikes  = spikeMovV*VSpikes(:,1:9);
eigImgsSpikes = reshape(projSpikes , nrow, ncol, size(projSpikes , 2));

figure;
for j = 1:9
    subplot(3,3,j)
    imshow(eigImgsSpikes(:,:,j), [])
end
sgtitle('Spatial Maps of First 9 Spike Components')

spikesFiltPCA = eigImgsSpikes(:,:,1:3);
spikesFiltPCAVec = reshape(spikesFiltPCA, [], size(spikesFiltPCA, 3)) * VSpikes(:,1:3)';
spikeMovPCA = reshape(spikesFiltPCAVec, nrow, ncol, size(spikesFiltPCAVec, 2));  % reconstruct the PCA filtered spike movie

% Make a nice movie of the PCA-filtered response.
nF = size(spikeMovPCA, 3);
baseline = mean(spikeMovPCA(:,:,1:10),3);
spikeMovPCA = spikeMovPCA - baseline;
spikeMovPCA = spikeMovPCA/max(spikeMovPCA(:));
playmov(spikeMovPCA, .1, 1)  

ampImg = mat2gray(std(spikeMovPCA, [], 3));
colormov = repmat(0.5 * mat2gray(ampImg), [1 1 3 nF]);
for j = 1:nF
    colormov(:,:,:,j) = colormov(:,:,:,j) + 3*ampImg.*grs2rgb(spikeMovPCA(:,:,j), colormap('jet'), 0, .4);
end
for j = 1:6
    subplot(2,3,j);
    imshow(colormov(:,30:110,:,j+48))
end
sgtitle('PCA-filtered Spike Movie – Montage of Frames 49–54')

saveas(gcf, '..\snapt_results\matlab\montage.fig')
saveas(gcf, '..\snapt_results\matlab\montage.png')

% Create video writer object
v = VideoWriter('..\snapt_results\matlab\spike_movie.mp4','MPEG-4');
v.FrameRate = 10;   % 10 frames per second (adjust as needed)
open(v);

figure;
for j = 1:nF
    imshow(colormov(:,:,:,j));
    title('PCA-filtered Spike Movie')
    text(5, 5, [num2str(j) ' ms'], 'Color', 'White', 'FontSize', 18, 'FontWeight','bold');
    
    % Grab frame from figure
    frame = getframe(gca);
    writeVideo(v, frame);
end

close(v);

%% Try SNAPT again, on the PCA filtered movie:
spikeMovFit = spikeMovPCA(:,:,(Pulse-13):(Pulse+17));

%Select a time just around the peak to use for the SNAPT fit
[~, kernel] = clicky(spikeMovPCA, avgImg);  % clicking on the soma and proximal dendrite.
kernel = kernel((Pulse-13):(Pulse+17));
kernel = mat2gray(kernel);  % Set the kernel to range from 0 to 1

spikeMovFit = imfilter(spikeMovFit, fspecial('Gaussian', [4 4], 1.5), 'replicate');  % filter a bit to smooth noise
spikeMovFit = (spikeMovFit - min(spikeMovFit(:)))/(max(spikeMovFit(:)) - min(spikeMovFit(:)));  % Set the max value of the spike movie to 1
[Vout, corrimg, weightimg, offsetimg] = extractV(spikeMovFit, kernel);
subplot(2,2,1); plot(Vout); hold on; plot(kernel);
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
figure;
imshow(goodfit,[], 'InitialMagnification', 'fit');
title('Dark pixels indicate failed fit')

% Check the raw amplitudes from the fit
figure;
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
figure;
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
figure;
colormap('gray');
subplot(2,3,1); imshow(ampimg, [], 'InitialMagnification', 'fit'); title('Amplitude image'); freezeColors;
subplot(2,3,2); imshow(baseimg, [], 'InitialMagnification', 'fit'); title('Offset image'); freezeColors;
subplot(2,3,4); imshow(widthimg, [], 'InitialMagnification', 'fit'); title('width image'); freezeColors;
subplot(2,3,5); imshow(dtimg, [], 'InitialMagnification', 'fit'); title('Delay image'); freezeColors;
subplot(2,3,6); rsquare_map(rsquaremat, 0.8); title('R^2'); % This function uses a nice colormap to make results easier to interpret
unfreezeColors

% Make a pretty color image of the delay
figure;
dtColImg = grs2rgb(dtimg, colormap('jet'), 0, 1);
dtColImg = dtColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(dtColImg)
colorbar
title('Delay map (ms)')
saveas(gcf, '..\snapt_results\matlab\delay map PCA.fig')
saveas(gcf, '..\snapt_results\matlab\delay map PCA.png')

% Make a pretty color image of the spike width
figure;
widthColImg = grs2rgb(widthimg, colormap('jet'), .6, 1.4);
widthColImg = widthColImg.*repmat(mat2gray(ampimg)*3 - .1, [1 1 3]);
imshow(widthColImg)
title('width map')
colorbar('Ticks',0:.25:1,...
    'TickLabels',{'0.6', '0.8', '1.0', '1.2', '1.4'})
saveas(gcf, '..\snapt_results\matlab\width map PCA.fig')
saveas(gcf, '..\snapt_results\matlab\width map PCA.png')


%% Build an interpolated time axis (5x finer than original)
T = length(kernel);
tInterp = linspace(1, T, 3*T);   % 5x temporal upsampling
nF = length(tInterp);

% Allocate interpolated SNAPT movie
spikeMovInterp = zeros(nrow, ncol, nF);

for y = 1:nrow
    for x = 1:ncol
        beta = squeeze(betamat2(y,x,:));
        if any(isnan(beta))
            continue
        end
        a  = beta(1);
        b  = beta(2);
        c  = beta(3);
        dt = beta(4);

        % Shifted + scaled kernel
        tShift = (tInterp - dt) * c;

        % Clip to kernel range
        tShift(tShift < 1) = 1;
        tShift(tShift > T) = T;

        % Interpolate kernel
        kInterp = interp1(1:T, kernel, tShift, 'linear', 'extrap');

        % Construct fitted trace
        spikeMovInterp(y,x,:) = a * kInterp + b;
    end
end

% Normalize movie for visualization
baseline = mean(spikeMovInterp(:,:,1:round(T/4)), 3);

% Subtract baseline from each frame
spikeMovInterpZ = spikeMovInterp - baseline;

% Robust scaling: clip at 1st–99th percentile to avoid outliers
lims = prctile(spikeMovInterpZ(:), [1 99]);
spikeMovInterpZ = (spikeMovInterpZ - lims(1)) ./ (lims(2) - lims(1));

% Ensure all values are between [0,1]
spikeMovInterpZ = min(max(spikeMovInterpZ,0),1);

spikeMovInterp = spikeMovInterpZ;  % replace original

% Normalize anatomy image once
avgImgN = mat2gray(avgImg);
avgRGB  = repmat(avgImgN, [1 1 3]);

% Video writer
v = VideoWriter('../snapt_results/matlab/snapt_interp_overlay.avi');
v.FrameRate = 20;   % adjust playback speed
open(v);

figure;
for j = 1:nF
    % Frame normalized to [0 1]
    frameNorm = spikeMovInterp(:,:,j);

    % Convert to jet RGB
    frameRGB  = ind2rgb(gray2ind(frameNorm, 256), jet(256));

    % Overlay: anatomy + colored activity
    overlay = 0.6*avgRGB + 0.8*frameRGB .* avgRGB;

    % Optional preview
    imshow(overlay);
    title(sprintf('Interpolated SNAPT frame %d / %d', j, nF));
    drawnow;

    % Write frame to video
    writeVideo(v, im2frame(im2uint8(overlay)));
end

close(v);
disp('Movie saved: snapt_interp_overlay.avi')