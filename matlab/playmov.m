function playmov(movie, frametime, contrast_flag)
% Play a 3D movie (height x width x frames)
% - frametime: time between frames (in seconds)
% - contrast_flag:
%       0 = no contrast scaling
%       1 = scale each frame independently
%       2 = scale entire movie globally (recommended)

    if nargin < 2, frametime = 0.1; end
    if nargin < 3, contrast_flag = 2; end

    [nrow, ncol, nframes] = size(movie);
    
    figure; clf
    colormap(gray);

    switch contrast_flag
        case 0
            clim = [0 1];  % default grayscale range
        case 1
            % Per-frame scaling
            for k = 1:nframes
                imagesc(movie(:, :, k));
                axis image off;
                title(sprintf('Frame %d / %d', k, nframes));
                drawnow;
                pause(frametime);
            end
            return
        case 2
            % Global scaling
            clim = [min(movie(:)), max(movie(:))];
    end

    for k = 1:nframes
        imagesc(movie(:, :, k), clim);
        axis image off;
        title(sprintf('Frame %d / %d', k, nframes));
        drawnow;
        pause(frametime);
    end
end
