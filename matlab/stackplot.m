function stackplot(data, spacing)
% STACKPLOT: Plots columns of `data` as vertically stacked lines
% 
% Usage:
%   stackplot(data)
%   stackplot(data, spacing)
%
% Inputs:
%   - data: [T x N] matrix (e.g., time x components)
%   - spacing: vertical space between traces (default: auto)

    if nargin < 2
        spacing = 1.2 * max(abs(data(:)));  % automatic spacing
    end

    [T, N] = size(data);
    offset = (0:N-1) * spacing;  % vertical offsets

    hold on
    for i = 1:N
        plot(1:T, data(:, i) + offset(i), 'k');
    end
    hold off

    ylim([-spacing, offset(end) + spacing])
    xlabel('Time')
    ylabel('Component #')
    title('Stacked Component Traces')
end