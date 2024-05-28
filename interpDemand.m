prompt = 'Enter powergrid IEEE24, IEEE39, IEEE118, UK:  ';
user_input = input(prompt, 's');
% Construct the full folder path
folder_path = fullfile('13_Power_system', user_input);

% Change the current folder
cd(folder_path);
figure; hold on; % Create a new figure and hold the plot
matrix = load('hourlyDemandBus.mat');
matrix = matrix.hourlyDemandBus
data_points = 35040;
hourlyDemandBusnew = zeros(size(matrix,1), data_points);
for i = 1:size(matrix, 1)
    if ~all(matrix(i, :) == 0) % Check if the row contains all zeros
        original_indices = linspace(1, length(matrix(i, :)), length(matrix(i, :)));

% Define the new indices
        new_indices = linspace(1, length(matrix(i, :)), data_points);

% Perform linear interpolation
        interpolated_data = interp1(original_indices, matrix(i, :), new_indices);
        hourlyDemandBusnew(i,:) = interpolated_data;
        plot(interpolated_data); % Plot the row
    end
end
save hourlyDemandBusnew.mat hourlyDemandBusnew;

hold off; % Release the hold
xlabel('Index'); % Label x-axis
ylabel('Value'); % Label y-axis
title('Plot of non-zero rows'); % Set title