%% Read Data
clear all; %#ok<CLALL>
addpath('./code/');
addpath('./code/dat/');

allFiles = dir('./code/dat/107*.pts');
numFiles = length(allFiles);
for i=1:numFiles
  sPts{i} = readPoints( strcat('dat/',allFiles(i).name ) ); %#ok<SAGROW>
end

%% Iter
% Initial
uPts = sPts{1};

max_iter = 100;
for iter = 1:max_iter
    % Compute Align
    uPts_last = uPts;
    for i=1:numFiles
      [hatPts{i}, ~] = getAlignedPts(uPts, sPts{i}); %#ok<SAGROW>
    end

    % Calculate the new mean
    uPts = 0;
    for i=1:numFiles
        uPts = uPts + hatPts{i};
    end
    uPts = uPts ./ numFiles;
    
    % Align Xu to X1
    uPts = getAlignedPts(sPts{1}, uPts);
    
    % Record Data
    uPts_norm = norm(uPts); 
    uPts_change_norm = norm(uPts - uPts_last); 

    fprintf('Iter: [%d]. L2-Norm of Xu is: [%.7f]. L2-Norm of Change of Xu is: [%.7f] \n', ...
        iter, uPts_norm, uPts_change_norm);
    
    % Check if Converged
    if uPts_change_norm < 1e-6
        fprintf('Iter End \n');
        break;
    end

end

%% Plot
figure
drawFaceParts( -sPts{1}, 'r-');
drawFaceParts( -uPts, 'k-');
axis off
axis equal
