%% PS3 prob1 - Read Data
fprintf("Compute aligned faces from PS3' s code firstly.\n")
clear all; %#ok<CLALL>
addpath('./code/');
addpath('./code/dat/');

allFiles = dir('./code/dat/107*.pts');
numFiles = length(allFiles);
for i=1:numFiles
  sPts{i} = readPoints( strcat('dat/',allFiles(i).name ) ); %#ok<SAGROW>
end

%% PS3 prob1 - Align Iteration

uPts = sPts{1};

max_iter = 20;

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
    [uPts, ~] = getAlignedPts(sPts{1}, uPts);
    
    % Record Data
    uPts_norm = norm(uPts); 
    uPts_change_norm = norm(uPts - uPts_last); 

    fprintf('Iter: [%d]. L2-Norm of Xu is: [%.7f]. L2-Norm of Change of Xu is: [%.7f]. \n', ...
        iter, uPts_norm, uPts_change_norm);
    
    % Check if Converged
    if uPts_change_norm < 1e-6
        fprintf('Iter ends at Step: %d. \n', iter);
        break;
    end
end

%% PS4 prob2
% The hatPts will be the aligned faces
features = 68*2;
hatPtsmatrix = zeros([features, numFiles]);

for i = 1:numFiles
    hatPtsmatrix(:, i) = hatPts{i}(:);
end

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions: [%.4f] \n", mseError);
figure(1); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data")
% saveas(gcf,'fromOriginal.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, 1);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for 1 percentage original data dimensions: [%.4f] \n", mseError);
figure(2); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From 1 percentage original data")
% saveas(gcf,'from1PerOriginal.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, 4);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for 3 percentage original data dimensions: [%.4f] \n", mseError);
figure(3); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From 3 percentage original data")
% saveas(gcf,'from3PerOriginal.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, 13);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for 10 percentage original data dimensions: [%.4f] \n", mseError);
figure(4); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From 10 percentage original data")
% saveas(gcf,'from10PerOriginal.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features, 'L2WeightRegularization',0.1);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions with 0.1 L2WeightRegularization: [%.4f] \n", mseError);
figure(5); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data with 0.1 L2WeightRegularization")
% saveas(gcf,'fromOriginalwith0.1.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features, 'L2WeightRegularization',0.2);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions with 0.2 L2WeightRegularization: [%.4f] \n", mseError);
figure(6); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data with 0.2 L2WeightRegularization")
% saveas(gcf,'fromOriginalwith0.2.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features, 'L2WeightRegularization',0.3);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions with 0.3 L2WeightRegularization: [%.4f] \n", mseError);
figure(7); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data with 0.3 L2WeightRegularization")
% saveas(gcf,'fromOriginalwith0.3.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features, 'L2WeightRegularization',0.4);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions with 0.4 L2WeightRegularization: [%.4f] \n", mseError);
figure(8); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data with 0.4 L2WeightRegularization")
% saveas(gcf,'fromOriginalwith0.4.png')

% ====================================== %
autoenc = trainAutoencoder(hatPtsmatrix, features, 'L2WeightRegularization',0.5);
hatPtsmatrixReconstructed = predict(autoenc,hatPtsmatrix);

mseError = mse(hatPtsmatrix-hatPtsmatrixReconstructed);
fprintf("mseError for original data dimensions with 0.5 L2WeightRegularization: [%.4f] \n", mseError);
figure(9); 
drawFaceParts( -reshape(hatPtsmatrixReconstructed(:, 1), [68, 2]), 'r-');
title("From original data with 0.5 L2WeightRegularization")
% saveas(gcf,'fromOriginalwith0.5.png')
