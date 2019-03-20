%% Read Data
clear all; %#ok<CLALL>
addpath('./code/');
addpath('./code/dat/');

allFiles = dir('./code/dat/107*.pts');
numFiles = length(allFiles);
for i=1:numFiles
  sPts{i} = readPoints( strcat('dat/',allFiles(i).name ) ); %#ok<SAGROW>
end

width = size(sPts{1}, 1);
height = size(sPts{1}, 2);

data_length = size(sPts{1}(:), 1);
Pts = zeros([data_length, numFiles]);
for i=1:numFiles
  Pts(:, i) = sPts{i}(:); 
end

%% Slove PCA
Pts_mean = mean(Pts, 2);
Pts_nomean = Pts - Pts_mean;
Pts_cov = cov(Pts');
[V,D] = eig(Pts_cov);
eigval = diag(D);

% sort eigenvalues/eigenvectors in descending order 
eigval = eigval(end:-1:1); 
V = fliplr(V);

% choice the most important three eigenvectors
v1 = V(:, 1); v2 = V(:, 2); v3 = V(:, 3); % Get eigenvectors

% Reconstruct data face data
rec1 = zeros([data_length, numFiles]);
rec2 = zeros([data_length, numFiles]);
rec3 = zeros([data_length, numFiles]);

for i=1:numFiles
  rec1(:, i) = ((sPts{i}(:) - Pts_mean)' * v1) * v1 + Pts_mean; 
  rec2(:, i) = ((sPts{i}(:) - Pts_mean)' * v2) * v2 + Pts_mean; 
  rec3(:, i) = ((sPts{i}(:) - Pts_mean)' * v3) * v3 + Pts_mean; 
end 

%% Visualization
rec1_mean = mean(rec1, 2);
rec1_std = std(rec1, 0, 2);

rec2_mean = mean(rec2, 2);
rec2_std = std(rec2, 0, 2);

rec3_mean = mean(rec3, 2);
rec3_std = std(rec3, 0, 2);

figure(1)
drawFaceParts( -reshape(rec1_mean + rec1_std, [width, height]), 'r-');
drawFaceParts( -reshape(rec1_mean - rec1_std, [width, height]), 'k-');
axis off
axis equal

figure(2)
drawFaceParts( -reshape(rec2_mean + rec2_std, [width, height]), 'r-');
drawFaceParts( -reshape(rec2_mean - rec2_std, [width, height]), 'k-');
axis off
axis equal

figure(3)
drawFaceParts( -reshape(rec3_mean + rec3_std, [width, height]), 'r-');
drawFaceParts( -reshape(rec3_mean - rec3_std, [width, height]), 'k-');
axis off
axis equal