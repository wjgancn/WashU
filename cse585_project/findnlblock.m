function [im_patches, nlblocks_index] = findnlblock(im_in, patch_size,... 
windows_size, number_nlblocks, step_nlblocks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Solve non-local blocks of input image.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convert *im_in* into overlapped *patches*.
fprintf("Begin: [Convert image to patches] \n");

feasible_row = size(im_in,1)- patch_size + 1;
feasible_col = size(im_in,2)- patch_size + 1;
number_im_patches = feasible_row * feasible_col;

im_patches = zeros(patch_size^2, number_im_patches, 'single');

k = 0;
for i  = 1:patch_size
    for j  = 1:patch_size
        k    =  k+1;

        im_patch = im_in(i:end-patch_size+i,j:end-patch_size+j);
        im_patches(k, :) =  im_patch(:)';
        
    end
end
im_patches = im_patches';
im_patches_squaresum = sum(im_patches.^2, 2);

im_patches_index = (1: number_im_patches);
im_patches_index = reshape(im_patches_index, [feasible_row, feasible_col]);

fprintf("End: [Convert image to patches] \n");

%% Find non-local blocks of each image patches
fprintf("Begin: [Compute non-local blocks] \n");

nbl_feasible_row = uint16(feasible_row / step_nlblocks);
nbl_feasible_col = uint16(feasible_col / step_nlblocks);

nlblocks_index = zeros(number_nlblocks, nbl_feasible_row * nbl_feasible_col);

k = 0;
for i = 1 : nbl_feasible_row
    
    for j = 1 : nbl_feasible_col
        
        % Prevent Crossing
        rowmin = max(i * step_nlblocks  - windows_size, 1);
        rowmax = min(i * step_nlblocks + windows_size, feasible_row);
        colmin = max(j * step_nlblocks - windows_size, 1);
        colmax = min(j * step_nlblocks  + windows_size, feasible_row);
        
        windows_indexs = im_patches_index(rowmin : rowmax, colmin : colmax);
        windows_indexs = windows_indexs(:);
        center_patch_indexs = (j * step_nlblocks - 1) * feasible_row + i  * step_nlblocks-1;
        
        windows_patches = im_patches(windows_indexs, :);
        center_patch = im_patches(center_patch_indexs, :);
        windows_patches_squaresum = im_patches_squaresum(windows_indexs, :);
        center_patch_squaresum = im_patches_squaresum(center_patch_indexs, :);
        
        distance =  (windows_patches_squaresum + center_patch_squaresum - ...
            2*windows_patches * center_patch') / (patch_size ^ 2);
        
        [~, sorted_index] = sort(distance);
        
        k = k + 1;
        nlblocks_index(:, k) = ...
            windows_indexs(sorted_index(1 : number_nlblocks));
        
        
    end
    
    fprintf("Procsee: [Compute non-local blocks] .. %.3f Per \n", 100*i/nbl_feasible_row);
    
end

fprintf("End: [Compute non-local blocks] \n");

end