function im_out = GBsimple(im_in)
    
    im_in_width = size(im_in, 1);
    im_in_height = size(im_in, 1);
    
    im_in = double(im_in);
    std_in = mean(std(im_in));
    
    patch_size = 9;
    windows_size = 30;
    number_nlblocks = 40;
    
    fprintf("Begin: [Group-Based Method] \n");
    
    [im_patches, im_blockes] = findnlblock(im_in, patch_size, ...
    windows_size, number_nlblocks, 8);

    im_patches_updated = zeros( size(im_patches) );
    im_patches_weight = zeros( size(im_patches) );

    for i = 1 : size(im_blockes, 2)

        im_patch_updated = im_patches(im_blockes(:, i), :);
        
        img_patch_upadtaed_processed = ADMMlasso(im_patch_updated, std_in);
        
        im_patches_updated(im_blockes(:, i), :) = ...
            im_patches_updated(im_blockes(:, i), :) + img_patch_upadtaed_processed;

         im_patches_weight(im_blockes(:, i), :) = ...
            im_patches_weight(im_blockes(:, i), :) + 1;
            
        fprintf("Procsee: [Group-Based Method] .. %.3f Per \n", 100*i/size(im_blockes, 2));
    
    end
        
    fprintf("End: [Group-Based Method] \n");

    fprintf("Begin: [Reconstruct Image] \n");
    im_out = zeros( size(im_in) );
    im_out_weight = zeros( size(im_in) );

    k = 0;
    for i = 1 : patch_size
        for j = 1 : patch_size
            k = k + 1;

            im_out(i:end-patch_size+i, j:end-patch_size+j) = ...
                im_out(i:end-patch_size+i, j:end-patch_size+j)...
                + reshape(im_patches_updated(:, k), ...
                [im_in_width - patch_size + 1, im_in_height - patch_size + 1]);

            im_out_weight(i:end-patch_size+i, j:end-patch_size+j) = ...
                im_out_weight(i:end-patch_size+i, j:end-patch_size+j)...
                + reshape(im_patches_weight(:, k),...
                [im_in_width - patch_size + 1, im_in_height - patch_size + 1]);

        end
    end
    
    im_out = im_out ./ im_out_weight;
 
    fprintf("End: [Reconstruct Image] \n");
    
    fprintf("All Processes are Done!!! \n");
    
end