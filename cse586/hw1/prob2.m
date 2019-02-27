% 2. Frequency smoothing

img = imread('lenaNoise.png');

img10 = main(img, 10);
img20 = main(img, 20);
img40 = main(img, 40);

subplot(2,2,1)
imshow(img, [])
title('Original u')
subplot(2,2,2)
imshow(img10, [])
title('Keep 10^2 low frequencies')
subplot(2,2,3)
imshow(img20, [])
title('Keep 20^2 low frequencies')
subplot(2,2,4)
imshow(img40, [])
title('Keep 40^2 low frequencies')

function img_output = main(img, filter_length)

img_height = size(img, 1);
img_width = size(img, 2);

img_fft = fft2(img);
img_fftshift = fftshift(img_fft);

low_filter = zeros(size(img_fftshift));
low_filter(img_height/2 - filter_length/2 : img_height/2 + filter_length/2 - 1,...
    img_width/2 - filter_length/2 : img_width/2 + filter_length/2 - 1) = 1;
img_output = low_filter.*img_fftshift;

img_output = ifftshift(img_output);
img_output = ifft2(img_output);

end