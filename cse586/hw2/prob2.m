% 2. Geodesic shooting
% Reading Given Data
addpath("./readData");
[Iv, originv, spacingv] = loadMETA('./data/velocity/v0Spatial.mhd', true);
[Is, origins, spacings] = loadMETA('./data/sourceImage/source.mhd', true);

h = 10; % After h step Euler' s method lead to t=1; step size is 1/h. 

%% 2.(a)
% Init Variables
vx = squeeze(Iv(1,:,:)); vy = squeeze(Iv(2,:,:));
vx0 = vx; vy0 = vy;

[rows,cols] = size(Is);
[phix, phiy] = meshgrid(1:rows, 1:cols);
phix0 = phix; phiy0 = phix;

fprintf("\n");

for i = 1:h
    fprintf("Solving (a) sub-problem. In [%d] Iterrations \n", i);
    
    % Compute and update v_t
    div = vx.^2 + vy.^2;
    div = Dx(div) + Dy(div);

    vx = vx + (1/h).* K((Dx(vx).*vx + Dy(vx).*vy + div), 16);
    vy = vy + (1/h).* K((Dx(vy).*vx + Dy(vy).*vy + div), 16);
    
    % Compute and update phi_t
    phix_temp = phix;
    phiy_temp = phiy;
    
    for r = 1:rows
        for c = 1:cols
            phix_temp(r, c) = phix_temp(r, c) + (1/h).* interp2(1:rows, 1:cols, vx, phix(r, c), phiy(r, c), 'spline');
            phiy_temp(r, c) = phiy_temp(r, c) + (1/h).* interp2(1:rows, 1:cols, vy, phix(r, c), phiy(r, c), 'spline');
        end
    end
    
    phix = phix_temp;
    phiy = phiy_temp;
end

vxa = vx; vya = vy;
phixa = phix; phiya = phiy;

%% 2.(b)
% Init Variables
vx = squeeze(Iv(1,:,:)); vy = squeeze(Iv(2,:,:));
vx0 = vx; vy0 = vy;

[rows,cols] = size(Is);
[phix, phiy] = meshgrid(1:rows, 1:cols);
phix0 = phix; phiy0 = phix;

fprintf("\n");

for i = 1:h
    fprintf("Solving (b) sub-problem. In [%d] Iterrations \n", i);
    
    % Compute and update v_t
    div = vx.^2 + vy.^2;
    div = Dx(div) + Dy(div);
    
    % Compute and update phi_t
    vx = vx - (1/h).* K((Dx(vx).*vx + Dy(vx).*vy + div), 16);
    vy = vy - (1/h).* K((Dx(vy).*vx + Dy(vy).*vy + div), 16);
    
    phix = phix - (1/h).*(Dx(phix).*vx + Dx(phiy).*vy);
    phiy = phiy - (1/h).*(Dy(phix).*vx + Dy(phiy).*vy);
end

vxb = vx; vyb = vy;
phixb = phix; phiyb = phiy;

%% 3.(c) Recover Image
imgs_a = zeros(size(Is));
imgs_b = zeros(size(Is));

fprintf("\n Solving (c) sub-problem.... \n");
for r = 1:rows
    for c = 1:cols
        imgs_a(r, c) = interp2(1:rows, 1:cols, Is, phixa(r, c), phiya(r, c), 'spline');
        imgs_b(r, c) = interp2(1:rows, 1:cols, Is, phixb(r, c), phiyb(r, c), 'spline');
    end
end

%% Show ALl Compare Results
figure(1)
subplot(2, 3, 1)
imshow(vx0, [])
title("Initial velocity ?eld v_x")
subplot(2, 3, 2)
imshow(vxa, [])
title("(a)sub-problem v_x")
subplot(2, 3, 3)
imshow(vxb, [])
title("(b)sub-problem v_x")
subplot(2, 3, 4)
imshow(vy0, [])
title("Initial velocity ?eld v_y")
subplot(2, 3, 5)
imshow(vya, [])
title("(a)sub-problem v_y")
subplot(2, 3, 6)
imshow(vyb, [])
title("(b)sub-problem v_y")

figure(2)
subplot(2, 3, 1)
imshow(phix0, [])
title("Initial transformation \phi_x")
subplot(2, 3, 2)
imshow(phixa, [])
title("(a)sub-problem \phi_x")
subplot(2, 3, 3)
imshow(phixb, [])
title("(b)sub-problem \phi_x")
subplot(2, 3, 4)
imshow(phiy0, [])
title("Initial transformation \phi_y")
subplot(2, 3, 5)
imshow(phiya, [])
title("(a)sub-problem \phi_y")
subplot(2, 3, 6)
imshow(phiyb, [])
title("(b)sub-problem \phi_y")

figure(3)
subplot(1, 3, 1)
imshow(Is, [])
title("Source Image")
subplot(1, 3, 2)
imshow(imgs_a, [])
title("(a)sub-problem Image")
subplot(1, 3, 3)
imshow(imgs_b, [])
title("(b)sub-problem Image")

function result = Dx(input)
    [rows,cols] = size(input);
    
    result = input;
    result(1:rows-1, 1:cols-1) = result(1:rows-1, 1:cols-1)- result(1:rows-1, 2:cols);
end

function result = Dy(input)
    [rows,cols] = size(input);
    
    result = input;
    result(1:rows-1, 1:cols-1) = result(1:rows-1, 1:cols-1)- result(2:rows, 1:cols-1);
end

function result = K(input, filter_length)

img_height = size(input, 1);
img_width = size(input, 2);

img_fft = fft2(input);
img_fftshift = fftshift(img_fft);

low_filter = zeros(size(img_fftshift));
low_filter(img_height/2 - filter_length/2 : img_height/2 + filter_length/2 - 1,...
    img_width/2 - filter_length/2 : img_width/2 + filter_length/2 - 1) = 1;
result = low_filter.*img_fftshift;

result = ifftshift(result);
result = ifft2(result);

result = abs(result);
end