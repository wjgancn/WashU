% 3. Gradient decent algorithm for ROF model with total variation minimization.
epoch = 200;
lambda = 0.01;
learning_rate = 0.5;

f = double(imread('lenaNoise.png'));
u = f;

[rows,cols] = size(f);
for i = 1:epoch

ux = u(1:rows-1, 1:cols-1)- u(1:rows-1, 2:cols);
uy = u(1:rows-1, 1:cols-1) - u(2:rows, 1:cols-1);

l1_u = sqrt(ux.^2 + uy.^2 + 1e-8);
l1_loss = sum(l1_u, 'all');
l1_loss_all(i) = l1_loss;

ux_ = ux ./ l1_u;
uy_ = uy ./ l1_u;

l1_gradient = zeros(size(u));
l1_gradient(1:rows-1, 1:cols-1) = ux_ + uy_;
l1_gradient(1:rows-1, 2:cols) = l1_gradient(1:rows-1, 2:cols) - ux_;
l1_gradient(2:rows, 1:cols-1) = l1_gradient(2:rows, 1:cols-1) - uy_;

l2_gradient = lambda.*(f - u);
l2_loss = lambda.*norm(f - u, 2);
l2_loss_all(i) = l2_loss;

loss = l1_loss + l2_loss;
loss_all(i) = loss;
u = u - learning_rate .* (l2_gradient + l1_gradient);

fprintf("Epoch: [%d], loss: [%.3f], l2_loss: [%.3f], l1_loss: [%.3f] \n", i, loss, l2_loss, l1_loss);

end

figure(1)
subplot(1,2,1)
imshow(f, [])
title('Original Image')
subplot(1,2,2)
imshow(u, [])
title('Final Results')

figure(2)
subplot(1,3,1)
plot(loss_all)
title('L1 + L2 Loss')
subplot(1,3,2)
plot(l1_loss_all)
title('L1 Loss Term')
subplot(1,3,3)
plot(l2_loss_all)
title('L2 Loss Term')