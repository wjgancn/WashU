data_src = readtable('iris.txt');
rows_virginica = strcmp(data_src.Species , 'virginica');
rows_versicolor = strcmp(data_src.Species , 'versicolor');

data_virginica = data_src{rows_virginica, 1:4};
data_virginica = [ones([size(data_virginica, 1), 1]), data_virginica, 0*ones([size(data_virginica, 1), 1])]; 

data_versicolor = data_src{rows_virginica, 1:4};
data_versicolor = [ones([size(data_versicolor, 1), 1]), data_versicolor, 1*ones([size(data_virginica, 1), 1])]; 

data_train = [data_virginica(1:30, :); data_versicolor(1:30, :)];
data_test = [data_virginica(31:50, :); data_versicolor(31:50, :)];

x_train = data_train(:, 1:5); y_train = data_train(:, 6);
x_test = data_test(:, 1:5); y_test = data_test(:, 6);

sigma = 200;
beta = ones(size(x_train,2),1);

for i = 1:1000
    
    obj = 0;
    grad = zeros(size(beta));
    for batch = 1:size(x_train, 1)
        obj = obj + (1 - y_train(batch, :)) * x_train(batch, :) * beta + log( 1 + exp(-1 * x_train(batch, :) * beta));
        
        h = exp(-1 * x_train(batch, :) * beta);
        grad = grad + (1 - y_train(batch, :) - h/(1+h)) * x_train(batch, :)';
    end
    
    obj = obj + norm(beta, 2)/(2*sigma^2);
    grad = grad + beta/(sigma^2);

    fprintf('Iter Step: [%d] Objective Function: [%.4f] \n', i, obj);
    beta = beta - 0.0015 * grad;
end

e = 0;
for i = 1:size(x_test, 1)
    pre_test = 1 / (1 + exp(-1 * x_test(i, :) * beta)) * exp(norm(beta, 2) / sigma^2);
    
    if pre_test >= 0.5
        pre_test = 1;
    else
        pre_test = 0;
    end
    
    if pre_test == y_test(i)
        e = e + 1;
    end
end
e = e/size(x_test, 1);