function test_example_NN

addpath('NN','data','util');
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% ex1 vanilla neural net
parallel.gpu.rng(0);
nn = nnsetup([784 100 10]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);

[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));

assert(er < 0.08, 'Too big error');


%% ex2 neural net with L2 weight decay
parallel.gpu.rng(0);
nn = nnsetup([784 100 10]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);

[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));
assert(er < 0.1, 'Too big error');


%% ex3 neural net with dropout
parallel.gpu.rng(0);
nn = nnsetup([784 100 10]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);

[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));
assert(er < 0.1, 'Too big error');

%% ex4 neural net with sigmoid activation function, and without normalizing inputs
parallel.gpu.rng(0);
nn = nnsetup([784 100 10]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.normalize_input = 0;             %  Don't normalize inputs
nn.learningRate = 1;                %  Sigm and non-normalized inputs require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);

[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));
assert(er < 0.18, 'Too big error');

%% ex5 neural net with normalized momentum after some iterations + Scaling learning rate
parallel.gpu.rng(0);
nn = nnsetup([784 100 10]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.normalize_input = 0;             %  Don't normalize inputs
nn.learningRate = 1;                %  Sigm and non-normalized inputs require a lower learning rate
opts.numepochs =  4;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples
nn.momentum  = 0.8; 
nn.it_no_momentum = 1; 
nn.normalize_momentum = 1;          %  Normalize momentum term (requires nn.it_no_momentum > 0)
nn.scaling_learningRate = 0.998;

nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);

[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));
assert(er < 0.07, 'Too big error');