clear all
close all 
clc

addpath('NN','data','util');

load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

nn = nnsetup([784 800 800 10]);

nn.weightPenaltyL2 = 1e-6; 
nn.activation_function = 'tanh_opt';    %  Sigmoid activation function
nn.output= 'softmax';                   %  Softmax for classification
nn.normalize_input = 0;                 %  Don't normalize inputs
nn.learningRate = 0.25;                 %  Learning rate
opts.numepochs =  700;                  %  Number of full sweeps through data
opts.batchsize = 100;                   %  Take a mean gradient step over this many samples

nn.momentum  = 0.5; 
nn.it_no_momentum = 10; 
nn.normalize_momentum = 0;              %  Normalize momentum term (requires nn.it_no_momentum > 0)
nn.scaling_learningRate = 0.998;        %  Reduce learning rate at each iteration
nn.dropoutFraction = 0.5; 

nn = nntrain(nn, gpuArray(train_x), gpuArray(train_y), opts);
[er, bad] = nntest(nn, gpuArray(test_x), gpuArray(test_y));
disp(['Test error: ',num2str(er)])
disp(['Results from Hinton''s paper: between 0.013 and 0.014'])