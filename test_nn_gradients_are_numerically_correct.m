function test_nn_gradients_are_numerically_correct

addpath('NN','data','util');
disp('WARNING: dropout is not checked due to problems in fixing the seed for random numbers in parallel.gpu.rng')

batch_x = gpuArray.rand(20, 5);
batch_y = gpuArray.rand(20, 2);

for output = {'sigm', 'linear', 'softmax'}
    y=batch_y;
    if(strcmp(output,'softmax'))
        % softmax output requires a binary output vector
        y=(y==repmat(max(y,[],2),1,size(y,2)));
    end
    
    for activation_function = {'sigm', 'tanh_opt'}
        for normalize_input = {0, 1}
            for dropoutFraction = {0}
%             for dropoutFraction = {0 rand()}
                nn = nnsetup([5 3 4 2]);

                nn.activation_function = activation_function{1};
                nn.output = output{1};
                nn.normalize_input = normalize_input{1};
                nn.dropoutFraction = dropoutFraction{1};


                rng(0)
                nn = nnff(nn, batch_x, y);
                nn = nnbp(nn);
                nnchecknumgrad(nn, batch_x, y);
            end
        end
    end
end
disp('Test successful')