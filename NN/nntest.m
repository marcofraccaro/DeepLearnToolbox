function [er, bad] = nntest(nn, x, y)

    assert(isa(x,'gpuArray'), 'x must be a gpuArray');
    assert(isa(y,'gpuArray'), 'y must be a gpuArray');
    
    if nn.normalize_input==1;
       x = zscore(x);
    end
    
    nn.testing = 1;
    nn = nnff(nn, x, y);
    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    [~, g] = max(y,[],2);
    bad = find(i ~= g);    
    er = numel(bad) / size(x, 1);
end
