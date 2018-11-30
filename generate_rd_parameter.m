function [p] = generate_rd_parameter(c,dim,scale)
% Generate random parameters {(weight_i,mu_i,sigma_i)}, i is number of mixture
%model.

% weight should be summed to 1.
p_weight = rand(c, 1);
p_weight = p_weight/sum(p_weight);

% scaled mu will help divide samples equally.
% this should be a small number;
p_mu = scale*randn(c, dim);

% variance matrix also random and diagnoal.
p_var = zeros(c, dim, dim);
for i = 1:c
    p_var(i, :, :) = rand(dim).*eye(dim);
end
p.weight = p_weight;
p.var = p_var;
p.mu = p_mu;
end

