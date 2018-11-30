function [likelihood] = EM_eval(data,p,dim)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
p_mu = p.mu;
p_weight = p.weight;
p_var = p.var;

c = size(p_mu, 1);
likelihood = zeros(size(data, 1),1);

data = data(:,1:dim);
p_mu = p_mu(:,1:dim);
p_var = p_var(:,1:dim,1:dim);

for j = 1:c
    likelihood_ = gaussian_likelihood(data, p_mu(j,:), p_var(j,:,:));
    likelihood = likelihood + likelihood_.* p_weight(j);
end
end

