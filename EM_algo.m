function [p, likelihood_eval] = EM_algo(data,p)
% EM algorithm using
%   Detailed explanation goes here
p_mu = p.mu;
p_weight = p.weight;
p_var = p.var;

c = size(p_mu, 1);
dim = size(p_mu, 2);

likelihood_sum_old = 0;
likelihood_sum = 100;
disp(['start ']);

while (likelihood_sum-likelihood_sum_old) > 1
%% Likelihood
    l_i = zeros(size(data, 1),1);
    h_ij = zeros(size(data, 1), c);
    
    for j = 1:c
        [likelihood] = gaussian_likelihood(data, p_mu(j,:), p_var(j,:,:));
        h_ij(:,j) = likelihood.* p_weight(j);
        l_i = l_i + likelihood.* p_weight(j);
    end
    h_ij = (h_ij'./(sum(h_ij')))';
    
    if status == "eval"
        likelihood_eval = l_i;
        break;
    end
    
    likelihood_sum_old = likelihood_sum;
    likelihood_sum = sum(log(l_i));
    disp(likelihood_sum);
    %% M
    p_mu_next = p_mu;
    p_weight_next = p_weight;
    p_var_next = p_var;

    for j = 1:c
        p_mu_next(j,:) = sum(data.*h_ij(:,j))/sum(h_ij(:,j));
        p_weight_next(j) = sum(h_ij(:,j))/size(data,1);
        tmp_var = zeros(dim,dim);
        for i = 1:size(data,1)
            tmp_var = tmp_var + h_ij(i,j)*eye(dim).*((data(i,:)-p_mu_next(j,:))'*(data(i,:)-p_mu_next(j,:)));
        end
        p_var_next(j,:,:) = tmp_var/sum(h_ij(:,j)) + 1e-5*eye(dim);
    end

    %% update
    p_mu = p_mu_next;
    p_weight = p_weight_next;
    p_var = p_var_next;
end
p.mu = p_mu;
p.weight = p_weight;
p.var = p_var;
end

