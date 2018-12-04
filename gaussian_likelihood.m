function [likelihood] = gaussian_likelihood(data, mu, var)
% calculate likelihood of gaussian model given mu and variance matrix
%   data: n by dim
%   mu  : 1 by dim
%   var : dim by dim
    likelihood = zeros(size(data,1),1);
    tmp_var = squeeze(var);
    tmp_var_inv = inv(tmp_var);
    tmp_var_det = (det(tmp_var))^(0.5);
    for i = 1:size(data, 1)
        tmp_x = data(i,:)-mu;
        likelihood(i) = 1/tmp_var_det * exp(-0.5*tmp_x*tmp_var_inv*tmp_x');
    end
end

