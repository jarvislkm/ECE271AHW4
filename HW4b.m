%% HW4 B
%% generate random parameters
clear all;
load('TrainingSamplesDCT_8_new.mat');
c = [1,2,4,8,16,32];
dim = 64;
rate_rec = [];

%%
scale = 0.0001;
for i = 1:size(c,2)
    p_FG_tmp = generate_rd_parameter(c(i),dim,scale);
    p_FG{i} = EM_algo(TrainsampleDCT_FG,p_FG_tmp);
    
    p_BG_tmp = generate_rd_parameter(c(i),dim,scale);
    p_BG{i} = EM_algo(TrainsampleDCT_BG,p_BG_tmp); 
end

%% load eval data
gt = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
img_p = im2double(padarray(img,[4 4],'symmetric','both'));
test_data = read_image(img,img_p);

%% inference
dim_eval = [1,2,4,8,16,24,32,40,48,56,64];
res = zeros([size(c,2),size(dim_eval,2),size(img)]);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));

for k = 1:size(dim_eval,2)
    disp(k);
    for i = 1:size(c,2)
        likelihood_bg = EM_eval(test_data, p_BG{i}, dim_eval(k));
        likelihood_fg = EM_eval(test_data, p_FG{i}, dim_eval(k));

        p_fg_x = likelihood_fg * p_fg;
        p_bg_x = likelihood_bg * p_bg;

        res_tmp = zeros(size(test_data,1),1);
        res_tmp(p_fg_x>p_bg_x) = 1;
        res(i,k,:,:) = reshape(res_tmp, size(img));
    end
end
%% error
rate = zeros(i,k);
for i = 1:size(c,2)
    for k = 1:size(dim_eval,2)
    diff = abs(squeeze(res(i,k,:,:))-im2double(gt));
    fg_num = sum(sum(im2double(gt)));
    bg_num = (size(img,1)*size(img,2)) - fg_num;
    error_fg = sum(sum(diff.*(im2double(gt))));
    error = sum(sum(diff));
    error_bg = (error-error_fg);
    
    rate(i,k) = error/(size(img,1)*size(img,2));
    rate_fg = error_fg/fg_num;
    rate_bg = error_bg/bg_num;
    end
end
rate_rec = [rate_rec, rate];
%%
figure
plot(dim_eval', rate','LineWidth', 2);
legend('1','2','4','8','16','32');
title(['Error rate of different number of mixture models']);
xlabel('Dimension of features');
ylabel('Error rate');
saveas(gcf,['number_of_mm.png']);
