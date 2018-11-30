%% generate random parameters
clear all;
load('TrainingSamplesDCT_8_new.mat');
c = 8;
dim = 64;

%%
scale = 0.0001;
for i = 1:5
    p_FG_tmp = generate_rd_parameter(c,dim,scale);
    p_FG{i} = EM_algo(TrainsampleDCT_FG,p_FG_tmp,"train");
    
    p_BG_tmp = generate_rd_parameter(c,dim,scale);
    p_BG{i} = EM_algo(TrainsampleDCT_BG,p_BG_tmp,"train"); 
end

%% load eval data
gt = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
img_p = im2double(padarray(img,[4 4],'symmetric','both'));
test_data = read_image(img,img_p);

%% inference
res = zeros([5,5,size(img)]);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));

for i = 1:5
    for j = 1:5
    [no_use, likelihood_bg] = EM_algo(test_data, p_BG{i}, "eval");
    [no_use, likelihood_fg] = EM_algo(test_data, p_FG{j}, "eval");

    p_fg_x = likelihood_fg * p_fg;
    p_bg_x = likelihood_bg * p_bg;

    res_tmp = zeros(size(test_data,1),1);
    res_tmp(p_fg_x>p_bg_x) = 1;
    res(i,j,:,:) = reshape(res_tmp, size(img));
    end
end

%% error
rate = zeros(5,5);
for i = 1:5
    for j = 1:5
        diff = abs(squeeze(res(i,j,:,:))-im2double(gt));
        fg_num = sum(sum(im2double(gt)));
        bg_num = (size(img,1)*size(img,2)) - fg_num;
        error_fg = sum(sum(diff.*(im2double(gt))));
        error = sum(sum(diff));
        error_bg = (error-error_fg);
        rate(i,j) = error/(size(img,1)*size(img,2));
        rate_fg = error_fg/fg_num;
        rate_bg = error_bg/bg_num;
    end
end
