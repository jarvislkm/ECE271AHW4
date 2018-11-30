%% HW4 A
%% generate random parameters
clear all;
load('TrainingSamplesDCT_8_new.mat');
c = 8;
dim = 64;

%%
scale = 0.0001;
for i = 1:5
    p_FG_tmp = generate_rd_parameter(c,dim,scale);
    p_FG{i} = EM_algo(TrainsampleDCT_FG,p_FG_tmp);
    
    p_BG_tmp = generate_rd_parameter(c,dim,scale);
    p_BG{i} = EM_algo(TrainsampleDCT_BG,p_BG_tmp); 
end

%% load eval data
gt = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
img_p = im2double(padarray(img,[4 4],'symmetric','both'));
test_data = read_image(img,img_p);

%% inference
dim_eval = [1,2,4,8,16,24,32,40,48,56,64];
res = zeros([5,5,size(dim_eval,2),size(img)]);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));

for k = 1:size(dim_eval,2)
    for i = 1:5
        for j = 1:5
        likelihood_bg = EM_eval(test_data, p_BG{i}, dim_eval(k));
        likelihood_fg = EM_eval(test_data, p_FG{j}, dim_eval(k));

        p_fg_x = likelihood_fg * p_fg;
        p_bg_x = likelihood_bg * p_bg;

        res_tmp = zeros(size(test_data,1),1);
        res_tmp(p_fg_x>p_bg_x) = 1;
        res(i,j,k,:,:) = reshape(res_tmp, size(img));
        end
    end
end
%% error
rate = zeros(5,5,size(dim_eval,2));
for i = 1:5
    for j = 1:5
        for k = 1:size(dim_eval,2)
            diff = abs(squeeze(res(i,j,k,:,:))-im2double(gt));
            fg_num = sum(sum(im2double(gt)));
            bg_num = (size(img,1)*size(img,2)) - fg_num;
            error_fg = sum(sum(diff.*(im2double(gt))));
            error = sum(sum(diff));
            error_bg = (error-error_fg);
            rate(i,j,k) = error/(size(img,1)*size(img,2));
            rate_fg = error_fg/fg_num;
            rate_bg = error_bg/bg_num;
        end
        figure
        plot(dim_eval, squeeze(rate(i,j,k)));
        title(['Error rate of set ' num2str(i) ' ' num2str(j)]);
        xlabel('Dimension of DCT used');
        saveas(gcf,['Set_' num2str(i) '_' num2str(j) '.png']);
    end
end
