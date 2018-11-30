%%
clear all;
%% find feature of training data
load('TrainingSamplesDCT_8_new.mat');
%%
mean_bg = mean(TrainsampleDCT_BG);
mean_fg = mean(TrainsampleDCT_FG);
sigma_bg = cov(TrainsampleDCT_BG);
sigma_fg = cov(TrainsampleDCT_FG);
%%
worse = [64, 63, 62, 60, 59, 6, 5, 3];
best = [1, 14, 15, 23, 25, 31, 32, 33];
figure
count = 1;
for i=1:64
    max_ = max(mean_bg(1,i)+5*(sigma_bg(i,i))^0.5, mean_fg(1,i)+5*(sigma_fg(i,i))^0.5);
    min_ = min(mean_bg(1,i)-5*(sigma_bg(i,i))^0.5, mean_fg(1,i)-5*(sigma_fg(i,i))^0.5);
    x = min_:(max_-min_)/100:max_-(max_-min_)/100;
    bg = makedist('Normal','mu', mean_bg(1,i),'sigma',(sigma_bg(i,i))^0.5);
    pdf_bg = pdf(bg,x);
    fg = makedist('Normal','mu', mean_fg(1,i),'sigma',(sigma_fg(i,i))^0.5);
    pdf_fg = pdf(fg,x);
    subplot(8, 8, count);
    count = count + 1;
    plot(x, pdf_bg); hold on;
    plot(x, pdf_fg);
    title([i]);
%     xlabel('value of component');
%     ylabel('P');
end

%% read test image, read zig zag pattern
index = textread('Zig-Zag Pattern.txt', '%d');
index = reshape(index+1, [8, 8])';

gt = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
img_p = im2double(padarray(img,[4 4],'symmetric','both'));
data = zeros([size(img),64]);
for i = 1:size(img_p,1)-8
    for j = 1:size(img_p,2)-8
        crop = img_p(i:i+7,j:j+7);
        crop_dct = dct2(crop);
        for ii = 1:8
            for jj = 1:8
                data(i,j,index(ii,jj)) = crop_dct(ii, jj);
            end
        end
    end
end
data = reshape(data, [size(data,1)*size(data,2), size(data,3)]);
%% 64
res_64 = zeros(size(data,1),1);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
sigma_fg_inv = sigma_fg^-1;
sigma_bg_inv = sigma_bg^-1;
for index = 1:size(data)
    v_fg = mean_fg-data(index,:);
    p_x_fg = 1/((det(sigma_fg))^0.5)*exp(-0.5*v_fg*sigma_fg_inv*v_fg');
    v_bg = mean_bg-data(index,:);
    p_x_bg = 1/((det(sigma_bg))^0.5)*exp(-0.5*v_bg*sigma_bg_inv*v_bg');
    
    p_fg_x = p_x_fg * p_fg;
    p_bg_x = p_x_bg * p_bg;
    
    if p_fg_x>=p_bg_x
        res_64(index) = 1;
    end
end

res_64 = reshape(res_64, size(img));
%% result
diff = abs(res_64-im2double(gt));
fg_num = sum(sum(im2double(gt)));
bg_num = (size(img,1)*size(img,2)) - fg_num;
error_fg = sum(sum(diff.*(im2double(gt))));
error = sum(sum(diff));
error_bg = (error-error_fg);
rate = error/(size(img,1)*size(img,2));
rate_fg = error_fg/fg_num;
rate_bg = error_bg/bg_num;

%% 8
res_8 = zeros(size(data,1),1);
sigma_fg_8 = sigma_fg(best,best);
sigma_bg_8 = sigma_bg(best,best);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
sigma_fg_inv = sigma_fg_8^-1;
sigma_bg_inv = sigma_bg_8^-1;
for index = 1:size(data)
    v_fg = mean_fg(1,best)-data(index,best);
    p_x_fg = 1/((det(sigma_fg_8))^0.5)*exp(-0.5*v_fg*sigma_fg_inv*v_fg');
    v_bg = mean_bg(1,best)-data(index,best);
    p_x_bg = 1/((det(sigma_bg_8))^0.5)*exp(-0.5*v_bg*sigma_bg_inv*v_bg');
    
    p_fg_x = p_x_fg * p_fg;
    p_bg_x = p_x_bg * p_bg;
    
    if p_fg_x>p_bg_x
        res_8(index) = 1;
    end
end

res_8 = reshape(res_8, size(img));

%% result
diff = abs(res_8-im2double(gt));
fg_num = sum(sum(im2double(gt)));
bg_num = (size(img,1)*size(img,2)) - fg_num;
error_fg = sum(sum(diff.*(im2double(gt))));
error = sum(sum(diff));
error_bg = (error-error_fg);
rate = error/(size(img,1)*size(img,2));
rate_fg = error_fg/fg_num;
rate_bg = error_bg/bg_num;