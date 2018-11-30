function data = read_image(img,img_p)
% read test image
index = textread('Zig-Zag Pattern.txt', '%d');
index = reshape(index+1, [8, 8])';

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
end

