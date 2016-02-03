% Linear Attenuation Coefficient Prediction by Using Deep Neural Networks
% Soysal Degirmenci
% 09/03/15

% This script attemps to generate data for the deep neural network to be
% trained.

close all; clear all; clc;
% Neural network info:
% Input: log normalized data
% Output: pixel values (attenuation coefficients)


% Attenuation coeff. list.
att_list = (0.01:0.01:0.5)/4;
size_att = numel(att_list);
% Possible shapes list.
% shape_list = {'ellipse','circle','rectangular'};
shape_list = {'ellipse','circle'};
size_shape = numel(shape_list);
% Max radius for ellipse and circle.

img_size_x = 64; img_size_y = img_size_x;
max_objects = 8;

I_0 = 1e4;

% Meshgrid stuff here
[y_inds_img, x_inds_img] = meshgrid(1:img_size_x, 1:img_size_y);



% Only allow them to be in a circle
boundary_center_x = (img_size_x-1)/2;
boundary_center_y = (img_size_x-1)/2;
rad_boundary = round(img_size_x/2.5);
pixels_allowed = (x_inds_img - boundary_center_x).^2 + (y_inds_img - boundary_center_y).^2 <= rad_boundary^2;
sum_pixels_allowed = sum(pixels_allowed(:));
[temp_x, temp_y] = find(pixels_allowed == 1);
min_center_allowed = min([temp_x; temp_y]);
max_center_allowed = max([temp_x; temp_y]);

imagesc(pixels_allowed); title('09-16-15 Dataset - ROI');
saveas(gcf, 'pixels_allowed_091615.jpg');
close all;

num_data = 55000;
img_truth_stack = zeros(img_size_x, img_size_y, num_data);
img_fbp_clean_stack = zeros(img_size_x, img_size_y, num_data);
img_fbp_noisy_stack = zeros(img_size_x, img_size_y, num_data);

theta_ind = 0:1:179;
max_rad = round(img_size_x/12);

ellipse_rotation_angle_list = 0:5:180;
size_angle_list = numel(ellipse_rotation_angle_list);

% pixel values inside ROI are called all_labels
size_label = sum(pixels_allowed(:));
all_labels = zeros(num_data, size_label);
img_temp = zeros(img_size_x, img_size_y);

temp_fp = radon(img_temp, theta_ind);
size_input = numel(temp_fp);
all_input = zeros(num_data, size_input);
i = 1;
max_temp_fp = 0;

while i<= num_data
    if mod(i-1, 1000) == 0
        disp(['Generating dataset, realization # = ' num2str(i) ' out of ' num2str(num_data)]);
    end
    img_temp = zeros(img_size_x, img_size_y);
    
    for j = 1:max_objects
        % Choose a shape
        shape_temp = shape_list{randi(size_shape)};
        % Choose an attenuation
        att_temp = att_list(randi(size_att));
        flag = 0;
        if strcmp(shape_temp, 'ellipse') == 1
            % Choose center, two radii, draw it, update the 1/0 list as well
            while flag == 0
                centers_temp = randi([min_center_allowed, max_center_allowed],1,2);
                rad_a = randi(max_rad);
                rad_b = randi(max_rad);
                % Rotated ellipse
                alpha = ellipse_rotation_angle_list(randi(size_angle_list));
                pixels_in = ((x_inds_img-centers_temp(1))*cos(alpha) + (y_inds_img-centers_temp(2))*sin(alpha)).^2/(rad_a)^2 ...
                    + ((x_inds_img-centers_temp(1))*sin(alpha) - (y_inds_img-centers_temp(2))*sin(alpha)).^2/(rad_b)^2 <= 1;
                     
                img_1 = zeros(size(img_temp));
                img_1(pixels_allowed) = 1;
                img_1(pixels_in) = 2;
                if sum(sum(or(pixels_in, pixels_allowed)))==sum_pixels_allowed
                    flag = 1;
                end
            end
            img_temp(pixels_in) = att_temp;
        elseif strcmp(shape_temp, 'circle') == 1
            % Choose center, one radius, draw it, update the 1/0 list as well
            while flag == 0
                centers_temp = randi([min_center_allowed, max_center_allowed],1,1);
                rad_a = randi(max_rad);
                pixels_in = (x_inds_img - centers_temp(1)).^2 + (y_inds_img - centers_temp(1)).^2 <= rad_a^2;
                img_1 = zeros(size(img_temp));
                img_1(pixels_allowed) = 1;
                img_1(pixels_in) = 2;
                if sum(sum(or(pixels_in, pixels_allowed)))==sum_pixels_allowed
                    flag = 1;
                end
            end
            img_temp(pixels_in) = att_temp;
        end
        %         elseif strcmp(shape_temp, 'rectangular') == 1
        %             % Choose center, one radius, draw it, update the 1/0 list as well
        %             while flag == 0
        %                 centers_temp = randi([min_center_allowed, max_center_allowed],1,2);
        %                 rad_a = randi(max_rad);
        %                 rad_b = randi(max_rad);
        %                 pixels_in = abs(x_inds_img - centers_temp(1)) <= 2*rad_a & abs(y_inds_img - centers_temp(2)) <= 2*rad_b;
        %                 img_1 = zeros(size(img_temp));
        %                 img_1(pixels_allowed) = 1;
        %                 img_1(pixels_in) = 2;
        %                 if sum(sum(or(pixels_in, pixels_allowed)))==sum_pixels_allowed
        %                     flag = 1;
        %                 end
        %             end
        %             img_temp(pixels_in) = att_temp;
        %
        %         end
    end
    
    
    
    img_truth_stack(:,:,i) = img_temp;
    imagesc(img_temp);
    temp_label = img_temp(pixels_allowed);
    temp_fp = radon(img_temp, theta_ind);
    if max_temp_fp < max(temp_fp(:))
        max_temp_fp = max(temp_fp(:));
        disp(['Maximum = ' num2str(max_temp_fp)]);
    end
    temp_2 = iradon(temp_fp, theta_ind);
    img_fbp_clean_stack(:,:,i) = temp_2(2:end-1,2:end-1);
    temp_input = poissrnd(I_0*exp(-temp_fp(:)));
    temp_input = -log(temp_input./I_0);
    temp_input(isnan(temp_input)) = 0; temp_input(isinf(temp_input)) = 0;
    temp_3 = iradon(reshape(temp_input, size(temp_fp)), theta_ind); temp_3 = temp_3(2:end-1,2:end-1);
%     temp_4 = iradon(reshape(temp_input, size(temp_fp)), theta_ind, 'linear', 'Shepp-Logan'); temp_4 = temp_4(2:end-1,2:end-1);
%     temp_5 = iradon(reshape(temp_input, size(temp_fp)), theta_ind, 'linear', 'Cosine'); temp_5 = temp_5(2:end-1,2:end-1);
%     temp_6 = iradon(reshape(temp_input, size(temp_fp)), theta_ind, 'linear', 'Hamming'); temp_6 = temp_6(2:end-1,2:end-1);
%     temp_7 = iradon(reshape(temp_input, size(temp_fp)), theta_ind, 'linear', 'Hann'); temp_7 = temp_7(2:end-1,2:end-1);
    img_fbp_noisy_stack(:,:,i) = temp_3;
%     subplot(1,3,1); imagesc(temp_2); title('FBP Non Noisy Data');
%     subplot(1,3,2); imagesc(img_temp); title('Truth');
%     subplot(1,3,3); imagesc(temp_3); title('Filtered BP Image');
%     imagesc(img_temp); title('Truth');
%     close all;
    
    all_labels(i,:) = temp_label;
    all_input(i,:) = temp_fp(:);
    i = i+1;
    
end
% imagesc(img_temp);

% TrainImages = (num_of_examples, size_of_input)
% TrainLabels = (num_of_examples, size_of_output)
save('img_dataset_101315_64x64_big.mat','img_truth_stack','all_labels','all_input','pixels_allowed','img_fbp_noisy_stack','img_fbp_clean_stack','-v7.3');
