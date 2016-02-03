load('img_dataset_101315_64x64_big.mat')
all_input = all_input/max(all_input(:));
train_data = all_input(1:50000, :);
train_label = all_labels(1:50000, :);
test_data = all_input(50001:end, :);
test_label = all_labels(50001:end, :);

train_data = train_data';
train_label = train_label';
test_data = test_data';
test_label = test_label';
save('compact_dataset_101315.mat','train_data','train_label','test_data','test_label','-v7.3');