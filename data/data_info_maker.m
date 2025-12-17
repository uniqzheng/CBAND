% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear,clc;

data_path = './LIVE_Banding_metadata.csv';
data = readtable(data_path);
video_names = data.Filename; % video names
scores = data.MOS; % subjective scores
framerate = data.framerate;
%clear data_path data

height = 1920; % video height
width = 1080; % video width
max_len = 1200; % maximum video length in the dataset
video_format = 'yuv420'; % video format
ref_ids = [1:length(scores)]'; % video content ids
% `random` train-val-test split index, 1000 runs
%index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
%    1:1000,'UniformOutput', false)'); 
ary_idx = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40];
index = zeros(1000,160);
for i = 1:1000 
    rng('shuffle');
    idx = randperm(40);
    idx_idx = idx(1:24);
    train_idx_index = ary_idx(idx_idx);
    subary_idx = setdiff(ary_idx , train_idx_index);
    subidx = randperm(16);
    subidx_idx = subidx(1:8);
    val_idx_index = subary_idx(subidx_idx);
    test_idx_index = setdiff(subary_idx,val_idx_index);
    for j = 1:24
        index(i,(j-1)*4+1:j*4) = linspace((train_idx_index(j)-1)*4+1,train_idx_index(j)*4,4);
    end
    for j = 1:8
        index(i,(j+24-1)*4+1:(j+24)*4) = linspace((val_idx_index(j)-1)*4+1,val_idx_index(j)*4,4);
    end
    for j = 1:8
        index(i,(j+32-1)*4+1:(j+32)*4) = linspace((test_idx_index(j)-1)*4+1,test_idx_index(j)*4,4);
    end
end
save('LIVE_Banding_contentaware','-v7.3')

% %% LIVE_Banding
% data_path = './LIVE_Banding_metadata.csv';
% data = readtable(data_path);
% video_names = data.Filename; % video names
% scores = data.MOS; % subjective scores
% clear data_path data
% 
% height = 1920; % video height
% width = 1080; % video width
% max_len = 1000; % maximum video length in the dataset
% video_format = 'yuv420'; % video format
% ref_ids = [1:length(scores)]'; % video content ids
% % `random` train-val-test split index, 1000 runs
% index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
%     1:1000,'UniformOutput', false)'); 
% save('LIVE_Banding','-v7.3')

