clear;close all;

folder = 'D:\zhaohengrui\data\VDSR\291';

savepath = 'train.h5';
size_input = 41;
size_label = 41;
stride = 41;

%% scale factors
scale = [2,3,4];
%% downsizing
downsizes = [1,0.7,0.5];

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

for i = 1 : length(filepaths)
    for flip = 1: 3
        for degree = 1 : 4
            for s = 1 : length(scale)
                for downsize = 1 : length(downsizes)
                    image = imread(fullfile(folder,filepaths(i).name));

                    if flip == 1
                        image = flipdim(image ,1);
                    end
                    if flip == 2
                        image = flipdim(image ,2);
                    end
                    
                    image = imrotate(image, 90 * (degree - 1));

                    image = imresize(image,downsizes(downsize),'bicubic');
                    
                    if size(image,3)==3            
                        image = rgb2ycbcr(image);
                        image = im2double(image(:, :, 1));

                        im_label = modcrop(image, scale(s));
                        [hei,wid] = size(im_label);
                        im_input = imresize(imresize(im_label,1/scale(s),'bicubic'),[hei,wid],'bicubic');
                        filepaths(i).name
                        for x = 1 : stride : hei-size_input+1
                            for y = 1 :stride : wid-size_input+1

                                subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
                                subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                                
                                count=count+1;

                                data(:, :, 1, count) = subim_input;
                                label(:, :, 1, count) = subim_label;
                            end
                        end
                    end
                end    
            end
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
