import torch.utils.data as data
import torch
import h5py
import numpy as np
import glob
import scipy.io as sio
import struct


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, driver=None):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, mode='r', driver=driver)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

class ndarrayLoader:
    def __init__(self,input,target,shuffle=True,batch_size=128):
        self.shape=[1111616,1,41,41]
        self.input=np.fromfile(input,dtype=np.float32).reshape(self.shape)
        self.target=np.fromfile(target,dtype=np.float32).reshape(self.shape)
        self.index=np.arange(self.shape[0])
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.sample_num=0
    def __iter__(self):
        return self
    def __len__(self):
        return self.shape[0]//self.batch_size
    def __next__(self):
        if self.shape[0]-self.sample_num<self.batch_size:
            self.sample_num=0
            raise StopIteration
        if self.sample_num==0:
            if self.shuffle:
                np.random.shuffle(self.index)
        batch_index=self.index[self.sample_num:self.sample_num+self.batch_size]
        data=torch.from_numpy(self.input[batch_index])
        target=torch.from_numpy(self.target[batch_index])
        self.sample_num+=self.batch_size
        return data,target

def dump_train_data(hdf5):
    with open('input.data','wb') as f1,open('target.data','wb')as f2:
        data=hdf5.data.value*255
        data=(data.round()-128)/255.0
        data.tofile(f1)
        data = hdf5.target.value-128/255.0
        data.tofile(f2)
    return

def dump_eval_data(path="Set5_mat/*.*"):
    scales = [2, 3, 4]
    image_list = glob.glob(path)
    for scale in scales:
        with open(str(scale)+'x.data','wb')as f:
            for image_name in image_list:
                if 'x'+str(scale) in image_name:
                    print("Processing ", image_name)
                    im_gt_y = sio.loadmat(image_name)['im_gt_y']
                    im_b_y = sio.loadmat(image_name)['im_b_y'].astype(np.float32)
                    f.write(struct.pack('2i',*im_gt_y.shape))
                    im_gt_y.tofile(f)
                    im_b_y.tofile(f)
    return

if __name__ == '__main__':
    #train_set = DatasetFromHdf5("data/train.h5")#,driver='core')
    dump_eval_data(path='Set14_mat/*.*')