# -*- coding: utf-8 -*-
"""
author: Hengrui Zhao
contact: zhenry@mail.ustc.edu.cn
"""

from PIL import Image
import time
import threading
from numpy import *
import configparser

cfg_files=[
    '../data/cfg/UCID1.ini',
    '../data/cfg/UCID2.ini',
    '../data/cfg/DIV2K.ini'
]
class train_data(threading.Thread):

    def __init__(self,QP,batch_size=32,batch_side=64):
        threading.Thread.__init__(self)
        self.bufferLock1=threading.Lock()
        self.bufferLock2=threading.Lock()
        self.buffer1=[]
        self.buffer2=[]
        self.current_buffer=0
        # structured data list
        self.list=[]
        self.batch_size=batch_size
        self.batch_side=batch_side
        self.stride=batch_side//2
        self.pieces=0
        # current position of sampling
        self.sample_num = 0
        cfg = configparser.ConfigParser()
        for cfg_file in cfg_files:
            cfg.read(cfg_file)
            data_dict=self.config2data(cfg,QP)
            self.list.append(data_dict)
        for i in range(len(self.list)):
            frame=self.list[i]['frame']
            self.list[i]['column']=(self.list[i]['height']-batch_side)//self.stride + 1
            self.list[i]['row']=(self.list[i]['width']-batch_side)//self.stride+1
            self.list[i]['pieces']=frame*self.list[i]['column']*self.list[i]['row']
            self.pieces+=self.list[i]['pieces']
        self.index=arange(0,self.pieces)
        self.RUN=True
        #self.start()

    def run(self):
        self.buffering()

    def config2data(self, cfg, QP):
        cfg_dict = {}
        cfg_dict['ori_file'] = cfg['path']['ori']
        cfg_dict['anchor_file'] = cfg['path']['file1'] + 'Q%d.yuv' % QP
        frame = cfg_dict['frame'] = cfg.getint('size', 'frame')
        height = cfg_dict['height'] = cfg.getint('size', 'height')
        width = cfg_dict['width'] = cfg.getint('size', 'width')
        # read data into memory
        print('start preparing data ' + cfg_dict['ori_file'])
        ori_fp = open(cfg_dict['ori_file'], 'rb')
        recon_fp = open(cfg_dict['anchor_file'], 'rb')
        ori_buf = ori_fp.read()
        recon_buf = recon_fp.read()
        ori_fp.close()
        recon_fp.close()
        # build data array
        cfg_dict['ori_Y'] = zeros((frame, height, width), uint8, 'C')
        cfg_dict['anchor_Y'] = zeros((frame, height, width), uint8, 'C')
        for i in range(frame):
            # get Y channel
            frame_size = round(height * (width + width / 2))
            Y_size = height * width
            cfg_dict['ori_Y'][i] = frombuffer(ori_buf[i * frame_size:i * frame_size + Y_size], dtype=uint8,
                                              count=Y_size).reshape(height, width)
            cfg_dict['anchor_Y'][i] = frombuffer(recon_buf[i * frame_size:i * frame_size + Y_size], dtype=uint8,
                                                 count=Y_size).reshape(height, width)
        del ori_buf, recon_buf
        print(cfg_dict['ori_file']+' prepared.')
        return cfg_dict

    def get_frame(self,seq=0,frame=0):
        if seq>len(self.list):
            print('sequence index overflow')
            exit(1)
        if frame>self.list[seq]['frame']:
            print('frame index overflow')
            exit(1)
        ori=self.list[seq]['ori_Y'][frame]
        recon=self.list[seq]['anchor_Y'][frame]
        return [ori,recon]

    def get_piece(self,piece_num):
        # piece_num is a global count
        i=0
        for i in range(len(self.list)):
            if piece_num<self.list[i]['pieces']:
                break
            else:
                piece_num-=self.list[i]['pieces']
        # local count now
        side_len=self.batch_side
        stride=self.stride
        column=self.list[i]['column']
        row=self.list[i]['row']
        frm=piece_num//(column*row)
        r=(piece_num%(row*column))//row
        col=(piece_num%(row*column))%row
        ori=self.list[i]['ori_Y'][frm,r*stride:r*stride+side_len,col*stride:col*stride+side_len]
        anchor=self.list[i]['anchor_Y'][frm,r*stride:r*stride+side_len,col*stride:col*stride+side_len]
        return [ori,anchor]

    def get_batch(self,size):
        if self.sample_num == 0:
            random.shuffle(self.index)
        ori = zeros((1, 64, 64), uint8, 'C')
        anchor = zeros((1, 64, 64), uint8, 'C')
        for i in range(size):
            index = self.index[self.sample_num]
            piece = self.get_piece(index)
            ori = concatenate((ori, piece[0].reshape(1, self.batch_side, self.batch_side)), 0)
            anchor = concatenate((anchor, piece[1].reshape(1, self.batch_side, self.batch_side)), 0)
            self.sample_num += 1
            self.sample_num %= self.pieces
        ori = delete(ori, 0, 0)
        anchor = delete(anchor, 0, 0)
        ori=ori.reshape(ori.shape[0],ori.shape[1],ori.shape[2],1)
        anchor=anchor.reshape(anchor.shape[0],anchor.shape[1],anchor.shape[2],1)
        return [ori,anchor]

    def buffering(self):
        while self.RUN:
            if self.bufferLock1.acquire(blocking=True,timeout=0.01):
                while len(self.buffer1) < 100:
                    self.buffer1.append(self.get_batch(self.batch_size))
                self.bufferLock1.release()
            if self.bufferLock2.acquire(blocking=True,timeout=0.01):
                while len(self.buffer2) < 100:
                    self.buffer2.append(self.get_batch(self.batch_size))
                self.bufferLock2.release()
            #time.sleep(0.01)

    def next_batch(self):
        if self.current_buffer==1:
            if len(self.buffer1)>0:
                [ori, anchor] = self.buffer1[0]
                del self.buffer1[0]
                return [ori, anchor]
            else:
                self.bufferLock1.release()
                while len(self.buffer2) == 0: pass
                self.bufferLock2.acquire(blocking=True)
                self.current_buffer=2
                [ori, anchor] = self.buffer2[0]
                del self.buffer2[0]
                return [ori, anchor]
        elif self.current_buffer==2:
            if len(self.buffer2)>0:
                [ori, anchor] = self.buffer2[0]
                del self.buffer2[0]
                return [ori, anchor]
            else:
                self.bufferLock2.release()
                while len(self.buffer1)==0:pass
                self.bufferLock1.acquire(blocking=True)
                self.current_buffer=1
                [ori, anchor] = self.buffer1[0]
                del self.buffer1[0]
                return [ori, anchor]
        else:
            while len(self.buffer1) == 0: pass
            self.bufferLock1.acquire()
            self.current_buffer=1
            [ori, anchor] = self.buffer1[0]
            del self.buffer1[0]
            return [ori, anchor]

if __name__ == '__main__':
    u=train_data(22)
    start_time=time.time()
    for idx in range(0, 10):
        batch = u.get_batch(64)
        print(idx)
    run_time=time.time()-start_time
    print('time=%.3f'%run_time)
    ori=batch[0]
    recon=batch[1]
    im=Image.fromarray(concatenate((ori[0],recon[0])))
    im.show()
    u.RUN=False

    '''fp = open('bsd100//BSD100_down_QP37_320x480.yuv', 'rb')
    buf=fp.read(320*480)
    y=Image.frombuffer('L',(320,480),buf,'raw','L',0,1).crop((0,0,160,240))
    y.show()
    #im=Image.fromarray(u.down_Y1[0])
    #print(u.down_Y1[0][192:200,192:200])'''