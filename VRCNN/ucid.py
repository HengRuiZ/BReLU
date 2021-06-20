# -*- coding: utf-8 -*-
"""
Created on Mar 9 15:50 2018

@author: Hengrui Zhao
"""

from numpy import *
from PIL import Image

class ucid:
    width,height=512,384
    ori_Y1 = zeros((453, 512,384), uint8, 'C')
    ori_Y2 = zeros((885, 384,512), uint8, 'C')
    up_Y1 = zeros((453,512,384), uint8,'C')
    up_Y2 = zeros((885, 384,512), uint8, 'C')
    down_Y1 = zeros((453, 512,384), uint8, 'C')
    down_Y2 = zeros((885, 384,512), uint8, 'C')

    sample_num=0
    def __init__(self,QP):
        ori1_fp = open('../data/HEVC_Sequence/UCID_384x512_453.yuv','rb')
        ori2_fp = open('../data/HEVC_Sequence/UCID_512x384_885.yuv', 'rb')
        up1_fp = open('../data/anchor16.0/UCID_384x512_453_intra_main_HM16.0_anchor_Q%d.yuv'%QP, 'rb')
        up2_fp = open('../data/anchor16.0/UCID_512x384_885_intra_main_HM16.0_anchor_Q%d.yuv'%QP, 'rb')
        down1_fp = open('../data/ucid/barc_down_rec_QP37_384x512.yuv', 'rb')
        down2_fp = open('../data/ucid/barc_down_rec_QP37_512x384.yuv', 'rb')
        ori1_buf=ori1_fp.read()
        ori2_buf=ori2_fp.read()
        up1_buf = up1_fp.read()
        up2_buf = up2_fp.read()
        down1_buf = down1_fp.read()
        down2_buf = down2_fp.read()
        ori1_fp.close()
        ori2_fp.close()
        up1_fp.close()
        up2_fp.close()
        down1_fp.close()
        down2_fp.close()
        for i in range(453):
            self.ori_Y1[i]=frombuffer(ori1_buf[i*384*(512+256):i*384*(512+256)+384*512],dtype=uint8,count=384*512).reshape(512,384)
            self.up_Y1[i] = frombuffer(up1_buf[i*384*(512+256):i*384*(512+256)+384*512],dtype=uint8,count=384*512).reshape(512,384)
            self.down_Y1[i] = frombuffer(down1_buf[i*384*(512+256):i*384*(512+256)+384*512],dtype=uint8,count=384*512).reshape(512,384)
            #if i%10==0:print(i)
        for i in range(885):
            self.ori_Y2[i] = frombuffer(ori2_buf[i * 384 * (512 + 256):i * 384 * (512 + 256) + 384 * 512], dtype=uint8,
                                        count=384 * 512).reshape(384,512)
            self.up_Y2[i] = frombuffer(up2_buf[i * 384 * (512 + 256):i * 384 * (512 + 256) + 384 * 512], dtype=uint8,
                                       count=384 * 512).reshape(384,512)
            self.down_Y2[i] = frombuffer(down2_buf[i * 384 * (512 + 256):i * 384 * (512 + 256) + 384 * 512],
                                         dtype=uint8, count=384 * 512).reshape(384,512)
            #if i%10==0:print(i)
        del ori1_buf,ori2_buf,up1_buf,up2_buf,down1_buf,down2_buf
        #print('init complete')

    def get_frame(self,num):
        if num>=453+885 or num<0:
            print("ucid:no such frame")
            return []
        elif num<453:
            ori=self.ori_Y1[num]
            up=self.up_Y1[num]
            down=self.down_Y1[num]
            return [ori,up,down]
        else:
            ori=self.ori_Y2[num-453]
            up=self.up_Y2[num-453]
            down=self.down_Y2[num-453]
            return [ori,up,down]

    def next_batch(self, batch_num,batch_size=64):
        #batch_size=64
        row=11
        column=15
        if batch_num>15*11*453:
            print('Error:too big batch!')
            exit(1)

        ori=zeros((1,64,64),uint8,'C')
        up=zeros((1,64,64),uint8,'C')
        down=zeros((1,32,32),uint8,'C')
        if batch_num+self.sample_num>row*column*(453+885):self.sample_num=0
        if batch_num+self.sample_num<=row*column*453:
                for i in range(batch_num):
                    ori=concatenate((ori,self.ori_Y1[self.sample_num//(row*column),
                        ((self.sample_num%(row*column))//row)*32:((self.sample_num%(row*column))//row)*32+64,
                        ((self.sample_num%(row*column))%row)*32:((self.sample_num%(row*column))%row)*32+64].reshape(1,64,64)),0)
                    up=concatenate((up,self.up_Y1[self.sample_num//(row*column),
                        ((self.sample_num%(row*column))//row)*32:((self.sample_num%(row*column))//row)*32+64,
                        ((self.sample_num%(row*column))%row)*32:((self.sample_num%(row*column))%row)*32+64].reshape(1,64,64)),0)
                    down=concatenate((down,self.down_Y1[self.sample_num//(row*column),
                        ((self.sample_num%(row*column))//row)*16:((self.sample_num%(row*column))//row)*16+32,
                        ((self.sample_num%(row*column))%row)*16:((self.sample_num%(row*column))%row)*16+32].reshape(1,32,32)),0)
                    self.sample_num += 1
                    if i%500==0 and i>1:print('batch512*384:%i'%i)
        else:
                n=self.sample_num-row*column*453
                for i in range(batch_num):
                    ori=concatenate((ori,self.ori_Y2[n//(row*column),
                        ((n%(row*column))//column)*32:((n%(row*column))//column)*32+64,
                        ((n%(row*column))%column)*32:((n%(row*column))%column)*32+64].reshape(1,64,64)),0)
                    up=concatenate((up,self.up_Y2[n//(row*column),
                        ((n%(row*column))//column)*32:((n%(row*column))//column)*32+64,
                        ((n%(row*column))%column)*32:((n%(row*column))%column)*32+64].reshape(1,64,64)),0)
                    down=concatenate((down,self.down_Y2[n//(row*column),
                        ((n%(row*column))//column)*16:((n%(row*column))//column)*16+32,
                        ((n%(row*column))%column)*16:((n%(row*column))%column)*16+32].reshape(1,32,32)),0)
                    n+=1
                    if i % 500 == 0 and i>1: print('batch384*512:%i' % i)
                self.sample_num+=batch_num
        ori=delete(ori,0,0)
        up=delete(up,0,0)
        down=delete(down,0,0)
        return [ori,up,down]

    def test(self):
        im = Image.frombytes('L', (512,384), self.ori_Y2[1].tostring())
        im.show()


if __name__ == '__main__':

    u=ucid(22)
    batch=u.next_batch(10)
    ori=batch[0]
    up=batch[1]
    down=batch[2]
    im=Image.fromarray(concatenate((ori[1],up[1])));im.show()

    '''fp = open('bsd100//BSD100_down_QP37_320x480.yuv', 'rb')
    buf=fp.read(320*480)
    y=Image.frombuffer('L',(320,480),buf,'raw','L',0,1).crop((0,0,160,240))
    y.show()
    #im=Image.fromarray(u.down_Y1[0])
    #print(u.down_Y1[0][192:200,192:200])'''