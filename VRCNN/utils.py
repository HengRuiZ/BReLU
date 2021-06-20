#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import h5py
import random
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS


def psnr(target, ref):
    target_data = np.asarray(target, 'f')
    ref_data = np.asarray(ref, 'f')
    target_data=np.round(target_data)
    ref_data=np.round(ref_data)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = np.mean(diff ** 2.)
    return 10 * math.log10(255 ** 2. / rmse)

def relu_sigma(array):
    sigma=0
    count=0
    for elem in np.nditer(array):
        if elem>0:
            count+=1
            sigma+=elem*elem
    sigma/=count
    sigma=np.sqrt(sigma)
    return sigma
def readYuvFile(filename,width,height,numfrm,startfrm,mode=1):#mode=1生成的数据是二维的,mode=2数据是一维的
    Y, U, V = [], [], []
    fp = open(filename, 'rb')
    blk_size = width * height * 3 // 2
    fp.seek(blk_size * startfrm, 0)
    uv_width = width // 2
    uv_height = height // 2

    # if mode == 2:
    Yt2 = np.zeros((numfrm, width * height))
    Ut2 = np.zeros((numfrm, uv_width * uv_height))
    Vt2 = np.zeros((numfrm, uv_width * uv_height))

    for i in range(numfrm):
        if mode == 1:
            Yt = np.zeros((height, width), np.uint8, 'C')
            Ut = np.zeros((uv_height, uv_width),np.uint8, 'C')
            Vt = np.zeros((uv_height, uv_width), np.uint8, 'C')
            for m in range(height):
                for n in range(width):
                    Yt[m, n] = ord(fp.read(1))
            for m in range(uv_height):
                for n in range(uv_width):
                    Vt[m, n] = ord(fp.read(1))
            for m in range(uv_height):
                for n in range(uv_width):
                    Ut[m, n] = ord(fp.read(1))
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]
        else:
            for m in range(height):
                for n in range(width):
                    Yt2[i, m * width + n] = ord(fp.read(1))
            for m in range(uv_height):
                for n in range(uv_width):
                    Vt2[i, m * uv_width + n] = ord(fp.read(1))
            for m in range(uv_height):
                for n in range(uv_width):
                    Ut2[i, m * uv_width + n] = ord(fp.read(1))
    fp.close()
    if mode == 1:
        return (Y, U, V)
    else:
        return (Yt2, Vt2, Ut2)

def getYasInput(filename, width,height, numfrm, startfrm,mode):
    YUV_data=readYuvFile(filename,width,height,numfrm,startfrm,mode)
    return YUV_data[0]

def getUasInput(filename, width,height, numfrm, startfrm,mode):
    YUV_data=readYuvFile(filename,width,height,numfrm,startfrm,mode)
    return YUV_data[1]

def getVasInput(filename, width,height, numfrm, startfrm,mode):
    YUV_data=readYuvFile(filename,width,height,numfrm,startfrm,mode)
    return YUV_data[2]


def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

def merge(config,images, size):
    img = np.zeros((config.yuv_height, config.yuv_width, 1))
    for idx, image in enumerate(images):  # size[1]是表示宽
        i = idx % size[1]
        j = idx // size[1]
        if i < size[1] - 1 and j < size[0] - 1:
            # print 1
            # print i*config.sub_image_size+config.sub_image_size
            img[j * config.sub_image_size:j * config.sub_image_size + config.sub_image_size,
            i * config.sub_image_size:i * config.sub_image_size + config.sub_image_size, :] = image
        elif i == size[1] - 1 and j != size[0] - 1:
            # print 2
            img[j * config.sub_image_size:j * config.sub_image_size + config.sub_image_size,
            config.yuv_width - config.sub_image_size:config.yuv_width, :] = image
        elif i != size[1] - 1 and j == size[0] - 1:
            # print 3
            img[config.yuv_height - config.sub_image_size:config.yuv_height,
            i * config.sub_image_size:i * config.sub_image_size + config.sub_image_size, :] = image
        else:
            # print 4
            img[config.yuv_height - config.sub_image_size:config.yuv_height,
            config.yuv_width - config.sub_image_size:config.yuv_width, :] = image

    return img

def make_data(data, label):
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'data/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'data/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def input_setup(config):
    input_=getYasInput(config.input,config.yuv_width,config.yuv_height,config.num_frames,0,1)
    label_=getYasInput(config.label,config.yuv_width,config.yuv_height,config.num_frames,0,1)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.sub_image_size - config.sub_label_size) / 2  # 6

    nx=ny=0
    if config.is_train:
        for frame in range(config.num_frames):
            for x in range(0, config.yuv_height - config.sub_image_size + 1, config.stride):
                for y in range(0, config.yuv_width - config.sub_image_size + 1, config.stride):
                    sub_input = input_[frame][x:x + config.sub_image_size, y:y + config.sub_image_size]/255.  # [33 x 33]
                    sub_label = label_[frame][x + padding:x + padding + config.sub_label_size,
                                y + padding:y + padding + config.sub_label_size]/255.  # [21 x 21]

                    sub_input = sub_input.reshape([config.sub_image_size, config.sub_image_size, 1])
                    sub_label = sub_label.reshape([config.sub_label_size, config.sub_label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:
        for frame in range(config.num_frames):
            for x in range(0, config.yuv_height, config.stride):
                nx+=1
                ny=0
                for y in range(0, config.yuv_width, config.stride):
                    ny+=1
                    if y+config.sub_image_size>config.yuv_width-1 and x+config.sub_image_size<config.yuv_height-1:
                        sub_input = input_[frame][x:x + config.sub_image_size,
                                    config.yuv_width-config.sub_image_size:config.yuv_width] / 255.
                        sub_label = label_[frame][x + padding:x + padding + config.sub_label_size,
                                    config.yuv_width-config.sub_image_size:config.yuv_width] / 255.

                        h, w = sub_input.shape

                        sub_input = sub_input.reshape([h, w, 1])
                        sub_label = sub_label.reshape([h, w, 1])

                        sub_input_sequence.append(sub_input)
                        sub_label_sequence.append(sub_label)

                    elif y+config.sub_image_size<config.yuv_width-1 and x+config.sub_image_size>config.yuv_height-1:
                        sub_input = input_[frame][config.yuv_height-config.sub_image_size:config.yuv_height,
                                    y:y + config.sub_image_size] / 255.  # [33 x 33]
                        sub_label = label_[frame][config.yuv_height-config.sub_image_size:config.yuv_height,
                                    y + padding:y + padding + config.sub_label_size] / 255.  # [21 x 21]

                        h, w = sub_input.shape

                        sub_input = sub_input.reshape([h, w, 1])
                        sub_label = sub_label.reshape([h, w, 1])

                        sub_input_sequence.append(sub_input)
                        sub_label_sequence.append(sub_label)


                    elif y+config.sub_image_size>config.yuv_width-1 and x+config.sub_image_size>config.yuv_height-1:
                        sub_input = input_[frame][config.yuv_height-config.sub_image_size:config.yuv_height,
                                    config.yuv_width-config.sub_image_size:config.yuv_width] / 255.
                        sub_label = label_[frame][config.yuv_height-config.sub_image_size:config.yuv_height,
                                    config.yuv_width-config.sub_image_size:config.yuv_width] / 255.

                        h, w = sub_input.shape

                        sub_input = sub_input.reshape([h, w, 1])
                        sub_label = sub_label.reshape([h, w, 1])

                        sub_input_sequence.append(sub_input)
                        sub_label_sequence.append(sub_label)

                    else:
                        sub_input = input_[frame][x:x + config.sub_image_size,
                                    y:y + config.sub_image_size] / 255.
                        sub_label = label_[frame][x + padding:x + padding + config.sub_label_size,
                                    y + padding:y + padding + config.sub_label_size] / 255.

                        h, w = sub_input.shape
                        # Make channel value
                        sub_input = sub_input.reshape([h, w, 1])
                        sub_label = sub_label.reshape([h, w, 1])

                        sub_input_sequence.append(sub_input)
                        sub_label_sequence.append(sub_label)

    data,label=[],[]
    if config.is_train:
        idx = [i for i in range(len(sub_input_sequence))]
        random.shuffle(idx)
        for i in idx:
            data.append(sub_input_sequence[i])
            label.append(sub_label_sequence[i])
    else:
        data=sub_input_sequence
        label=sub_label_sequence

    # Make list to numpy array. With this transform
    arrdata = np.asarray(data)
    arrlabel = np.asarray(label)


    if config.is_train:
        make_data(arrdata, arrlabel)
    else:
        return nx, ny,input_[0],label_[0],arrdata,arrlabel

def imsave(image, path):
  return scipy.misc.imsave(path, image)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
