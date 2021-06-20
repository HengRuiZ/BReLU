# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import struct
#import matplotlib.pyplot as plt
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def NCHW2NCHW_VECT_C(NCHW):
    n,c,h,w=NCHW.shape
    outchannel=math.ceil(c/4.0)
    VECT_C=np.zeros(shape=[n,outchannel,h,w,4],dtype=NCHW.dtype)
    for i in range(n):
        for j in range(c):
            VECT_C[i,j//4,:,:,j%4]=NCHW[i,j]
    return VECT_C

def NCHW_VECT_C2NCHW(NCHW_VECT_C,outChannel):
    n,c,h,w,_=NCHW_VECT_C.shape
    NCHW=np.zeros(shape=[n,outChannel,h,w],dtype=NCHW_VECT_C.dtype)
    for i in range(n):
        for j in range(outChannel):
            NCHW[i,j]=NCHW_VECT_C[i,j//4,:,:,j%4]
    return NCHW

class Quant_param(object):
    def __init__(self):
        self.mul=1
        self.shift=0
        self.blu=10000
    def load_param(self,f,conv):
        n,c,h,w=conv.weight.shape
        weight_VECT_C=np.fromfile(f,np.int8,n*((c+3)//4)*h*w*4).reshape(n,(c+3)//4,h,w,4)
        weight_NCHW=NCHW_VECT_C2NCHW(weight_VECT_C,c)
        weight_tensor=torch.from_numpy(weight_NCHW.astype(np.float32))
        conv.weight.data.copy_(weight_tensor)
        bias=np.fromfile(f,np.int32,n)
        conv.bias.data.copy_(torch.from_numpy(bias))
        self.blu,self.mul,self.shift=struct.unpack('3i',f.read(struct.calcsize('3i')))
        return

    def load_param_fc(self,f,fc):
        n,c=fc.weight.shape
        weight=np.fromfile(f,np.int8,n*c).reshape(n,c)
        weight_tensor=torch.from_numpy(weight.astype(np.float32))
        fc.weight.data.copy_(weight_tensor)
        bias=np.fromfile(f,np.int32,n)
        fc.bias.data.copy_(torch.from_numpy(bias))
        self.blu,self.mul,self.shift=struct.unpack('3i',f.read(struct.calcsize('3i')))
        return

    def quantize(self,x):
        x_int=x.round_().to(torch.int32)
        x_int=torch.clamp(x_int,0,self.blu)
        x_int=x_int*self.mul+2**(self.shift-1)
        x_int=x_int>>self.shift
        return x_int.to(torch.float32)

def quantize_d(x,mul,shift):
    x_int = x.to(torch.int32)
    x_int=x_int*mul+2**(shift-1)
    x_int=x_int>>shift
    return x_int.to(torch.float32)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        #quant variables
        self.quant1=Quant_param()
        self.quant2=Quant_param()
        self.mul_d=1
        self.shift_d=0
        self.quant3=Quant_param()
        if self.downsample is not None:
            self.quantd = Quant_param()

    def load_param(self,f):
        self.quant1.load_param(f,self.conv1)
        self.quant2.load_param(f,self.conv2)
        self.mul_d,self.shift_d=struct.unpack('2i',f.read(struct.calcsize('2i')))
        self.quant3.load_param(f,self.conv3)
        if self.downsample is not None:
            self.quantd.load_param(f,self.downsample._modules['0'])
        return
    def acc_update(self):
        self.quant1.acc_update(self.conv1.weight,self.bn1.bias)
        self.quant2.acc_update(self.conv2.weight,self.bn2.bias)
        self.quant3.acc_update(self.conv3.weight,self.bn3.bias)
        if self.downsample is not None:
            self.quantd.acc_update(self.downsample._modules['0'].weight,self.downsample._modules['1'].bias)
    def forward_int(self, x_u, x_v):
        residual = x_u
        out = self.conv1(x_v)
        out = self.quant1.quantize(out)
        out = self.conv2(out)
        out = self.quant2.quantize(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x_v)
        residual=quantize_d(residual,self.mul_d,self.shift_d)
        out += residual
        out_v=self.quant3.quantize(out)
        return out.clamp(0,self.quant3.blu),out_v
    def forward_avg_pool(self,x_u,x_v,avg_pool):
        residual = x_u
        out = self.conv1(x_v)
        out = self.quant1.quantize(out)
        out = self.conv2(out)
        out = self.quant2.quantize(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x_v)
        residual = quantize_d(residual, self.mul_d, self.shift_d)
        out += residual
        out = avg_pool(out)
        out = self.quant3.quantize(out)
        return out
    def forward_quant(self,x_u,x_v,normal=True):
        residual = x_u
        out = self.conv1(x_v)
        out = self.bn1(out)
        out = torch.clamp(out, 0, self.quant1.blu)
        out = torch.round(out / (self.quant1.blu / 127))
        out=out*(self.quant1.blu/127)
        # fig, ax = plt.subplots()
        # ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.clamp(out, 0, self.quant2.blu)
        out = torch.round(out / (self.quant2.blu / 127))
        out=out*(self.quant2.blu/127)
        # fig, ax = plt.subplots()
        # ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant2.blu))
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x_v)
        residual=torch.round(residual*self.quant3.conv_ratio)
        residual=residual/self.quant3.conv_ratio
        out += residual
        out = torch.clamp(out, 0, self.quant3.blu)
        if normal:
            out_v = torch.round(out / (self.quant3.blu / 127))
            out_v = out_v*(self.quant3.blu/127)
        else:
            out_v=out
        # fig, ax = plt.subplots()
        # ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant3.blu))
        return out,out_v


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # quant variable
        self.quant1=Quant_param()
        self.quantfc=Quant_param()
        self.layers=layers
        # initialize params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def acc_update(self):
        self.quant1.acc_update(self.conv1.weight,self.bn1.bias)
        for block in self.layer1._modules:
            self.layer1._modules[block].acc_update()
        for block in self.layer2._modules:
            self.layer2._modules[block].acc_update()
        for block in self.layer3._modules:
            self.layer3._modules[block].acc_update()
        for block in self.layer4._modules:
            self.layer4._modules[block].acc_update()
        self.quantfc.acc_update(self.fc.weight,self.fc.bias)
    def load_param(self,fn):
        f = open(fn, 'rb')
        self.quant1.load_param(f,self.conv1)
        for block in self.layer1._modules:
            self.layer1._modules[block].load_param(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].load_param(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].load_param(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].load_param(f)
        self.quantfc.load_param_fc(f,self.fc)
        f.close()
    def forward_quant(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = torch.clamp(x, 0, self.quant1.blu)
        x_v=torch.round(x/(self.quant1.blu/127))
        x_v=x_v*(self.quant1.blu/127)
        for block in self.layer1._modules:
            x,x_v = self.layer1._modules[block].forward_quant(x,x_v)
        for block in self.layer2._modules:
            x,x_v = self.layer2._modules[block].forward_quant(x,x_v)
        for block in self.layer3._modules:
            x,x_v = self.layer3._modules[block].forward_quant(x,x_v)
        for i in range(self.layers[3]-1):
            x,x_v = self.layer4._modules[str(i)].forward_quant(x,x_v)
        x,x_v=self.layer4._modules[str(self.layers[3]-1)].forward_quant(x,x_v,normal=False)
        x = self.avgpool(x)
        x = torch.clamp(x, 0, self.avg_blu_avgpool.avg)
        x=torch.round(x/(self.avg_blu_avgpool.avg/127))
        x=x*(self.avg_blu_avgpool.avg/127)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def forward(self,x):
        x = self.conv1(x)
        x_u = self.maxpool(x)
        x_v=self.quant1.quantize(x_u)
        for block in self.layer1._modules:
            x_u, x_v = self.layer1._modules[block].forward_int(x_u, x_v)
        for block in self.layer2._modules:
            x_u, x_v = self.layer2._modules[block].forward_int(x_u, x_v)
        for block in self.layer3._modules:
            x_u, x_v = self.layer3._modules[block].forward_int(x_u, x_v)
        for i in range(self.layers[3] - 1):
            x_u, x_v = self.layer4._modules[str(i)].forward_int(x_u, x_v)
        x = self.layer4._modules[str(self.layers[3] - 1)].forward_avg_pool(x_u,x_v,self.avgpool)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'],model_dir='resnet152'))
    return model

def moduledict_to_dict(moduledict):
    dict={}
    for key in moduledict:
        key_new=key.split('module.')[-1]
        dict[key_new]=moduledict[key]
    return dict
if __name__ == '__main__':
    model=resnet152(pretrained=False).cuda()
    model.load_param('resnet152_blu101.data')
    pass
