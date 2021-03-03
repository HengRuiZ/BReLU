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
def n_sigma_of(x,n=3.5):
    if n==3:
        rate=0.99865
    elif n==3.5:
        rate=0.99977
    else:#default
        rate=0.99865
    sorted_x,_=torch.sort(x.view(-1))
    index=round(sorted_x.size()[0]*rate)
    bound=sorted_x[index].item()
    return bound
def mul_shift(max_u):
    for i in range(1,28):
        max_int=127.5*2**i
        if max_int>max_u:
            mul=round(max_int/max_u)
            temp=max_u*mul/2**i
            if(abs(temp-127)<0.5):
                shift=i
                return mul,shift
    return mul,i
def mul_shift_f(ratio,precision=0.02):
    for i in range(0,28):
        max_int=2**i
        if max_int>ratio:
            temp=max_int/ratio
            mul=round(temp)
            if abs(max_int/mul-ratio)<precision*ratio:
                shift=i
                return mul,shift
    return mul,i
class quant_params:
    # torch cpu version
    def __init__(self,conv):
        self.weight=torch.rand_like(conv.weight.detach().cpu())
        self.weight_n=torch.rand_like(self.weight)
        self.weight_q=torch.rand_like(self.weight)
        self.grad=torch.rand_like(self.weight)
        self.channel=self.weight.size()[0]
        self.step_m=0.1
        self.step=[]
        self.blu=1.5
        self.blu_q=10000
        self.mul=1
        self.shift=1
    def quantize_weight(self,weight_src):
        for i in range(self.channel):
            torch.div(self.weight[i], self.step[i], out=self.weight_q[i])
            self.weight_q[i].round_().clamp_(-128, 127)
            self.weight_q[i].mul_(self.step[i])
        weight_src.data.copy_(self.weight_q)
    def quantize_layer(self,conv,bn,f):
        # merge weight to cpu
        self.weight.copy_(conv.weight.detach())
        gama = bn.weight.detach().cpu()
        var = bn.running_var.cpu()
        self.blu=torch.mean(gama).item()
        weight_m = torch.rand_like(self.weight)
        for i in range(self.channel):
            weight_m[i] = self.weight[i] * gama[i] / torch.sqrt(var[i] + bn.eps)
        # get step_m and step
        min_val = weight_m.min().item()
        max_val = weight_m.max().item()
        if abs(min_val / 128) > abs(max_val / 127):
            self.step_m = abs(min_val) / 128
        else:
            self.step_m = abs(max_val) / 127
        for i in range(self.channel):
            self.step.append(self.step_m * math.sqrt(var[i].item() + bn.eps) / gama[i].item())
        # quantize weight
        self.quantize_weight(conv.weight)
        bn.eval()
        bn.weight.detach_()
        f.write(struct.pack('%sf'%self.channel,*self.step))
        f.write(struct.pack('1f',self.step_m))
    def quantize_fc_layer(self,fc,f):
        self.weight.copy_(fc.weight.detach())
        # get step_m and step
        min_val = self.weight.min().item()
        max_val = self.weight.max().item()
        if abs(min_val / 128) > abs(max_val / 127):
            self.step_m = abs(min_val) / 128
        else:
            self.step_m = abs(max_val) / 127
        for i in range(self.channel):
            self.step.append(self.step_m)
        # quantize weight
        self.quantize_weight(fc.weight)
        f.write(struct.pack('%sf' % self.channel, *self.step))
        f.write(struct.pack('1f', self.step_m))
    def quantize_layer_from(self,conv,bn,f):
        bytes=struct.calcsize('%sf'%self.channel)
        buf=f.read(bytes)
        self.step=list(struct.unpack('%sf'%self.channel,buf))
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)[0]
        self.weight.copy_(conv.weight.detach())
        self.quantize_weight(conv.weight)
        bn.eval()
        bn.weight.detach_()
    def quantize_fc_layer_from(self,fc,f):
        bytes = struct.calcsize('%sf' % self.channel)
        buf = f.read(bytes)
        self.step = list(struct.unpack('%sf' % self.channel, buf))
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)[0]
        self.weight.copy_(fc.weight.detach())
        self.quantize_weight(fc.weight)
    def acc_update(self, weight_src):  # inplace
        self.weight_n.copy_(weight_src.detach())
        torch.add(-self.weight_q, self.weight_n, out=self.grad)
        self.weight.add_(self.grad)
        self.quantize_weight(weight_src)
class quant_params_np:
    # numpy version
    def __init__(self,conv):
        self.weight=conv.weight.clone().detach().cpu().numpy()
        self.weight_n=np.empty_like(self.weight)
        self.weight_q=np.empty_like(self.weight)
        self.grad=np.empty_like(self.weight)
        self.channel=self.weight.shape[0]
        self.step_m=0.1
        self.step=[]
    def quantize_weight(self,weight_src):
        for i in range(self.channel):
            self.weight_q[i]=np.clip(np.around(self.weight[i]/self.step[i]),-128, 127)*self.step[i]
        weight_src.data.copy_(torch.from_numpy(self.weight_q))
    def quantize_layer(self,conv,bn,f):
        # merge weight to cpu
        self.weight=conv.weight.detach().clone().cpu().numpy()
        gama = bn.weight.detach().cpu().numpy()
        var = bn.running_var.cpu().numpy()
        weight_m = np.empty_like(self.weight)
        for i in range(self.channel):
            weight_m[i] = self.weight[i] * gama[i] / math.sqrt(var[i] + bn.eps)
        # get step_m and step
        min_val = np.max(weight_m)
        max_val = np.min(weight_m)
        if abs(min_val / 128) > abs(max_val / 127):
            self.step_m = abs(min_val) / 128
        else:
            self.step_m = abs(max_val) / 127
        for i in range(self.channel):
            self.step.append(self.step_m * math.sqrt(var[i] + bn.eps) / gama[i])
        # quantize weight
        self.quantize_weight(conv.weight)
        bn.track_running_stats = False
        bn.weight.detach_()
        f.write(struct.pack('%sf'%self.channel,*self.step))
        f.write(struct.pack('1f',self.step_m))
    def quantize_layer_from(self,conv,bn,f):
        bytes=struct.calcsize('%sf'%self.channel)
        buf=f.read(bytes)
        self.step=struct.unpack('%sf'%self.channel,buf)
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)
        self.weight=conv.weight.detach().clone().cpu().numpy()
        self.quantize_weight(conv.weight)
        bn.track_running_stats = False
        bn.weight.detach_()
    def acc_update(self, weight_src):
        self.weight_n=weight_src.detach().clone().cpu().numpy()
        self.grad=self.weight_n-self.weight_q
        self.weight+=self.grad
        self.quantize_weight(weight_src)
class quant_params_gpu:
    # torch gpu version
    def __init__(self,conv):
        self.weight=torch.rand_like(conv.weight.detach(),device=torch.device('cuda:0'))
        self.weight_n=torch.rand_like(self.weight)
        self.weight_q=torch.rand_like(self.weight)
        self.grad=torch.rand_like(self.weight)
        self.channel=self.weight.size()[0]
        self.step_m=0.1
        self.step=[]
    def quantize_weight(self,weight_src):
        for i in range(self.channel):
            torch.div(self.weight[i], self.step[i], out=self.weight_q[i])
            self.weight_q[i].round_().clamp_(-128, 127)
            self.weight_q[i].mul_(self.step[i])
        weight_src.data.copy_(self.weight_q)
    def quantize_layer(self,conv,bn,f):
        # merge weight to cpu
        self.weight.copy_(conv.weight.detach())
        gama = bn.weight.detach().cpu()
        var = bn.running_var.cpu()
        weight_m = torch.rand_like(self.weight)
        for i in range(self.channel):
            weight_m[i] = self.weight[i] * gama[i] / torch.sqrt(var[i] + bn.eps)
        # get step_m and step
        min_val = weight_m.min().item()
        max_val = weight_m.max().item()
        if abs(min_val / 128) > abs(max_val / 127):
            self.step_m = abs(min_val) / 128
        else:
            self.step_m = abs(max_val) / 127
        for i in range(self.channel):
            self.step.append(self.step_m * math.sqrt(var[i].item() + bn.eps) / gama[i].item())
        # quantize weight
        self.quantize_weight(conv.weight)
        bn.track_running_stats = False
        bn.weight.detach_()
        f.write(struct.pack('%sf'%self.channel,*self.step))
        f.write(struct.pack('1f',self.step_m))
    def quantize_layer_from(self,conv,bn,f):
        bytes=struct.calcsize('%sf'%self.channel)
        buf=f.read(bytes)
        self.step=struct.unpack('%sf'%self.channel,buf)
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)
        self.weight.copy_(conv.weight.detach())
        self.quantize_weight(conv.weight)
        bn.track_running_stats = False
        bn.weight.detach_()
    def acc_update(self, weight_src):  # inplace
        self.weight_n.copy_(weight_src.detach())
        torch.add(-self.weight_q, self.weight_n, out=self.grad)
        self.weight.add_(self.grad)
        self.quantize_weight(weight_src)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        # quant variable
        self.in_ratio=1
        self.out_ratio=1
        self.quant1=quant_params(self.conv1)
        self.quant2=quant_params(self.conv2)
        self.mul_d=1
        self.shift_d=1
        if self.downsample is not None:
            self.quantd=quant_params(self.downsample._modules['0'])
        self.avg_blu1=AverageMeter()
        self.avg_blu2=AverageMeter()
    def quantize(self,f):
        self.quant1.quantize_layer(self.conv1,self.bn1,f)
        self.quant2.quantize_layer(self.conv2,self.bn2,f)
        if self.downsample is not None:
            self.quantd.quantize_layer(self.downsample._modules['0'],self.downsample._modules['1'],f)
    def quantize_from(self,f):
        self.quant1.quantize_layer_from(self.conv1, self.bn1, f)
        self.quant2.quantize_layer_from(self.conv2, self.bn2, f)
        if self.downsample is not None:
            self.quantd.quantize_layer_from(self.downsample._modules['0'], self.downsample._modules['1'], f)
    def acc_update(self):
        self.quant1.acc_update(self.conv1.weight)
        self.quant2.acc_update(self.conv2.weight)
        if self.downsample is not None:
            self.quantd.acc_update(self.downsample._modules['0'].weight)
    def blu(self,pre_ratio,in_ratio,f):
        self.in_ratio=in_ratio
        # n sigma clip
        #std1 = torch.sqrt(torch.var(self.bn1.bias) + torch.mean(self.bn1.weight**2)).item()
        #std2 = torch.sqrt(torch.var(self.bn2.bias) + torch.mean(self.bn2.weight ** 2)).item()
        #mean1=torch.mean(self.bn1.bias).item()
        #mean2=torch.mean(self.bn2.bias).item()
        #self.quant1.blu=std1*3+mean1
        #self.quant2.blu=std2*3*1.414+mean2 # var after residual
        self.quant1.blu=self.avg_blu1.avg
        self.quant2.blu=self.avg_blu2.avg
        # get blu for conv1
        ratio_conv1 = in_ratio / self.quant1.step_m
        self.quant1.blu_q = self.quant1.blu * ratio_conv1
        self.quant1.mul, self.quant1.shift = mul_shift(self.quant1.blu_q)
        self.quant1.blu = 127 * 2 ** self.quant1.shift / self.quant1.mul / ratio_conv1
        self.quant1.blu_q = round(self.quant1.blu * ratio_conv1)
        ratio_conv1_q=ratio_conv1*self.quant1.mul/2**self.quant1.shift
        if self.downsample is None:
            # synchronize dowmsample
            self.mul_d,self.shift_d=mul_shift_f(pre_ratio/(ratio_conv1_q/self.quant2.step_m), precision=0.01)
            self.quant2.step_m=ratio_conv1_q/(pre_ratio*self.mul_d/2**self.shift_d)
        else:
            # synchronize dowmsample
            ratio_convd=in_ratio/self.quantd.step_m
            self.mul_d, self.shift_d = mul_shift_f(ratio_convd/(ratio_conv1_q / self.quant2.step_m ),precision=0.01)
            self.quant2.step_m = ratio_conv1_q / (ratio_convd * self.mul_d / 2 ** self.shift_d)
        # get blu for conv2
        ratio_conv2 = ratio_conv1_q / self.quant2.step_m
        self.quant2.blu_q = self.quant2.blu * ratio_conv2
        self.quant2.mul, self.quant2.shift = mul_shift(self.quant2.blu_q)
        self.quant2.blu = 127 * 2 ** self.quant2.shift / self.quant2.mul/ratio_conv2
        self.quant2.blu_q = round(self.quant2.blu * ratio_conv2)
        f.write(struct.pack('3f', self.quant1.blu,self.quant2.blu,self.quant2.step_m))
        f.write(struct.pack('2f',in_ratio,ratio_conv1_q))
        f.write(struct.pack('9i',self.quant1.blu_q,self.quant1.mul,self.quant1.shift,
                        self.quant2.blu_q,self.quant2.mul,self.quant2.shift,
                        0,self.mul_d,self.shift_d))
        return ratio_conv2,ratio_conv2 * self.quant2.mul / 2 ** self.quant2.shift
    def load_blu(self,f):
        #load blu
        self.quant1.blu, self.quant2.blu,self.quant2.step_m=struct.unpack('3f',f.read(struct.calcsize('3f')))
        in_ratio,ratio_conv1_q=struct.unpack('2f',f.read(struct.calcsize('2f')))
        bluq1,mul1,shift1,bluq2,mul2,shift2,blud,muld,shiftd = struct.unpack('9i',f.read(struct.calcsize('9i')))
        #and modify steps
        gama = self.bn2.weight.detach().cpu()
        var = self.bn2.running_var.cpu()
        for i in range(gama.size()[0]):
            self.quant2.step[i]=self.quant2.step_m * math.sqrt(var[i].item() + self.bn2.eps) / gama[i].item()
        return
    def forward_blu(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.clamp(out,0,self.quant1.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = torch.clamp(out,0,self.quant2.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant2.blu))
        return out
    def forward_stat_blu(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        self.avg_blu1.update(n_sigma_of(out))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        self.avg_blu2.update(n_sigma_of(out))
        return out
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # quant variable
        self.in_ratio = 1
        self.out_ratio = 1
        self.quant1 = quant_params(self.conv1)
        self.quant2 = quant_params(self.conv2)
        self.quant3 = quant_params(self.conv3)
        self.mul_d = 1
        self.shift_d = 1
        if self.downsample is not None:
            self.quantd = quant_params(self.downsample._modules['0'])
        self.avg_blu1 = AverageMeter()
        self.avg_blu2 = AverageMeter()
        self.avg_blu3 = AverageMeter()

    def quantize(self, f):
        self.quant1.quantize_layer(self.conv1, self.bn1, f)
        self.quant2.quantize_layer(self.conv2, self.bn2, f)
        self.quant3.quantize_layer(self.conv3, self.bn3, f)
        if self.downsample is not None:
            self.quantd.quantize_layer(self.downsample._modules['0'], self.downsample._modules['1'], f)

    def quantize_from(self, f):
        self.quant1.quantize_layer_from(self.conv1, self.bn1, f)
        self.quant2.quantize_layer_from(self.conv2, self.bn2, f)
        self.quant3.quantize_layer_from(self.conv3, self.bn3, f)
        if self.downsample is not None:
            self.quantd.quantize_layer_from(self.downsample._modules['0'], self.downsample._modules['1'], f)

    def acc_update(self):
        self.quant1.acc_update(self.conv1.weight)
        self.quant2.acc_update(self.conv2.weight)
        self.quant3.acc_update(self.conv3.weight)
        if self.downsample is not None:
            self.quantd.acc_update(self.downsample._modules['0'].weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # quant variable
        self.layers=layers
        self.quant1=quant_params(self.conv1)
        self.quantfc=quant_params(self.fc)
        self.avg_blu=AverageMeter()
        self.avg_fc_blu=AverageMeter()
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
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def quantize(self,fn):
        f = open(fn,'wb')
        self.quant1.quantize_layer(self.conv1,self.bn1,f)
        for block in self.layer1._modules:
            self.layer1._modules[block].quantize(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].quantize(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].quantize(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].quantize(f)
        self.quantfc.quantize_fc_layer(self.fc,f)
        f.close()
    def quantize_from(self,fn):
        f=open(fn,'rb')
        self.quant1.quantize_layer_from(self.conv1, self.bn1, f)
        for block in self.layer1._modules:
            self.layer1._modules[block].quantize_from(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].quantize_from(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].quantize_from(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].quantize_from(f)
        self.quantfc.quantize_fc_layer_from(self.fc,f)
        f.close()
    def blu(self,fn):
        f=open(fn,'wb')
        ratio=255*0.226/self.quant1.step_m
        self.quant1.blu=self.avg_blu.avg
        self.quant1.blu_q = self.quant1.blu * ratio
        self.quant1.mul, self.quant1.shift = mul_shift(self.quant1.blu_q)
        self.quant1.blu = 127 * 2 ** self.quant1.shift / self.quant1.mul / ratio
        self.quant1.blu_q = round(self.quant1.blu * ratio)
        f.write(struct.pack('f',self.quant1.blu))
        f.write(struct.pack('3i',self.quant1.blu_q,self.quant1.mul,self.quant1.shift))
        ratio_q=ratio*self.quant1.mul/2**self.quant1.shift
        for block in self.layer1._modules:
            ratio,ratio_q=self.layer1._modules[block].blu(ratio,ratio_q,f)
        for block in self.layer2._modules:
            ratio,ratio_q=self.layer2._modules[block].blu(ratio,ratio_q,f)
        for block in self.layer3._modules:
            ratio,ratio_q=self.layer3._modules[block].blu(ratio,ratio_q,f)
        for block in self.layer4._modules:
            ratio,ratio_q=self.layer4._modules[block].blu(ratio,ratio_q,f)
        self.quantfc.blu=self.avg_fc_blu.avg
        self.quantfc.blu_q=self.quantfc.blu*ratio
        self.quantfc.mul, self.quantfc.shift = mul_shift(self.quantfc.blu_q)
        self.quantfc.blu = 127 * 2 ** self.quantfc.shift / self.quantfc.mul / ratio
        self.quantfc.blu_q = round(self.quantfc.blu * ratio)
        f.write(struct.pack('2f',self.quantfc.blu, ratio*self.quantfc.mul/2**self.quantfc.shift))
        f.write(struct.pack('3i',self.quantfc.blu_q,self.quantfc.mul,self.quantfc.shift))
        f.close()
    def load_blu(self,fn):
        f = open(fn, 'rb+')
        self.quant1.blu=struct.unpack('f',f.read(struct.calcsize('f')))[0]
        f.read(struct.calcsize('3i'))
        for block in self.layer1._modules:
            self.layer1._modules[block].load_blu(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].load_blu(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].load_blu(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].load_blu(f)
        self.layer4._modules[str(self.layers[3]-1)].quant2.blu=1000 # simulate relu
        self.quantfc.blu=struct.unpack('f',f.read(struct.calcsize('f')))[0]
        f.close()
    def forward_blu(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        #fig, ax = plt.subplots()
        #ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        x = self.maxpool(x)
        x = torch.clamp(x,0,self.quant1.blu)
        #fig, ax = plt.subplots()
        #ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        for block in self.layer1._modules:
            x=self.layer1._modules[block].forward_blu(x)
        for block in self.layer2._modules:
            x=self.layer2._modules[block].forward_blu(x)
        for block in self.layer3._modules:
            x=self.layer3._modules[block].forward_blu(x)
        for block in self.layer4._modules:
            x=self.layer4._modules[block].forward_blu(x)
        x = self.avgpool(x)
        x = torch.clamp(x, 0, self.quantfc.blu)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def forward_stat_blu(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        self.avg_blu.update(n_sigma_of(x))
        for block in self.layer1._modules:
            x = self.layer1._modules[block].forward_stat_blu(x)
        for block in self.layer2._modules:
            x = self.layer2._modules[block].forward_stat_blu(x)
        for block in self.layer3._modules:
            x = self.layer3._modules[block].forward_stat_blu(x)
        for block in self.layer4._modules:
            x = self.layer4._modules[block].forward_stat_blu(x)
        x = self.avgpool(x)
        self.avg_fc_blu.update(n_sigma_of(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def acc_update(self):
        self.quant1.acc_update(self.conv1.weight)
        for block in self.layer1._modules:
            self.layer1._modules[block].acc_update()
        for block in self.layer2._modules:
            self.layer2._modules[block].acc_update()
        for block in self.layer3._modules:
            self.layer3._modules[block].acc_update()
        for block in self.layer4._modules:
            self.layer4._modules[block].acc_update()
        self.quantfc.acc_update(self.fc.weight)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='resnet18'))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'],model_dir='resnet34'))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],model_dir='resnet50'))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],model_dir='resnet101'))
    return model


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

def save_onnx():
    dummy_input = torch.randn(50, 3, 224, 224, device='cuda')
    model = resnet152(pretrained=True).cuda()
    input_names = ["actual_input_1"]
    output_names = ["output1"]

    torch.onnx.export(model, dummy_input, "resnet152.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)
if __name__ == '__main__':
    save_onnx()
    #model=resnet152(pretrained=True)
    #checkpoint = torch.load('resnet18\\checkpoint_renorm89.pth.tar')
    #model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
    #model.quantize('quant_param152.data')
