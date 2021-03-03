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
    for i in range(1,29):
        max_int=127.5*2**i
        if max_int>max_u:
            mul=round(max_int/max_u)
            temp=max_u*mul/2**i
            if(abs(temp-127)<0.5):
                shift=i
                return mul,shift
    return mul,i
def mul_shift_f(div,precision=0.02):
    for i in range(0,28):
        max_int=2**i
        if max_int>div:
            temp=max_int/div
            mul=round(temp)
            if abs(max_int/mul-div)<precision*div:
                shift=i
                return mul,shift
    return mul,i
def NCHW2NCHW_VECT_C(NCHW):
    n,c,h,w=NCHW.shape
    outchannel=math.ceil(c/4.0)
    VECT_C=np.zeros(shape=[n,outchannel,h,w,4],dtype=NCHW.dtype)
    for i in range(n):
        for j in range(c):
            VECT_C[i,j//4,:,:,j%4]=NCHW[i,j]
    return VECT_C
class quant_params:
    # torch cpu version
    def __init__(self,conv):
        self.channel=conv.weight.size()[0]
        self.weight=torch.rand_like(conv.weight.detach().cpu())
        self.weight_n=torch.rand_like(self.weight)
        self.weight_q=torch.rand_like(self.weight)
        self.grad=torch.rand_like(self.weight)

        self.bias_l=torch.zeros(self.channel)
        self.bias_n=torch.zeros(self.channel)
        self.bias_q=torch.zeros(self.channel)
        self.bias_grad=torch.zeros(self.channel)
        self.bias_base=torch.zeros(self.channel)

        self.step_m=0.1
        self.step=[]

        self.in_ratio=0.1
        self.conv_ratio=0.1
        self.out_ratio=0.1
        self.avg_blu=AverageMeter()

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
        return
    def quantize_bias(self,bias_src):
        for i in range(self.channel):
            torch.mul(self.bias_l[i]-self.bias_base[i],self.conv_ratio,out=self.bias_q[i])
            self.bias_q[i].round_()
            self.bias_q[i].div_(self.conv_ratio)
            self.bias_q[i]+=self.bias_base[i]
        bias_src.data.copy_(self.bias_q)
        return
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
        return
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
        return
    def quantize_layer_from(self,conv,bn,f):
        bytes=struct.calcsize('%sf'%self.channel)
        buf=f.read(bytes)
        self.step=list(struct.unpack('%sf'%self.channel,buf))
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)[0]
        self.weight.copy_(conv.weight.detach())
        self.weight_q.copy_(conv.weight.detach())
        #self.quantize_weight(conv.weight)
        self.mean=bn.running_mean.cpu().numpy()
        self.var = bn.running_var.cpu().numpy()
        self.gama = bn.weight.detach().cpu().numpy()
        self.bias=bn.bias.detach().cpu().numpy()
        self.bias_l.copy_(bn.bias.detach())
        bn.eval()
        bn.weight.detach_()
        return
    def quantize_fc_layer_from(self,fc,f):
        bytes = struct.calcsize('%sf' % self.channel)
        buf = f.read(bytes)
        self.step = list(struct.unpack('%sf' % self.channel, buf))
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)[0]
        self.weight.copy_(fc.weight.detach())
        self.weight_q.copy_(fc.weight.detach())
        #self.quantize_weight(fc.weight)
        self.mean=np.zeros(shape=self.channel)
        self.var=np.ones(shape=self.channel)
        self.gama=np.ones(shape=self.channel)
        self.bias=fc.bias.detach().cpu().numpy()
        self.bias_l.copy_(fc.bias.detach())
        return
    def acc_update(self, weight_src,bias_src):  # inplace
        self.weight_n.copy_(weight_src.detach())
        torch.add(-self.weight_q, self.weight_n, out=self.grad)
        self.weight.add_(self.grad)
        self.quantize_weight(weight_src)
        self.bias_n.copy_(bias_src.detach())
        torch.add(-self.bias_q,self.bias_n,out=self.bias_grad)
        self.bias_l.add_(self.bias_grad)
        self.quantize_bias(bias_src)
        return

    def stat_blu(self,x):
        self.avg_blu.update(n_sigma_of(x))
        return

    def quant_blu(self,in_ratio,f):
        self.blu=self.avg_blu.avg
        self.in_ratio=in_ratio
        self.conv_ratio=in_ratio/self.step_m
        self.blu_q=round(self.blu*self.conv_ratio)
        self.mul,self.shift=mul_shift(self.blu_q)
        self.out_ratio=self.conv_ratio*self.mul/(2**self.shift)
        self.blu=127/self.out_ratio
        self.blu_q=round(self.blu*self.conv_ratio)
        f.write(struct.pack('2f', self.conv_ratio,self.blu))
        f.write(struct.pack('3i', self.blu_q,self.mul,self.shift))
        return self.conv_ratio,self.out_ratio

    def load_blu(self,f):
        self.conv_ratio,self.blu=struct.unpack('2f',f.read(struct.calcsize('2f')))
        self.avg_blu.avg=self.blu
        self.blu_q,self.mul,self.shift=struct.unpack('3i',f.read(struct.calcsize('3i')))
        for i in range(self.channel):
            self.bias_base[i]=self.mean[i]*self.gama[i]/math.sqrt(self.var[i]+1e-5)
        self.quantize_bias(self.bias_l)
        return
    def integer_model(self,f):
        weight_i=np.array(self.weight_q.numpy())
        bias_i=np.array(self.bias)
        for i in range(bias_i.shape[0]):
            weight_i[i]=weight_i[i]/self.step[i]
            bias_i[i]=(self.bias[i]-self.mean[i]*self.gama[i]/math.sqrt(self.var[i]+1e-5))*self.conv_ratio
        weight_i=np.round(weight_i).astype(np.int8)
        weight_i=NCHW2NCHW_VECT_C(weight_i)
        weight_i.tofile(f)
        bias_i=np.round(bias_i).astype(np.int32)
        bias_i.tofile(f)
        f.write(struct.pack('3i',self.blu_q,self.mul,self.shift))
        return
    def integer_model_fc(self,f):
        weight_i = np.array(self.weight_q.numpy())
        bias_i = np.array(self.bias)
        weight_i = weight_i / self.step_m
        bias_i = self.bias * self.conv_ratio
        weight_i = np.round(weight_i).astype(np.int8)
        weight_i.tofile(f)
        bias_i = np.round(bias_i).astype(np.int32)
        bias_i.tofile(f)
        f.write(struct.pack('3i', self.blu_q, self.mul, self.shift))
        return


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
        self.quant1.acc_update(self.conv1.weight,self.bn1.bias)
        self.quant2.acc_update(self.conv2.weight,self.bn2.bias)
        self.quant3.acc_update(self.conv3.weight,self.bn3.bias)
        if self.downsample is not None:
            self.quantd.acc_update(self.downsample._modules['0'].weight,self.downsample._modules['1'].bias)
    def quant_blu(self,pre_ratio,in_ratio,f):
        conv_ratio,out_ratio=self.quant1.quant_blu(in_ratio,f)
        conv_ratio,out_ratio=self.quant2.quant_blu(out_ratio,f)
        if self.downsample is None:
            # synchronize dowmsample
            self.mul_d,self.shift_d=mul_shift_f(pre_ratio/(out_ratio/self.quant3.step_m), precision=0.01)
            self.quant3.step_m=out_ratio/(pre_ratio*self.mul_d/2**self.shift_d)
        else:
            # synchronize dowmsample
            ratio_convd=in_ratio/self.quantd.step_m
            self.mul_d, self.shift_d = mul_shift_f(ratio_convd/(out_ratio / self.quant3.step_m ),precision=0.01)
            self.quant3.step_m = out_ratio / (ratio_convd * self.mul_d / 2 ** self.shift_d)
            self.quantd.avg_blu.avg=1.5
        f.write(struct.pack('f', self.quant3.step_m))
        f.write(struct.pack('2i',self.mul_d,self.shift_d))

        conv_ratio,out_ratio=self.quant3.quant_blu(out_ratio,f)
        if self.downsample is not None:
            _ = self.quantd.quant_blu(in_ratio, f)

        return conv_ratio,out_ratio
    def load_blu_new(self,f):
        self.quant1.load_blu(f)
        self.quant2.load_blu(f)
        self.quant3.step_m=struct.unpack('f',f.read(struct.calcsize('f')))[0]
        self.mul_d,self.shift_d=struct.unpack('2i',f.read(struct.calcsize('2i')))
        #and modify steps
        gama = self.bn3.weight.detach().cpu()
        var = self.bn3.running_var.cpu()
        for i in range(gama.size()[0]):
            self.quant3.step[i] = self.quant3.step_m * math.sqrt(var[i].item() + self.bn3.eps) / gama[i].item()
        self.quant3.load_blu(f)
        if self.downsample is not None:
            self.quantd.load_blu(f)
        return
    def load_blu(self,f):
        self.quant1.load_blu(f)
        self.quant2.load_blu(f)
        self.quant3.load_blu(f)
        self.quant3.step_m=struct.unpack('f',f.read(struct.calcsize('f')))[0]
        self.mul_d,self.shift_d=struct.unpack('2i',f.read(struct.calcsize('2i')))
        #and modify steps
        gama = self.bn3.weight.detach().cpu()
        var = self.bn3.running_var.cpu()
        for i in range(gama.size()[0]):
            self.quant3.step[i] = self.quant3.step_m * math.sqrt(var[i].item() + self.bn3.eps) / gama[i].item()
        return
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
    def forward_stat_blu(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        self.quant1.stat_blu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        self.quant2.stat_blu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        self.quant3.stat_blu(out)
        return out
    def forward_blu(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.clamp(out,0,self.quant1.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.clamp(out,0,self.quant2.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant2.blu))
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = torch.clamp(out,0,self.quant3.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant3.blu))
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
    def integer_model(self,f):
        self.quant1.integer_model(f)
        self.quant2.integer_model(f)
        f.write(struct.pack('2i',self.mul_d,self.shift_d))
        self.quant3.integer_model(f)
        if self.downsample is not None:
            self.quantd.integer_model(f)
        return


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
        self.avg_blu_avgpool=AverageMeter()
        self.quantfc=quant_params(self.fc)
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
    def forward_stat_blu(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        self.quant1.stat_blu(x)
        for block in self.layer1._modules:
            x = self.layer1._modules[block].forward_stat_blu(x)
        for block in self.layer2._modules:
            x = self.layer2._modules[block].forward_stat_blu(x)
        for block in self.layer3._modules:
            x = self.layer3._modules[block].forward_stat_blu(x)
        for block in self.layer4._modules:
            x = self.layer4._modules[block].forward_stat_blu(x)
        x = self.avgpool(x)
        self.avg_blu_avgpool.update(n_sigma_of(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def quant_blu(self,fn):
        f=open(fn,'wb')
        in_ratio=255*0.226
        conv_ratio,out_ratio=self.quant1.quant_blu(in_ratio,f)
        for block in self.layer1._modules:
            conv_ratio,out_ratio=self.layer1._modules[block].quant_blu(conv_ratio,out_ratio,f)
        for block in self.layer2._modules:
            conv_ratio,out_ratio=self.layer2._modules[block].quant_blu(conv_ratio,out_ratio,f)
        for block in self.layer3._modules:
            conv_ratio,out_ratio=self.layer3._modules[block].quant_blu(conv_ratio,out_ratio,f)
        self.layer4._modules[str(self.layers[3]-1)].quant3.avg_blu.avg=self.avg_blu_avgpool.avg
        for block in self.layer4._modules:
            conv_ratio,out_ratio=self.layer4._modules[block].quant_blu(conv_ratio,out_ratio,f)
        self.quantfc.avg_blu.avg=1.5#avoid zero division
        self.quantfc.quant_blu(out_ratio,f)
        f.close()
    def load_blu(self,fn):
        f = open(fn, 'rb')
        self.quant1.load_blu(f)
        for block in self.layer1._modules:
            self.layer1._modules[block].load_blu(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].load_blu(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].load_blu(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].load_blu(f)
        self.avg_blu_avgpool.avg=self.layer4._modules[str(self.layers[3]-1)].quant3.blu
        self.layer4._modules[str(self.layers[3]-1)].quant3.blu=10000.0 # simulate relu
        self.quantfc.avg_blu.avg=1.5
        f.close()
    def load_blu_new(self,fn):
        f = open(fn, 'rb')
        self.quant1.load_blu(f)
        for block in self.layer1._modules:
            self.layer1._modules[block].load_blu_new(f)
        for block in self.layer2._modules:
            self.layer2._modules[block].load_blu_new(f)
        for block in self.layer3._modules:
            self.layer3._modules[block].load_blu_new(f)
        for block in self.layer4._modules:
            self.layer4._modules[block].load_blu_new(f)
        self.avg_blu_avgpool.avg=self.layer4._modules[str(self.layers[3]-1)].quant3.blu
        self.layer4._modules[str(self.layers[3]-1)].quant3.blu=10000.0 # simulate relu
        self.quantfc.load_blu(f)
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
        x = torch.clamp(x, 0, self.avg_blu_avgpool.avg)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
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
    def forward_plain(self,x):
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
    def forward(self, x):
        #return self.forward_plain(x)
        return self.forward_blu(x)
        #return self.forward_quant(x)
    def integer_model(self,fn):
        with open(fn,'wb') as f:
            self.quant1.integer_model(f)
            for block in self.layer1._modules:
                x = self.layer1._modules[block].integer_model(f)
            for block in self.layer2._modules:
                x = self.layer2._modules[block].integer_model(f)
            for block in self.layer3._modules:
                x = self.layer3._modules[block].integer_model(f)
            for block in self.layer4._modules:
                x = self.layer4._modules[block].integer_model(f)
            self.quantfc.integer_model_fc(f)
        return


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

def dump_onnx(onnx_name):
    model = resnet152(pretrained=True).cuda()
    dummy_input = torch.randn(50, 3, 224, 224, device='cuda')
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, "resnet152.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)
if __name__ == '__main__':
    model=resnet152(pretrained=False).cuda()
    checkpoint = torch.load('resnet152\\checkpoint_blu3_105.pth.tar')
    model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
    model.quantize_from('quant_param152.data')
    model.load_blu_new('blu152_n3.data')
    #model.avg_blu_avgpool.avg+=2
    #model.quant_blu('blu152_relu6.data')
    model.integer_model('resnet152_blu3_105.data')
    pass
