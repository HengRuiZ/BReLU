import torch
import torch.nn as nn
import struct
from math import sqrt

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
        self.in_ratio=0.1
        self.step_m=0.1
        self.blu=1.5
        self.blu_q=10000
        self.mul=1
        self.shift=1
    def quantize_weight(self,weight_src):
        torch.div(self.weight, self.step_m, out=self.weight_q)
        self.weight_q.round_().clamp_(-128, 127)
        self.weight_q.mul_(self.step_m)
        weight_src.data.copy_(self.weight_q)
    def quantize_layer(self,conv,f):
        # merge weight to cpu
        self.weight.copy_(conv.weight.detach())
        # get step_m and step
        min_val = self.weight.min().item()
        max_val = self.weight.max().item()
        if abs(min_val / 128) > abs(max_val / 127):
            self.step_m = abs(min_val) / 128
        else:
            self.step_m = abs(max_val) / 127
        # quantize weight
        self.quantize_weight(conv.weight)
        f.write(struct.pack('1f',self.step_m))
    def quantize_layer_from(self,conv,f):
        bytes = struct.calcsize('1f')
        buf = f.read(bytes)
        self.step_m = struct.unpack('1f', buf)[0]
        self.weight.copy_(conv.weight.detach())
        self.quantize_weight(conv.weight)
    def acc_update(self, weight_src):  # inplace
        self.weight_n.copy_(weight_src.detach())
        torch.add(-self.weight_q, self.weight_n, out=self.grad)
        self.weight.add_(self.grad)
        self.quantize_weight(weight_src)
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.quant=quant_params(self.conv)
        self.avg_blu=AverageMeter()
    def forward(self, x):
        return self.relu(self.conv(x))

    def quantize(self,f):
        self.quant.quantize_layer(self.conv,f)
    def quantize_from(self,f):
        self.quant.quantize_layer_from(self.conv, f)
    def acc_update(self):
        self.quant.acc_update(self.conv.weight)
    def blu(self,in_ratio,f):
        self.in_ratio=in_ratio
        self.quant.blu=self.avg_blu.avg
        # get blu for conv1
        ratio_conv = in_ratio / self.quant.step_m
        self.quant.blu_q = self.quant.blu * ratio_conv
        self.quant.mul, self.quant.shift = mul_shift(self.quant.blu_q)
        self.quant.blu = 127 * 2 ** self.quant.shift / self.quant.mul / ratio_conv
        self.quant.blu_q = round(self.quant.blu * ratio_conv)
        ratio_conv_q=ratio_conv*self.quant.mul/2**self.quant.shift
        f.write(struct.pack('2f', self.quant.blu,ratio_conv))
        f.write(struct.pack('3i',self.quant.blu_q,self.quant.mul,self.quant.shift))
        return ratio_conv_q
    def load_blu(self,f):
        #load blu
        self.quant.blu,self.quant.in_ratio=struct.unpack('2f',f.read(struct.calcsize('2f')))
        _=struct.unpack('3i',f.read(struct.calcsize('3i')))
        return
    def forward_blu(self, x):
        out = self.conv(x)
        out = torch.clamp(out,0,self.quant.blu)
        #fig, ax = plt.subplots()
        #ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        return out
    def forward_quant(self, x):
        out = self.conv(x)
        out = torch.clamp(out, 0, self.quant.blu)
        # fig, ax = plt.subplots()
        # ax.hist(out.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out=torch.round(out/(self.quant.blu/255))
        out=out*(self.quant.blu/255)
        return out
    def forward_stat_blu(self, x):
        out = self.relu(self.conv(x))
        self.avg_blu.update(n_sigma_of(out))
        return out
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res_layers=18
        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.res_layers)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.input_quant=quant_params(self.input)
        self.output_quant=quant_params(self.output)
        self.input_avg_blu=AverageMeter()
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return out
    def quantize(self,fn):
        f = open(fn,'wb')
        self.input_quant.quantize_layer(self.input,f)
        for block in self.residual_layer._modules:
            self.residual_layer._modules[block].quantize(f)
        self.output_quant.quantize_layer(self.output,f)
        f.close()
    def quantize_from(self,fn):
        f=open(fn,'rb')
        self.input_quant.quantize_layer_from(self.input, f)
        for block in self.residual_layer._modules:
            self.residual_layer._modules[block].quantize_from(f)
        self.output_quant.quantize_layer_from(self.output,f)
        f.close()
    def blu(self,fn):
        f=open(fn,'wb')
        ratio=255.0/self.input_quant.step_m
        self.input_quant.blu=self.input_avg_blu.avg
        self.input_quant.blu_q = self.input_quant.blu * ratio
        self.input_quant.mul, self.input_quant.shift = mul_shift(self.input_quant.blu_q)
        self.input_quant.blu = 127 * 2 ** self.input_quant.shift / self.input_quant.mul / ratio
        self.input_quant.blu_q = round(self.input_quant.blu * ratio)
        f.write(struct.pack('2f',self.input_quant.blu,ratio))
        f.write(struct.pack('3i',self.input_quant.blu_q,self.input_quant.mul,self.input_quant.shift))
        ratio_q=ratio*self.input_quant.mul/2**self.input_quant.shift
        for block in self.residual_layer._modules:
            ratio_q=self.residual_layer._modules[block].blu(ratio_q,f)
        ratio_out=ratio_q/self.output_quant.step_m
        self.output_quant.mul, self.output_quant.shift = 0,0
        self.output_quant.blu = 0
        self.output_quant.blu_q = 0
        f.write(struct.pack('2f', self.output_quant.blu, ratio_out))
        f.write(struct.pack('3i', self.output_quant.blu_q, self.output_quant.mul, self.output_quant.shift))
        f.write(struct.pack('f', ratio_out))
        f.close()
    def load_blu(self,fn):
        f = open(fn, 'rb')
        self.input_quant.blu,self.input_quant.in_ratio=struct.unpack('2f',f.read(struct.calcsize('2f')))
        f.read(struct.calcsize('3i'))
        for block in self.residual_layer._modules:
            self.residual_layer._modules[block].load_blu(f)
        f.close()
    def forward_blu(self,x):
        out = self.input(x)
        out=torch.clamp(out,0,self.input_quant.blu)
        #fig, ax = plt.subplots()
        #ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        for block in self.residual_layer._modules:
            out=self.residual_layer._modules[block].forward_blu(out)
            #fig, ax = plt.subplots()
            #ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out = self.output(out)
        return out
    def forward_stat_blu(self, x):
        out = self.relu(self.input(x))
        self.input_avg_blu.update(n_sigma_of(out))
        for block in self.residual_layer._modules:
            out = self.residual_layer._modules[block].forward_stat_blu(out)
        out = self.output(out)
        return out
    def forward_quant(self, x):
        out = self.input(x)
        out = torch.clamp(out, 0, self.input_quant.blu)
        # fig, ax = plt.subplots()
        # ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out=torch.round(out/(self.input_quant.blu/255))
        out=out*(self.input_quant.blu/255)
        for block in self.residual_layer._modules:
            out = self.residual_layer._modules[block].forward_quant(out)
            # fig, ax = plt.subplots()
            # ax.hist(x.detach().cpu().numpy().flatten(),bins=300,range=(0.0001,self.quant1.blu))
        out = self.output(out)
        return out
    def acc_update(self):
        self.input_quant.acc_update(self.input.weight)
        for block in self.residual_layer._modules:
            self.residual_layer._modules[block].acc_update()
        self.output_quant.acc_update(self.output.weight)
    def dump(self,fn):
        with open(fn, 'wb') as f:
            self.input.weight.detach().cpu().numpy().tofile(f)
            self.input.bias.detach().cpu().numpy().tofile(f)
            for i in range(self.res_layers):
                self.residual_layer._modules[str(i)]._modules['conv'].weight.detach().cpu().numpy().tofile(f)
                self.residual_layer._modules[str(i)]._modules['conv'].bias.detach().cpu().numpy().tofile(f)
            self.output.weight.detach().cpu().numpy().tofile(f)
            self.output.bias.detach().cpu().numpy().tofile(f)

if __name__ == '__main__':
    model=Net()
    checkpoint = torch.load('model\\model_blu35_epoch_50.pth', map_location='cpu')
    model.load_state_dict(checkpoint["model"].state_dict())
    #model.quantize_from('quant.data')
    model.dump('model.data')
    pass