# -*- coding: UTF-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--quant', dest='quantize', action='store_true',
                    help='use quantized model')
parser.add_argument('--stat_blu', dest='stat_blu', action='store_true',
                    help='get statistical blu value')
parser.add_argument('--blu', dest='blu', action='store_true',
                    help='use quantized blu model')
parser.add_argument('--quant_param', default='', type=str, metavar='PATH',
                    help='path to quantize parameters (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def moduledict_to_dict(moduledict):
    dict={}
    for key in moduledict:
        key_new=key.split('module.')[-1]
        dict[key_new]=moduledict[key]
    return dict

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    ngpus_per_node = torch.cuda.device_count()
    print('Total GPUs:%d, selected GPUs:%d'%(ngpus_per_node, args.gpu))
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, num_gpus, args):
    global best_acc1
    args.gpu = gpu
    # create model
    if args.pretrained:
        model = resnet.__dict__[args.arch](pretrained=True)
        print("=> created pre-trained model '{}'".format(args.arch))
    else:
        model = resnet.__dict__[args.arch]()
        print("=> created model '{}'".format(args.arch))

    if num_gpus==1 or args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print('warning:multiple GPUs is used')
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            #model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)

    if args.quantize:
        if args.quant_param:
            if os.path.isfile(args.quant_param):
                model.quantize_from(args.quant_param)
                print('model quantized from '+args.quant_param)
            else:
                print("=> no quantize checkpoint found at '{}'".format(args.quant_param))
                exit(1)
        else:
            model.quantize(args.resume+'.quant_param')
            print('model quantized')

    if args.blu:
        model.load_blu('3sigma.blu')
        print('loaded blu from '+'3sigma.blu')

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[128/255,128/255,128/255], std=[0.226, 0.226, 0.226])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    validate(val_loader, model, criterion, args)
    if args.evaluate:
        return

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = False#acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=args.arch+'/checkpoint_renorm%d.pth.tar'%(epoch+1))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    #model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        else:
            input=input.cuda()
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        if args.blu:
            output=model.forward_blu(input)
        else:
            output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.quantize:
            model.acc_update()
        # print(model.layer1._modules['0'].conv1.weight[0,0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    #model.eval()
    if args.stat_blu:
        print('count blu while validating')
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            else:
                input=input.cuda()
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.blu:
                output=model.forward_blu(input)
            else:
                if args.stat_blu:
                    output = model.forward_stat_blu(input)
                else:
                    output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if args.stat_blu:
            model.blu('3sigma.blu')
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_best')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def dump_val_data():
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_data=datasets.ImageFolder('../data/imagenet/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #np.array
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    f_img=open('val_image_NCHW_50.data','wb')
    #f_label=open('val_label.data','wb')
    for i, (input, target) in enumerate(val_loader):
        # input(NCHW) to img_np(NCHW)
        img_np=np.transpose(input.numpy(),(0,3,1,2))
        # input(NCHW) to img_np(NCHW_VECT_C)
        #img_np = input.numpy()
        #alpha=np.zeros((img_np.shape[0],img_np.shape[1],img_np.shape[2],1),dtype=np.uint8)
        #img_np=np.concatenate((img_np, alpha),axis=3)
        #img_pil=Image.fromarray(np.transpose(img_np[0],(1,2,0)))
        #img_pil.show()
        img_np[0:50].tofile(f_img)
        exit(0)
        #target_np=target.numpy()
        #target_np.tofile(f_label)
        #break
    f_img.close()
    #f_label.close()
    return 0
def dump_model():
    def get_bn_params(bn):
        return [bn.running_mean,bn.running_var,bn.weight.data,bn.bias.data]
    def get_block_params(block):
        params=[block.conv1.weight.data]+get_bn_params(block.bn1)+[block.conv2.weight.data]+get_bn_params(block.bn2)
        if block.downsample:
            params+=[block.downsample._modules['0'].weight.data]+get_bn_params(block.downsample._modules['1'])
        return params
    def get_BN_block_params(block):
        params=[block.conv1.weight.data]+get_bn_params(block.bn1)+[block.conv2.weight.data]+get_bn_params(block.bn2)+[block.conv3.weight.data]+get_bn_params(block.bn3)
        if block.downsample:
            params+=[block.downsample._modules['0'].weight.data]+get_bn_params(block.downsample._modules['1'])
        return params
    def get_layer_params(layer):
        params=[]
        for block in layer._modules:
            params+=get_block_params(layer._modules[block])
        return params
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    # model.load('resnet18\\mergedresnet18.sd')
    checkpoint = torch.load('resnet18\\checkpoint_blu35103.pth.tar')
    #model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
    model.load_state_dict(checkpoint['state_dict'])
    #model.quantize_from('quant_param89.data')
    params=[model.conv1.weight.data]+get_bn_params(model.bn1)#(64,3,7,7)
    params+=get_layer_params(model.layer1)
    params+=get_layer_params(model.layer2)
    params += get_layer_params(model.layer3)
    params += get_layer_params(model.layer4)
    params+=[model.fc.weight.data,model.fc.bias.data]
    with open('checkpoint_blu35103.data','wb') as f:
        for para_tensor in params:
            para_np=para_tensor.numpy()
            para_np.tofile(f)
    return
def conv_validation():
    #model = resnet.__dict__['resnet18'](pretrained=True)
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    #model.load('resnet18\\mergedresnet18.sd')
    checkpoint = torch.load('resnet18\\checkpoint_renorm89.pth.tar')
    model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
    #model.load_state_dict(checkpoint['state_dict'])
    model.quantize('quant_param89.data')
    model.load_blu('3sigma.blu')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[128/255,128/255,128/255], std=[0.226, 0.226, 0.226])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('..\\data\\imagenet\\val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10, shuffle=False,
        num_workers=4, pin_memory=True)

    acc1_t = torch.tensor([0.0])
    acc5_t = torch.tensor([0.0])
    for i, (input, target) in enumerate(val_loader):
        #input=(input-0.5)/0.226
        output=model(input)
        #output=model.forward_blu(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        print('{}:acc1:{},acc5:{}'.format(i,acc1,acc5))
        acc1_t+=acc1
        acc5_t+=acc5
    print('acc1:{}\nacc5:{}'.format(acc1_t/(i+1), acc5_t/(i+1)))

if __name__ == '__main__':
    main()
    #dump_val_data()
    #dump_model()
    #conv_validation()
