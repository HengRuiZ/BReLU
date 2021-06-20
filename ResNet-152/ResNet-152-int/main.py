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

def transpose_img(nhwc):
    return nhwc.transpose(2,0,1).astype(np.float32)

def NHWC2VECT_C_img(nhwc):
    alpha = np.zeros((nhwc.shape[0],nhwc.shape[1],1), dtype=np.uint8)
    vect_c=np.concatenate((nhwc, alpha),axis=2)
    return vect_c.astype(np.float32)

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
    print('Total GPUs:%d, selected GPUs:'%ngpus_per_node, args.gpu)
    main_worker(ngpus_per_node, args)

def main_worker(num_gpus, args):
    global best_acc1
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
            if num_gpus == 1 or args.gpu is not None:
                model.load_param(args.resume)
            else:
                model.module.load_param(args.resume)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)

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
            np.array,
            transpose_img,
            #transforms.ToTensor(),
            #normalize,
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
            }, is_best, filename=args.arch+'/checkpoint_blu%d.pth.tar'%(epoch+1))


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
            #model.acc_update()
            model.module.acc_update()
        # print(model.layer1._modules['0'].conv1.weight[0,0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            message='Epoch: [%d][%d/%d]\t'%(epoch, i, len(train_loader))+\
                    ('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'+\
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'+\
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'+\
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'+\
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(message)
            with open('log.txt','a') as f:
                f.write(message+'\n')


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    #model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input=input-128
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            else:
                input=input.cuda()
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)

            with open('pred_torch.data','wb') as f:
                output.detach().cpu().numpy().tofile(f)
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
                message='Test: [%d/%d]\t'%(i, len(val_loader))+\
                'Time %.3f (%.3f)\t'%(batch_time.val,batch_time.avg)+\
                'Loss %.4f (%.4f)\t'%(losses.val, losses.avg)+\
                'Acc@1 %.3f (%.3f)\t'%(top1.val, top1.avg)+\
                'Acc@5 %.3f (%.3f)'%(top5.val, top5.avg)
                print(message)
                with open('log.txt', 'a') as f:
                    f.write(message+'\n')
        message=' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        print(message)
        with open('log.txt', 'a') as f:
            f.write(message+'\n')
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
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('D:\\zhaohengrui\\data\\imagenet_raw\\val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            np.array,
            NHWC2VECT_C_img
        ])),
        batch_size=50, shuffle=False,
        num_workers=4, pin_memory=True)
    f_img=open('val_image_NCHW_50.data','wb')
    for i, (input, target) in enumerate(val_loader):
        img_np=input.numpy().astype(np.uint8)
        img_np.tofile(f_img)
    f_img.close()
    return 0

def conv_validation():
    model = resnet.resnet152(pretrained=False)
    checkpoint = torch.load('resnet152\\checkpoint_blu100.pth.tar')
    model.load_state_dict(moduledict_to_dict(checkpoint['state_dict']))
    model.quantize_from('quant_param152.data')
    model.load_blu('blu152.data')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[128/255,128/255,128/255], std=[0.226, 0.226, 0.226])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('..\\data\\imagenet_test\\val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=50, shuffle=False,
        num_workers=4, pin_memory=True)

    acc1_t = torch.tensor([0.0])
    acc5_t = torch.tensor([0.0])
    for i, (input, target) in enumerate(val_loader):
        #input=(input-0.5)/0.226
        #output=model.forward_stat_blu(input)
        #model.blu('blu152.data')
        output=model.forward_quant(input)
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
