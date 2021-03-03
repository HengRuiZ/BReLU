import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from vdsr import Net,AverageMeter
import eval
from dataset import ndarrayLoader

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--quant", action="store_true", help="Quant model?")
parser.add_argument("--quant_param", default="quant.data", type=str, help="Path to quant param (default: quant.data)")
parser.add_argument("--blu", action="store_true", help="Use BLU model?")
parser.add_argument("--stat_blu", action="store_true", help="stat BLU?")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--gpu", default=3, type=int, help="gpu ids (default: 0)")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(reduction='sum')

    print("===> Using GPU %d"%opt.gpu)
    torch.cuda.set_device(opt.gpu)
    model = model.cuda(opt.gpu)
    criterion = criterion.cuda(opt.gpu)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint["model"].state_dict(),strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.quant:
        if os.path.isfile(opt.quant_param):
            model.quantize_from(opt.quant_param)
            print('model quantized from ' + opt.quant_param)
        else:
            print("=> no quantize checkpoint found at '{}'".format(opt.quant_param))
            exit(1)

    if opt.blu:
        model.load_blu('blu_train.data')
        print('loaded blu from '+'blu_train.data')

    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("data/train.h5")
    #training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    training_data_loader=ndarrayLoader('data\\input.data','data\\target.data',shuffle=True,batch_size=opt.batchSize)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        train(training_data_loader, optimizer, model, criterion, epoch, opt)
        result=eval.main(model,opt.blu,'Set5')
        with open('result.txt','a') as f:
            f.write('epoch:%d\n'%epoch)
            f.write(result)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, opt):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    loss_avg=AverageMeter()

    for iteration, batch in enumerate(training_data_loader):
        input = (batch[0].round()-128)/255.0
        input = input.cuda()
        target = batch[1].cuda()

        if opt.blu:
            residual = model.forward_blu(input)
        elif opt.stat_blu:
            residual = model.forward_stat_blu(input)
            if iteration%100==0:
                print(iteration)
            continue
        else:
            residual = model(input)
        recon = batch[0].cuda() + residual * 255.0
        loss = criterion(recon, target)
        loss_avg.update(loss.data.item())

        optimizer.zero_grad()
        loss.backward() 
        #nn.utils.clip_grad_value_(model.parameters(),opt.clip/lr)
        optimizer.step()
        if opt.quant:
            model.acc_update()

        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.1f}({:.1f})".format(epoch, iteration, len(training_data_loader), loss_avg.avg, loss_avg.val))
    if opt.stat_blu:
        model.blu('blu_train.data')
        exit(0)

def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_bias_blu_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()