
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os
import shutil

import utils
from model import Models
from data_process import data

parser = argparse.ArgumentParser(description='PyTorch classifier Training')

parser.add_argument('--gpu', type=str, default=None,
                    help='ID of GPUs to use, eg. 1,3')

parser.add_argument('--train_data', type=str, default='./data/train',
                    help='train data path')

parser.add_argument('--valid_data', type=str, default='./data/valid',
                    help='valid data path')

parser.add_argument('--batch_size', type=int,
                    default=32, help='input batch size')

parser.add_argument('--workers', type=int,
                    default=4, help='workers for reading datasets')

parser.add_argument('--arch', '-a', metavar='ARCH', default='wide_resnet50_2',
                    help='model architecture')

parser.add_argument('--resume_path', type=str, default=None,
                    help='model file to resume to train')

parser.add_argument('--optim', type=str, default='SGD',
                    help='optim for training, Adam / SGD (default)')

parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate for training')

parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for SGD')

parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight_decay for SGD / Adam')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--model_path', type=str, default='weights',
                    help='model file to save')

parser.add_argument('--num_classes', type=int, default=2,
                    help='model file to resume to train')

parser.add_argument('--displayInterval', type=int,
                    default=1, help='Interval to be displayed')
args = parser.parse_args()
print(args)


if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = list(range(len(args.gpu.split(','))))
else:
    gpus = [0] 


def train(train_loader, model, criterion, optimizer, epoch, lr_cur):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(gpus[0], async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        optimizer.zero_grad()
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()
        # measure utils.accuracy and record loss
        prec1 = utils.accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        if (i+1) % args.displayInterval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: {lr:.4e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch,
                                                                    i+1,
                                                                    len(train_loader),
                                                                    lr=lr_cur,
                                                                    loss=losses,
                                                                    top1=top1
                                                                    ))

def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(gpus[0], async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        print("model output :", output)

        # measure utils.accuracy and record loss
        prec1 = utils.accuracy(output.data, target, topk=(1, ))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        if (i+1) % args.displayInterval == 0:
            print('Valid: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i+1, 
                                                                    len(val_loader),
                                                                    loss=losses,
                                                                    top1=top1
                                                                    ))
    # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def run():

    # read data
    train_loader, val_loader = data(args.train_data, 
                                    args.valid_data,
                                    args.batch_size, 
                                    args.workers)

    # set up model -- without imagenet pre-training
    model = Models(args.arch, 2, gpus).Model()
    cudnn.benchmark = True

    # resume pretrain model
    if args.resume_path is not None:
        pretrained_model = torch.load(args.resume_path)
        model.load_state_dict(pretrained_model['state_dict'])
        best_prec1 = pretrained_model['best_prec1']
        print('Load resume model done.')
    else:
        best_prec1 = 0
    print('Best top-1: {:.4f}'.format(best_prec1))

    # optimizer
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     args.lr,
                                     weight_decay=args.weight_decay)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # learning rate optimal
    lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20))

    model_path = args.model_path
    def save_checkpoint(state, is_best, filename='checkpoint'):
        torch.save(state, os.path.join(model_path, filename + '_latest.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(model_path, filename + '_latest.pth.tar'),
                            os.path.join(model_path, filename + '_best.pth.tar'))

    # start 
    for epoch in range(args.start_epoch, args.epochs):
        lr_cur = lr_opt(args.lr, epoch)  # speed change
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_cur

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, lr_cur)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.arch)


if __name__ == '__main__':
    run()
