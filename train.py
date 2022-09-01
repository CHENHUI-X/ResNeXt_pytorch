# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.model import CifarResNeXt
from torchsummary.torchsummary import summary
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from tqdm import  tqdm

def getparser():
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=16,help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.05, help='Weight decay (L2 penalty).')
    # https://paperswithcode.com/method/weight-decay

    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./checkpoint/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group or like the width).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default = 2, help='Widen factor. base channel = 64 , then 64*2,64*2*2...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=0, help='0 for CPU , >=1 for GPU')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./log', help='Log folder.')

    return parser

# train function (forward, backward, update)
def train(net,train_loader,optimizer,state,epoch,args):

    net.train()
    loss_avg = 0.0
    epoch_loss = 0.0
    correct = 0.0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

        data, target = data.to(device) , target.to(device)
        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + float(loss) * 0.8
        epoch_loss +=loss

        # accuracy
        pred = output.data.max(1)[1]
        correct += float(pred.eq(target.data).sum())

    state['train_loss'] = epoch_loss
    state['train_ema_loss'] = loss_avg
    state['train_accuracy'] = correct / len(train_loader.dataset)
    print(
        f'[{epoch}/{args.epochs}] --epoch_loss : {epoch_loss} --EMA_loss : {loss_avg} --{state["train_accuracy"]}'
    )

# test function (forward only)
def test(net,test_loader,state):
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device),target.to(device)
        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += float(pred.eq(target.data).sum())

        # test loss average
        loss_avg += float(loss)

    # state['test_loss'] = loss_avg / len(test_loader)
    state['test_loss'] = loss_avg
    state['test_accuracy'] = correct / len(test_loader.dataset)
    print(
        f'EMA_loss : {loss_avg} --{state["test_accuracy"]}'
    )


if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    print('Saved args parameters !')


    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), # Whether flipped
         transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100
    else:
        raise NotImplementedError('You need create your customer dataset !')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    net = net.to(device)

    summary(net,input_size=(3,32,32),batch_size=args.batch_size,device=device)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        if epoch in args.schedule: # lr decay
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                # update the lr
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train(net,train_loader,optimizer,state,epoch,args)
        test(net,test_loader,state)
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)

    log.close()
