from __future__ import print_function, absolute_import, division

import os
from os.path import join, dirname
import sys
import time
from pprint import pprint
import numpy as np
from progress.bar import Bar as Bar
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from src.opt import Options
import src.log as log
import src.utils as utils
from model import weight_init       # TODO: Do I need this???
from src.data import ToTensor, ClassificationDataset, PETA, People3D
from src.data_utils import one_hot


def train(train_loader, model, criterion, optimizer, num_kpts=15, num_classes=200,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
    losses = utils.AverageMeter()

    model.train()

    errs, accs, confs = [], [], []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    for i, sample in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        inputs = sample['X'].cuda()
        targets = sample['Y'].reshape(-1).cuda()

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        
        # Set outputs to [0, 1].
        softmax = nn.Softmax()
        outputs = softmax(outputs)

        outputs = outputs.data.cpu().numpy()
        targets = one_hot(targets.data.cpu().numpy(), num_classes)

        errs.append(np.mean(np.abs(outputs - targets)))
        accs.append(accuracy_score(
            np.argmax(targets, axis=1),
            np.argmax(outputs, axis=1))
        )
        confs.append(np.mean(outputs[np.argmax(outputs, axis=1)]))

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()

    err = np.mean(np.array(errs, dtype=np.float32))
    acc = np.mean(np.array(accs, dtype=np.float32))
    conf = np.mean(np.array(confs, dtype=np.float32))
    print (">>> train error: {} <<<".format(err))
    print (">>> train accuracy: {} <<<".format(acc))
    return glob_step, lr_now, losses.avg, err, acc, conf


def test(test_loader, model, criterion, num_kpts=15, num_classes=2, 
        batch_size=64, inference=False):
    losses = utils.AverageMeter()

    model.eval()

    errs, accs, confs = [], [], []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, sample in enumerate(test_loader):
        inputs = sample['X'].cuda()
        targets = sample['Y'].reshape(-1).cuda()
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        
        # Set outputs to [0, 1].
        softmax = nn.Softmax()
        outputs = softmax(outputs)

        outputs = outputs.data.cpu().numpy()
        targets = one_hot(targets.data.cpu().numpy(), num_classes)

        # TODO: This is not completely correct (per batch).
        errs.append(np.mean(np.abs(outputs - targets)))
        accs.append(accuracy_score(
            np.argmax(targets, axis=1),
            np.argmax(outputs, axis=1))
        )
        confs.append(np.mean(outputs[np.argmax(outputs, axis=1)]))

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    err = np.mean(np.array(errs))
    acc = np.mean(np.array(accs))
    conf = np.mean(np.array(confs))
    bar.finish()

    print('>>> test error: {} <<<'.format(err))
    print('>>> test accuracy: {} <<<'.format(acc)) 
    return losses.avg, err, acc, conf


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)
    writer = SummaryWriter(log_dir=f'./exp/{opt.name}')
    exp_dir_ = dirname(opt.load)

    # create model
    print(">>> creating model")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=opt.num_classes)
    # TODO: This is how to avoid weird data reshaping for non-3-channel inputs.
    # Have ResNet model take in grayscale rather than RGB
#    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'err_train', 'acc_train', 
            'loss_test', 'err_test', 'acc_test'])
        
    transforms = [
            ToTensor(), 
    ]

    train_datasets = []
    for dataset_name in opt.datasets:
        if 'peta' in dataset_name:
            train_datasets.append(PETA(
                name=dataset_name,
                num_kpts=opt.num_kpts,
                transforms=transforms,
                data_type='train'))
        elif 'people3d' in dataset_name:
            train_datasets.append(People3D(
                name=dataset_name,
                num_kpts=opt.num_kpts,
                transforms=transforms,
                data_type='train'))
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch,
                        shuffle=True, num_workers=opt.job)

    set_ = False
    for dataset_name in opt.datasets:
        if 'peta' in dataset_name:
            test_dataset = PETA(
                    name=dataset_name,
                    num_kpts=opt.num_kpts, 
                    transforms=transforms,
                    data_type='test')
            set_ = True
            break
    if not set_:
        for dataset_name in opt.datasets:
            if 'people3d' in dataset_name:
                test_dataset = People3D(
                        num_kpts=opt.num_kpts, 
                        transforms=transforms,
                        dataset=dataset_name,
                        data_type='test')

    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch,
                        num_workers=opt.job)

    if opt.test:
        # TODO: This is currently broken.
        scores = test(test_loader, model, 
                criterion, num_kpts=opt.num_kpts, 
                num_classes=opt.num_classes, inference=True)
        test_dataset.report(scores)
        sys.exit()

    cudnn.benchmark = True

    for epoch in range(start_epoch, opt.epochs):
        torch.cuda.empty_cache()
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train, err_train, acc_train, conf_train = \
                train(train_loader, model, criterion, optimizer, 
                        num_kpts=opt.num_kpts, num_classes=opt.num_classes, 
                        lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, 
                        lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
                        max_norm=opt.max_norm)
        loss_test, err_test, acc_test, conf_test = test(test_loader, model, 
                criterion, num_kpts=opt.num_kpts, num_classes=opt.num_classes,
                batch_size=opt.test_batch)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, err_train, acc_train,
            loss_test, err_test, acc_test],
            ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Loss/test', loss_test, epoch)
        writer.add_scalar('Error/train', err_train, epoch)
        writer.add_scalar('Error/test', err_test, epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy/test', acc_test, epoch)
        writer.add_scalar('Confidence/train', conf_train, epoch)
        writer.add_scalar('Confidence/test', conf_test, epoch)

    logger.close()
    writer.close()


if __name__ == '__main__':
    option = Options().parse()
    main(option)
