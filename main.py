from __future__ import print_function, absolute_import, division

import os
from os.path import join, dirname
import sys
import time
from pprint import pprint
import numpy as np
from progress.bar import Bar as Bar

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from src.opt import Options
import src.log as log
import src.utils as utils
from src.model import LinearModel, weight_init
from src.data import ThreeDPeopleDataset, ToTensor, \
        NumViewsReductionTransformation, H36MDataset


def train(train_loader, model, criterion, optimizer, num_kpts=17,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
        losses = utils.AverageMeter()

        model.train()

        all_dist = []

        start = time.time()
        batch_time = 0
        bar = Bar('>>>', fill='>', max=len(train_loader))

        for i, sample in enumerate(train_loader):
            glob_step += 1
            if glob_step % lr_decay == 0 or glob_step == 1:
                lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
            inputs = sample['X'].cuda()
#            targets = sample['Y'].cuda(async=True)
            targets = sample['Y'].cuda()

            outputs = model(inputs)

            # calculate loss
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            if max_norm:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            sqerr = (outputs - targets) ** 2

            # NOTE: sqerr.shape[0] is batch dimensions.
            distance = np.zeros((sqerr.shape[0], num_kpts))
            dist_idx = 0
            for k in np.arange(0, num_kpts * 3, 3):
                distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
                dist_idx += 1
            all_dist.append(distance)

            # update summary
            if (i + 1) % 100 == 0:
                batch_time = time.time() - start
                start = time.time()

            bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
                .format(batch=i + 1,
                        size=len(train_loader),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg)
            bar.next()

        bar.finish()

        all_dist = np.vstack(all_dist)
        ttl_err = np.mean(all_dist)

        print (">>> train error: {} <<<".format(ttl_err))

        return glob_step, lr_now, losses.avg, ttl_err


def test(test_loader, model, criterion, num_kpts=17, inference=False):
    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))
    sample_output_saved = False

    for i, sample in enumerate(test_loader):
        inputs = sample['X'].cuda()
#        targets = sample['Y'].cuda(async=True)
        targets = sample['Y'].cuda()

        outputs = model(inputs)

        # calculate loss
        #outputs_coord = outputs
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        if inference and not sample_output_saved:
            sample_output_saved = True
            np.save('outputs.npy', outputs)

        # calculate erruracy
        #targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        #outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

        # remove dim ignored
        #dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        #outputs_use = outputs_unnorm[:, dim_use]
        #targets_use = targets_unnorm[:, dim_use]

        '''
        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 51)
        '''
        sqerr = (outputs - targets) ** 2

        # NOTE: sqerr.shape[0] is a batch dimension.
        distance = np.zeros((sqerr.shape[0], num_kpts))
        dist_idx = 0
        for k in np.arange(0, num_kpts * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

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

    all_dist = np.vstack(all_dist)
    ttl_err = np.mean(all_dist)
    bar.finish()
    print (">>> test error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    exp_dir_ = dirname(opt.load)

    # create model
    print(">>> creating model")
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=opt.num_kpts * 3)
    #model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=51, groups=2)
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
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
        logger.set_names(['epoch', 'lr', 'loss_train', 'err_train', 
            'loss_test', 'err_test'])
        
    transforms = [
            ToTensor(), 
#            NumViewsReductionTransformation(opt.epochs, opt.num_views)
        ]

    if opt.dataset == '3dpeople':
        dataset_classes = [ThreeDPeopleDataset]
    elif opt.dataset == 'h36m':
        dataset_classes = [H36MDataset]
    else:
        dataset_classes = [ThreeDPeopleDataset, H36MDataset]

    train_datasets = []
    for dataset_class in dataset_classes:
        train_datasets.append(dataset_class(num_views=opt.num_views, 
            num_kpts=opt.num_kpts, transforms=transforms, 
            data_type='train', mode=opt.data_mode))
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch,
                        shuffle=True, num_workers=opt.job)

    if opt.test_set == '3dpeople':
        test_dataset = ThreeDPeopleDataset(num_views=opt.num_views,
            num_kpts=opt.num_kpts, transforms=transforms, 
            data_type='test', mode=opt.data_mode)
    else:
        test_dataset = H36MDataset(num_views=opt.num_views,
            num_kpts=opt.num_kpts, transforms=transforms,
            data_type='test', mode=opt.data_mode)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch,
                        shuffle=True, num_workers=opt.job)
    if opt.test:
        loss_test, err_test = test(test_loader, model, criterion, num_kpts=opt.num_kpts, inference=True)
        sys.exit()

    cudnn.benchmark = True

    for epoch in range(start_epoch, opt.epochs):
        # These epochs are set to decrease the num
        train_dataset.epoch = epoch
        test_dataset.epoch = epoch
        torch.cuda.empty_cache()
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train, err_train = train(
            train_loader, model, criterion, optimizer, num_kpts=opt.num_kpts,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        loss_test, err_test = test(test_loader, model, criterion, num_kpts=opt.num_kpts)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, err_train, 
            loss_test, err_test],
            ['int', 'float', 'float', 'float', 'float', 'float'])

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

    logger.close()


if __name__ == '__main__':
    option = Options().parse()
    main(option)
