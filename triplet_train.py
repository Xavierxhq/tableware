import os, random
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import time

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import utils.loss as loss

from datasets.data_manager import Tableware
from datasets.data_loader import ImageData
from models import get_baseline_model
from evaluator import Evaluator
from utils.meters import AverageMeter
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform

from triplet_tester import get_feature, pickle_read, pickle_write, write_feature_map, dist, do_get_feature_and_t


def load_model(base_model, model_path):
    model_parameter = torch.load(model_path)
    base_model.load_state_dict(model_parameter['state_dict'])
    base_model = base_model.cuda()
    print('model', model_path.split('/')[-1], 'loaded.')
    return base_model


def triplet_example(input1, labels, distmat):
    labels_set = set(labels.numpy())
    label_to_indices = {label: np.where(labels.numpy() == label)[0] for label in labels_set}
    random_state = np.random.RandomState(29)
    input2 = torch.Tensor(input1.size())
    input3 = torch.Tensor(input1.size())

    dis_mp, dis_mn, p_inds, n_inds = loss.hard_example_extream(distmat, labels, True)

    for i, label in enumerate(labels.numpy()):
        input2[i] = input1[int(p_inds[i])]
        input3[i] = input1[int(n_inds[i])]

    return input1, input2, input3


def compute_center_with_label(model, label, data_pth):
    # model.eval()
    temp_name = './evaluate_result/feature_map/%f.pkl' % time.time()
    processed_imgs = [data_pth + x for x in os.listdir(data_pth) if x.split('.')[0].split('_')[-1] == str(label)]
    for img_path in processed_imgs:
        f = get_feature(img_path, model)
        _store_features_temp(temp_name, str(label), f)
    obj = pickle_read(temp_name)

    _avg_feature = None
    for _f in obj[str(label)]:
        if _avg_feature is None:
            _avg_feature = _f
        else:
            _avg_feature = _f.add(_avg_feature)
    _avg_feature /= len(obj[str(label)])

    min_dis, max_dis = 1e6, -1
    for _f in obj[str(label)]:
        _d = dist(_avg_feature, _f)
        if _d > max_dis:
            max_dis = _d
            _max_feature = _f
        if _d < min_dis:
            min_dis = _d
            _min_feature = _f

    os.remove(temp_name)
    return _avg_feature, _max_feature, _min_feature


def _store_features_temp(feature_map_name, label, feature):
    if os.path.exists(feature_map_name):
        obj = pickle_read(feature_map_name)
        obj[label].append(feature)
    else:
        obj = {
            label: [feature]
        }
    pickle_write(feature_map_name, obj)


def _get_avg_feature_for_labels(model, labels, data_pth):
    prefix = '%f' % time.time()
    _labels = set(labels)
    for label in _labels:
        _avg_feature, _max_feature, _min_feature = compute_center_with_label(model, label, data_pth)
        write_feature_map('./evaluate_result/feature_map/%s.center.pkl' % prefix, str(label - 1), _avg_feature)
        write_feature_map('./evaluate_result/feature_map/%s.furthest_from_center.pkl' % prefix, str(label - 1), _max_feature)
        write_feature_map('./evaluate_result/feature_map/%s.near_to_center.pkl' % prefix, str(label - 1), _min_feature)
    return prefix


def get_center_anchor(labels, prefix):
    inputs = []
    obj = pickle_read('./evaluate_result/feature_map/%s.center.pkl' % prefix)
    for label in labels:
        # if str(label) not in obj:
        #     print(label, 'center is not computed. exit')
        #     exit(500)
        inputs.append(obj[(str(label))])
    return inputs


def get_furthest_from_center(labels, prefix):
    inputs = []
    obj = pickle_read('./evaluate_result/feature_map/%s.furthest_from_center.pkl' % prefix)
    for label in labels:
        inputs.append(obj[(str(label))])
    return inputs


def get_nearest_to_center(labels, prefix):
    inputs = []
    obj = pickle_read('./evaluate_result/feature_map/%s.near_to_center.pkl' % prefix)
    for label in labels:
        inputs.append(obj[(str(label))])
    return inputs


def clean_cache(prefix):
    if prefix is None:
        return
    os.remove('./evaluate_result/feature_map/%s.center.pkl' % prefix)
    os.remove('./evaluate_result/feature_map/%s.furthest_from_center.pkl' % prefix)
    os.remove('./evaluate_result/feature_map/%s.near_to_center.pkl' % prefix)


def train(model, optimizer, criterion, epoch, print_freq, data_loader, data_pth):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    evaluator = Evaluator(model)
    distmat = evaluator.calDistmat(data_loader)
    is_add_margin = False

    # _t1 = time.time()
    # _get_avg_feature_for_labels(model, [x for x in range(1, 41)], data_pth)
    # print('time for getting center: %.1f s' % (time.time() - _t1))

    # model.train()
    prefix = None
    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - start)

        # model optimizer
        # parse data
        imgs, pids = inputs
        labels = [x.item() for x in pids]

        if i % 1 == 0:
            # clean_cache(prefix)
            _t1 = time.time()
            prefix = _get_avg_feature_for_labels(model, [x for x in range(1, 41)], data_pth)
            # print('time for getting center: %.1f s' % (time.time() - _t1), i)

        img1, img2, img3 = triplet_example(imgs, pids, distmat)
        # input1 = img1.cuda()
        input1 = get_center_anchor(labels, prefix)
        feat1 = torch.stack(input1)
        # print(input1)
        if random.randint(1, 10) > 5:
            input2 = get_furthest_from_center(labels, prefix)
            # input2 = get_nearest_to_center(labels, prefix)
            feat2 = torch.stack(input2)
        else:
            feat2 = img2.cuda()
            feat2 = model(feat2)
        input3 = img3.cuda()

        # forward
        # feat1 = model(input1)
        # feat2 = model(input2)
        feat3 = model(input3)
        # print(feat1)
        # print(feat3)

        loss = criterion(feat1, feat2, feat3)

        optimizer.zero_grad()
        # backward
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        losses.update(loss.item())

        start = time.time()

        if (i + 1) % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Batch Time {:.3f} ({:.3f})\t'
                    'Data Time {:.3f} ({:.3f})\t'
                    'Loss {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        if losses.val < 1e-5:
            is_add_margin = True

    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.6f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
    print()
    return is_add_margin


def trainer(data_pth, a, b, _time=0, layers=18):
    seed = 0

    # dataset options
    height = 128
    width = 128

    # optimization options
    optim = 'Adam'
    max_epoch = 20
    train_batch = 64
    test_batch = 64
    lr = 0.1
    step_size = 40
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    test_margin = b
    margin = a
    num_instances = 4
    num_gpu = 1

    # model options
    last_stride = 1
    pretrained_model_18 = 'model/resnet18-5c106cde.pth'
    pretrained_model_50 = 'model/resnet50-19c8e357.pth'
    pretrained_model_34 = 'model/resnet34-333f7ec4.pth'
    pretrained_model_101 = 'model/resnet101-5d3b4d8f.pth'
    pretrained_model_152 = 'model/resnet152-b121ed2d.pth'

    # miscs
    print_freq = 20
    eval_step = 1
    save_dir = 'model/pytorch-ckpt/time%d' % _time
    workers = 1
    start_epoch = 0


    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print('currently using cpu')

    pin_memory = True if use_gpu else False

    print('initializing dataset {}'.format('Tableware'))
    dataset = Tableware(data_pth)

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(height, width)),
        batch_size=train_batch, num_workers=workers,
        pin_memory=pin_memory, drop_last=True
    )

    # testloader = DataLoader(
    #     ImageData(dataset.test, TestTransform(height, width)),
    #     batch_size=test_batch, num_workers=workers,
    #     pin_memory=pin_memory, drop_last=True
    # )

    # model, optim_policy = get_baseline_model(model_path=pretrained_model)
    if layers == 18:
        model, optim_policy = get_baseline_model(model_path=pretrained_model_18, layers=18)
    else:
        model, optim_policy = get_baseline_model(model_path=pretrained_model_50, layers=50)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_18, layers=18)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_34, layers=34)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_101, layers=101)
    # model = load_model(model, model_path='./model/pytorch-ckpt/87_layers18_margin20_epoch87.tar')
    print('model\'s parameters size: {:.5f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    inner_dist = 0
    outer_dist = 0
    max_outer = 0
    min_outer = 0
    max_iner = 0
    min_iner = 0

    tri_criterion = TripletLoss(margin)

    # get optimizer
    optimizer = torch.optim.Adam(
        optim_policy, lr=lr, weight_decay=weight_decay
    )

    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * num_gpu
        elif ep < 180:
            lr = 1e-4 * num_gpu
        elif ep < 300:
            lr = 1e-5 * num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) *num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * num_gpu
        else:
            lr = 1e-5 * num_gpu
        for p in optimizer.param_groups:
            p['lr'] = lr

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    evaluator = Evaluator(model)

    for epoch in range(start_epoch, max_epoch):
        if step_size > 0:
            adjust_lr(optimizer, epoch + 1)
        next_margin = margin


        # skip if not save model
        if eval_step > 0 and (epoch + 1) % eval_step == 0 or (epoch + 1) == max_epoch:
            save_record_path = 'margin_'+ str(margin) + '_epoch_' + str(epoch + 1) + '.txt'
            _t1 =time.time()
            train(model, optimizer, tri_criterion, epoch, print_freq, trainloader, data_pth=data_pth)
            _t2 = time.time()
            print('time for training:', '%.2f' % (_t2 - _t1), 's')

            """
            acc, inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner = evaluator.evaluate(testloader, test_margin, save_record_path)
            print('margin:{}, epoch:{}, acc:{}'.format(margin, epoch+1, acc))
            f = open('record.txt', 'a')
            f.write('margin:{}, epoch:{}, acc:{}\n'.format(margin, epoch+1, acc))
            f.close()
            """

            is_best = False
            # save_model_path = 'new_margin({})_epoch({}).pth.tar'.format(margin, epoch+1)
            save_model_path = 'time{}_layers{}_margin{}_epoch{}.tar'.format(_time, layers, margin, epoch+1)
            # save_model_path = 'layers34_margin{}_epoch{}.tar'.format(margin, epoch+1)
            # save_model_path = 'layers101_margin{}_epoch{}.tar'.format(margin, epoch+1)
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()


            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=is_best, save_dir=save_dir, filename=save_model_path)
            
            model.eval()
            acc = do_get_feature_and_t(model, margin=20, epoch=1)

            margin = next_margin
    return save_model_path, inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner


if __name__ == "__main__":

    # for margin in range(1, 500):
    #     f = open('margin_dist.txt', 'a')
    #     inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner = trainer('/home/ubuntu/Program/Tableware/reid_tableware/datas/dishes_dataset/', margin, 0)
    #     f.write("{},{},{},{},{},{},{}\r\n".format(margin,inner_dist,outer_dist, max_outer, min_outer, max_iner, min_iner))
    #     print(margin, inner_dist, outer_dist)
    #     f.close()

    for _i in range(1):
        # trainer(data_pth, 20, 0, _time=_i+1, layers=18)
        trainer('/home/ubuntu/Program/xhq/dataset/temp/train_data/', 20, 0, _time=_i, layers=50)
    # _ = don't care.

    # model_path = '1_margin(10)_epoch(1).pth.tar'


    # it will cost you lots of time.
