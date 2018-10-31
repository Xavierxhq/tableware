import os, random, sys, time
from os import path as osp
from pprint import pprint
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import utils.loss as loss
from datasets.data_manager import Tableware
from datasets.data_loader import ImageData
from models import get_baseline_model
from utils.meters import AverageMeter
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform
from tester import get_feature, pickle_read, pickle_write, dist, do_get_feature_and_t


def load_model(base_model, model_path):
    model_parameter = torch.load(model_path)
    base_model.load_state_dict(model_parameter['state_dict'])
    base_model = base_model.cuda()
    print('model', model_path.split('/')[-1], 'loaded.')
    return base_model


def _compute_center_with_label(model, label, data_pth, sample_count):
    # model.eval()
    temp_name = './evaluate_result/feature_map/%f.pkl' % time.time()
    processed_imgs = [data_pth + x for x in os.listdir(data_pth) if x.split('.')[0].split('_')[-1] == str(label)]
    random.shuffle(processed_imgs)
    for img_path in processed_imgs[:sample_count*4]:
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

    dist_dict = {}
    for _i, _f in enumerate(obj[str(label)]):
        _d = dist(_avg_feature, _f)
        dist_dict[str(_i)] = _d
    dist_dict = sorted(dist_dict.items(), key=lambda d: d[1])

    _max_feature_ls, _min_feature_ls = [], []
    for index, _d in dist_dict[:sample_count]:
        _min_feature_ls.append(obj[str(label)][int(index)])
    for index, _d in dist_dict[-1*sample_count:]:
        _max_feature_ls.append(obj[str(label)][int(index)])

    os.remove(temp_name)
    return _avg_feature, _max_feature_ls, _min_feature_ls


def _store_features_temp(feature_map_name, label, feature):
    if os.path.exists(feature_map_name):
        obj = pickle_read(feature_map_name)
        obj[label].append(feature)
    else:
        obj = {
            label: [feature]
        }
    pickle_write(feature_map_name, obj)


def _get_negative_feature(model, label, data_pth):
    all_nagative_imgs = [data_pth + x for x in os.listdir(data_pth) if x.split('.')[0].split('_')[-1] != str(label)]
    random.shuffle(all_nagative_imgs)
    return get_feature(all_nagative_imgs[0], model)


def _get_input_samples(model, labels, data_pth, sample_count=64):
    if sample_count % len(labels) != 0:
        print('sample_count should times the count of labels')
        return
    sample_count_for_one_label = int(sample_count / len(labels))
    candidate_count_for_one_label = sample_count_for_one_label * 4
    anchors, positives, negatives = [], [], []
    for label in labels:
        _avg_feature, _max_feature_ls, _min_feature_ls = _compute_center_with_label(model, label, data_pth, sample_count=sample_count_for_one_label)
        for i in range(len(_max_feature_ls)):
            anchors.append(_avg_feature)
            positives.append(_max_feature_ls[i])
            # positives.append(_min_feature_ls[i])
            negatives.append(_get_negative_feature(model, label, data_pth))
    return anchors, positives, negatives


def train(model, optimizer, criterion, epoch, print_freq, data_loader, data_pth):
    model.train()
    losses = AverageMeter()
    is_add_margin = False
    labels_count_in_one_batch = 8
    labels_count_for_all = 40

    start = time.time()
    for _j in range(20):
        for i in range(int(labels_count_for_all / labels_count_in_one_batch)):
            start_label = i * labels_count_in_one_batch + 1
            feat1, feat2, feat3 = _get_input_samples(model, [x for x in range(start_label, start_label + labels_count_in_one_batch)], data_pth)

            loss = criterion(feat1, feat2, feat3)

            optimizer.zero_grad()
            # backward
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            if (_j * int(labels_count_for_all / labels_count_in_one_batch) + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                        'Loss {:.6f} ({:.6f})\t'
                          .format(epoch, _j * int(labels_count_for_all / labels_count_in_one_batch) + 1, 20 * int(labels_count_for_all / labels_count_in_one_batch),
                                  losses.val, losses.mean))
            if losses.val < 1e-5:
                is_add_margin = True

    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.1f} s\tLoss {:.6f}\t'
              'Lr {:.2e}'
              .format(epoch, (time.time() - start), losses.mean, param_group[0]['lr']))
    print()
    return is_add_margin


def trainer(data_pth, a, b, _time=0, layers=18):
    seed = 0
    # dataset options
    height, width = 128, 128
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

    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print('currently using cpu')

    pin_memory = True if use_gpu else False

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

    max_acc = .0
    for epoch in range(max_epoch):
        if step_size > 0:
            adjust_lr(optimizer, epoch + 1)
        next_margin = margin
        # skip if not save model
        if eval_step > 0 and (epoch + 1) % eval_step == 0 or (epoch + 1) == max_epoch:
            _t1 =time.time()
            train(model, optimizer, tri_criterion, epoch, print_freq, None, data_pth=data_pth)
            _t2 = time.time()
            print('time for training:', '%.2f' % (_t2 - _t1), 's')

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_model_name = 'layers{}_margin{}_epoch{}.tar'.format(layers, margin, epoch+1)
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=False, save_dir=save_dir, filename=save_model_name)

            model.eval()
            acc = do_get_feature_and_t(model, margin=20, epoch=1)
            if acc > max_acc:
                print('max acc:', acc, ', epoch:', epoch + 1)

            margin = next_margin
    return save_model_name


if __name__ == "__main__":
    for _i in range(1):
        trainer('/home/ubuntu/Program/xhq/dataset/temp/train_data/', 20, 0, _time=_i, layers=50)
