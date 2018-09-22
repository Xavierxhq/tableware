import torch
from models.networks import get_baseline_model
from datasets import data_loader
from torch.autograd import Variable
import numpy as np
import os
from utils.transforms import TestTransform
import re
import shutil
import pickle
from collections import OrderedDict
import json
import xlwt
import time


pretrained_model = 'model/resnet50-19c8e357.pth'
base_model,  optim_policy = get_baseline_model(model_path=pretrained_model)
model_parameter = torch.load('model/pytorch-ckpt/1_test_checkpoint_ep1.pth.tar')
base_model.load_state_dict(model_parameter['state_dict'])



def dist(y1, y2):  # ok
    return torch.sqrt(torch.sum(torch.pow(y1 - y2, 2)))


def get_proper_input(img_path):  # ok
    if not os.path.exists(img_path):
        return None
    pic_data = data_loader.read_image(img_path)
    lst = list()
    HEIGHT = 128
    WIDTH = 128
    test = TestTransform(WIDTH, HEIGHT)
    lst.append(np.array(test(pic_data)))
    lst = np.array(lst)
    pic_data = Variable(torch.from_numpy(lst))
    return pic_data


def get_feature(img_path, base_model):  # ok
    x = get_proper_input(img_path)
    y = base_model(x)
    return y


def get_dis(img_path_1, img_path_2, base_model):  # ok
    y1 = get_feature(img_path_1, base_model)
    y2 = get_feature(img_path_2, base_model)
    return dist(y1, y2)


def get_feature_map(base_model, lst, sample_num_each_cls=5, test_file_dir='datas/test_chawdoe/sample_data_'):
    # ok

    if os.path.exists('feature_map_'+str(len(lst)) + '_' + str(sample_num_each_cls)):
        return None
    feature_map = dict()
    # lst is a list which includes class index as int array.
    test_file_dir += str(sample_num_each_cls)
    for i in lst:
        cls_idx = str(i)
        feature_map[cls_idx] = list()
        dir_full_path = os.path.join(test_file_dir, cls_idx)  # open the directory in order.
        dir_file_list = os.listdir(dir_full_path)
        for file_name in dir_file_list:
            file_full_path = os.path.join(dir_full_path, file_name)
            if len(feature_map[cls_idx]) < sample_num_each_cls:
                feature_map[cls_idx].append(get_feature(file_full_path, base_model))

    f = open('feature_map_'+str(len(lst)) + '_' + str(sample_num_each_cls), 'wb+')
    pickle.dump(feature_map, f)
    f.close()

    return feature_map


def get_sample_std_file(sample_num_each_cls=5, directory = 'datas/dishes_dataset/test_std', save_dir_path = 'datas/test_chawdoe/sample_data_'):
    # half complete
    save_dir_path += str(sample_num_each_cls)
    file_list = os.listdir(directory)
    sample_list = []
    copy_file_name_list = []
    sample_num_dict = dict()
    for i in file_list:
        line_list = re.split('_', i)
        class_index = line_list[-1][:-4]
        if class_index not in sample_num_dict:
            sample_num_dict[class_index] = 1
        elif sample_num_dict[class_index] == sample_num_each_cls:
            continue
        else:
            sample_num_dict[class_index] += 1
        sample_list.append(os.path.join(directory, i))
        copy_file_name_list.append(i)
    for i in range(len(sample_list)):
        line_list = re.split('_', copy_file_name_list[i])
        class_index = line_list[-1][:-4]
        save_dir = os.path.join(save_dir_path, class_index)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, copy_file_name_list[i])
        shutil.copyfile(sample_list[i], save_path)


def rename_t_file():  # rename test file. no need any more.
    test_file_dir = 'datas/dishes_dataset/test'
    save_file_dir = 'datas/dishes_dataset/test_std'
    for i in os.listdir(test_file_dir):
        file_full_path = os.path.join(test_file_dir, i)
        line = re.split('_', file_full_path[:-4])
        cls_idx = int(line[-1]) + 40
        new_line = line[1][-5:] + '_' + str(cls_idx) + '.png'
        new_line = os.path.join(save_file_dir, new_line)
        # print(new_line)
        shutil.copyfile(file_full_path, new_line)


def evaluate_single_file(file_path, feature_map, base_model):
    result_dict = {}
    file_feature = get_feature(file_path, base_model)
    for k, v in feature_map.items():
        for _feature in v:
            if k not in result_dict:
                result_dict[k] = dist(file_feature, _feature)
            else:
                result_dict[k] += dist(file_feature, _feature)
    for k, v in result_dict.items():
        result_dict[k] = np.asarray(v.detach().numpy())
        for i in np.nditer(result_dict[k]):
            result_dict[k] = float(str(i))

    my_map = sorted(result_dict.items(), key=lambda d: d[1])

    new_map = dict()
    for i in range(len(my_map)):
        new_map[str(my_map[i][0])] = i

    return new_map


def t(base_model, lst, sample_num_each_cls = 5, test_dir = 'datas/dishes_dataset/test_std'):  # test

    feature_map_name = 'feature_map' + '_' + str(len(lst)) + '_' + str(sample_num_each_cls)
    feature_dir = './'
    f = open(os.path.join(feature_dir, feature_map_name), 'rb')
    feature_map = pickle.load(f)
    f.close()

    test_file_name_list = os.listdir(test_dir)

    rank_map = dict()
    num_map = dict()
    positive_num = dict()
    first_num = dict()

    for i in lst:

        first_num[str(i)] = dict()
        positive_num[str(i)] = 0
        num_map[str(i)] = 0

        for j in lst:
            first_num[str(i)][str(j)] = 0


    t1 = time.time()
    j = 0  # no need


    for i in test_file_name_list:
        file_path = os.path.join(test_dir, i)
        cls_idx = re.split('_', file_path)[-1][:-4]  # accroding to the directory name

        if int(cls_idx) not in lst:  # ugly code except the class we do not need in test
            continue
        # print(cls_idx)
        tmp_dict = evaluate_single_file(file_path, feature_map, base_model)
        if tmp_dict[cls_idx] == 0:  # compute the correct num of the class
            positive_num[cls_idx] += 1
        if cls_idx not in rank_map:  # compute the rank
            rank_map[cls_idx] = tmp_dict
        else:
            for k in tmp_dict.keys():
                rank_map[cls_idx][k] += tmp_dict[k]
                if tmp_dict[k] == 0:
                    first_num[cls_idx][k] += 1

        num_map[cls_idx] += 1  # compute all test num of the class
        j += 1


    houzhui = str(len(lst)) + '_' + str(sample_num_each_cls)
    # print(houzhui)
    t2 = time.time()
    print(t2 - t1)
    print(first_num)
    save_path = "evaluate_result/all_result/"

    f = open(os.path.join(save_path, 'result_map_' + houzhui), 'wb+')  # average rank
    pickle.dump(rank_map, f)
    f.close()

    f = open(os.path.join(save_path, 'num_map_' + houzhui), 'wb+')  # evaluate num of each class
    pickle.dump(num_map, f)
    f.close()

    f = open(os.path.join(save_path, 'positive_num_' + houzhui), 'wb+')  # evaluate
    pickle.dump(positive_num, f)
    f.close()

    f = open(os.path.join(save_path, 'first_num_' + houzhui), 'wb+')
    pickle.dump(first_num, f)
    f.close()

    rate_dict = dict()

    for k, v in positive_num.items():
        rate_dict[k] = v / (num_map[k] + 1e-12)  # to avoid 0
    # print(rate_dict)
    for k, v in rank_map.items():
        for cls_idx in v.keys():
            rank_map[k][cls_idx] /= num_map[k]

    f = open(os.path.join(save_path, 'all_map_' + houzhui), 'wb+')
    pickle.dump(rank_map, f)
    f.close()

    return rate_dict, rank_map


def get_all_feature_map():
    lst = [i for i in range(1, 55)]
    lst2 = [i for i in range(1, 41)]
    lst3 = [i for i in range(41, 55)]
    lst4 = []
    for i in range(1, 55):
        if i % 2 == 0:
            lst4.append(i)
    lst_all = [lst, lst2, lst3, lst4]
    for i in [5, 10]:
        for j in lst_all:
            get_feature_map(base_model, j, i, 'datas/test_chawdoe/sample_data_' + str(i))


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('pickle read error: not exits {}'.format(file_path))
        return None


def get_text_result():  # just for a test temproraly
    base_dir = './'

    class_num = ['46']  # need to be modified accroding to the classes num.
    sample_num = ['_5', '_10']  #  5 or 10 samples each class
    positive_map_str = 'positive_num_'
    num_map_str = 'num_map_'
    all_map_str = 'all_map_'
    for i in class_num:
        for j in sample_num:
            positive_map_name = positive_map_str + str(i) + str(j)
            num_map_name = num_map_str + str(i) + str(j)
            all_map_name = all_map_str + str(i) + str(j)
            positive_map_path = os.path.join(base_dir, positive_map_name)
            num_map_path = os.path.join(base_dir, num_map_name)
            all_map_path = os.path.join(base_dir, all_map_name)
            positive_map = pickle_read(positive_map_path)
            num_map = pickle_read(num_map_path)
            all_map = pickle_read(all_map_path)
            result_dict = OrderedDict()
            write_excel(all_map, 'result_' + str(i) + str(j) + '.xls')
            for cls in range(55):  # read the class index in order to make the result in order.
                cls_idx = str(cls)
                if cls_idx in positive_map:
                    result_dict[cls_idx] = positive_map[cls_idx] / num_map[cls_idx]
                    f = open(i + j + '.txt', 'w+')
                    f.write('accuracy:')
                    f.write(json.dumps(result_dict, indent=4))
                    f.write('average rank of each class')
                    f.write(json.dumps(all_map, indent=4))
                    f.close()


def write_excel(new_all_map, file_name):
    book = xlwt.Workbook()
    sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True)
    row_index = 1
    col_index = 1
    for i in range(55):
        if str(i) in new_all_map.keys():
            for j in range(55):
                if str(j) in new_all_map[str(i)]:
                    sheet.write(row_index, col_index, new_all_map[str(i)][str(j)])
                    col_index += 1
            row_index += 1
            col_index = 1
    assert file_name[-4:] == '.xls'
    book.save(file_name)


def do_get_feature_and_t():
    # lst_except = [4, 7, 14, 18, 21, 24, 26, 30]
    lst = [i for i in range(1, 55)]

    lst_all = [lst]
    for i in lst_all:
        for j in [5, 10]:
            get_feature_map(base_model, i, j)
            # print(len(a1['8']))
            t(base_model, i, j)  # rate_map records the accuracy, and the all_map records the average rank.


def get_num_of_each_class_in_dir():
    train_dir = 'datas/dishes_dataset/train'
    file_list = os.listdir(train_dir)
    num_dict = dict()
    for i in range(41):
        num_dict[str(i)] = 0
    for file_name in file_list:
        # file_path = os.path.join(train_dir, file_name)
        line = re.split('_', file_name)
        cls = line[-1][:-4]
        num_dict[cls] += 1
    return num_dict


if __name__ == '__main__':
    # do_get_feature_and_t()
    # mapping_dict = pickle_read('evaluate_result/all_result/mapping_dict')
    #
    # the_dict = get_num_of_each_class_in_dir()
    # print(json.dumps(the_dict, indent=4))
    # get_sample_std_file(5)
    # get_sample_std_file(10)
    do_get_feature_and_t()
    pass