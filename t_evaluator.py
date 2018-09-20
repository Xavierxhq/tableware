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


pretrained_model = 'model/resnet50-19c8e357.pth'
base_model,  optim_policy = get_baseline_model(model_path=pretrained_model)
model_parameter = torch.load('model/pytorch-ckpt/model_best.pth.tar')
base_model.load_state_dict(model_parameter['state_dict'])



def dist(y1, y2):
    return torch.sqrt(torch.sum(torch.pow(y1 - y2, 2)))


def get_proper_input(img_path):
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


def get_feature(img_path, base_model):
    x = get_proper_input(img_path)
    y = base_model(x)
    return y


def get_dis(img_path_1, img_path_2, base_model):
    y1 = get_feature(img_path_1, base_model)
    y2 = get_feature(img_path_2, base_model)
    return dist(y1, y2)


def get_feature_map(base_model, lst, sample_num_each_cls=5, test_file_dir='datas/test_chawdoe/sample_data_5'):
    feature_map = dict()
    # lst is a list which includes class index as int array.

    for i in lst:
        cls_idx = str(i)
        feature_map[cls_idx] = list()
        dir_full_path = os.path.join(test_file_dir, cls_idx)
        dir_file_list = os.listdir(dir_full_path)
        for file_name in dir_file_list:
            file_full_path = os.path.join(dir_full_path, file_name)
            if len(feature_map[cls_idx]) < sample_num_each_cls:
                feature_map[cls_idx].append(get_feature(file_full_path, base_model))

    f = open('feature_map_'+str(len(lst)) + '_' + str(sample_num_each_cls), 'wb+')
    pickle.dump(feature_map, f)
    f.close()

    return feature_map


def get_sample_std_file(sample_num_each_cls=5, directory = 'datas/dishes_dataset/test', save_dir_path = 'datas/test_chawdoe/sample_data_5'):
    file_list = os.listdir(directory)
    sample_list = []
    copy_file_name_list = []
    sample_num_dict = dict()
    for i in file_list:
        line_list = re.split('_', i)
        class_index = line_list[-1][:-4]
        class_index = str(int(class_index) + 40)
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
        class_index = str(int(class_index) + 40)
        save_dir = os.path.join(save_dir_path, class_index)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, copy_file_name_list[i])
        shutil.copyfile(sample_list[i], save_path)


def rename_t_file():  # rename test file
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
    # print(new_map)
    return new_map


def t(base_model, lst, sample_num_each_cls = 5, test_dir = 'datas/dishes_dataset/test_std'):  # test

    feature_map_name = 'feature_map' + '_' + str(len(lst)) + '_' + str(sample_num_each_cls)
    feature_dir = r'/home/ubuntu/Program/Tableware/reid_tableware/evaluate_result/feature_map'
    f = open(os.path.join(feature_dir, feature_map_name), 'rb')
    feature_map = pickle.load(f)
    f.close()

    test_file_name_list = os.listdir(test_dir)

    all_map = dict()
    num_map = dict()
    positive_num = dict()

    import time
    t1 = time.time()
    j = 0

    for cls_idx in lst:
        cls_idx = str(cls_idx)
        positive_num[cls_idx] = 0
        num_map[cls_idx] = 0

    for i in test_file_name_list:
        file_path = os.path.join(test_dir, i)
        cls_idx = re.split('_', file_path)[-1][:-4]

        if int(cls_idx) not in lst:  # ugly code except the class we do not need test
            continue
        # print(cls_idx)
        tmp_dict = evaluate_single_file(file_path, feature_map, base_model)
        if tmp_dict[cls_idx] == 0:  # compute the correct num of the class
            positive_num[cls_idx] += 1
        if cls_idx not in all_map:  # compute the rank
            all_map[cls_idx] = tmp_dict
        else:
            for k in tmp_dict.keys():
                all_map[cls_idx][k] += tmp_dict[k]
        num_map[cls_idx] += 1  # compute all test num of the class
        j += 1
        # if j == 2:  # for quick test.
        #     break
    houzhui = str(len(lst)) + '_' + str(sample_num_each_cls)
    print(houzhui)
    t2 = time.time()
    print(t2 - t1)

    save_path = "evaluate_result/all_result/"

    f = open(os.path.join(save_path, 'result_map_' + houzhui), 'wb+')
    pickle.dump(all_map, f)
    f.close()

    f = open(os.path.join(save_path, 'num_map_' + houzhui), 'wb+')
    pickle.dump(num_map, f)
    f.close()

    f = open(os.path.join(save_path, 'positive_num_' + houzhui), 'wb+')
    pickle.dump(positive_num, f)
    f.close()

    rate_dict = dict()

    for k, v in positive_num.items():
        rate_dict[k] = v / (num_map[k] + 1e-12)
    # print(rate_dict)
    for k, v in all_map.items():
        for cls_idx in v.keys():
            all_map[k][cls_idx] /= num_map[k]

    f = open(os.path.join(save_path, 'all_map_' + houzhui), 'wb+')
    pickle.dump(all_map, f)
    f.close()

    return rate_dict, all_map


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


def get_text_result():


    pass


if __name__ == '__main__':
    get_text_result()
    # import time
    # t1 = time.time()
    # pic_path = 'datas/dishes_dataset/test_std/10001_1.png'
    # get_feature(pic_path, base_model)
    # t2 = time.time()
    # print(t2-t1)
    # lst = [i for i in range(1, 55)]
    # lst2 = [i for i in range(1, 41)]
    # lst3 = [i for i in range(41, 55)]
    # lst4 = []
    # for i in range(1, 55):
    #     if i % 2 == 0:
    #         lst4.append(i)
    # lst_all = [lst, lst2, lst3, lst4]
    # for i in lst_all:
    #     for j in [5, 10]:
    #         t(base_model, i, j)  # rate_map records the accuracy, and the all_map records the average rank.
    #
    # print(rate_dict['27'])
    # print(rate_dict['51'])
    # print(all_map['51']['51'])
    # print(all_map['27']['27'])

    # get_feature_map(base_model, lst, 10)


    # feature_map = get_feature_map(base_model, lst, 10, 'datas/test_chawdoe/sample_data_10')