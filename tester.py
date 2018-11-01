import os, re, sys, shutil, pickle, time
import torch
from models.networks import get_baseline_model
from datasets import data_loader
from torch.autograd import Variable
from utils.transforms import TestTransform
import numpy as np
from get_xls_from_map import init_dict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(0)

dish_map_dict = {  # we need the dict.
    '32':[0, 10],
    '35': [4],
    '12': [4],
    '26': [4],
    '50': [1],
    '21': [4],
    '29': [9],
    '8': [4],
    '4': [5],
    '3': [9],
    '47': [4],
    '38': [9],
    '42': [5],
    '33': [5],
    '28': [1],
    '16': [9],
    '43': [9],
    '15': [9],
    '6': [4],
    '37': [4],
    '17': [9],
    '11': [4],
    '18': [4],
    '41': [8],
    '5': [1],
    '14': [4],
    '24': [8],
    '46': [5],
    '22': [6],
    '48': [4],
    '27': [2],
    '44': [9],
    '54': [4],
    '9': [4],
    '1': [9],
    '30': [4],
    '23': [9],
    '40': [5],
    '53': [4],
    '2': [4],
    '31': [9],
    '10': [1],
    '51': [3],
    '13': [8],
    '36': [9],
    '34': [1],
    '20': [10, 4],
    '45': [11],
    '39': [9],
    '19': [5],
    '7': [5],
    '25': [4],
    '52': [1],
    '49': [4]
}


def dist(y1, y2):  # ok
    y1 = y1.cuda()
    y2 = y2.cuda()
    return torch.sqrt(torch.sum(torch.pow(y1 - y2, 2))).item()


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


def get_feature(img_path, base_model, use_cuda=True):  # ok
    x = get_proper_input(img_path)
    if use_cuda:
        x = x.cuda()
    y = base_model(x)
    if use_cuda:
        y = y.cuda()
    return y


def get_dis(img_path_1, img_path_2, base_model):  # ok
    y1 = get_feature(img_path_1, base_model)
    y2 = get_feature(img_path_2, base_model)
    return dist(y1, y2)


def load_model(model_path=None, layers=50):
    if not model_path:
        model_path = 'model/resnet50-19c8e357.pth'
    base_model, optim_policy = get_baseline_model(model_path=model_path, layers=layers)
    model_parameter = torch.load(model_path)
    base_model.load_state_dict(model_parameter['state_dict'])
    base_model = base_model.cuda()
    print('model', model_path.split('/')[-1], 'loaded.')
    return base_model


def evaluate_single_file_with_average_feature_map(file_path, feature_map, base_model):
    result_dict = {}
    file_feature = get_feature(file_path, base_model)
    for k, v in feature_map.items():
        if type(v) == dict:
            continue
        _feature = torch.FloatTensor(v)
        result_dict[k] = dist(file_feature, _feature)

    for k, v in result_dict.items():
        for i in np.nditer(result_dict[k]):
            result_dict[k] = float(str(i))

    my_map = sorted(result_dict.items(), key=lambda d: d[1])

    new_map = dict()
    rank_list = list()
    for i in range(len(my_map)):
        new_map[str(my_map[i][0])] = i
        rank_list.append(str(my_map[i][0]))
    return new_map, rank_list, my_map[0][0]


def transform_feature_map_to_everage(origin_feature_map, output_map_path = '', feature_num_each_class=5, _range=55):
    new_feature_map = init_dict(_range, _range, 0)
    lst = [i for i in range(_range)]
    for index in lst:
        index = str(index)
        if str(index) in origin_feature_map:
            _avg_feature = np.zeros(shape=origin_feature_map[index][0].shape)
            for _feature in origin_feature_map[index]:
                _feature = _feature.cpu().detach().numpy()
                _avg_feature += _feature
            _avg_feature /= feature_num_each_class
            new_feature_map[index] = torch.FloatTensor(_avg_feature)
    pickle_write(output_map_path, new_feature_map)
    return new_feature_map


def get_feature_map_average(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_file_dir='./base_sample/', save_dir='evaluate_result/feature_map'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    save_path = os.path.join(save_dir, save_file_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    start_time = time.time()
    # lst is a list which includes class index as int array.
    test_file_dir += str(sample_num_each_cls)
    if not os.path.exists(test_file_dir):
        print('You must use get sample_std_file() firstly')
        return None

    for i in lst:
        # print(i)
        ground_truth_label = str(i)
        # feature_map[ground_truth_label] = list()
        features = []
        dir_full_path = os.path.join(test_file_dir, ground_truth_label)  # open the directory in order.
        dir_file_list = os.listdir(dir_full_path)
        for file_name in dir_file_list:
            file_full_path = os.path.join(dir_full_path, file_name)
            # if len(feature_map[ground_truth_label]) < sample_num_each_cls:
            if len(features) < sample_num_each_cls:
                feature_on_gpu = get_feature(file_full_path, base_model)
                # feature_map[ground_truth_label].append(f)
                features.append(feature_on_gpu)
        write_feature_map(save_path, ground_truth_label, features)
        features = None

    feature_map = pickle_read(save_path)
    new_feature = transform_feature_map_to_everage(feature_map, save_path, _range=55)
    # print('feature map of avg has been saved in ' + save_path)
    print('time for generating feature map:', '%.1f' % (time.time() - start_time), 's')
    return new_feature


def write_feature_map(feature_map_name, label, features):
    if os.path.exists(feature_map_name):
        obj = pickle_read(feature_map_name)
        obj[label] = features
    else:
        obj = {
            label: features
        }
    pickle_write(feature_map_name, obj)


def get_sample_std_file(sample_num_each_cls=5, directory='/home/ubuntu/Program/xhq/dataset/temp/test_data/', save_dir_path='./base_sample/', refresh=True):
    # half complete. Usually only use once in your first training.
    save_dir_path += str(sample_num_each_cls)

    if os.path.exists(save_dir_path) and not refresh:
        print('base file exists, and no force to refresh.')
        return
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)

    sample_list, copy_file_name_list, sample_num_dict = [], [], {}

    for i in os.listdir(directory):
        class_index = i.split('.')[0].split('_')[-1]
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
        class_index = copy_file_name_list[i].split('.')[0].split('_')[-1]
        save_dir = os.path.join(save_dir_path, class_index)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, copy_file_name_list[i])
        shutil.copyfile(sample_list[i], save_path)


def t_save_file(feature_map, base_model, lst, sample_num_each_cls=5, margin=5, epoch=1,
                test_dir='/home/ubuntu/Program/xhq/dataset/temp/test_data/',
                feature_dir='evaluate_result/feature_map/'):  # test

    t1 = time.time()
    test_file_name_list = os.listdir(test_dir)
    rank_map = dict()
    num_map = dict()
    positive_num = dict()
    first_num = dict()
    rate_dict = dict()

    for i in lst:
        first_num[str(i)], positive_num[str(i)], num_map[str(i)] = dict(), 0, 0
        for j in lst:
            first_num[str(i)][str(j)] = 0

    all_count, positive_count = 0, 0
    mapping_dict = pickle_read('./evaluate_result/all_result/train_mapping_dict.pkl')

    for i in test_file_name_list:
        file_path = os.path.join(test_dir, i)
        cls_idx = i.split('.')[0].split('_')[-1] # accroding to the directory name
        # print('cls_idx:', cls_idx)

        if int(cls_idx) not in lst:  # ugly code except the class we do not need in test
            # print('test dont handle class like:', cls_idx)
            continue
        all_count += 1

        tmp_dict, _, prid_label = evaluate_single_file_with_average_feature_map(file_path, feature_map, base_model)
        if tmp_dict[str(cls_idx)] == 0:  # compute the correct num of the class
            positive_num[str(cls_idx)] += 1
            positive_count += 1
        else:
            pass
        if cls_idx not in rank_map:  # compute the rank
            rank_map[cls_idx] = tmp_dict
        else:
            for k in tmp_dict.keys():
                rank_map[cls_idx][k] += tmp_dict[k]
                if tmp_dict[k] == 0:
                    first_num[cls_idx][k] += 1

        num_map[str(cls_idx)] += 1  # compute all test num of the class

        if all_count % 500 == 0:
            print('now all:', all_count, ', and positive:', positive_count)
            pass

    for i in range(1, len(lst) + 1):
        _key = str(i)
        if _key in positive_num.keys():
            _acc = positive_num[_key] / (num_map[_key] + 1e-12)

    suffix = '{}_{}'.format(len(lst), sample_num_each_cls)
    prefix = 'margin({})_epoch({})_'.format(margin, epoch)

    t2 = time.time()
    print('time for testing', '%.2f' % (t2 - t1), 's')  # the time we use in the test

    save_path = "evaluate_result/all_result/"
    for k, v in positive_num.items():
        rate_dict[k] = v / (num_map[k] + 1e-12)  # to avoid 0

    for k, v in rank_map.items():
        for cls_idx in v.keys():
            rank_map[k][cls_idx] /= num_map[k]
    # pickle_write(os.path.join(save_path, prefix + 'num_map_' + suffix), num_map)  # all prediction of each class
    # pickle_write(os.path.join(save_path, prefix + 'positive_num_' + suffix), positive_num)  # correct prediction of each class
    # pickle_write(os.path.join(save_path, prefix + 'first_num_' + suffix), first_num)   # rank 1st num of each class
    # pickle_write(os.path.join(save_path, prefix + 'all_map_' + suffix), rank_map)  # average rank
    return rate_dict, rank_map, positive_num, num_map


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('pickle read error: not exits {}'.format(file_path))
        return None


def pickle_write(file_path, what_to_write):
    try:
        with open(file_path, 'wb+') as f:
            pickle.dump(what_to_write, f)
    except:
        print('pickle write error: {}'.format(file_path))


def get_accuracy_from_map(positive_num, num_map):
    start_time = time.time()
    if type(positive_num) == str:
        positive_num = pickle_read(positive_num)
    if type(num_map) == str:
        num_map = pickle_read(num_map)
    all_positive_num = 0
    all_num = 0

    mapping_dict = pickle_read('./evaluate_result/all_result/train_mapping_dict.pkl')
    for key, value in positive_num.items():
        all_positive_num += positive_num[key]
        all_num += num_map[key]
    print('time for get_accuracy_from_map:', '%.1f' % (time.time() - start_time), 's')
    return all_positive_num / (all_num + 1e-12)

def evaluate_model(model, margin, epoch):
    model.eval()
    lst_all = [i for i in range(1, 55)]
    lst = [lst_all]
    for i in lst:
        for j in [5]:  # choose 5, 10 samples as the database
            feature_map = get_feature_map_average(model, i, j, margin=margin, epoch=epoch)
            _, _, positive_num, num_map = t_save_file(feature_map, model, i, j, margin=margin, epoch=epoch)
            _accuracy = get_accuracy_from_map(positive_num, num_map)
            print('Accuracy: {} under {} classes, {} samples/class'.format(_accuracy, len(i), j))
    return _accuracy


if __name__ == '__main__':
    get_sample_std_file(5) # Do this to get 5 sample pictures for every class

    # model = load_model(model_path='./model/pytorch-ckpt/time2/time2_layers50_margin20_epoch1.tar', layers=50)
    # acc = evaluate_model(model, margin=20, epoch=1)
