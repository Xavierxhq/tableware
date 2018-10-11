import os
import cv2
import pickle


if __name__ == "__main__":
    rootdir = "/home/ubuntu/Program/Tableware/data/2018043000/样本/样本"
    train_save_dir = "../datas/dishes_dataset/train/"
    test_save_dir = "../datas/dishes_dataset/test_std/"
    mapping_dict = dict()
    train_num_dict = dict()
    test_num_dict = dict()
    class_list = os.listdir(rootdir)
    each_class_train_num = 300
    each_class_test_num = 200
    train_class_limit = 54

    # should be careful because the index is [0, 53]
    # but the class is [1, 54]
    test_class_list = [i for i in range(0, 54)]

    # prepare training data
    index = 10000
    label = 1

    for cls_name in class_list:  # it will in order by index.
        cls_path = os.path.join(rootdir, cls_name)
        file_list_of_cls = os.listdir(cls_path)
        mapping_dict[str(label)] = cls_name
        train_num_dict[str(label)] = 0
        while train_num_dict[str(label)] < each_class_train_num:
            for file_name in file_list_of_cls:
                file_path = os.path.join(cls_path, file_name)
                img = cv2.imread(file_path)
                cv2.imwrite(os.path.join(train_save_dir, str(index) + '_' + str(label) + '.png'), img)
                index += 1
                train_num_dict[str(label)] += 1
                if train_num_dict[str(label)] == each_class_train_num:
                    break
        label += 1
        if label == train_class_limit:  # beacuse we only train the class in [1, 41)
            break

    index = 10000  # reset

    for cls_idx in test_class_list:
        label = cls_idx + 1
        cls_name = class_list[cls_idx]
        cls_path = os.path.join(rootdir, cls_name)
        file_list_of_cls = os.listdir(cls_path)
        mapping_dict[str(label)] = cls_name
        test_num_dict[str(label)] = 0
        for file_name in file_list_of_cls:
            file_path = os.path.join(cls_path, file_name)
            img = cv2.imread(file_path)
            cv2.imwrite(os.path.join(test_save_dir, str(index) + '_' + str(label) + '.png'), img)
            index += 1
            test_num_dict[str(label)] += 1
            if test_num_dict[str(label)] == each_class_test_num:
                break
        label += 1

    save_dir = '../evaluate_result/all_result/'

    f = open(os.path.join(save_dir, 'mapping_dict'), 'wb+')
    pickle.dump(mapping_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'test_num_dict'), 'wb+')
    pickle.dump(test_num_dict, f)
    f.close()









