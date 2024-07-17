import pickle
import numpy as np
import os
import random as rd
import shutil
import csv

def split_images(folder_path, dest_path, pos_rate, images_per_split):
    # 将文件夹里的图片随机分到多个文件夹中

    # 获取文件夹中的图片文件列表
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    rd.shuffle(image_files)  # 随机打乱图片顺序
    total_num = len(image_files)//images_per_split + 1

    for i in range(total_num):
        if i <= pos_rate * total_num:
        # 创建新的文件夹来存储每份图片
            new_folder = os.path.join(dest_path, f'pos_{i + 1}')
        else:
            new_folder = os.path.join(dest_path, f'neg_{i - int(pos_rate * total_num)}')
        os.makedirs(new_folder, exist_ok=True)

        # 选取当前份的图片
        start_index = i * images_per_split
        end_index = start_index + images_per_split
        current_split_images = image_files[start_index:end_index]

        # 将图片复制到新文件夹
        for j, image in enumerate(current_split_images):
            source_path = os.path.join(folder_path, image)
            destination_path = os.path.join(new_folder, f'{i+1}_{j+1}.png')
            shutil.copy(source_path, destination_path)

def all_pseudo_label_init(folder_path):
    # 将文件夹下的图片自动随机生成标注，即初始化随机伪代码生成
    result_dict = {}
    for root, dirs, files in os.walk(folder_path):
        current_folder = os.path.basename(root)
        file_dict = {}
        pos_loc = []
        if 'pos' in root:
            for idx in range(len(files)):
                decision = rd.randint(0, 1)
                if decision:
                    pos_loc.append(idx)
            if not len(pos_loc):
                pos_loc.append(rd.choice([i for i in range(len(files))]))
            # print("pos_loc", pos_loc)
        for index, file in enumerate(files):
            if file.endswith('.png'):
                if 'pos' in root:
                    if index in pos_loc:
                        file_dict[file.split('.')[0]] = 1
                    else:
                        file_dict[file.split('.')[0]] = 0
                else:
                    file_dict[file.split('.')[0]] = 0
            if file_dict:
                result_dict[current_folder] = file_dict
    return result_dict

def split_dataset(folder_path, target_path, train_ratio, test_ratio, val_ratio):
    # 收集文件夹及其标签
    folder_info = {}
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            label = dir_name.split('_')[0]  # 假设标签在文件夹名称中，通过特定方式提取
            if label not in folder_info:
                folder_info[label] = []
            folder_info[label].append(dir_name)

    # 计算每个标签应分配到各集的数量
    train_counts = {}
    test_counts = {}
    val_counts = {}
    for label, folder_list in folder_info.items():
        total_count = len(folder_list)
        train_count = int(total_count * train_ratio)
        test_count = int(total_count * test_ratio)
        val_count = int(total_count * val_ratio)

        train_counts[label] = train_count
        test_counts[label] = test_count
        val_counts[label] = val_count

    # 随机分配文件夹到各集
    train_set = []
    test_set = []
    val_set = []

    for label, folder_list in folder_info.items():
        rd.shuffle(folder_list)  # 打乱文件夹列表顺序

        for i, folder in enumerate(folder_list):
            if i < train_counts[label]:
                train_set.append(folder)
            elif i < train_counts[label] + test_counts[label]:
                test_set.append(folder)
            else:
                val_set.append(folder)

    # 创建目标文件夹
    train_folder = os.path.join(target_path, 'training')
    test_folder = os.path.join(target_path, 'testing')
    val_folder = os.path.join(target_path, 'val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 移动文件夹到对应的目标文件夹
    for folder in train_set:
        source = os.path.join(folder_path, folder)
        destination = os.path.join(train_folder, folder)
        shutil.move(source, destination)

    for folder in test_set:
        source = os.path.join(folder_path, folder)
        destination = os.path.join(test_folder, folder)
        shutil.move(source, destination)

    for folder in val_set:
        source = os.path.join(folder_path, folder)
        destination = os.path.join(val_folder, folder)
        shutil.move(source, destination)

    return train_set, test_set, val_set

def define_split_labels(all_labels_dict, name_set, flag):
    new_dict = {}
    for set in name_set:
        if set in all_labels_dict:
            new_dict[flag+set] = all_labels_dict[set]
        if flag == "testing/":
            save_keys_to_csv(new_dict)
    return new_dict

def save_keys_to_csv(data_dict):
    with open('reference.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for parent_key, child_dict in data_dict.items():
            left_content = parent_key.split('/')[1]
            if "pos" in parent_key:
                right_content = "pos"
            else:
                right_content = "neg"
            writer.writerow([left_content, right_content])



if __name__ == '__main__':
    gt_ins_labels_train = {"training/tumor_1": {"1_1": 1, "1_2": 0},
                           "training/tumor_2": {"2_1": 1, "2_2": 1},
                           "training/tumor_3": {"3_1": 0, "3_2": 1},
                           "training/normal_1": {"3_1": 0,	"3_2": 0},
                           "val/tumor_4": {"5_1": 1, "5_2": 1},
                           "val/tumor_5": {"6_1": 0, "6_2": 1},
                           "val/normal_2": {"7_1": 0, "7_2": 0},}
    gt_ins_labels_test = {"testing/test_1": {"1_1": 1, "1_2": 1},
                          "testing/test_2": {"1_1": 0, "1_2": 0},
                          "testing/test_3": {"1_1": 1, "1_2": 0},
                          "testing/test_4": {"1_1": 0, "1_2": 1},}
    dict = {
        "file_1": {
            "pic_1": 0,
            "pic_2": 0
        },
        "file_2": {
            "pic_1": 0,
            "pic_2": 0
        },
    }

    pos_rate = 0.2
    images_per_split = 5

    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1

    folder_path = r"E:/Desktop Files/Work/高测股份/Q-091-20240215-6"
    dest_path = r"E:/Desktop Files/Work/高测股份/dataset"

    split_images(folder_path, dest_path, pos_rate=pos_rate, images_per_split=images_per_split)


    all_labels_dict = all_pseudo_label_init(dest_path)
    print("all_labels_dict:", all_labels_dict)

    train_set, test_set, val_set = split_dataset(dest_path, "single/", train_ratio, test_ratio, val_ratio)

    train_dict = define_split_labels(all_labels_dict, train_set, "training/")
    test_dict = define_split_labels(all_labels_dict, test_set, "testing/")
    val_dict = define_split_labels(all_labels_dict, val_set, "val/")
    print("train_dict", train_dict)
    print("test_dict", test_dict)
    print("val_dict", val_dict)

    new_all_labels_dict = {**train_dict, **test_dict, **val_dict}
    new_train_dict = {**train_dict, **val_dict}

    with open('single/annotation/gt_ins_labels_train.p', 'wb') as f:
        pickle.dump(new_train_dict, f)

    with open('single/annotation/gt_ins_labels_test.p', 'wb') as f:
        pickle.dump(test_dict, f)

    with open('single/annotation/init_ins_pseudo_label.p', 'wb') as f:
        pickle.dump(new_all_labels_dict, f)

    with open("dataset/train_bags.txt", "w") as file:
        for key, value in train_dict.items():
            if key == list(train_dict.keys())[-1]:  # 判断是否为最后一个键
                file.write(f"/{key}/")
            else:
                file.write(f"/{key}/\n")

    with open("dataset/test_bags.txt", "w") as file:
        for key, value in test_dict.items():
            if key == list(test_dict.keys())[-1]:  # 判断是否为最后一个键
                file.write(f"/{key}/")
            else:
                file.write(f"/{key}/\n")

    with open("dataset/val_bags.txt", "w") as file:
        for key, value in val_dict.items():
            if key == list(val_dict.keys())[-1]:  # 判断是否为最后一个键
                file.write(f"/{key}/")
            else:
                file.write(f"/{key}/\n")