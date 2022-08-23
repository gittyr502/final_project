from ctypes.wintypes import RGB

from PIL import Image
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

import create_data as cd
import config as cnfg


def show_10_image_of_class(class_label:int) -> None:
    """
    show 10 images of choosen class
    :param class_label:
    :param cifar:
    """
    DATA = pd.read_csv(cnfg.csv_path)
    # if cifar == cnfg.cifar100:
    #     class_label = class_label+cnfg.num_classes_cifar10
    image_pathes=DATA[DATA['label']==class_label][:10]
    fig = plt.figure(figsize=(10, 7))

    for row in range(len(image_pathes)):
        fig.add_subplot(2, 5, row+1)
        img = Image.open(image_pathes.iloc[row]['image_path']+image_pathes.iloc[row]['image_name'])
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_classes_count()->None:
    '''
    returns the spread of the data
    :return:None
    '''
    DATA = pd.read_csv(cnfg.csv_path)
    value_count_dict=dict(DATA['label'].value_counts())
    value_count_dict={k:value_count_dict[k] for k in sorted(value_count_dict)}
    plt.figure(figsize=(10, 7))
    labels_array = cd.get_model_classes_dict()
    c = [(70/255,130/255,180/255), (99/255,184/255,255/255), (92/255,172/255,238/255), (79/255,148/255,205/255), (54/255,100/255,139/255)]
    plt.bar(value_count_dict.keys(), value_count_dict.values(),  width=0.4, color=c)
    plt.xticks(np.arange(len(labels_array)),labels_array.values(),rotation='vertical')
    plt.xlabel("classes")
    plt.ylabel("sum of images")
    plt.title("sum of images for each class")
    plt.tight_layout()
    plt.show()

def show_splited_classes_count()->None:
    '''
    show data after split train valisation and test
    :return: None
    '''
    # data = pd.read_csv(cnfg.csv_path)
    # value_count_dict = dict(data['label'].value_counts())
    # value_count_dict = {k: value_count_dict[k] for k in sorted(value_count_dict)}
    x_train, x_validation, x_test, y_train, y_validation, y_test=cd.split_train_test_validation()
    train=y_train.value_counts()
    train = {k: train[k] for k in sorted(train.keys())}
    validation = y_validation.value_counts()
    validation = {k: validation[k] for k in sorted(validation.keys())}
    test = y_test.value_counts()
    test = {k: test[k] for k in sorted(test.keys())}
    labels_array=cd.get_model_classes_dict()
    df = pd.DataFrame({"train": train,  "validation": validation, "test":test})
    ax = df.plot.bar(rot=0, figsize=(10, 7))
    plt.xticks(np.arange(len(labels_array.keys())), labels_array.values(), rotation='vertical')
    plt.tight_layout()
    plt.show()

