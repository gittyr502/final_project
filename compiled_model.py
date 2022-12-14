import cv2
import pandas as pd
import keras
import numpy as np
from keras.utils import load_img , img_to_array, save_img

import config as cnfg
import create_data as cd

def load_compiled_model():
    '''
    load model training
    :return: extract model
    '''
    model = keras.models.load_model(cnfg.model_path)
    return model

def load_history():
    '''
    read history
    :return: read history
    '''
    history=pd.read_csv(cnfg.history_path)
    return history

def load_data():
    '''
    read data from zip file
    :return: data after extract
    '''
    loaded_data = np.load('./'+cnfg.z_file_path)
    x_train = loaded_data['train'].astype('float32')/255
    x_validation = loaded_data['validation'].astype('float32')/255
    x_test = loaded_data['test'].astype('float32')/255
    y_train = loaded_data['ytrain']
    y_test = loaded_data['ytest']
    y_validation = loaded_data['yvalidation']
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    return x_train, x_validation, x_test, y_train, y_validation, y_test

def load_our_labels():
    '''
    load labels from zip file
    :return: labels after extract
    '''
    return cd.get_model_classes_dict()

#load image with keras
def predict_by_image1(image):
    '''
    predict of images
    :param image:
    :return: prediction
    '''
    model = load_compiled_model()
    if isinstance(image, str):
        image = load_img(image, target_size=(32,32))
    image=img_to_array(image)
    image = image.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    prediction = model.predict(image,verbose=0)
    print(prediction)
    pred = np.argsort(prediction)
    print(pred)
    pred = pred[0][-3:]
    print(pred)
    labels = [cd.get_model_classes_dict()[pred[-1]], cd.get_model_classes_dict()[pred[-2]],
              cd.get_model_classes_dict()[pred[-3]]]
    print(labels)
    percent = ["%5.2f" % (float(prediction[0][pred[-1]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-2]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-3]]) * 100) + "%"]
    res_dict= {labels[i]: percent[i] for i in range(len(percent))}
    print(res_dict)
    return res_dict

#load image with cv2
def predict_by_image(image):
    '''
    predict of images
    :param image:
    :return: prediction
    '''
    model = load_compiled_model()
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    prediction = model.predict(image,verbose=0)
    pred = np.argsort(prediction)
    pred = pred[0][-3:]
    labels = [cd.get_model_classes_dict()[pred[-1]], cd.get_model_classes_dict()[pred[-2]],
              cd.get_model_classes_dict()[pred[-3]]]
    percent = ["%5.2f" % (float(prediction[0][pred[-1]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-2]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-3]]) * 100) + "%"]
    res_dict= {labels[i]: percent[i] for i in range(len(percent))}
    return res_dict