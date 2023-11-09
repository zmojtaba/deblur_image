import cv2 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from create_dataset import normalize_image
from torch.utils.data import TensorDataset, DataLoader
import torch,torchvision, torch.nn as nn




def get_batch( dir_lists, image_size, image_number ):
    data_batch = []
    y  = []
    image_files = [i for i in os.listdir(dir_lists[1])[:image_number]]
    print('--------------', np.shape(image_files))
    for sample_file in image_files:
        real_sample_dir =  dir_lists[0] + sample_file
        blur_sample_dir =  dir_lists[1] +'/'+ sample_file
        data_batch.append(normalize_image(blur_sample_dir, image_size))
        y.append(normalize_image(real_sample_dir, image_size))

    y = np.array(y)
    data_batch = np.array(data_batch)
    

    return data_batch, y



def tensor_dataset(data, label, image_size, device):
    data_set = TensorDataset( torch.tensor( data ).reshape(-1, 3, image_size, image_size).float().to(device), torch.tensor(label).reshape(-1, 3, image_size, image_size).float().to(device))
    return data_set

def save_dataset(dataset, path):
    torch.save(dataset, path)


def split_data(data, label, image_size, device, save, save_path):
    print('-----------------------------reading data is finished--------------------------')
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1, shuffle=True)
    test_data, validation_data, test_label, validation_label  = train_test_split(test_data, test_label, test_size=0.5, shuffle=True)

    print(np.shape(train_data), np.shape(test_data), np.shape(validation_data))

    train_dataset = tensor_dataset(train_data, train_label, image_size, device)
    test_dataset = tensor_dataset(test_data, test_label, image_size, device)
    validation_dataset = tensor_dataset(validation_data, validation_label, image_size, device)
    if save:
        print('--------------savinng data---------------')
        save_dataset(train_dataset,      save_path[0])
        save_dataset(test_dataset,       save_path[1])
        save_dataset(validation_dataset, save_path[2])

    return train_dataset, test_dataset, validation_dataset

def read_dataset(dir_list, image_size, image_number, device, save, save_path):
    data, label = get_batch(dir_list, image_size, image_number)
    train_dataset, test_dataset, validation_dataset = split_data(data, label, image_size, device, save, save_path)
    return train_dataset, test_dataset, validation_dataset

def load_dataset(dataset_path):
    dataset    =       torch.load(dataset_path)
    return dataset

def data_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size, drop_last=True, shuffle=True)
    return loader




