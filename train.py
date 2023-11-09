from sklearn.model_selection import train_test_split
import torch,torchvision, torch.nn as nn
import torchvision.transforms.functional as fn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import Dataset, DataLoader,random_split
import os
import cv2
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm
import os
from read_data import read_dataset, data_loader, load_dataset
from model import create_model
from visualizing import visualizing_data, saving_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'

real_images_dir = "D:\\backup\\images_stunmaster\\3pd_align\\"
blur_images_dir = "D:\\backup\\images_stunmaster\\3pd_blur\\"

dir_list = [real_images_dir, blur_images_dir]
image_size = 256
image_number = 128
save = True
batch_size = 8
load = True

save_dataset_path = [
    "project//stunmaster//improve_quality//my_model/saved_data/128train_dataset.pt", 
    "project//stunmaster//improve_quality//my_model/saved_data/128test_dataset.pt", 
    "project//stunmaster//improve_quality//my_model/saved_data/128validation_dataset.pt" 
] 
if load:
    train_dataset       = load_dataset(save_dataset_path[0])
    test_dataset        = load_dataset(save_dataset_path[1])
    validation_dataset  = load_dataset(save_dataset_path[2])
else:
    train_dataset, test_dataset, validation_dataset = read_dataset(dir_list, image_size, image_number, device, save, save_dataset_path)

train_loader        = data_loader(train_dataset,2)
test_loader         = data_loader(test_dataset, 2)
validation_loader   = data_loader(validation_dataset, 2)

up_model, down_model, loss_function, optimizer = create_model(device)


def train_model(up_model, down_model ,epochs):
    train_loss=[]
    validation_loss = []
    for epoch in range(epochs):
        train_batch_loss = []
        up_model.train()
        down_model.train()

        for train_data, train_label in train_loader:
            y_up = up_model(train_data)
            y_down = down_model(train_data)
            y_down = fn.resize(y_down, (256, 256))

            y_up = fn.resize(y_up, y_down.shape[2:])
            y_hat = y_down + y_up

            # up_loss = loss_function(y_up, train_label)
            train_label = fn.resize(train_label, (y_hat.shape[2:]))


            # accumulate_loss = up_loss + down_loss
            loss = loss_function(y_hat, train_label)

            ##################################### train up net ################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_loss.append(loss.item())
            

        up_model.eval()
        down_model.eval()

        validation_batch_loss = []

        for val_data, val_label in validation_loader:

            y_up = up_model(val_data)
            y_down = down_model(val_data)

            y_up = fn.resize(y_up, y_down.shape[2:])
            y_hat = y_down + y_up

            val_label = fn.resize(val_label, (y_hat.shape[2:]))


            loss = loss_function(y_hat, val_label)

            validation_batch_loss.append(loss.item())
            
        
        train_loss.append(np.mean(train_batch_loss))
        validation_loss.append(np.mean(validation_batch_loss))

        ############################### testing model ################################
        test_data, test_label = next(iter(test_loader))
        y_up = up_model(test_data)

        y_down = down_model(test_data)

        y_up = fn.resize(y_up, y_down.shape[2:])

        y_predicted = y_down + y_up



        visualizing_data(y_predicted, test_data, test_label,test_data.shape[2], y_predicted.shape[2], epoch)
        saving_model(up_model.state_dict(), 'project//stunmaster//improve_quality//my_model/saved_model/up_model.pt' )
        saving_model(down_model.state_dict(), 'project//stunmaster//improve_quality//my_model/saved_model/down_model.pt' )
        
        print(f"train loss is {np.mean( train_batch_loss )} & validation loss is {np.mean(validation_batch_loss)} in {epoch}th epoch")

    plt.figure()
    plt.plot(train_loss,'s-', label='train')
    plt.plot(validation_loss,'o-', label='validation')
    plt.legend()
    plt.savefig(f"project//stunmaster//improve_quality//my_model/results/loss.png")
    plt.show()
    return up_model, down_model,  train_loss, validation_loss


up_model,down_model, train_loss, validation_loss = train_model(up_model, down_model, 2)