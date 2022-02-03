import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from dual_pixel_test_datasets import *
import model_test 
import scipy.misc
from PIL import Image
import PIL.Image as pil
import skimage
import PIL.Image as pil
from tqdm import tqdm
import torch.optim as optim
import time
import os
import copy
import shutil
from data_preparation import *
import matplotlib.pyplot as plt
import os.path
import time
import datetime

parser = argparse.ArgumentParser(description="DDD")
parser.add_argument('--start_epoch',type = int, default = 1)
parser.add_argument('--batchsize',type = int, default = 8)
parser.add_argument('--gpu',type=int, default=1)
parser.add_argument('--input_test_file', type=str, default ="./data/simudata/NYU/")
parser.add_argument('--img_list_t', type=str, default ="./data/nyu_test.txt")
parser.add_argument('--output_file', type=str, default ="test_results/DDDsys/")
parser.add_argument('--modelname', type=str, default = "model_nyu", help="model_nyu")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
args = parser.parse_args()


#Hyper Parameters
METHOD = args.modelname
OUT_DIR = args.output_file
GPU = range(args.gpu)
TEST_DIR = args.input_test_file

 
if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


# mse = nn.MSELoss().cuda()

Estd_stereo = model_test.YRStereonet_3D()
Estd_stereo = torch.nn.DataParallel(Estd_stereo, device_ids=GPU)
Estd_stereo.cuda()


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, \
    criterion, use_cuda, checkpoint_path, best_model_path, log_file):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path
    
    returns trained model
    """
    log_file = open(log_file, 'w')
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 
    
    for epoch in range(start_epochs, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        transform = transforms.ToTensor()
        ###################
        # train the model #
        ###################
        # path_to_data = enumerate(loaders['train'])
        # # print(batch_idx)
        # print(path_to_data)
        
        for batch_idx, path_to_data in enumerate(loaders['train']):
            start = time.time()
            model.train()
            # print(path_to_data['left'])
            # print(path_to_data['right'])
            # print(path_to_data['gt'])
            number_of_samples = len(path_to_data['left'])
            for i in range(number_of_samples):
            # print(path_to_data)
                left_img = transform(Image.open(path_to_data['left'][i])).unsqueeze(0)
                right_img = transform(Image.open(path_to_data['right'][i])).unsqueeze(0)
                gt = transform(Image.open(path_to_data['gt'][i]))

                # move to GPU
                if use_cuda:
                    left, right, target = left_img.cuda(), right_img.cuda(), gt.cuda()
                ## find the loss and update the model parameters accordingly
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(left, right)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                train_loss = train_loss + loss.data
            train_loss = train_loss / number_of_samples
        
            ######################    
            # validate the model #
            ######################
            model.eval()
            path_to_data = next(iter(loaders['test']))
            number_of_samples = len(path_to_data['left'])
            for i in range(number_of_samples):    

                left_img = transform(Image.open(path_to_data['left'][i])).unsqueeze(0)
                right_img = transform(Image.open(path_to_data['right'][i])).unsqueeze(0)
                gt = transform(Image.open(path_to_data['gt'][i]))
                # move to GPU
                if use_cuda:
                    left, right, target = left_img.cuda(), right_img.cuda(), gt.cuda()
                ## update the average validation loss
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(left, right)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss = valid_loss + loss.data
            valid_loss = valid_loss / number_of_samples
            end = time.time()
            info = f'batch idx = {batch_idx}, training loss = {train_loss}, validation loss = {valid_loss}, took {end - start} second'
            print(info)
            log_file.writelines(info)
            
        # print training/validation statistics
        info = 'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            )
        print(info)
        log_file.writelines(info)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            info = 'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss)
            print(info)
            print(info)
            log_file.writelines(info)
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
        
        log_file.close()
            
    # return trained model
    return model


print("Preparing dataset")
dataset = dpd_disp_dataset("./data/ICCP2020_DP_dataset_new/left", \
        "./data/ICCP2020_DP_dataset_new/right", "./data/ICCP2020_DP_dataset_new/gt", transform = transforms.ToTensor())

dp_dataset, train_set, validation_set, test_set = dataset.create_dataset()
print(f'training set: {len(train_set)}, validation_set: {len(validation_set)} test set: {len(test_set)}')


training_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0)
testing_loader = torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=True, num_workers=0)
loaders = {'train':training_loader, 'test':testing_loader}

criterion = nn.MSELoss().cuda()
use_cuda = torch.cuda.is_available()
optimizer = optim.Adam(Estd_stereo.parameters(), lr=0.0001)
epoch = 3

print("Start training ... ")
path_to_best_model = './checkpoints/best_model.pt'
if os.path.isfile(path_to_best_model):
    best_model = torch.load(path_to_best_model)
    valid_loss_min_input = best_model['valid_loss_min']
    start_epoch = best_model['epoch']
    Estd_stereo.load_state_dict(best_model['state_dict'])
    optimizer.load_state_dict(best_model['optimizer'])
else:
    valid_loss_min_input = np.Inf
    start_epoch = 0
    Estd_stereo.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Estd" + ".pkl")), strict=False)
    print ("File not exist")

current_time = str(datetime.datetime.now())
log_filename = current_time.replace("-", "_")
log_filename = log_filename.replace(":", "_")
log_filename = log_filename.replace(".", "_") + ".txt"

trained_model = train(start_epoch, epoch, valid_loss_min_input, loaders, Estd_stereo, optimizer, criterion, use_cuda,\
     "./checkpoints/current_checkpoint.pt", "./checkpoints/best_model.pt", log_filename)