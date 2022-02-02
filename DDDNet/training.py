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

print("init data folders - NYU dataset")

test_dataset = GoProDataset_test(
    img_list = args.img_list_t,
    root_dir = args.input_test_file,
    transform = transforms.ToTensor()
    )
test_dataloader = DataLoader(test_dataset, batch_size = args.batchsize, shuffle=False, num_workers=args.workers)

mse = nn.MSELoss().cuda()

Estd_stereo = model_test.YRStereonet_3D()
Estd_stereo = torch.nn.DataParallel(Estd_stereo, device_ids=GPU)
Estd_stereo.cuda()

checkpoint = torch.load(str('./checkpoints/' + METHOD + "/Estd" + ".pkl"))

# print(checkpoint.keys())

Estd_stereo.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Estd" + ".pkl")), strict=False)
print("ini load Estd " + " success")

# print(Estd_stereo)







# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in Estd_stereo.state_dict():
#     print(param_tensor, "\t", Estd_stereo.state_dict()[param_tensor].size())





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
    criterion, use_cuda, checkpoint_path, best_model_path):
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
        model.train()
        for batch_idx, path_to_data in enumerate(loaders['train']):
            # print(batch_idx)
            # print(path_to_data)
            left_img = transform(Image.open(path_to_data['left'])).unsqueeze(0)
            right_img = transform(Image.open(path_to_data['right'])).unsqueeze(0)
            gt = transform(Image.open(path_to_data['gt'])).unsqueeze(0)

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
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################    
        # validate the model #
        ######################
        # model.eval()
        # for batch_idx, (data, target) in enumerate(loaders['test']):
        #     # move to GPU
        #     if use_cuda:
        #         data, target = data.cuda(), target.cuda()
        #     ## update the average validation loss
        #     # forward pass: compute predicted outputs by passing inputs to the model
        #     output = model(data)
        #     # calculate the batch loss
        #     loss = criterion(output, target)
        #     # update average validation loss 
        #     valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # # calculate average losses
        # train_loss = train_loss/len(loaders['train'].dataset)
        # valid_loss = valid_loss/len(loaders['test'].dataset)

        # # print training/validation statistics 
        # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #     epoch, 
        #     train_loss,
        #     valid_loss
        #     ))
        
        # # create checkpoint variable and add important data
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'valid_loss_min': valid_loss,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        
        # # save checkpoint
        # save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        # ## TODO: save the model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        #     # save checkpoint as best model
        #     save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        #     valid_loss_min = valid_loss
            
    # return trained model
    return model



dataset = dpd_disp_dataset("./data/ICCP2020_DP_dataset_new/left", \
        "./data/ICCP2020_DP_dataset_new/right", "./data/ICCP2020_DP_dataset_new/gt", transform = transforms.ToTensor())

dp_dataset, train_set, test_set = dataset.create_dataset()

loaders = {'train':train_set, 'test':test_set}

criterion = nn.MSELoss()
optimizer = optim.Adam(Estd_stereo.parameters(), lr=0.0001)

use_cuda = torch.cuda.is_available()
trained_model = train(1, 3, np.Inf, loaders, Estd_stereo, optimizer, criterion, use_cuda,\
     "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")