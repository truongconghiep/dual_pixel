from sysconfig import get_path
from turtle import clear
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from PIL import Image
import os
import torchvision
from torch.utils.data import Dataset, random_split
from os import walk
import matplotlib.pyplot as plt

def check_path(path):
    directory = os.path.dirname(path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

def read_image_filename(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    f.sort()
    # print(f)
    return f


class RandomCropImage():
    def __init__(self, root_dir, path_to_img_list, out_dir, width, height, num_img):
        img_file = open(path_to_img_list, 'r')
        self.img_list = img_file.readlines()
        self.out_dir = out_dir
        self.root_dir = root_dir
        self.width = width
        self.height = height
        self.number_of_crop_img = num_img

    def crop(self):
        for dp_img in range(len(self.img_list)):
            img_id = self.img_list[dp_img][0:-1].split(' ')[0]
            imagel = self.img_list[dp_img][0:-1].split(' ')[1]
            imager = self.img_list[dp_img][0:-1].split(' ')[2]
            dispm = self.img_list[dp_img][0:-1].split(' ')[3]

            left = (Image.open(os.path.join(self.root_dir + imagel)).convert('RGB'))
            right = (Image.open(os.path.join(self.root_dir + imager)).convert('RGB'))
            gt = (Image.open(os.path.join(self.root_dir + dispm)).convert('L'))

            # Random crop
            for k in range(self.number_of_crop_img):
                i, j, h, w = transforms.RandomCrop.get_params(left, output_size=(self.width, self.height))
                image_left = TF.crop(left, i, j, h, w)
                image_right = TF.crop(right, i, j, h, w)
                image_disp = TF.crop(gt, i, j, h, w)

                left_path = self.out_dir + 'left/' + img_id + '_L_' + str(k).zfill(3) + '.jpg'
                check_path(left_path)
                image_left.save(left_path)

                right_path = self.out_dir + 'right/' + img_id + '_R_' + str(k).zfill(3) + '.jpg'
                check_path(right_path)
                image_right.save(right_path)

                gt_path = self.out_dir + 'gt/' + img_id + '_D_' + str(k).zfill(3) + '.jpg'
                check_path(gt_path)
                image_disp.save(gt_path)

class dpd_disp_dataset(Dataset):
    def __init__(self, left_img_path, right_img_path, gt_paths, transform=None, target_transform=None):
        self.left_images = read_image_filename(left_img_path)
        self.right_images = read_image_filename(right_img_path)
        self.gt_images = read_image_filename(gt_paths)
        self.transform = transform
        self.target_transform = target_transform
        self.left_dir = left_img_path
        self.right_dir = right_img_path
        self.gt_dir = gt_paths
        print(self.left_dir)
        
    def getitem(self, index):
        # print(self.left_images[index])
        left = (self.left_dir + '/' + self.left_images[index])
        right = (self.right_dir + '/' + self.right_images[index])
        gt = (self.gt_dir + '/' + self.gt_images[index])

        retval = {'left': left, 'right': right, 'gt': gt}

        return retval
    
    def create_dataset(self):
        dataset = []
        self.full_dataset = len(self.left_images)
        for i in range(self.full_dataset):
            item = self.getitem(i)
            dataset.append(item)

        train_size = int(0.8 * self.full_dataset)
        test_size = self.full_dataset - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        return dataset, train_dataset, test_dataset


if __name__ == "__main__": 
    # crop_img = RandomCropImage("./data/ICCP2020_DP_dataset/", "./data/dpd_disp.txt", "./data/ICCP2020_DP_dataset_new/", \
    #     480, 640, 100)

    # crop_img.crop()


    # f = read_image_filename("./data/ICCP2020_DP_dataset_new/left")
    # print(f)

    dataset = dpd_disp_dataset("./data/ICCP2020_DP_dataset_new/left", \
        "./data/ICCP2020_DP_dataset_new/right", "./data/ICCP2020_DP_dataset_new/gt", transform = transforms.ToTensor())

    dp_dataset, train_set, test_set = dataset.create_dataset()

    image = dp_dataset[0]['gt']

    print(image)
    print("train_set ", len(train_set))
    print("test_set ", len(test_set))


            






