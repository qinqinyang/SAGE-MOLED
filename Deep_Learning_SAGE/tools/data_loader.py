import os
import random
import numpy as np
import scipy.io
from torch.utils import data
from torchvision import transforms as T

def load_data_Charles(image_path, config):
    #print(image_path)
    data_in = np.fromfile(image_path, dtype=np.float32)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)
    input_sets = data_pairs[:, :, 0:6]
    #input_sets = input_sets / input_sets.max()
    #print(input_sets.max())
    #input_sets = data_pairs[:, :, 2:3]
    label_sets = data_pairs[:, :, 8:9]
    # label_sets[:, :, 0:1] = label_sets[:, :, 0:1] * 10
    # label_sets[:, :, 1:2] = label_sets[:, :, 1:2] * 10
    #label_sets2 = data_pairs[:,:,11:12]

    #label_sets = np.concatenate((label_sets1,label_sets2),axis=2)

    return input_sets,label_sets

def load_data_Charles_test(image_path, config):
    #print(image_path)
    data_in = np.fromfile(image_path, dtype=np.float32)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, 6)
    input_sets = data_pairs[:, :, 0:6]
    #input_sets = input_sets / input_sets.max()
    #input_sets = data_pairs[:, :, 2:3]
    label_sets = data_pairs[:, :, 0:1]

    return input_sets,label_sets

class ImageFolder(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, crop_key, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))
        self.image_paths.sort(reverse=True)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        if self.mode == 'brain':
            image,GT = load_data_Charles_test(image_path, self.config)
            #image,GT = load_data_Mat_test(image_path, self.config)
        else:
            image,GT = load_data_Charles(image_path, self.config)
            #image, GT = load_data_Mat(image_path, self.config)

        if self.crop_key:
            # -----RandomCrop----- #
            (w, h, c) = image.shape
            th, tw = self.crop_size, self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - th)
            if w <= th and h <= th:
                print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                return
            image = image[i:i + th, j:j + th,:]
            #image = image / image.max()
            GT = GT[i:i + th, j:j + th,:]

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, config, crop_key, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, config=config, crop_key=crop_key, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader

def load_data_Charles_fast(image_path, config):
    filen = len(image_path)
    input_list = []
    output_list = []

    for filei in range(0, filen):
        filename = image_path[filei]
        data_in = np.fromfile(filename, dtype=np.float32)
        data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)

        input_sets = data_pairs[:, :, 0:2]
        input_sets = input_sets / input_sets.max()
        label_sets = data_pairs[:, :, 6:7]
        input_list.append(input_sets)
        output_list.append(label_sets)
        print(filename)

    return input_list,output_list

def load_data_Charles_test_fast(image_path, config):
    filen = len(image_path)
    input_list = []
    output_list = []

    for filei in range(0, filen):
        filename = image_path[filei]
        data_in = np.fromfile(filename, dtype=np.float32)
        data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, 6)

        input_sets = data_pairs[:, :, 0:2]
        input_sets = input_sets / input_sets.max()
        label_sets = data_pairs[:, :, 0:1]
        input_list.append(input_sets)
        output_list.append(label_sets)
        print(filename)

    return input_list, output_list

class ImageFolderfast(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, crop_key, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))
        self.image_paths.sort(reverse=True)
        print(self.image_paths)
        if self.mode == 'train':
            self.input_list, self.label_list = load_data_Charles_fast(self.image_paths,config)
        elif self.mode == 'brain':
            self.input_list, self.label_list = load_data_Charles_test_fast(self.image_paths,config)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = self.input_list[index]
        GT = self.label_list[index]

        if self.crop_key:
            # -----RandomCrop----- #
            (w, h, c) = image.shape
            th, tw = self.crop_size, self.crop_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - th)
            if w <= th and h <= th:
                print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                return
            image = image[i:i + th, j:j + th,:]
            #image = image / image.max()
            GT = GT[i:i + th, j:j + th,:]

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader_fast(image_path, config, crop_key, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolderfast(root=image_path, config=config, crop_key=crop_key, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader