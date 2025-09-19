# -*- coding: UTF-8 -*-
'''
Created on Wed Oct 9 20:15:00 2019

@author: Qinqin Yang
'''
import os
import argparse
import scipy.io as matio

from network.R2AttUNet_T2 import Inference
#from network.UNet import Inference

from tools.evaluation import *
import numpy as np

from torch.utils import data
from torchvision import transforms as T

def load_data_Charles_test(image_path, config):
    print(image_path)
    data_in = np.fromfile(image_path, dtype=np.float32)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)
    input_sets = data_pairs[:, :, 0:6]
    input_sets = input_sets / input_sets.max()
    label_sets = data_pairs[:, :, :1]

    return input_sets,label_sets


class ImageFolder(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, crop_key):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE
        self.image_dir = root

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in dir :{}".format(len(self.image_paths)))
        self.image_paths.sort(reverse=True)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image, GT = load_data_Charles_test(image_path, self.config)

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(config, crop_key, num_workers, shuffle=True):
    """Builds and returns Dataloader."""

    print(config.test_dir)

    dataset = ImageFolder(root=config.test_dir, config=config, crop_key=crop_key)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader

def test(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    np.random.seed(1)
    torch.manual_seed(1)

    model_dir = os.path.join(config.model_path, config.model_name)
    if not os.path.exists(model_dir):
        print('Model not found, please check you path to model')
        print(model_dir)
        os._exit(0)

    test_batch = get_loader(config, crop_key=False, num_workers=1, shuffle=False)

    net = Inference(config.INPUT_C,config.OUTPUT_C,config.FILTERS)

    if torch.cuda.is_available():
       net.cuda()

    net.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))
    print('Model parameters loaded!')

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # ********************************************test*****************************************************#
    net.eval()
    for i,(images, GT) in enumerate(test_batch):
        images = images.type(torch.FloatTensor)
        GT = GT.type(torch.FloatTensor)

        images = images.to(device)

        SR = net(images)  # forward

        if i == 0:
            OUT_test = SR.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            OUT_test = np.concatenate((SR.permute(0, 2, 3, 1).cpu().detach().numpy(),OUT_test),axis=0)

    print('.' * 30)
    print('OUT_test:', OUT_test.shape)
    print('.' * 30)
    matio.savemat(
        os.path.join(config.result_dir),
        {
            'output': OUT_test
        })
    print('Save result in ',config.result_dir)
    print('.' * 30)
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--GPU_NUM', type=str, default='0')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=256)
    parser.add_argument('--INPUT_W', type=int, default=256)
    parser.add_argument('--INPUT_C', type=int, default=6)
    parser.add_argument('--OUTPUT_C', type=int, default=2)
    parser.add_argument('--LABEL_C', type=int, default=1)
    parser.add_argument('--DATA_C', type=int, default=9)
    parser.add_argument('--FILTERS', type=int, default=64)
    parser.add_argument('--CROP_SIZE', type=int, default=96)
    # test hyper-parameters
    parser.add_argument('--BATCH_SIZE', type=int, default=1)

    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='models.pth')
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--test_dir', type=str, default='')

    config = parser.parse_args()

    #config.model_name = 'OLED_paper_1954_t1t2t2star_t2t2star_norm_R2AttUNet_lownoise.pth'
    config.model_name = 'OLED_paper_1954_t1t2t2star_Mz_norm_R2AttUNet.pth'

    config.test_dir = 'meas_MID00871_FID1901364_a_sage_oled_dyn_Charles/'
    config.result_dir = 'meas_MID00871_FID1901364_a_sage_oled_dyn_t2t2star/'


    test(config)