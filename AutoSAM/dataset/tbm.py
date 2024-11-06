import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
import cv2
import sys
import random
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.tfs import get_tbm_transform
def cv2_loader(path, is_mask):
    if is_mask:
        img = cv2.imread(path, 0) 
        img[img > 0] = 1
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.array(img)


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader,
                 sam_trans=None, augmentation_factor=1):
        assert os.path.isdir(root), f'not a valid root: {root}'
        self.root = root
        self.imgs_root = os.path.join(self.root, 'images')
        self.masks_root = os.path.join(self.root, 'masks')

        self.paths = os.listdir(self.imgs_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.augmentation_factor = augmentation_factor
        self.sam_trans = sam_trans
        print('num of data:{}'.format(len(self.paths)))

    def __getitem__(self, index):
        index = index % len(self.paths)
        file_path = self.paths[index]
        mask_path = file_path.split('.')[0] + '.png'
        img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False)
        mask = self.loader(os.path.join(self.masks_root, mask_path), is_mask=True)
        
        img, mask = self.transform(img, mask)
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(img), self.sam_trans.apply_image_torch(mask)


        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        image_size = tuple(img.shape[1:3])

        if not self.train:
            return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
                original_size), torch.Tensor(image_size), file_path
        else:
            return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
                original_size), torch.Tensor(image_size)

    def __len__(self):
        return len(self.paths) * self.augmentation_factor


def get_tbm_dataset(args, sam_trans):
    transform_train, transform_test = get_tbm_transform()
    ds_train = ImageLoader(args['train_data_root'], train=True, transform=transform_train, sam_trans=sam_trans, augmentation_factor=2)
    ds_test = ImageLoader(args['test_data_root'], train=False, transform=transform_test, sam_trans=sam_trans, augmentation_factor=1)
    print(f"Number of train images: {len(ds_train)}")
    print(f"Number of test images: {len(ds_test)}")
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import os
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Idim', '--Idim', default=256, help='Image dimension', required=False)
    parser.add_argument('-pSize', '--pSize', default=4, help='Patch size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='Scale factor 1', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='Scale factor 2', required=False)
    parser.add_argument('-rotate', '--rotate', default=20, help='Rotation factor', required=False)
    args = vars(parser.parse_args())

    sam_args = {
        'sam_checkpoint': args['sam_checkpoint'],
        'model_type': args['model_type'],
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)
    ds_train, ds_test = get_tbm_dataset(args, sam_trans)
    ds = torch.utils.data.DataLoader(ds_train,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)
    
