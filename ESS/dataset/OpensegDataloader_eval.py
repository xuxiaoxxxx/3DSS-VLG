import pickle
import os
import sys
import numpy as np
import torch

from torch.utils.data import Dataset


class OpensegDataset(Dataset):
    def __init__(self, path, openseg_data_path, split='train'):
        super().__init__()
        self.data_root = path
        self.split = split

        self.clouds_path = np.loadtxt(os.path.join('/data/xuxiaoxu/dataset/scannet', 'scannetv2_train.txt'), dtype=np.str_)

        self.openseg_data_path = openseg_data_path
        self.train_path = path

        self.openseg_data_files = np.sort([os.path.join(self.openseg_data_path, f + '.pth') for f in self.clouds_path])
        self.data_files = np.sort([os.path.join(self.data_root, f + '_vh_clean_2.pth') for f in self.clouds_path])

    def __getitem__(self, index):
        path_data = self.openseg_data_files[index]

        scene_name = path_data.split('/')[-1]

        path_data = os.path.join(self.openseg_data_path, scene_name.split('.')[0] + '_0.pt')
        processed_data = torch.load(path_data)
        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
    
        _, _, gt_labels = torch.load(self.data_files[index])
        gt_labels = gt_labels[mask_chunk]

            
        return torch.tensor(feat_3d).float(), torch.tensor(gt_labels.astype(np.int32)), mask_chunk, scene_name

    def __len__(self):
        return len(self.data_files)