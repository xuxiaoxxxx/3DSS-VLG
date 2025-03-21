import pickle
import os
import sys
import numpy as np
import torch

from torch.utils.data import Dataset


class OpensegDataset(Dataset):
    def __init__(self, path, npoints, openseg_data_path, pesudo_label_path, split='train'):
        super().__init__()
        self.data_root = path
        self.split = split
        self.npoints = npoints

        if split == 'train':
            self.clouds_path = np.loadtxt(os.path.join('/data/xuxiaoxu/dataset/scannet', 'scannetv2_train.txt'), dtype=np.str_)
        else:
            self.clouds_path = np.loadtxt(os.path.join('/data/xuxiaoxu/dataset/scannet', 'scannetv2_val.txt'), dtype=np.str_)
        
        self.openseg_data_path = openseg_data_path
        self.train_path = path
        self.pesudo_label_path = pesudo_label_path

        self.openseg_data_files = np.sort([os.path.join(self.openseg_data_path, f + '.pth') for f in self.clouds_path])
        self.pesudo_files = np.sort([os.path.join(self.pesudo_label_path, f + '_pred.npy') for f in self.clouds_path])
        self.data_files = np.sort([os.path.join(self.data_root, f + '_vh_clean_2.pth') for f in self.clouds_path])

    def __getitem__(self, index):
        path_data = self.openseg_data_files[index]

        scene_name = path_data.split('/')[-1]

        path_data = os.path.join(self.openseg_data_path, scene_name.split('.')[0] + '_0.pt')
        processed_data = torch.load(path_data)
        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
    
        _, _, gt_labels = torch.load(self.data_files[index])
        gt_labels = gt_labels[mask_chunk]
        if self.split == 'train':
            path_pesudo = self.pesudo_files[index]
            pesudo_labels = np.load(path_pesudo)
            pesudo_labels = pesudo_labels[mask_chunk]
        
        num_points = feat_3d.shape[0]

        if num_points < self.npoints:
            choice = np.random.choice(num_points, self.npoints)   
        else:
            choice = np.random.choice(num_points, self.npoints, replace=False)

        feat_3d = feat_3d[choice, :].float()
        gt_labels = gt_labels[choice]
        gt_labels = torch.from_numpy(gt_labels.astype(np.int32))

        if self.split == 'train':
            pesudo_labels = pesudo_labels[choice]
            pesudo_labels = torch.from_numpy(pesudo_labels.astype(np.int32))
            return feat_3d, gt_labels, pesudo_labels
        return feat_3d, gt_labels

    def __len__(self):
        return len(self.data_files)