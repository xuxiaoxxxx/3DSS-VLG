'''Dataloader for fused point features.'''

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA
from dataset.voxelizer import Voxelizer

# from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(torch.utils.data.Dataset):
    '''Dataloader for fused point features.'''

    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):

        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.data_paths = sorted(glob(join(datapath_prefix, split, '*.pth')))
        self.datapath_feat = datapath_prefix_feat

        dataset_name = datapath_prefix.split('/')[-1]
        self.dataset_name = dataset_name

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        # Precompute the occurances for each scene
        self.list_occur = []
        for data_path in self.data_paths:
            if 'scan' in self.dataset_name:
                scene_name = data_path[:-15].split('/')[-1]
            else:
                scene_name = data_path[:-4].split('/')[-1]
            # import pdb; pdb.set_trace()
            # file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
            file_dirs = glob(join(self.datapath_feat, scene_name + '.pt'))
            self.list_occur.append(len(file_dirs))
        
        # import pdb; pdb.set_trace()
        ind = np.where(np.array(self.list_occur) != 0)[0]
        if np.any(np.array(self.list_occur)==0):
            data_paths, list_occur = [], []
            for i in ind:
                data_paths.append(self.data_paths[i])
                list_occur.append(self.list_occur[i])
            self.data_paths = data_paths
            self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        if np.isscalar(feats_in) and feats_in == 0:
            # no color in the input point cloud, e.g nuscenes lidar
            feats_in = np.zeros_like(locs_in)
        else:
            feats_in = (feats_in + 1.) * 127.5

        # load 3D features
        if self.dataset_name == 'scannet_3d':
            scene_name = self.data_paths[index][:-15].split('/')[-1]
        else:
            scene_name = self.data_paths[index][:-4].split('/')[-1]

        # no repeated file
        # processed_data = torch.load(join(self.datapath_feat, scene_name+'_0.pt'))
        processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))

        feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
        # import pdb; pdb.set_trace()
        if isinstance(mask_chunk, np.ndarray): # if the mask itself is a numpy array
            mask_chunk = torch.from_numpy(mask_chunk)
        mask = copy.deepcopy(mask_chunk)
        # if self.split != 'train': # val or test set
        feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
        feat_3d_new[mask] = feat_3d
        feat_3d = feat_3d_new
        mask_chunk = torch.ones_like(mask_chunk) # every point needs to be evaluted

        if len(feat_3d.shape)>2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        # calculate the corresponding point features after voxelization
        
        locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
            locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
        vox_ind = torch.from_numpy(vox_ind)
        feat_3d = feat_3d[vox_ind]
        mask = mask[vox_ind]

        labels = labels_in

        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        return coords, feats, labels, feat_3d, mask, torch.from_numpy(inds_reconstruct).long()

    def __len__(self):
        return len(self.data_paths)


def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    '''
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    '''
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_recons)