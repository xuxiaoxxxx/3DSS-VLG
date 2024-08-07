'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet as model3D
from torch import nn
import MinkowskiEngine as ME
import numpy as np
import torch

def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    return model


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, text_feature, cfg=None):
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        net3d = constructor3d(in_channels=3, out_channels=last_dim, D=3, arch=cfg.arch_3d)
        self.net3d = net3d

        c_in = 768
        reduction = 4
        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.Linear(c_in // reduction, c_in, bias=True),
        )
        self.text_feature = text_feature

    def forward(self, sparse_3d, sparse_2d=None):
        '''Forward method.'''

        if sparse_2d == None:
            return self.net3d(sparse_3d)
        else:
            adapter_feat_2d = self.adapter(sparse_2d)
            adapter_feat_2d = (adapter_feat_2d/(adapter_feat_2d.norm(dim=-1, keepdim=True)+1e-5))
            adapter_pred =  adapter_feat_2d @ self.text_feature.t()

            return self.net3d(sparse_3d), adapter_feat_2d,  adapter_pred
