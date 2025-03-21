import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OpenSeg_Adapter(nn.Module):

    def __init__(self):
        super().__init__()
        c_in = 768
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            # nn.ReLU(inplace=True),
            # nn.GELU(),

            nn.Linear(c_in // reduction, c_in, bias=True),
            # nn.ReLU(inplace=True)
        )
        self.text_feature = np.load('/data/xuxiaoxu/code/openvocabulary/ovdet_2d/3DSS-VLG/saved_text_embeddings/scannet_openseg_768.npy')
        self.text_feature = torch.from_numpy(self.text_feature).float().cuda()
        

    
    def forward(self, feat):
        B, N, C = feat.shape
        feat = feat.reshape(B*N, -1)
        alpha = 0.05

        feat = self.fc(feat) * (1 - alpha) + feat * alpha
        predictions = feat.reshape(B, N, -1)
        pred =  predictions @ self.text_feature.t()
        pred = pred.permute(0, 2, 1)

        return pred, feat