import os
import random
import numpy as np
import logging
import argparse
import urllib

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo
import torch.nn.functional as F

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm

from dataset.label_constants import *


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    mapper = None

    args.prompt_eng = False
    text_features = extract_text_feature(labelset, args)
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette

def main():
    '''Main function.'''

    args = get_parser()
    cudnn.benchmark = True
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))
    main_worker(args)


def main_worker(argss):
    global args
    args = argss

    global logger
    logger = get_logger()
    logger.info(args)


    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False


    from vlm_ps_generation.dataset.generation_featute_loader import FusedFeatureLoader,collation_fn_eval_all
    print("args.data_root:", args.data_root, args.data_root_2d_fused_feature)
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset
    labelset_name = 'scannet'
    print("process the dataset name is:", labelset_name)
    evaluate(val_loader, labelset_name)

def evaluate(val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    
    mark_no_feature_to_unknown = True
    text_features, labelset, mapper, palette = \
        precompute_text_related_properties(labelset_name)

    rep_i = 0
    preds, gts = [], []
    val_data_loader.dataset.offset = rep_i
    # if main_process():
    logger.info( "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

    # repeat the evaluation process
    # to account for the randomness in MinkowskiNet voxelization
    if rep_i>0:
        seed = np.random.randint(10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if mark_no_feature_to_unknown:
        masks = []

    for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
        coords = coords[inds_reverse, :]

        predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
        pred = predictions.half() @ text_features.t()
        
        # # add the filter
        pred = F.softmax(pred, dim=1)

        # using the scene-level labels as mask
        cloud_label = torch.unique(label)
        if cloud_label[-1] == 255:
            cloud_label = cloud_label[:-1]
        mask_cloud = torch.zeros(pred.shape[1]).cuda()
        mask_cloud[cloud_label] = 1
        pred[:, mask_cloud == 0] = -999

        logits_pred = torch.max(pred, 1)[1].detach().cpu()
        max_logits_pred = torch.max(pred, 1)[0].detach().cpu()
        
        if mark_no_feature_to_unknown:
        # some points do not have 2D features from 2D feature fusion.
        # Directly assign 'unknown' label to those points during inference.
            logits_pred[~mask[inds_reverse]] = len(labelset)-1


        scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
        scene_name = scene_name[:-11]

        ### save the predition label
        logits_save = logits_pred.cpu().numpy()
        path_save_pred = os.path.join(save_folder, scene_name + '_pred.npy')
        np.save(path_save_pred, logits_save)

        max_logits_save = max_logits_pred.cpu().numpy()
        path_save_max_pred = os.path.join(save_folder, scene_name + '_prob_pred.npy')
        np.save(path_save_max_pred, max_logits_save)

        # print(a)
        if eval_iou:
            if mark_no_feature_to_unknown:
                masks.append(mask[inds_reverse])

            if args.test_repeats==1:
                # save directly the logits
                preds.append(logits_pred)
            else:
                # only save the dot-product results, for repeat prediction
                preds.append(pred.cpu())

            gts.append(label.cpu())
    
    print("Begin to calculate the IoU.....")
    if eval_iou:
        gt = torch.cat(gts)
        pred = torch.cat(preds)

        pred_logit = pred
        if args.test_repeats>1:
            pred_logit = pred.float().max(1)[1]

        if mapper is not None:
            pred_logit = mapper[pred_logit]

        if mark_no_feature_to_unknown:
            mask = torch.cat(masks)
            pred_logit[~mask] = 256

        if args.test_repeats==1:
            current_iou = metric.evaluate(pred_logit.numpy(),
                                        gt.numpy(),
                                        dataset=labelset_name,
                                        stdout=True)
            print("The IoU is: ", current_iou)
                

if __name__ == '__main__':
    main()
