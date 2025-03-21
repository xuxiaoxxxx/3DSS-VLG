
import os
import sys
import torch
import torch.nn as nn
import numpy as np

import datetime
import logging
import importlib
import argparse

from pathlib import Path
from tqdm import tqdm
from dataset.OpensegDataloader_eval import OpensegDataset
from util.misc import SmoothedValue
from util import metric
import config
import time
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='Openseg evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/pointnet_pl.yaml',
                    help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def main():
    def log_string(str):
        logger.info(str)
        print(str)
    
    args = get_parser()
    '''HYPER PARAMETER'''
    # cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('openseg_pretrain') # 第一个地方
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = get_parser()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    train_data_path = args.train_data_path
    train_openseg_embedding_path = args.train_openseg_embedding_path
    model_path = args.model_path
    save_base = args.save_base

    train_dataset = OpensegDataset(path=train_data_path, openseg_data_path=train_openseg_embedding_path, split='val')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = importlib.import_module(args.model)
    classifier = model.OpenSeg_Adapter()
    criterion_pesudo = nn.CrossEntropyLoss(ignore_index=20)
    criterion_pesudo = criterion_pesudo.cuda()
    classifier = classifier.cuda()

    os.makedirs(save_base, exist_ok=True)
    try:
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')

    global_epoch = 0
    '''TRANING'''
    logger.info('Start evaling...')

    with torch.no_grad():
        classifier = classifier.eval()
        log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
        predictions = []
        gt = []
        for i, (feats, gt_label, mask_chunk, scene_name) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            feats, gt_label = feats.cuda(), gt_label.long().cuda()
            pred_logit, prediction = classifier(feats)
            pred_logit = pred_logit.permute(0, 2, 1)
            
            torch.save({"feat": prediction.half().cpu(),
                "mask_full": mask_chunk[0]
            },  os.path.join(save_base, scene_name[0][:-4] +'.pt'))

            pred = torch.argmax(pred_logit, axis=2)
            
            predictions.append(pred[0])
            gt.append(gt_label[0])
        predictions = torch.cat(predictions)
        gt = torch.cat(gt)

        current_iou, current_acc = metric.evaluate(predictions.cpu().numpy(),
                                            gt.cpu().numpy(),
                                            dataset='scannet_3d',
                                            stdout=True,
                                            logger=logger)
        

        log_string("Mean IoU = {:.4f}".format(current_iou))
        log_string("Mean Acc = {:.4f}".format(current_acc))
        log_string(current_iou)

if __name__ == '__main__':
    main()