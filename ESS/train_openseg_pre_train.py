
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
from dataset.OpensegDataloader import OpensegDataset
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

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
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
    # train_data_path = '/data/xuxiaoxu/code/openvocabulary/ovdet_2d/openscene/data/scannet_3d/train'
    # val_data_path = '/data/xuxiaoxu/code/openvocabulary/ovdet_2d/openscene/data/scannet_3d/val'

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    pesudo_label_path = args.pesudo_label_path

    train_openseg_embedding_path = args.train_openseg_embedding_path
    val_openseg_embedding_path = args.val_openseg_embedding_path

    # train_dataset = OpensegDataset(path=train_data_path, npoints=40000, split='train')
    # test_dataset = OpensegDataset(path=val_data_path, npoints=40000, split='val')

    train_dataset = OpensegDataset(path=train_data_path, npoints=args.num_points, openseg_data_path=train_openseg_embedding_path, pesudo_label_path=pesudo_label_path, split='train')
    test_dataset = OpensegDataset(path=val_data_path, npoints=args.num_points, openseg_data_path=val_openseg_embedding_path, pesudo_label_path=pesudo_label_path, split='val')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = importlib.import_module(args.model)
    classifier = model.OpenSeg_Adapter()
    criterion_pesudo = nn.CrossEntropyLoss(ignore_index=20)
    criterion_pesudo = criterion_pesudo.cuda()
    classifier = classifier.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_iou_acc = 0.0
    loss_sum = 0
    num_batches = len(trainDataLoader)
    global_epoch = 0
    '''TRANING'''
    logger.info('Start training...')
    iter_count = 0
    train_max_iters = args.epoch * len(trainDataLoader)
    train_max_epoch = args.epoch

    for epoch in range(start_epoch, args.epoch):
        loss_sum = 0
        classifier = classifier.train()
        scheduler.step()
        time_delta = SmoothedValue(window_size=10)
        loss_avg = SmoothedValue(window_size=10)

        for batch_id, (feats, gt_labels, pesudo_labels) in enumerate(trainDataLoader):
            optimizer.zero_grad()
            curr_time = time.time()

            feats, pesudo_labels = feats.cuda(), pesudo_labels.long().cuda()
            
            pred, _ = classifier(feats)
            loss = criterion_pesudo(pred, pesudo_labels)
        
            loss.backward()
            optimizer.step()
            loss_avg.update(loss.item())
            time_delta.update(time.time() - curr_time)

            if batch_id % 2 == 0:
                curr_iter = epoch * len(trainDataLoader) + batch_id
                eta_seconds = (train_max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_string(f"Epoch [{epoch}/{train_max_epoch}]; Iter [{curr_iter}/{train_max_iters}]; Loss {loss_avg.avg:0.2f}; Iter time {time_delta.avg:0.2f}; ETA {eta_str};")
        
        with torch.no_grad():
            classifier = classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            predictions = []
            gt = []
            for i, (feats, gt_label) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
                feats, gt_label = feats.cuda(), gt_label.long().cuda()

                pred_logit, _ = classifier(feats)
                pred_logit = pred_logit.permute(0, 2, 1)
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

            if current_iou > best_iou_acc:
                best_iou_acc = current_iou
                best_acc = current_acc
                best_epoch = epoch
                logger.info('Save model...')
                
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s ' % savepath)
                state = {
                    'epoch': epoch,
                    'miou': current_iou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            log_string("Best IoU = {:.4f}, Best Acc = {:.4f}, Best Epoch = {}".format(best_iou_acc, best_acc, best_epoch))
            global_epoch += 1

if __name__ == '__main__':
    main()