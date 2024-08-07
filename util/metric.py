'''IoU'''
import numpy as np
from dataset.label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives

    
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom

def evaluate_200(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'scan_200' in dataset:
        CLASS_LABELS = SCANNET_LABELS_200
    else:
        raise NotImplementedError

    HEAD_CATS_SCANNET_200 = ['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture', 'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet', 'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair', 'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity', 'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel', 'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe', 'clothes dryer', 'radiator', 'shelf']
    COMMON_CATS_SCANNET_200 = ["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"]
    TAIL_CATS_SCANNET_200 = ["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", "fire alarm", "power strip", "calendar", "poster", "luggage"]

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0
    head_mean_iou = 0
    head_mean_acc = 0
    commom_mean_iou = 0
    commom_mean_acc = 0
    tail_mean_iou = 0
    tail_mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        # print("process label:", label_name)
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

        # sperate the label into head, commom, tail split
        if label_name in HEAD_CATS_SCANNET_200:
            head_mean_iou += class_ious[label_name][0]
            head_mean_acc += class_accs[label_name]
        elif label_name in COMMON_CATS_SCANNET_200:
            commom_mean_iou += class_ious[label_name][0]
            commom_mean_acc += class_accs[label_name]
        elif label_name in TAIL_CATS_SCANNET_200:
            tail_mean_iou += class_ious[label_name][0]
            tail_mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES

    head_mean_iou /= 66
    head_mean_acc /= 66
    commom_mean_iou /= 68
    commom_mean_acc /= 68
    tail_mean_iou /= 66
    tail_mean_acc /= 66

    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
        print('Head Mean IoU', head_mean_iou)
        print('Head Mean Acc', head_mean_acc)
        print('Commom Mean IoU', commom_mean_iou)
        print('Commom Mean Acc', commom_mean_acc)
        print('Tail Mean IoU', tail_mean_iou)
        print('Tail Mean Acc', tail_mean_acc)
    return mean_iou


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'scan_200' in dataset:
        CLASS_LABELS = SCANNET_LABELS_200
    elif 's3dis' in dataset:
        CLASS_LABELS = S3DIS_DETAILS
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou
