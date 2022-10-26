import os
import random
import torch
import wandb
import numpy as np

from torchvision.ops import box_iou
from tqdm import tqdm
from data import MOT17_Dataset


def initialise_wandb(params):
    """Initialises wandb logging"""
    if params['restore_checkpoint'] == True:
        wandb.init(project="semantic-eye-tracking", resume="allow", id=params['logging']['wandb_id'])
    else:
        wandb.init(project="semantic-eye-tracking", name=params['logging']['wandb_name'], config=params)


def set_seed(seed):
    """Set seed for consistent experiments."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataloaders(params):
    """Get dataloaders for training and validation set by splitting across participants/videos"""

    # Splitting the dataset across participants
    if params['split_across_participants'] == True:
        datasets = []
        for i in params['participant_ids']:
            dataset = MOT17_Dataset(participant_id=i, videos=params['videos'], **params['dataset'])
            if dataset.participant_mean_proportion() < params['max_missing_proportion']:
                datasets.append(dataset)
        # Splitting the dataset across participants
        split_idx = len(datasets) - int(params['validation_split'] * len(datasets))
        random.shuffle(datasets)
        # Creating datalaoders for training and validation set
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets[:split_idx]), **params['dataloader'])
        val_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets[split_idx:]), **params['dataloader'])
        return train_dataloader, val_dataloader

    # Splitting the dataset across videos
    else:
        split_idx = len(params['videos']) - int(params['validation_split'] * len(params['videos']))
        random.shuffle(params['videos'])
        train_datasets = []
        val_datasets = []
        for i in params['participant_ids']:
            train_data = MOT17_Dataset(participant_id=i, videos=params['videos'][:split_idx], **params['dataset'])
            if train_data.participant_mean_proportion() < params['max_missing_proportion']:
                train_datasets.append(train_data)
            val_data = MOT17_Dataset(participant_id=i, videos=params['videos'][split_idx:], **params['dataset'])
            if val_data.participant_mean_proportion() < params['max_missing_proportion']:
                val_datasets.append(val_data)
        # Creating datalaoders for training and validation set
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_datasets), **params['dataloader'])
        val_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_datasets), **params['dataloader'])
        return train_dataloader, val_dataloader


def compute_iou(box1, box2):
    """
    Computes the intersection over union of two boxes.

    box1 : shape = (batch, 4) : (x1, y1, x2, y2)
    box2 : shape = (batch, 4) : (x1, y1, x2, y2)
    """
    return torch.mean(box_iou(box1, box2))


def model_log(checkpoint, params, num_iters):
    """Saves model and logs given metrics"""
    # Saving model at specified interval as well as at the end of every epoch
    if checkpoint['model_state']['itr'] % params['logging']['ckp_save_interval']  == 0 or checkpoint['model_state']['itr'] % num_iters == 0 :
        ckp_path = os.path.join(
            params['logging']['ckp_dir'], 'model_ckp_epoch_{}_itr_{}.pth'.format(
                checkpoint['model_state']['epoch'], checkpoint['model_state']['itr']
        ))
        torch.save(checkpoint['model_state'], ckp_path)

    wandb.log(
        checkpoint['wandb_log'],
        step = checkpoint['model_state']['itr']
    )


def evaluate(dataloader, model, params, bbx_loss, cls_loss = None):
    """Evaluates model on given dataloader returning loss"""
    model.eval()
    avg_loss, avg_iou = 0.0, 0.0
    with torch.no_grad():
        for val_batch in tqdm(dataloader):
            img, gaze, gt_cls, gt_bbx, _ = val_batch
            if params['model']['use_cls_head'] == True:
                bbx, cls = model(img.squeeze(0).to(params['device']), gaze.squeeze(0).to(params['device']))
                bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0).to(params['device']))
                cls_loss_val = cls_loss(cls, gt_cls.squeeze(0).to(params['device']))
                loss = bbx_loss_val + cls_loss_val
            else:
                bbx = model(img.squeeze(0).to(params['device']), gaze.squeeze(0).to(params['device']))
                bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0).to(params['device']))
                loss = bbx_loss_val
            avg_loss += loss.item()
            avg_iou += compute_iou(bbx.detach(), gt_bbx.squeeze(0).detach().to(params['device']))
    avg_loss /= len(dataloader)
    avg_iou /= len(dataloader)
    return avg_loss, avg_iou


def load_ckp(checkpoint_fpath, model, optimizer):
    """Loads model and optimizer state from given checkpoint"""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['itr'], checkpoint['val_loss_min']