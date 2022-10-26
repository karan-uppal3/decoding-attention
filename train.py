import torch
from torchvision import transforms

import wandb
import os
from tqdm import tqdm

from model import Gaze_Attention_Model
from train_utils import (
    set_seed, 
    initialise_wandb,
    get_dataloaders,
    model_log,
    compute_iou,
    evaluate,
    load_ckp
)

PARTICIPANT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]


def main():

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    params = {

        # Put False to split across videos
        'split_across_participants': True,

        'dataset' : {
            'csv_folder_path': '../Semantic Eye-Tracking/data/experiment1',
            'video_folder_path': '../Semantic Eye-Tracking/data/MOT17_video_frames',
            'gaze_img_path' : '../Semantic Eye-Tracking/data/sequences',
            'sequence_length': 5,
            'sigma': 500,
            'transform': trans
        },

        'dataloader' : {
            'batch_size' : 1,
            'shuffle' : True,
            'num_workers' : 4,
            'pin_memory' : True
        },

        'model' : {
            'num_classes' : 20,
            'hidden_layer_dim' : [512, 256],
            'layer_dim_rnn' : 1,
            'use_cls_head' : False
        },

        'loss' : {
            'bbx_loss' : 'L1 Loss',
            'cls_loss' : 'Cross Entropy Loss',
            'optimizer' : 'Adam',
            'lr' : 0.001,
            'weight_decay' : 0.0001
        },

        'logging' : {
            'ckp_dir' : '../checkpoints/exp58',
            'ckp_save_interval' : 20000,
            'ckp_restore_path' : None,          # None if no restore
            'wandb_id' : None,                  # None if no restore
            'wandb_name' : 'vid-1_fasterrcnn_dist_sigma-500'
        },

        'seed' : 0,
        'max_epochs' : 30,

        'max_missing_proportion' : 1.0,
        'validation_split' : 0.2,
        'evaluation_interval' : 20000,

        'restore_checkpoint' : False,
        'pretrained_weights' : None,           # None if no initialistion of weights

        'participant_ids' : PARTICIPANT_IDS,
        'videos' : [1],

        'device' : 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    set_seed(params['seed'])

    print("\nInitialising wandb\n")
    initialise_wandb(params)

    print("\nLoading data\n")
    train_dataloader, val_dataloader = get_dataloaders(params)
        
    # Loading model, loss and optimizer
    model = Gaze_Attention_Model(**params['model']).to(params['device'])
    bbx_loss = torch.nn.L1Loss().to(params['device'])
    if params['model']['use_cls_head'] == True:
        cls_loss = torch.nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['loss']['lr'], weight_decay=params['loss']['weight_decay'])

    start_epoch = 0
    start_itr = 0
    val_loss_min = float('inf')

    if params['restore_checkpoint'] == True:
        print("\nLoading checkpoint from {}\n".format(params['logging']['ckp_restore_path']))
        model, optimizer, start_epoch, start_itr, val_loss_min = load_ckp(params['logging']['ckp_restore_path'], model, optimizer)

    model.train()

    # Tracking model gradients
    wandb.watch(model, log_freq=100)

    print("\nStarting training\n")

    itr = start_epoch * len(train_dataloader)

    # Training loop
    for epoch in range(start_epoch, params['max_epochs']):

        print("Epoch: ", epoch)

        for train_batch in tqdm(train_dataloader):

            if itr < start_itr:
                itr += 1
                continue

            model.train()
            optimizer.zero_grad()
            
            img, gaze, gt_cls, gt_bbx, _ = train_batch

            # Forward pass
            if params['model']['use_cls_head'] == True:
                bbx, cls = model(img.squeeze(0).to(params['device']), gaze.squeeze(0).to(params['device']))
                bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0).to(params['device']))
                cls_loss_val = cls_loss(cls, gt_cls.squeeze(0).to(params['device']))
                loss = bbx_loss_val + cls_loss_val

            else:     
                bbx = model(img.squeeze(0).to(params['device']), gaze.squeeze(0).to(params['device']))   
                bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0).to(params['device']))
                loss = bbx_loss_val
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            itr += 1

            # Logging
            checkpoint = {
                'model_state' : {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'itr': itr,
                    'val_loss_min': val_loss_min
                },
                'wandb_log' : {
                    'bbx_loss_val': bbx_loss_val.item(),
                    'cls_loss_val': cls_loss_val.item() if params['model']['use_cls_head'] == True else None,
                    'iou' : compute_iou(bbx.cpu().detach(), gt_bbx.cpu().detach().squeeze(0)),
                    'loss': loss.item()
                }
            }
            model_log(checkpoint, params, len(train_dataloader))

            if itr % 100 == 0 :
                print("\nIteration: ", itr)
                print("Predicted Bounding Box: \n", bbx)
                print("Actual Bounding Box: \n", gt_bbx.squeeze())
                print("IoU: ", checkpoint['wandb_log']['iou'])
                print("\n")

            if itr % params['evaluation_interval'] == 0 :
                print("Evaluating on validation set")
                val_loss, val_iou = evaluate(
                    val_dataloader, model, params, bbx_loss, 
                    cls_loss if params['model']['use_cls_head'] == True else None
                )
                print("Validation Loss = {}".format(val_loss))
                wandb.log({
                    "val_loss": val_loss,
                    "val_iou": val_iou
                }, step = itr)

                # Saving checkpoint if validation loss is minimum
                if val_loss < val_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        val_loss_min, val_loss))
                    val_loss_min = val_loss
                    checkpoint['model_state']['val_loss_min'] = val_loss
                    ckp_path = os.path.join(
                        params['logging']['ckp_dir'], 'ckp_epoch_{}_itr_{}_val_loss_{:.6f}.pth'.format(
                            epoch, itr, val_loss
                    ))
                    torch.save(checkpoint['model_state'], ckp_path)


if __name__ == "__main__":
    main()