import os
import torch
from torchvision import transforms

from tqdm import tqdm

from data import MOT17_Dataset
from model import Gaze_Attention_Model

DEVICE = 'cuda'

DISPLAY_RESOLUTION = (1920, 1200)
MODEL_NAME = 'vid-1_fasterrcnn_dist_sigma-500'
MODEL_PATH = '../checkpoints/model.pth'
PARTICIPANT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
VIDEOS = [1]


def main():

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    params = {

        'dataset' : {
            'csv_folder_path': '../Semantic Eye-Tracking/data/experiment1',
            'video_folder_path': '../Semantic Eye-Tracking/data/MOT17_video_frames',
            'gaze_img_path' : '../Semantic Eye-Tracking/data/sequences',
            'sequence_length': 1,
            'sigma': 500,
            'transform': trans
        },

        'dataloader' : {
            'batch_size' : 1,
            'shuffle' : False,
            'num_workers' : 2,
            'pin_memory' : False
        },

        'model' : {
            'num_classes' : 20,
            'hidden_layer_dim' : [512, 256],
            'layer_dim_rnn' : 1,
            'use_cls_head' : False
        },

        'device' : DEVICE
    }

    model = Gaze_Attention_Model(**params['model']).to(params['device'])
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    datasets = []
    for participant_id in PARTICIPANT_IDS:
        dataset = MOT17_Dataset(participant_id=participant_id, videos=VIDEOS, **params['dataset'])
        datasets.append(dataset)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), **params['dataloader'])

    # Create empty lists to store the predictions
    data = {
        k:{ v:{} for v in VIDEOS } for k in PARTICIPANT_IDS   
    }

    for batch in tqdm(data_loader):
        img, gaze, cls_gt, bbx_gt, sequence_info = batch

        model.eval()
        with torch.no_grad():
            bbx_pred = model(
                img.squeeze(0).to(params['device']),
                gaze.squeeze(0).to(params['device'])
            )

        bbx_pred = bbx_pred.cpu().detach().numpy()
        bbx_gt = bbx_gt.squeeze(0).cpu().detach().numpy()

        for i in range(bbx_pred.shape[0]):
            bbx_pred[i][0] = int(bbx_pred[i][0] * DISPLAY_RESOLUTION[0]) 
            bbx_pred[i][1] = int(bbx_pred[i][1] * DISPLAY_RESOLUTION[1])
            bbx_pred[i][2] = int(bbx_pred[i][2] * DISPLAY_RESOLUTION[0])
            bbx_pred[i][3] = int(bbx_pred[i][3] * DISPLAY_RESOLUTION[1])
            bbx_gt[i][0] = int(bbx_gt[i][0] * DISPLAY_RESOLUTION[0]) 
            bbx_gt[i][1] = int(bbx_gt[i][1] * DISPLAY_RESOLUTION[1])
            bbx_gt[i][2] = int(bbx_gt[i][2] * DISPLAY_RESOLUTION[0])
            bbx_gt[i][3] = int(bbx_gt[i][3] * DISPLAY_RESOLUTION[1])

        for i in range(bbx_pred.shape[0]):

            if int(sequence_info['video_frame_start'])+i in data[int(sequence_info['participant_id'])][int(sequence_info['video_idx'])].keys():
                    data[int(sequence_info['participant_id'])][int(sequence_info['video_idx'])][int(sequence_info['video_frame_start'])+i]['pred_bbx'] += bbx_pred[i]
                    data[int(sequence_info['participant_id'])][int(sequence_info['video_idx'])][int(sequence_info['video_frame_start'])+i]['ctr'] += 1

            else:
                data[int(sequence_info['participant_id'])][int(sequence_info['video_idx'])][int(sequence_info['video_frame_start'])+i] = {
                    'pred_bbx': bbx_pred[i],
                    'gt_bbx': bbx_gt[i],
                    'gaze_x' : int(sequence_info['gaze'][i][0]),
                    'gaze_y' : int(sequence_info['gaze'][i][1]),
                    'ctr' : 1
                }

    # Calculate the average
    for participant_id in PARTICIPANT_IDS:
        for video_idx in VIDEOS:
            for frame_idx in data[participant_id][video_idx].keys():
                data[participant_id][video_idx][frame_idx]['pred_bbx'] /= data[participant_id][video_idx][frame_idx]['ctr']

    # Save the predictions in a CSV file
    path = os.path.join("../output/predictions/", MODEL_NAME)
    if not os.path.exists(path):
        os.makedirs(path)
    for participant_id in PARTICIPANT_IDS:
        with open(os.path.join(path, "pred_bbx_participant_" + str(participant_id) + ".csv"), "a") as f:
            f.write("video_idx,frame_idx,pred_x1,pred_y1,pred_x2,pred_y2,gt_x1,gt_y1,gt_x2,gt_y2,gaze_x,gaze_y\n")
            for video_idx in VIDEOS:
                for frame_idx in data[participant_id][video_idx].keys():
                    f.write(
                        str(video_idx)+","+
                        str(frame_idx)+","+
                        str(data[participant_id][video_idx][frame_idx]['pred_bbx'][0])+","+
                        str(data[participant_id][video_idx][frame_idx]['pred_bbx'][1])+","+
                        str(data[participant_id][video_idx][frame_idx]['pred_bbx'][2])+","+
                        str(data[participant_id][video_idx][frame_idx]['pred_bbx'][3])+","+
                        str(data[participant_id][video_idx][frame_idx]['gt_bbx'][0])+","+
                        str(data[participant_id][video_idx][frame_idx]['gt_bbx'][1])+","+
                        str(data[participant_id][video_idx][frame_idx]['gt_bbx'][2])+","+
                        str(data[participant_id][video_idx][frame_idx]['gt_bbx'][3])+","+
                        str(data[participant_id][video_idx][frame_idx]['gaze_x'])+","+
                        str(data[participant_id][video_idx][frame_idx]['gaze_y'])+"\n"
                    )

if __name__ == '__main__':
    main()