import os
import sys
import random
import torch
from torchvision import transforms, models

from tqdm import tqdm

sys.path.append('./')
from data import MOT17_Dataset

DEVICE = 'cuda'

DISPLAY_RESOLUTION = (1920, 1200)
MODEL_NAME = 'obj_det_baseline'
PARTICIPANT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
VIDEOS = list(range(1, 15))


def inside_bbx(bbx, p):
    x, y = p
    x1, y1, x2, y2 = bbx
    x1, x2 = x1 * DISPLAY_RESOLUTION[0] / 512, x2 * DISPLAY_RESOLUTION[0] / 512
    y1, y2 = y1 * DISPLAY_RESOLUTION[1] / 512, y2 * DISPLAY_RESOLUTION[1] / 512
    return x1 <= x <= x2 and y1 <= y <= y2


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

        'device' : DEVICE
    }

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(params['device'])
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
            bbx_preds = model(
                img.squeeze(0).to(params['device'])
            )[0]['boxes']

        gaze_p = ( int(sequence_info['gaze'][0][0]), int(sequence_info['gaze'][0][1]) )

        bbx_pred = None
        for b in bbx_preds:
            if inside_bbx(b, gaze_p):
                bbx_pred = b
                break

        if bbx_pred is None:
            idx = random.randint(0, len(bbx_preds) - 1)
            bbx_pred = bbx_preds[idx]

        bbx_pred = bbx_pred.cpu().detach().numpy()
        bbx_gt = bbx_gt.squeeze().cpu().detach().numpy()

        bbx_pred[0] = int(bbx_pred[0] * DISPLAY_RESOLUTION[0] / 512) 
        bbx_pred[1] = int(bbx_pred[1] * DISPLAY_RESOLUTION[1] / 512)
        bbx_pred[2] = int(bbx_pred[2] * DISPLAY_RESOLUTION[0] / 512)
        bbx_pred[3] = int(bbx_pred[3] * DISPLAY_RESOLUTION[1] / 512)
        bbx_gt[0] = int(bbx_gt[0] * DISPLAY_RESOLUTION[0]) 
        bbx_gt[1] = int(bbx_gt[1] * DISPLAY_RESOLUTION[1])
        bbx_gt[2] = int(bbx_gt[2] * DISPLAY_RESOLUTION[0])
        bbx_gt[3] = int(bbx_gt[3] * DISPLAY_RESOLUTION[1])

        data[int(sequence_info['participant_id'])][int(sequence_info['video_idx'])][int(sequence_info['video_frame_start'])] = {
            'pred_bbx': bbx_pred,
            'gt_bbx': bbx_gt,
            'gaze_x' : int(sequence_info['gaze'][0][0]),
            'gaze_y' : int(sequence_info['gaze'][0][1]),
        }

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