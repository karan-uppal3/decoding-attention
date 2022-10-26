import os
import sys
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.append('./')
from data import MOT17_Dataset

DEVICE = 'cuda'

DISPLAY_RESOLUTION = (1920, 1200)
MODEL_NAME = 'fixed_box_baseline'
PARTICIPANT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
VIDEOS = list(range(1, 15))

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

    datasets = []
    for participant_id in PARTICIPANT_IDS:
        dataset = MOT17_Dataset(participant_id=participant_id, videos=VIDEOS, **params['dataset'])
        datasets.append(dataset)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), **params['dataloader'])

    # Create empty lists to store the predictions
    data = {
        k:{ v:{} for v in VIDEOS } for k in PARTICIPANT_IDS   
    }

    avg_width, avg_height = 0, 0
    for batch in tqdm(data_loader):
        _, bbx_gt, _ = batch
        bbx_gt = bbx_gt.squeeze().cpu().detach().numpy()
        avg_width += bbx_gt[2] - bbx_gt[0]
        avg_height += bbx_gt[3] - bbx_gt[1]
    avg_width = avg_width * DISPLAY_RESOLUTION[0] / len(data_loader)
    avg_height = avg_height * DISPLAY_RESOLUTION[1] / len(data_loader)

    for batch in tqdm(data_loader):
        _, bbx_gt, sequence_info = batch

        bbx_gt = bbx_gt.squeeze().cpu().detach().numpy()
        bbx_gt[0], bbx_gt[2] = bbx_gt[0] * DISPLAY_RESOLUTION[0], bbx_gt[2] * DISPLAY_RESOLUTION[0]
        bbx_gt[1], bbx_gt[3] = bbx_gt[1] * DISPLAY_RESOLUTION[1], bbx_gt[3] * DISPLAY_RESOLUTION[1]
        bbx_pred = [
            int( int(sequence_info['gaze'][0][0]) - avg_width / 2 ),
            int( int(sequence_info['gaze'][0][1]) - avg_height / 2 ),
            int( int(sequence_info['gaze'][0][0]) + avg_width / 2 ),
            int( int(sequence_info['gaze'][0][1]) + avg_height / 2 )
        ]

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