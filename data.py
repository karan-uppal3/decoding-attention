import torch
import os.path as osp
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from data_utils import (
    preprocess_data, 
    remove_transition_frames, 
    create_FDM
)

GAZE_DISTANCE_THRESHOLD = 100

class MOT17_Dataset(Dataset):
    """Custom dataset class for the data of a single participant."""

    def __init__(self, participant_id, videos, csv_folder_path, video_folder_path, gaze_img_path, sequence_length, sigma = 500, impute_max_len = 10, transform = None): 
        """Constructor function to initiate the dataset object."""
        self.participant_id = participant_id
        self.videos = videos
        self.csv_folder_path = csv_folder_path
        self.video_folder_path = video_folder_path
        self.gaze_img_path = gaze_img_path
        self.sequence_length = sequence_length
        self.sigma = sigma
        self.impute_max_len = impute_max_len
        self.transform = transform

        print("Loading data for Participant {}".format(participant_id))
        self.prepare_data()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        data_img = []
        data_gaze = []
        data_gt_cls = []
        data_gt_bbx = []

        for i in range(len(self.sequences[idx])):
            tmp = self.get_single_frame(idx, i)
            data_img.append(tmp[0])
            data_gaze.append(tmp[1])
            data_gt_cls.append(tmp[2]['class_id'])
            data_gt_bbx.append(tmp[2]['target_roi'])

        data_gaze_img = torch.tensor(create_FDM(data_gaze, data_img[0].size()[1:], self.sigma), dtype=torch.float32)
        data_img = torch.stack(data_img)
        data_gt_cls = torch.tensor(data_gt_cls)
        data_gt_bbx = torch.tensor(data_gt_bbx, dtype=torch.float32)

        sequence_info = {
            'participant_id': self.participant_id,
            'video_idx': self.sequences[idx][0]['video_idx'],
            'video_frame_start': self.sequences[idx][0]['video_frame'],
            'video_frame_end': self.sequences[idx][-1]['video_frame'],
            'gaze' : data_gaze
        }

        return data_img, data_gaze_img, data_gt_cls, data_gt_bbx, sequence_info

    def get_single_frame(self, idx_sequence, idx_frame):
        """Utility function to get a single frame from the dataset."""
        itm_data = self.sequences[idx_sequence][idx_frame]
        img_path = osp.join(
            self.video_folder_path, 
            '{:02d}'.format(itm_data['video_idx']), 
            '{:05d}.jpg'.format(itm_data['video_frame']+1)
        )
        img = Image.open(img_path)
        img = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        return img.to(torch.float32), itm_data['gaze'], itm_data['target']
        
    def prepare_data(self):
        """Utility function to prepare the data."""
        # Preprocessing
        self.frames = preprocess_data(self.participant_id, self.csv_folder_path, self.impute_max_len)

        # Skipping frames during transition periods
        preprocessed_frames = remove_transition_frames(self.frames)

        # Skipping frames with missing gaze and frames with gaze too far away
        processed_frames = [ f for f in preprocessed_frames if f['gaze_is_missing'] == False and f['gaze_distance'] <= GAZE_DISTANCE_THRESHOLD ]

        # Separating video frames into sequences (sliding window)
        self.sequences = []
        for video_idx in self.videos:
            video_frames = [ frame for frame in processed_frames if video_idx == frame['video_idx'] ]
            video_frames.sort(key = lambda x: x['video_frame'])
            for i in range(0, len(video_frames) - self.sequence_length + 1):
                # Check frames are consecutive
                if video_frames[i]['video_frame'] + self.sequence_length - 1 == video_frames[i+self.sequence_length-1]['video_frame']:
                   self.sequences.append(video_frames[i:i+self.sequence_length])

        print("Loaded {} sequences from {} frames".format(len(self.sequences), len(processed_frames)))

    def return_id(self):
        return self.participant_id

    def participant_mean_proportion(self):
        return np.mean([frame['gaze_is_missing'] for frame in self.frames])


if __name__ == '__main__':

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    tmp_dataset = MOT17_Dataset(
        participant_id = 1,
        videos = list(range(1,15)),
        csv_folder_path = '../Semantic Eye-Tracking/data/experiment1',
        video_folder_path = '../Semantic Eye-Tracking/data/MOT17_video_frames',
        gaze_img_path = '../Semantic Eye-Tracking/data/sequences',
        sequence_length = 5,
        sigma = 500, 
        transform = trans
    )

    tmp_dataloader = DataLoader(
        tmp_dataset, 
        batch_size = 1, 
        shuffle = False, 
        num_workers = 0
    )

    it = iter(tmp_dataloader)
    data = next(it)

    print("\nSize of image batch (Batch Size x Sequence Length x Channels x Height x Width) : ", data[0].size())
    print("Size of Fixation Density Map : ", data[1].size())
    print("Class gt : ", data[2][0])
    print("Bounding box gt : ", data[3][0])
    print("Sequence info : ", data[4])