import cv2
import torch
import numpy as np
import csv
import time
import os 

from train_utils import compute_iou

VIDEO_DIR = '../Semantic Eye-Tracking/data/MOT17_videos/'
VIDEO_FRAMES_DIR = '../Semantic Eye-Tracking/data/MOT17_video_frames/'
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200


def _rescale_video_to_screen(frame: np.ndarray):
    """Rescale video to fit the screen on which the experiment was displayed."""
    video_height, video_width, _ = frame.shape
    scale = min(SCREEN_HEIGHT/video_height, SCREEN_WIDTH/video_width)
    scaled_width = int(scale * video_width)
    scaled_height = int(scale * video_height)
    frame = cv2.resize(frame, (scaled_width, scaled_height))

    # Pad the remaining screen space with black space
    top_padding = max(0, int((SCREEN_HEIGHT - scaled_height)/2))
    left_padding = max(0, int((SCREEN_WIDTH - scaled_width)/2))
    return cv2.copyMakeBorder(frame, top_padding,
                                SCREEN_HEIGHT - (scaled_height + top_padding),
                                left_padding,
                                SCREEN_WIDTH - (scaled_width + left_padding),
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _plot_object(frame: np.ndarray, obj, color):
    pt_1 = [int(float(obj[0])), int(float(obj[1]))]
    pt_2 = [int(float(obj[2])), int(float(obj[3]))]
    cv2.rectangle(frame, pt_1, pt_2, color = color, thickness = 2)  


def play_experiment_video(model_name, participant_idx, video_idx, csv_name = None, save_name = None):

    video_idx_str = str(video_idx).zfill(2)
    video_fname = VIDEO_DIR + video_idx_str + '.mp4'
    video = cv2.VideoCapture(video_fname)

    # Load csv from predictions folder
    csv_fname = "../output/predictions/" + model_name + "/pred_bbx_participant_" + str(participant_idx) + ".csv" if csv_name is None else csv_name
    csv_file_reader = csv.reader(open(csv_fname), delimiter=',')
    _ = next(csv_file_reader)

    data = []
    for row in csv_file_reader:
        try:
            if int(row[0]) == video_idx:
                data.append(row)
        except:
            pass
    data.sort(key=lambda x: int(x[1]))

    final_data = {
        int(data[i][1]) : data[i][2:] for i in range(len(data))
    }

    num_frames = len(os.listdir(os.path.join(VIDEO_FRAMES_DIR, '{:02d}'.format(video_idx))))

    # Set the inter-frame delay based on the video's natural framerate
    FPS = video.get(cv2.CAP_PROP_FPS) # natural frame rate
    # print('Video framerate: ' + str(FPS))
    delay = 1.0/FPS

    current_frame = 0
    videoStartTime = time.time()
    nextFrameExists, frame = video.read() # Load first video frame

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        model_name + "__participant-" + str(participant_idx) + "_vid-" + str(video_idx) + ".avi" if save_name is None else save_name, 
        fourcc, FPS, (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    avg_iou = 0
    
    while nextFrameExists and current_frame < num_frames:
        if time.time() > videoStartTime + current_frame * delay:
            frame = _rescale_video_to_screen(frame)

            if current_frame in final_data.keys():

                gaze = (int(float(final_data[current_frame][8])), int(float(final_data[current_frame][9])))
                cv2.circle(frame, center=gaze, radius=10, color=(255, 255, 255),
                        thickness = 3)

                # Plot true target and predicted target
                pred = final_data[current_frame][0:4]
                target = final_data[current_frame][4:8]

                pred = torch.Tensor(np.array([[int(float(itm)) for itm in pred]]))
                target = torch.Tensor(np.array([[int(float(itm)) for itm in target]]))

                iou = compute_iou(pred, target)

                avg_iou += iou

                _plot_object(frame, pred.squeeze(), color=(0, 255, 0))
                _plot_object(frame, target.squeeze(), color=(255, 0, 0))

            else:
                # When eye-tracking is missing, plot a red square in the top-left corner
                cv2.rectangle(frame, (0, 0), (20, 20), (0, 0, 255), 20)

            out.write(frame)

            # cv2.imshow('Video Frame', frame) # Display current frame
            cv2.waitKey(1)

            nextFrameExists, frame = video.read() # Load next video frame
            current_frame += 1

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video : ", video_idx, "| Participant :", participant_idx, "| Average Iou: " + str(avg_iou/len(final_data.keys()))) if len(final_data.keys()) != 0 else None


if __name__ == '__main__':

    for video_idx in range(1, 15):
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]:
            play_experiment_video(
                model_name = "vid-1_fasterrcnn_dist_sigma-500",
                participant_idx = idx,
                video_idx = video_idx,
            )