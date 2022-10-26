import sys
import math
import numpy as np
import cv2
import os
import collections
import csv
import json


DISPLAY_RESOLUTION = (1920, 1200)
DETECTION_THRESHOLD = 60.0
FRAME_RATE = {
  1: 30.0,
  2: 30.0,
  3: 30.0,
  4: 30.0,
  5: 14.0,
  6: 14.0,
  7: 30.0,
  8: 30.0,
  9: 30.0,
  10: 30.0,
  11: 30.0,
  12: 30.0,
  13: 25.0,
  14: 25.0
}
TRANSITION_DURATION = 300.0 / 1000.0 # 300ms


def preprocess_data(participant_id, csv_folder_path, impute_max_len : int = 10):
  """Utility function to preprocess eyetracking and stimulus data."""
  eyetrack = load_eyetrack(participant_id, csv_folder_path)
  frames = load_stimulus(participant_id, csv_folder_path)
  eyetrack = impute_missing_data_D(eyetrack, max_len = impute_max_len)
  frames = synchronize_eyetracking_with_stimulus(eyetrack, frames)
  return frames


def remove_transition_frames(frames):
  """Utility function to remove frames during transition between object of interest"""
  # Return if all frames are missing gaze data
  if len(frames) == 0:
    return frames
  # Skipping frames during transition of object of interest
  video_idx = frames[0]['video_idx']
  i = int( FRAME_RATE[video_idx] * TRANSITION_DURATION )
  processed_frames = [ frames[i-1 if i > 0 else 0] ]
  prev_target_idx = frames[i-1 if i > 0 else 0]['target']['target_object_index']
  while i < len(frames):
    if frames[i]['video_idx'] == video_idx and frames[i]['target']['target_object_index'] != prev_target_idx:
      prev_target_idx = frames[i]['target']['target_object_index']
      i += int( FRAME_RATE[video_idx] * TRANSITION_DURATION )
    else:
      video_idx = frames[i]['video_idx']
      prev_target_idx = frames[i]['target']['target_object_index']
      processed_frames.append(frames[i])
      i += 1
  return processed_frames


def load_eyetrack(participant_id, csv_folder_path):
  """Utility function to load eyetracking data."""
  fname = os.path.join(csv_folder_path, str(participant_id).zfill(2) + '_eyetracking.csv')
  with open(fname, 'r') as f:
      reader = csv.reader(f, delimiter=',')
      eyetrack = []
      EyetrackRow = collections.namedtuple('EyetrackRow', next(reader))
      for row in map(EyetrackRow._make, reader):
          timestamp = float(row.ComputerClock_Timestamp)
          gaze_x = get_best(float(row.LeftEye_GazeX), float(row.RightEye_GazeX))
          gaze_y = get_best(float(row.LeftEye_GazeY), float(row.RightEye_GazeY))
          diam = get_best(float(row.LeftEye_Diam), float(row.RightEye_Diam))
          eyetrack.append([timestamp, gaze_x, gaze_y, diam])
  print('Loading {} rows of eyetracking data from {}.'.format(len(eyetrack), fname))
  return np.array(eyetrack)


def load_stimulus(participant_id, csv_folder_path):
  """
  Utility function to load stimulus data, which returns a list of dictionaries
  with the following keys:
    'video_idx': int
    'time': float
    'video_frame': int
    'target' : { 'class_id' : int, 'target_roi' : [x, y, x_radius, y_radius] }

  """
  with open("labels.json", "r") as json_file:
    COCO_MAPPING = json.load(json_file)
  fname = os.path.join(csv_folder_path, str(participant_id).zfill(2) + '_stimulus.csv')
  with open(fname, 'r') as f:
      frames = []
      current_video = None
      reader = csv.reader(f, delimiter=',')
      next(reader) # Skip experiment metadata row
      StimulusRow = collections.namedtuple('StimulusRow', next(reader))

      for row in map(StimulusRow._make, reader):
          video_idx = int(row.Video_Index)
          if video_idx != current_video:
              video_frame = 0
              current_video = video_idx
          target_class_name = row.Target_Name.split('_')[0]
          target_object_index = int(row.Target_Name.split('_')[1])
          t = float(row.ComputerClock_Timestamp)
          if t < 1e12:
              # Due to a bug, some stimulus were recorded in seconds rather than ms
              t *= 1000
          # Scaling the target ROI to the resolution of the display
          target_RoI = [ 
            ( int(row.TargetX) - int(row.TargetXRadius) ) / DISPLAY_RESOLUTION[0] , # x1
            ( int(row.TargetY) - int(row.TargetYRadius) ) / DISPLAY_RESOLUTION[1] , # y1
            ( int(row.TargetX) + int(row.TargetXRadius) ) / DISPLAY_RESOLUTION[0] , # x2
            ( int(row.TargetY) + int(row.TargetYRadius) ) / DISPLAY_RESOLUTION[1]   # y2
          ]

          try:  # Only newer recordings include confidence data
            object_detection_threshold = float(row.Object_Detection_Threshold)
          except AttributeError:
            object_detection_threshold = 60.0

          if object_detection_threshold != DETECTION_THRESHOLD:
            continue

          frames.append({
              'video_idx': video_idx,
              'time': t,
              'video_frame': video_frame,
              'target': {
                'class_id': COCO_MAPPING[target_class_name],
                'target_roi': target_RoI,
                'target_object_index': target_object_index
              }
          })
          video_frame += 1
  print('Loading {} rows of stimulus data from {}.'.format(len(frames), fname))
  return frames


def get_best(left: float, right: float):
  """
  For gaze and diameter we get separate left and right eye measurements.
  We recode missing values from 0.0 to NaN. If one eye's data is missing,
  take the other eye's data; else, take the average.
  """
  if left < sys.float_info.epsilon and right < sys.float_info.epsilon:
    return float('nan')
  if left < sys.float_info.epsilon:
    return right
  if right < sys.float_info.epsilon:
    return left
  return (left + right)/2


def impute_missing_data_D(X, max_len = 10):
  """
  Given a sequence X of D-dimensional vectors, performs __impute_missing_data
  (independently) on each dimension of X.

  X is N X D, where D is the dimensionality and N is the sample length
  """
  D = X.shape[1]
  for d in range(D):
    X[:, d] = __impute_missing_data(X[:, d], max_len)
  return X


def __impute_missing_data(X, max_len):
  """
  Given a sequence X of floats, replaces short streches (up to length
  max_len) of NaNs with linear interpolation. For example, if
    X = np.array([1, NaN, NaN,  4, NaN,  6])
  then
    impute_missing_data(X, max_len = 1) == np.array([1, NaN, NaN, 4, 5, 6])
  and
    impute_missing_data(X, max_len = 2) == np.array([1, 2, 3, 4, 5, 6]).
  """
  last_valid_idx = -1
  for n in range(len(X)):
    if not math.isnan(X[n]):
      if last_valid_idx < n - 1: # there is missing data and we have seen at least one valid eyetracking sample
        if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
          if last_valid_idx == -1: # No previous valid data (i.e., first timepoint is missing)
            X[0:n] = X[n] # Just propogate first valid data point
          else:
            first_last = np.array([X[last_valid_idx], X[n]]) # initial and final values from which to linearly interpolate
            new_len = n - last_valid_idx + 1
            X[last_valid_idx:(n + 1)] = np.interp([float(x)/(new_len - 1) for x in range(new_len)], [0, 1], first_last)
      last_valid_idx = n
    elif n == len(X) - 1: # if n is the last index of X and X[n] is NaN
      if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
        X[last_valid_idx:] = X[last_valid_idx]
  return X


def synchronize_eyetracking_with_stimulus(eyetrack, frames):
  """Interpolates eyetracking frames to same timepoints as stimulus frames."""
  eyetrack_idx = 0
  if eyetrack[0, 0] >= frames[0]['time']:
    error = (eyetrack[0, 0] - frames[0]['time'])/1000
    print('EYE-TRACKING STARTS {} SECONDS AFTER STIMULUS.'.format(error))
    eyetrack[0] = np.array([frames[0]['time'] - 1, float('nan'), float('nan'), float('nan')])
  if eyetrack[-1, 0] <= frames[-1]['time']:
    error = (frames[-1]['time'] - eyetrack[-1, 0])/1000
    print('EYE-TRACKING ENDS {} SECONDS BEFORE STIMULUS.'.format(error))
    eyetrack[-1] = np.array([frames[-1]['time'] + 1, float('nan'), float('nan'), float('nan')])
  for frame in frames:
    while eyetrack[eyetrack_idx, 0] < frame['time']:
      eyetrack_idx += 1
    t0, t1 = eyetrack[(eyetrack_idx - 1):(eyetrack_idx + 1), 0]
    # At this point, t0 <= frame['time'] < t1. Linearly interpolate x and y based on
    # the surrounding x0, x1, y0, and y1.
    theta = (frame['time'] - t0)/(t1 - t0)
    gaze_x, gaze_y, diam = ((1 - theta) * eyetrack[eyetrack_idx - 1, 1:]
                          +   theta   * eyetrack[eyetrack_idx, 1:])
    frame['gaze'] = [gaze_x, gaze_y]
    frame['gaze_is_missing'] = math.isnan(gaze_x) or math.isnan(gaze_y)

    bbx = frame['target']['target_roi'][:]
    bbx[0], bbx[2] = bbx[0] * DISPLAY_RESOLUTION[0], bbx[2] * DISPLAY_RESOLUTION[0]
    bbx[1], bbx[3] = bbx[1] * DISPLAY_RESOLUTION[1], bbx[3] * DISPLAY_RESOLUTION[1]
    if bbx[0] <= gaze_x <= bbx[2] and bbx[1] <= gaze_y <= bbx[3]:
      frame['gaze_distance'] = 0.0
    else:
      frame['gaze_distance'] = min(
        abs(gaze_x - bbx[0]) if bbx[1] <= gaze_y <= bbx[3] else float('inf'),
        abs(gaze_x - bbx[2]) if bbx[1] <= gaze_y <= bbx[3] else float('inf'),
        abs(gaze_y - bbx[1]) if bbx[0] <= gaze_x <= bbx[2] else float('inf'),
        abs(gaze_y - bbx[3]) if bbx[0] <= gaze_x <= bbx[2] else float('inf'),
        _euclidean_distance((gaze_x, gaze_y), (bbx[0], bbx[1])),
        _euclidean_distance((gaze_x, gaze_y), (bbx[2], bbx[3])),
        _euclidean_distance((gaze_x, gaze_y), (bbx[0], bbx[3])),
        _euclidean_distance((gaze_x, gaze_y), (bbx[2], bbx[1]))
      )
  return frames

def _euclidean_distance(x, y):
  return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def GaussianMask(sizex, sizey, center = None, sigma = 500):
  """Uses a Gaussian mask to create a 2D array of size sizex x sizey."""
  
  x = np.arange(0, sizex, 1, float)
  y = np.arange(0, sizey, 1, float)
  x, y = np.meshgrid(x,y)
  
  if center is None:
      x0, y0 = sizex // 2,  sizey // 2
  else:
      x0, y0 = center[0], center[1]

  return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def create_FDM(gaze_points, img_size, sigma = 500):
  """Creates a fixation detection map from gaze points."""
  
  if img_size == (512, 512):
    width, height = 128, 128

  # Resolution of the stimulus display
  FDM = np.zeros(DISPLAY_RESOLUTION[::-1], np.float32)
  for n_subject in range(len(gaze_points)):
    FDM += GaussianMask(*DISPLAY_RESOLUTION, gaze_points[n_subject], sigma)
  FDM = FDM / np.max(FDM)

  return cv2.resize(FDM, (width, height))