import numpy as np
import cv2
import dlib
import time
import json

from pathlib import Path
from face import detection

PROJECT_DIR = Path(Path(__file__).resolve()).parents[1]
DATA_DIR = PROJECT_DIR/'data'


def landmarks_LR(filename, extension='jpg'):
    # Left Image Setup
    left_img_name = '{}{}.{}'.format(filename, 'L', extension)
    left_img_path = '{}/{}'.format(str(DATA_DIR), left_img_name)

    # Right Image Setup
    right_img_name = '{}{}.{}'.format(filename, 'R', extension)
    right_img_path = '{}/{}'.format(str(DATA_DIR), right_img_name)

    # Reading images
    left_img = cv2.imread(left_img_path, 0)
    right_img = cv2.imread(right_img_path, 0)

    # Detection Processing
    left_xy = detection(left_img)
    right_xy = detection(right_img)

    # Construction Object Data
    landmarks_info = {
        'filename': filename,
        'left_img_name': left_img_name,
        'right_img_name': right_img_name,
        'left_landmarks': left_xy,
        'right_landmarks': right_xy
    }

    return landmarks_info


def landmarks_to_json(landmark_info):
    file_path = str(DATA_DIR/landmark_info['filename'])+'.json'
    with open(file_path, 'w') as outfile:
        json.dump(landmark_info, outfile, indent=4)


def main():
    LM = landmarks_LR('139')
    landmarks_to_json(LM)
    # cv2.imshow('Face detection', post_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
