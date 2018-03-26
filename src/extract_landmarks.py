import numpy as np
import cv2
import dlib
import time
import json

from pathlib import Path
from face import detection, get_delaunay, draw_delaunay, indexing_delaunay

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
    left_xy, left_box = detection(left_img)
    right_xy, right_box = detection(right_img)

    # Construction Object Data
    landmarks_info = {
        'filename': filename,
        'left_img_name': left_img_name,
        'right_img_name': right_img_name,
        'left_landmarks': left_xy,
        'right_landmarks': right_xy,
        'left_img': left_img,
        'right_img': right_img,
        'left_box': left_box,
        'right_box': right_box,
    }

    return landmarks_info


def landmarks_to_json(landmark_info):
    file_path = str(DATA_DIR/landmark_info['filename'])+'.json'
    with open(file_path, 'w') as outfile:
        json.dump(landmark_info, outfile, indent=4)


def main():
    # Create Landmarks
    LM = landmarks_LR('139')

    # Get delauany model point
    triangle_list = get_delaunay(
        LM['left_img'],
        LM['left_landmarks'],
        LM['left_box']
    )

    # Add 'triangle_index' key to LM
    LM['triangle_index'] = indexing_delaunay(
        LM['left_landmarks'],
        triangle_list
    )

    # Drawing Triangle
    draw_delaunay(LM['left_img'], LM['triangle_index'], LM['left_landmarks'])
    draw_delaunay(LM['right_img'], LM['triangle_index'], LM['right_landmarks'])

    # Delete Images from LM dictionary
    del LM['left_img']
    del LM['right_img']

    # Write information to json file
    landmarks_to_json(LM)


if __name__ == '__main__':
    main()
