import numpy as np
import cv2
import dlib
import time
import json

from pathlib import Path
from face import detection

PROJECT_DIR = Path(Path(__file__).resolve()).parents[1]
DATA_DIR = PROJECT_DIR/'data'


def write_landmarks(
        leftFilename='leftFilename',
        rightFilename='rightFilename',
        landmarks=[]
):
    dict_data = {
        'leftFilename': leftFilename,
        'rightFilename': rightFilename,
        'landmarks': landmarks
    }
    json_xy = json.dumps(dict_data)
    print(json_xy)


def main():
    leftFilename = '139L.jpg'
    rightFilename = '139R.jpg'
    filepath = str(DATA_DIR/leftFilename)
    image = cv2.imread(filepath)
    post_image, landmarks = detection(image, draw=True)
    write_landmarks(leftFilename=leftFilename,
                    rightFilename=rightFilename,
                    landmarks=landmarks
                    )
    cv2.imshow('Face detection', post_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
