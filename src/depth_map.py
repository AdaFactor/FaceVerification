import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(Path(__file__).resolve()).parents[1]/'data'
DATA_DIR = str(DATA_DIR)


def add_z_axis(img, depth):
    cols, rows = img.shape
    data = []
    for col in np.arange(0, cols):
        for row in np.arange(0, rows):
            z = depth[col, row]
            data.append([col, row, z])
    return np.array(data, np.int32)


def show_disperities(image_no, numDisparities=16, blockSize=15):
    image_no = '/{}'.format(image_no)
    path_left = ''.join([DATA_DIR, image_no, 'L', '.jpg'])
    path_right = ''.join([DATA_DIR, image_no, 'R', '.jpg'])
    img_left = cv2.imread(path_left, 0)
    img_right = cv2.imread(path_right, 0)

    # Pre-Pro cessing
    dwn_left = cv2.pyrDown(img_left)
    dwn_right = cv2.pyrDown(img_right)

    stereo = cv2.StereoSGBM_create(
        numDisparities=numDisparities,
        blockSize=blockSize
    )
    disparity = stereo.compute(dwn_left, dwn_right).astype(np.uint8)
    disparity = cv2.pyrUp(disparity)
    # print(disparity)
    coor = add_z_axis(img_left, disparity)
    x = coor[:, 0]
    y = coor[:, 1]
    z = coor[:, 2]
    print(x*y*z)

    plt.figure(1, figsize=(50, 50))

    plt.subplot(131)
    plt.title('Original Left image')
    plt.imshow(img_left, cmap='gray')

    plt.subplot(132)
    plt.title('numDisp={}, blockSize={}'.format(numDisparities, blockSize))
    plt.imshow(disparity, cmap='gray')

    plt.subplot(133)
    plt.title('Original Right image')
    plt.imshow(img_right, cmap='gray')

    plt.show()


def main():
    image_no = sys.argv[1]
    numDisparities = 16*int(sys.argv[2])
    blockSize = int(sys.argv[3])
    # for block in np.arange(15, 30, 2):
    show_disperities(image_no, numDisparities, blockSize=blockSize)


if __name__ == '__main__':
    main()
