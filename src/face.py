import numpy as np
import cv2
import dlib
import time

from pathlib import Path

# Path Configurations
PROJECT_DIR = Path(Path(__file__).resolve()).parents[1]
RESOURCE_DIR = PROJECT_DIR/'resource'
DATA_DIR = PROJECT_DIR/'data'
predictor_path = str(RESOURCE_DIR/'shape_predictor_68_face_landmarks.dat')
filename = '114L.jpg'


def shape_point(img, detector):
    # Impoer predictor
    predictor = dlib.shape_predictor(predictor_path)

    # Convert to gray scale image
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract shape part to (x, y)
    shape = predictor(img, detector)
    coordinate = []
    for i in np.arange(0, shape.num_parts):
        x, y = shape.part(i).x, shape.part(i).y
        coordinate.append((x, y))

    return coordinate


def detection(image, draw=False):
    # Setting up Detector and Predictor
    detector = dlib.get_frontal_face_detector()
    green_color = (0, 255, 0)

    # Detection Processing
    # Convert to gray scale image
    if len(image.shape) > 2:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # img = cv2.pyrDown(image)
    dets = detector(gray_img, 0)
    shapes = []
    coordinate = []
    # print("Number of faces detected: {}".format(len(dets)))

    # Get Shape from Face Detected
    for index, det in enumerate(dets):
        # Box Boundary
        left_top = (det.left(), det.top())
        right_bottom = (det.right(), det.bottom())

        # Draw a box
        if draw:
            cv2.rectangle(image, left_top, right_bottom, green_color, 3)

        # Shapes
        coordinate = shape_point(gray_img, det)

    # Draw landmarks
    if draw:
        for center in coordinate:
            cv2.circle(image, center, 2, green_color, -1)
        return image, coordinate

    return coordinate

# def main():
    # Video processing
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 600)
    # cap.set(4, 400)

    # vw = int(cap.get(3))
    # vh = int(cap.get(4))

    # while(cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     # Our operations on the frame come here
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     post_image = detection(frame)

    #     # Display the resulting frame
    #     cv2.imshow('frame', post_image)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
