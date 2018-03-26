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
filename = '139L.jpg'


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
    landmarks = []
    # print("Number of faces detected: {}".format(len(dets)))

    # Get Shape from Face Detected
    for index, det in enumerate(dets):
        # Box Boundary
        left_top = (det.left(), det.top())
        right_bottom = (det.right(), det.bottom())
        rectangle = (det.left(), det.top(), det.right(), det.bottom())

        # Draw a box
        if draw:
            cv2.rectangle(image, left_top, right_bottom, green_color, 3)

        # Shapes
        landmarks = shape_point(gray_img, det)

    # Draw landmarks
    if draw:
        for center in landmarks:
            cv2.circle(image, center, 2, green_color, -1)
        return image, landmarks

    return landmarks, rectangle


def is_contains(rect, point):
    # Check is point contain in rectangle
    if point[0] < 0 or point[1] < 0:
        return False

    if point[0] < rect[0]:
        return False

    elif point[1] < rect[1]:
        return False

    elif point[0] > rect[2]:
        return False

    elif point[1] > rect[3]:
        return False

    return True


def get_delaunay(img, landmarks, face_box):
    # Initialize Subdiv2d
    subdiv = cv2.Subdiv2D(face_box)
    for point in landmarks:
        subdiv.insert(point)

    # Get an Triangle List
    triangle_list = subdiv.getTriangleList()

    # Init list for filter triangle list
    filtered_list = []

    # Filter processing for select only in bound points
    for i, tri in enumerate(triangle_list):
        p0 = (tri[0], tri[1])
        p1 = (tri[2], tri[3])
        p2 = (tri[4], tri[5])

        is_mesh = is_contains(face_box, p0) and is_contains(
            face_box, p1) and is_contains(face_box, p2)

        if is_mesh:
            filtered_list.append(tri)

    return filtered_list


def mapping_to_index(point, landmarks):
    for index, lm in enumerate(landmarks):
        if point == lm:
            return index
    return -1


def indexing_delaunay(landmarks, triangle_list):
    result = []
    for i, tri in enumerate(triangle_list):
        p0 = (tri[0], tri[1])
        p1 = (tri[2], tri[3])
        p2 = (tri[4], tri[5])
        indexing = (
            mapping_to_index(p0, landmarks),
            mapping_to_index(p1, landmarks),
            mapping_to_index(p2, landmarks),
        )
        result.append(indexing)

    return result


def draw_delaunay(img, triangle_index, landmarks):
    for idx_set in triangle_index:
        p0 = landmarks[idx_set[0]]
        p1 = landmarks[idx_set[1]]
        p2 = landmarks[idx_set[2]]
        cv2.line(img, p0, p1, (0, 0, 255), 1)
        cv2.line(img, p1, p2, (0, 0, 255), 1)
        cv2.line(img, p2, p0, (0, 0, 255), 1)
    cv2.imshow('Delauany', img)
    cv2.waitKey(0)


def main():
    img = cv2.imread(str(DATA_DIR/'139L.jpg'))
    w, h = img.shape[0], img.shape[1]
    points, rect = detection(img)
    tri_list = get_delaunay(img, points, rect)
    tri_index = indexing_delaunay(points, tri_list)
    print(tri_index)


if __name__ == '__main__':
    main()
