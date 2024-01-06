# face_morphing_functions.py

import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt

def load_and_show_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load the image from {image_path}.")
    else:
        show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(show_img)
        plt.show()
    return img

def resize_image(image, common_size):
    return cv2.resize(image, common_size)

def get_points(image):
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detected_face = face_detector(image, 1)[0]
    pose_landmarks = face_pose_predictor(image, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])

    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.extend([[0, 0], [x // 2, 0], [x, 0], [x, y // 2], [x, y], [x // 2, y], [0, y], [0, y // 2]])

    return np.array(points)

def get_triangles(points):
    return Delaunay(points).simplices

def affine_transform(input_image, input_triangle, output_triangle, size):
    warp_matrix = cv2.getAffineTransform(np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image


