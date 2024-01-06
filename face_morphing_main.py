# face_morphing_main.py

import numpy as np
import cv2
from matplotlib import pyplot as plt
from face_morphing_functions import load_and_show_image, resize_image, get_points, get_triangles, affine_transform

# Load the first image
img1 = load_and_show_image("1.png")

# Access the webcam
def capture_and_save_image():
    camera = cv2.VideoCapture(0)

    # Capture an image from the webcam
    return_value, image = camera.read()

    # Save the captured image as '2.png'
    cv2.imwrite('2.png', image)

    # Release the webcam
    del(camera)

capture_and_save_image()

# Load the second image
img2 = load_and_show_image("2.png")

# Resize images to a common size
common_size = (200, 200)
img1 = resize_image(img1, common_size)
img2 = resize_image(img2, common_size)

# Print face landmarks in the first image
points1 = get_points(img1)

# Print face landmarks in the second image
points2 = get_points(img2)

# Define alpha as the merge percentage of the two images
alpha = 0.5

# Calculate the average coordinates of points in two images
points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

# Define an all-zero matrix img_morphed to store the merged image
img1 = np.float32(img1)
img2 = np.float32(img2)
img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

# Get triangles for the first image
triangles1 = get_triangles(points1)

# Get triangles for the second image
triangles2 = get_triangles(points2)

# Get triangles for the merged image
triangles = get_triangles(points)

# Apply affine transform to triangles and merge images
for i in triangles:
    x = i[0]
    y = i[1]
    z = i[2]

    tri1 = [points1[x], points1[y], points1[z]]
    tri2 = [points2[x], points2[y], points2[z]]
    tri = [points[x], points[y], points[z]]

    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)

    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] * (1 - mask) + img_rect * mask

img_morphed = np.uint8(img_morphed)

# Display the merged image
plt.title('Merged Image')
show_img_morphed = cv2.cvtColor(img_morphed, cv2.COLOR_BGR2RGB)
plt.imshow(show_img_morphed)
plt.show()

# Save the merged image
cv2.imwrite("merged_image.png", img_morphed)

# Print a message indicating that the image has been saved
print("Merged image saved as 'merged_image.png' in the same directory.")

# Compare the result with the original images
print("Compare the result:")
plt.subplot(1, 3, 1)
plt.title('Original image')
img1_original = cv2.imread("1.png")
show_img = cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB)
plt.imshow(show_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Similar image')
img2_original = cv2.imread("2.png")
show_img = cv2.cvtColor(img2_original, cv2.COLOR_BGR2RGB)
plt.imshow(show_img)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Merged Image')
show_img = cv2.cvtColor(img_morphed, cv2.COLOR_BGR2RGB)
plt.imshow(show_img)
plt.axis('off')
plt.show()

