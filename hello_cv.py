# simple opencv program to display an image

import cv2

# load the image

image = cv2.imread("datasets/kitti_dataset/extracted/odometry/dataset/sequences/00/image_2/000000.png")

# display the image

cv2.imshow("image", image)
cv2.waitKey(0)
