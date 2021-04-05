import numpy as np
import cv2
from pathlib import Path
import os
import time

path = Path.cwd()
homography_source = path / "homography_src"
homography_dst = path / "homography_dst"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#dimensions of chessboard
chessboard = (6, 8)
cheesboard_origin_respect_to_device = (-255.0, 148.75)   # (x, y) mm
chessboard_square_edge_length = 42.5   # mm

objpoints = None
imgpoints = None
raw_img = None

# prepare object points
objpoints = np.zeros((chessboard[0]*chessboard[1],2), np.float32)
objpoints = (np.mgrid[0:chessboard[0],0:chessboard[1]].T.reshape(-1,2)) * chessboard_square_edge_length
objpoints[:,0] += cheesboard_origin_respect_to_device[0]
objpoints[:,1] -= cheesboard_origin_respect_to_device[1]

h = 0
w = 0
images = [str(p) for p in homography_source.glob("*.jpg")]
for image in images:
    print("loading image:", image)
    img = cv2.imread(image)
    raw_img = img

    if h == 0:
        h, w = img.shape[:2]
    
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grey, chessboard)

    # If found, add object points, image points (after refining them)
    if ret == True:
        imgpoints = cv2.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria).reshape(-1, 2)

for (im_coords, obj_coords) in zip(imgpoints, objpoints):
    print("img: {0} obj: {1}".format(im_coords, obj_coords))


# initial Homography matrix where the chessboard origin is wrt the device
# the next part is creating a matrix that warps and translates the raw image to a birds eye view that
# could be used for visualising to the user or stithing together to get a very large top down view of the entire shed floor
init_H, status = cv2.findHomography(imgpoints, objpoints)

# getting coords to translate the birds eye view image into the full frame without cropping
pixel_coords_of_raw_image_corners = np.array([[[0, 0], [raw_img.shape[1], 0], \
                                         [0, raw_img.shape[0]], [raw_img.shape[1], raw_img.shape[0]]]], np.float32)
print("pixel coords of corners: {0}".format(pixel_coords_of_raw_image_corners))

# coords of those image corners wrt to device. some are negative so image gets cropped
real_world_coords_of_raw_image_corners = cv2.perspectiveTransform(pixel_coords_of_raw_image_corners, init_H, pixel_coords_of_raw_image_corners.shape)[0]
print("real world coords of corners: {0}".format(real_world_coords_of_raw_image_corners))

# working out the translation to get the origin of the raw image into the top left corner so nothing gets cropped
warped_translation = (max(abs(real_world_coords_of_raw_image_corners[0:3:1, 0])), max(abs(real_world_coords_of_raw_image_corners[0:3:1, 1])))
print("translation_x: {0} translation_y: {1}".format(warped_translation[0], warped_translation[1]))

# after warping the entire image will now fit into the resolution image
warped_res_x = int(round(warped_translation[0]) + round(max(abs(real_world_coords_of_raw_image_corners[1][0]), abs(real_world_coords_of_raw_image_corners[3][0]))))
warped_res_y = int(round(warped_translation[1]) + round(max(abs(real_world_coords_of_raw_image_corners[2][1]), abs(real_world_coords_of_raw_image_corners[3][1]))))
print("warped res: ({0}, {1})".format(warped_res_x, warped_res_y))

# combining the intial homography transform with the translation transform
warped_translation_M = np.array([[1, 0, warped_translation[0]], [0, 1, warped_translation[1]], [0, 0, 1]])
print("translation M: {0}".format(warped_translation_M))
warped_H = np.matmul(warped_translation_M, init_H)
print("warped H: {0}".format(warped_H))

# warping raw image to a full non-cropped birds eye view image
im_out = cv2.warpPerspective(raw_img, warped_H, (warped_res_x, warped_res_y))

# saving matricies and relavent images
np.savetxt(path / "results" / "init_H.txt'", init_H, fmt='%.8f', delimiter=',')
np.savetxt(path / "results" / "warped_H.txt'", warped_H, fmt='%.8f', delimiter=',')
cv2.imwrite(os.path.join(homography_dst, "warped and translated.jpg"), im_out)


## pixel to real world (relative to device) test
# pix_x = 1170
# pix_y = 1305
# p = np.array([[[pix_x, pix_y]]], np.float32)
# res = cv2.perspectiveTransform(p, full_frame_H, p.shape)[0]
# print(res)
# test_img = cv2.circle(im_out, (pix_x, pix_y), radius=0, color=(0, 255, 0), thickness=20)
# cv2.imwrite(os.path.join(homography_dst, "test_3.jpg"), test_img)

# real world (relative to device) to pixel test
# real_x = 0
# real_y = 0
# p = np.array([[[real_x, real_y]]], np.float32)
# res = cv2.perspectiveTransform(p, full_frame_H, p.shape)[0]
# print(res)
# test_img = cv2.circle(im_out, (real_x, real_y), radius=0, color=(0, 255, 0), thickness=20)
# cv2.imwrite(os.path.join(homography_dst, "test_3.jpg"), test_img)