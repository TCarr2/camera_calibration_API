from pathlib import Path
import cv2
import numpy as np
import time

# directory names
mydir = Path.cwd()
intrinsic_source_image_dir = 'intrinsic_src_images'
intrinsic_destination_image_dir = 'intrinsic_dst_images'
extrinsic_source_image_dir = 'extrinsic_src_images'
extrinsic_destination_image_dir = 'extrinsic_dst_images'
params_dir = 'params'


# create path objects to directories
intrinsic_source_image_dir_path = mydir / intrinsic_source_image_dir
intrinsic_destination_image_dir_path = mydir / intrinsic_destination_image_dir
extrinsic_source_image_dir_path = mydir / extrinsic_source_image_dir
extrinsic_destination_image_dir_path = mydir / extrinsic_destination_image_dir
params_dir_path = mydir / params_dir

# checking directories exist
intrinsic_source_image_dir_path.mkdir(parents=True, exist_ok=True)
intrinsic_destination_image_dir_path.mkdir(parents=True, exist_ok=True)
params_dir_path.mkdir(parents=True, exist_ok=True)

# loading camera params from file
cam_dist = np.loadtxt(params_dir_path / 'cam_dist.txt', dtype=float, delimiter=',')
print(type(cam_dist))
print(cam_dist)
cam_mtx = np.loadtxt(params_dir_path / 'cam_mtx.txt', dtype=float, delimiter=',')
cam_mtx_new = np.loadtxt(params_dir_path / 'cam_mtx_new.txt', dtype=float, delimiter=',')
x, y, w, h = np.loadtxt(params_dir_path / 'roi.txt', dtype=int, unpack=True)
rvecs = np.loadtxt(params_dir_path / 'rvecs.txt', dtype=float)
tvecs = np.loadtxt(params_dir_path / 'tvecs.txt', dtype=float)

# create undistort map (all of this is for images of size (4056, 3040))
mapx, mapy = cv2.initUndistortRectifyMap(cam_mtx, cam_dist, None, cam_mtx_new, (4056,3040), 5)


time.sleep(9999999)

# loop through images in source directory, udistort, then place new image into destination directory
for image in intrinsic_source_image_dir_path.glob('*.jpg'):
    try:
        img = cv2.imread(str(image))
        print('found: ', image)
    except:
        continue

    #remapping
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    #crop
    dst = dst[y:y+h, x:x+w]

    try:
        cv2.imwrite(str(mydir / intrinsic_destination_image_dir_path / str(image.stem + '_crop.jpg')), dst)
    except:
        pass




# chessboard_dimensions = (6, 8)  #num rows, num columns
# square_edge_length = 42.50  # mm

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# chess_x, chess_y = np.meshgrid(range(chessboard_dimensions[0]),range(chessboard_dimensions[1]))
# prod = chessboard_dimensions[0] * chessboard_dimensions[1]
# objp = np.hstack((chess_x.reshape(prod,1), chess_y.reshape(prod,1),np.zeros((prod,1)))).astype(np.float32)
# objp = objp * square_edge_length

# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.


# for image in extrinsic_source_image_dir_path.glob('*.jpg'):
#     try:
#         img = cv2.imread(str(image), 0)
#         print('found: ', image)
#     except:
#         continue

#     ret, corners = cv2.findChessboardCorners(img, chessboard_dimensions, None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)

# R, T = cv2.solvePnP(objpoints, imgpoints, cam_mtx, cam_dist)

# print(R)
# print('\n')
# print(T)
# print('n')

# print(objpoints)
# print('\n')
# print(imgpoints)