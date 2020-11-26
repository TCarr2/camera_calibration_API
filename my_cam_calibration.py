import sys
import os
sys.path.append("../../")
from camera_calibration import Camera_Calibration_API
import glob
import cv2
import numpy as np

images_path_list = glob.glob("C:/Users/Terry/Documents/GitHub/camera_calibration_API/images/*.jpg")
path = "C:/Users/Terry/Documents/GitHub/camera_calibration_API"
print(len(images_path_list))

chessboard_dimensions = (6, 8)  #num rows, num columns
square_edge_length = 19.94  # mm
refine = True
refine_threshold = 0.3

chessboard = Camera_Calibration_API(pattern_type="chessboard", pattern_rows=chessboard_dimensions[1], pattern_columns=chessboard_dimensions[0], distance_in_world_units=square_edge_length, debug_dir=os.path.join(path, "debug"))
results = chessboard.calibrate_camera(images_path_list)

np.savetxt(os.path.join(path, 'results', 'cam_mtx.txt'), results['intrinsic_matrix'], fmt='%.5f', delimiter=',')
np.savetxt(os.path.join(path, 'results', 'cam_dist.txt'), results['distortion_coefficients'], fmt='%.5f', delimiter=',')

h, w = cv2.imread(images_path_list[0], 0).shape[:2]
newcammtx, roi = cv2.getOptimalNewCameraMatrix(results['intrinsic_matrix'], results['distortion_coefficients'], (w,h), 1, (w,h))
mapx, mapy = cv2.initUndistortRectifyMap(results['intrinsic_matrix'], results['distortion_coefficients'], None, newcammtx, (w,h), 5)
print("New Camera Matrix: \n")
print(newcammtx)

np.savetxt(os.path.join(path, 'results', 'cam_mtx_new.txt'), newcammtx, fmt='%.5f', delimiter=',')
np.savetxt(os.path.join(path, 'results', 'roi.txt'), roi, fmt='%i', delimiter=',')

for image in images_path_list:
    img = cv2.imread(image)

    #remapping
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    try:
        cv2.imwrite(image + "_1remap.jpg", dst)
    except:
        pass

    #crop
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    try:
        cv2.imwrite(image + "_2crop.jpg", dst)
    except:
        pass

    break

if refine:
    refined_images_paths = [img_path for i,img_path in enumerate(chessboard.calibration_df.image_names) if chessboard.calibration_df.reprojection_error[i] < refine_threshold]
    print(len(refined_images_paths))
    refined_chessboard = Camera_Calibration_API(pattern_type="chessboard", pattern_rows=chessboard_dimensions[1], pattern_columns=chessboard_dimensions[0], distance_in_world_units=square_edge_length, debug_dir=os.path.join(path, "debug_refine"))
    refined_results = refined_chessboard.calibrate_camera(refined_images_paths)

    np.savetxt(os.path.join(path, 'results', 'cam_mtx_refined.txt'), refined_results['intrinsic_matrix'], fmt='%.5f', delimiter=',')
    np.savetxt(os.path.join(path, 'results', 'cam_dist_refined.txt'), refined_results['distortion_coefficients'], fmt='%.5f', delimiter=',')

    h, w = cv2.imread(images_path_list[0], 0).shape[:2]
    refined_newcammtx, refined_roi = cv2.getOptimalNewCameraMatrix(refined_results['intrinsic_matrix'], refined_results['distortion_coefficients'], (w,h), 1, (w,h))
    refined_mapx, refined_mapy = cv2.initUndistortRectifyMap(refined_results['intrinsic_matrix'], refined_results['distortion_coefficients'], None, refined_newcammtx, (w,h), 5)
    print("New Refined Camera Matrix: \n")
    print(refined_newcammtx)

    np.savetxt(os.path.join(path, 'results', 'cam_mtx_new_refined.txt'), refined_newcammtx, fmt='%.5f', delimiter=',')
    np.savetxt(os.path.join(path, 'results', 'roi_refined.txt'), refined_roi, fmt='%i', delimiter=',')

    for image in images_path_list:
        img = cv2.imread(image)

        #remapping
        dst = cv2.remap(img, refined_mapx, refined_mapy, cv2.INTER_LINEAR)
        try:
            cv2.imwrite(image + "_1remap_1refined.jpg", dst)
        except:
            pass

        #crop
        x, y, w, h = refined_roi
        dst = dst[y:y+h, x:x+w]
        try:
            cv2.imwrite(image + "_2crop_2refined.jpg", dst)
        except:
            pass

        break