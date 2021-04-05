import numpy as np
import cv2
from pathlib import Path
import os

path = Path.cwd()
homography_source = path / "homography_src"
homography_dst = path / "homography_dst"

#dimensions of circle grid
circle_grid = (6, 8)
circle_grid_origin_respect_to_device = (1500, 1000)   # top left point of grid from device origin (x, y) mm
circle_grid_c2c_col_distance = 240   # mm
circle_grid_c2c_row_distance = 290    # mm

objpoints = None
imgpoints = None
raw_img = None

# prepare object points
objpoints = (np.mgrid[0:circle_grid[0],0:circle_grid[1]].T.reshape(-1,2))
objpoints[:,0] = objpoints[:,0] * circle_grid_c2c_col_distance + circle_grid_origin_respect_to_device[0]
objpoints[:,1] = objpoints[:,1] * circle_grid_c2c_row_distance * -1 + circle_grid_origin_respect_to_device[1]   # -1 because y axis positive is up and blobs are detected right to left up to down

# circle detection params
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 1000
params.maxArea = 5000
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.55
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

images = [str(p) for p in homography_source.glob("*.jpg")]
for image in images:
    print("loading image:", image)
    raw_img = cv2.imread(image)

    #convert image to greyscale
    grey = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    # convert image to hsv
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

    # thresholding values
    r_b_mask = cv2.inRange(hsv, (100, 40, 0), (179, 255, 210))
    y_g_mask = cv2.inRange(hsv, (0, 62, 42), (91, 255, 255))
    mask_combined = r_b_mask + y_g_mask
 
    #morphilogical operations
    # opening
    kernel = np.ones((3,3), np.uint8)
    mask_morphed = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    # filling holes
    mask_floodfill = mask_morphed.copy()
    h, w = mask_morphed.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, mask, (0,0), 255)
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
    mask_morphed |= mask_floodfill_inv
    mask_morphed = cv2.bitwise_not(mask_morphed)

    # test and show mask and detected blobs
    cv2.imshow("img", mask_morphed)
    cv2.waitKey(0)
    keypoints = detector.detect(mask_morphed)
    # Draw detected blobs as red circles.
    im_with_keypoints = cv2.drawKeypoints(grey, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("img", im_with_keypoints)
    cv2.waitKey(0)

    # Find the circle grid centres
    ret, corners = cv2.findCirclesGrid(mask_morphed, circle_grid, flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, blobDetector=detector)

    # If found, add object points, image points
    if ret:
        imgpoints = np.flipud(corners)

        # draw circle centres
        circle_img = raw_img.copy()
        cv2.drawChessboardCorners(circle_img, circle_grid, corners, ret)
        cv2.imshow("img", circle_img)
        cv2.waitKey(0)

        # print out pixel coords and real world coords
        for (im_coords, obj_coords) in zip(imgpoints, objpoints):
            print("img: {0} obj: {1}".format(im_coords, obj_coords))


        # initial Homography matrix where the  circle grid origin is wrt the device
        # the next part is creating a matrix that warps and translates the raw image to a birds eye view that
        # could be used for visualising to the user or stitching together to get a high resolution top down view of the entire shed floor
        init_H, status = cv2.findHomography(imgpoints, objpoints)

        # getting coords to translate the birds eye view image into the full frame without cropping
        pixel_coords_of_raw_image_corners = np.array([[[0, 0], [raw_img.shape[1], 0], \
                                                [0, raw_img.shape[0]], [raw_img.shape[1], raw_img.shape[0]]]], np.float32)
        print("pixel coords of corners: {0}".format(pixel_coords_of_raw_image_corners))

        # coords of those image corners wrt to device. some are negative so image gets cropped
        real_world_coords_of_raw_image_corners = cv2.perspectiveTransform(pixel_coords_of_raw_image_corners, init_H, pixel_coords_of_raw_image_corners.shape)[0]
        print("real world coords of corners: {0}".format(real_world_coords_of_raw_image_corners))

        # working out the translation to get the origin of the raw image into the top left corner so nothing gets cropped
        warped_translation = (max(abs(real_world_coords_of_raw_image_corners[0][0]), abs(real_world_coords_of_raw_image_corners[2][0])), \
                                                max(abs(real_world_coords_of_raw_image_corners[2][1]), abs(real_world_coords_of_raw_image_corners[3][1])))
        print("translation_x: {0} translation_y: {1}".format(warped_translation[0], warped_translation[1]))

        # after warping the entire image will now fit into the resolution image
        warped_res_x = int(warped_translation[0] + max(abs(real_world_coords_of_raw_image_corners[1][0]), abs(real_world_coords_of_raw_image_corners[3][0])))
        warped_res_y = int(warped_translation[1] + max(abs(real_world_coords_of_raw_image_corners[0][1]), abs(real_world_coords_of_raw_image_corners[1][1])))
        print("warped res: ({0}, {1})".format(warped_res_x, warped_res_y))

        # combining the intial homography transform with the translation transform
        warped_translation_M = np.array([[1, 0, warped_translation[0]], [0, 1, warped_translation[1]], [0, 0, 1]])
        print("translation M: {0}".format(warped_translation_M))
        warped_H = np.matmul(warped_translation_M, init_H)
        print("warped H: {0}".format(warped_H))

        # warping raw image to a full non-cropped birds eye view image and flipping to correct orientation
        im_out = cv2.flip(cv2.warpPerspective(raw_img, warped_H, (warped_res_x, warped_res_y)), 0)

        #resize image
        scale_percent = 100 # percent of original size
        width = int(im_out.shape[1] * scale_percent / 100)
        height = int(im_out.shape[0] * scale_percent / 100)
        im_out = cv2.resize(im_out, (width, height), interpolation = cv2.INTER_AREA)

        # saving matricies and relavent images
        np.savetxt(path / "results" / "init_H.txt", init_H, fmt='%.8f', delimiter=',')
        np.savetxt(path / "results" / "warped_H.txt", warped_H, fmt='%.8f', delimiter=',')
        cv2.imwrite(os.path.join(homography_dst, "warped and translated.jpg"), im_out)


        ## pixel to real world (relative to device) test
        pix_x = 1431
        pix_y = 432
        p = np.array([[[pix_x, pix_y]]], np.float32)
        real_coords = cv2.perspectiveTransform(p, init_H, p.shape)[0][0]
        print("pix_x: {0} pix_y: {1} real_x:{2} real_y:{3}".format(pix_x, pix_y, int(real_coords[0]), int(real_coords[1]), 1))
        #test_img = cv2.circle(im_out, (pix_x, pix_y), radius=0, color=(0, 255, 0), thickness=20)
        #cv2.imwrite(os.path.join(homography_dst, "test_3.jpg"), test_img)
        break