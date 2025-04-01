##########################
# Main Processing Script #
##########################

from glob import glob
import cv2
import os
from pathlib import Path
from src.utils import manualThreshold, manualFiltering, show_image

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import pandas as pd
import argparse
# import imutils
import cv2

ROOT_DIR = 'images'
OUT_DIR = 'output'

sp=20
sr=30

hsv_min = (1, 54, 0)
hsv_max = (155, 228, 210)

# Set area thresholds
min_area = 1000

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

#check image
# img_name = 'TLI_SGG3201.JPG'
# img_path = Path(ROOT_DIR) / img_name
# img = cv2.imread(img_path)


# shifted = cv2.pyrMeanShiftFiltering(img, sp, sr)
# cv2.imwrite(Path(OUT_DIR)/ ('pyr_shifted_' + img_name), shifted)

# show_image(shifted)

# Set thresholding values
# thr_img, hsv_values = manualThreshold(shifted, invert=False, output='both')
# hsv_min = hsv_values['HSVmin']
# hsv_max = hsv_values['HSVmax']

# print(hsv_values)


# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# thr_img = cv2.inRange(hsv_img, hsv_min, hsv_max)
# cv2.imwrite(Path(OUT_DIR) / ('threshed_' + img_name), thr_img)

# Find contours
# contours, _ = cv2.findContours(thr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area
# filtered_contours = []
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if min_area <= area:
#         filtered_contours.append(contour)

# Draw filtered contours on a new mask
# mask = np.zeros(img.shape[:2]).astype(np.uint8)
# cv2.drawContours(mask, filtered_contours, -1, (255), cv2.FILLED)
# cv2.imwrite(Path(OUT_DIR) / ('mask_' + img_name), mask)

# Maximize using dilation
# image_max = ndimage.maximum_filter(thr_img, size=5, mode='constant')

# print(image_max)
# print(mask)

# mask_binary = cv2.bitwise_and(image_max, image_max, mask=mask)


# cv2.imwrite(Path(OUT_DIR) / ('binary_' + img_name), mask_binary)


# show_image(mask_binary)


# Distance matrix and watershed
#Calc distance
# D = ndimage.distance_transform_edt(mask_binary)

# localMax = peak_local_max(image=D, min_distance=30, labels=mask_binary)


# for pt in localMax:
    # img = cv2.circle(img, tuple(pt[::-1]), radius=10, color=(255, 0, 0), thickness=-1)

# show_image(img)


# mask = np.zeros(D.shape, dtype=bool)
# mask[tuple(localMax.T)] = True
# markers, _ = ndimage.label(mask)
# labels = watershed(-D, markers, mask=mask_binary)

# print(labels)

# show_image(labels.astype(np.uint8))



img_names = glob('*.JPG', root_dir='images')
res_dict = {'img_names': [], 'num_seeds': []}
for img_name in img_names:
    print(f"Processing img {img_name}")

    #Read img
    img_path = Path('images') / img_name
    img = cv2.imread(img_path)

    #Mean Shift
    shifted = cv2.pyrMeanShiftFiltering(img, sp, sr)
    cv2.imwrite(Path(OUT_DIR)/ ('mean_shifted_' + img_name), shifted)

    #HSV threshold
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thr_img = cv2.inRange(hsv_img, hsv_min, hsv_max)
    cv2.imwrite(Path(OUT_DIR) / ('threshed_' + img_name), thr_img)

    # Find contours
    contours, _ = cv2.findContours(thr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area:
            filtered_contours.append(contour)

    # Draw filtered contours on a new mask
    mask = np.zeros(img.shape[:2]).astype(np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, (255), cv2.FILLED)
    cv2.imwrite(Path(OUT_DIR) / ('cnt_mask_' + img_name), mask)

    # Maximize using dilation
    image_max = ndimage.maximum_filter(thr_img, size=5, mode='constant')
    mask_binary = cv2.bitwise_and(image_max, image_max, mask=mask)
    cv2.imwrite(Path(OUT_DIR) / ('binary_' + img_name), mask_binary)

    # Distance matrix and watershed
    D = ndimage.distance_transform_edt(mask_binary)
    localMax = peak_local_max(image=D, min_distance=30, labels=mask_binary)

    for pt in localMax:
        img = cv2.circle(img, tuple(pt[::-1]), radius=12, color=(255, 0, 0), thickness=-1)
    cv2.imwrite(Path(OUT_DIR) / ('detections_' + img_name), img)

    # Run Watershed
    mask = np.zeros(D.shape, dtype=bool)
    mask[tuple(localMax.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-D, markers, mask=mask_binary)

    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    res_dict['img_names'].append(img_name)
    res_dict['num_seeds'].append(len(np.unique(labels))-1)

df = pd.DataFrame(res_dict)
df.to_csv('seed_counts.csv')