import cv2 as cv
import os
import sys
import numpy as np

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'coins.jpg'))
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    threshold = 30
    segmented_image = recursive_split(gray, threshold)
    segmented_image = merge_regions(segmented_image)

    cv.imshow("Segmented image",segmented_image*255)
    cv.waitKey(0)

def is_homogeneous(region, threshold):
    min_val, max_val = np.min(region), np.max(region)
    return (max_val - min_val) <= threshold

def recursive_split(region, threshold):
    rows, cols = region.shape
    if rows <= 1 or cols <= 1:
        return np.zeros_like(region, dtype=np.uint8)
    
    if is_homogeneous(region, threshold):
        return np.ones_like(region, dtype=np.uint8)
    
    # Split and merge into four quadratns 
    mid_row, mid_col = rows // 2, cols // 2

    top_left = region[:mid_row, :mid_col]
    top_right = region[:mid_row, mid_col:]
    bottom_left = region[mid_row:, :mid_col]
    bottom_right = region[mid_row:, mid_col:]

    segmented_quadrants = np.zeros_like(region, dtype=np.uint8)

    segmented_quadrants[:mid_row, :mid_col] = recursive_split(top_left, threshold)
    segmented_quadrants[:mid_row, mid_col:] = recursive_split(top_right, threshold)
    segmented_quadrants[mid_row:, :mid_col] = recursive_split(bottom_left, threshold)
    segmented_quadrants[mid_row:, mid_col:] = recursive_split(bottom_right, threshold)

    return segmented_quadrants

def merge_regions(segmented):
        return segmented

if __name__=="__main__":
    main()
    