import cv2
import numpy as np
from common import chessboard_3DPoints, detect_chessboard_corners

if __name__ == "__main__":
    """
    TODO: Summarise all the steps here
    """

    """
    - Find the chessboard corners in all the images
    - Compute the real world 3D positions of the chessboard corners
    - See AraSLAM calibration https://github.com/Lxrd-AJ/AraSLAM/blob/master/src/calibration/calibrator.cc for similar code
        for doing this in OpenCV
    """
    pattern = cv2.imread('uncalibrated_pattern_2.png', 0)
    points3D = chessboard_3DPoints((9,9), 15)
    _ = detect_chessboard_corners(pattern, (9,9), 15)
    
    # cv2.imshow('image',pattern)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    """
    For computing the reprojections, see
        - https://towardsdatascience.com/inverse-projection-transformation-c866ccedef1c
    """