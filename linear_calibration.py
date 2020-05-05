import cv2
import numpy as np
from common import chessboard_3DPoints, detect_chessboard_corners

def min_SVD(A, full_matrices=True):
    U, D, V = np.linalg.svd(A, full_matrices=full_matrices)
    min_eigenvalue_idx = np.argmin(D)
    min_eigenvector = V[:,min_eigenvalue_idx]
    return min_eigenvector

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
    pattern = cv2.imread('./calib_data_linear/IMG_3824.jpg')
    pattern = cv2.resize(pattern, (1008, 756), interpolation=cv2.INTER_AREA)
    cv2.imshow('image',pattern)
    cv2.waitKey(1000)
    
    points3D = chessboard_3DPoints((8,6), 15)
    #Actual chessboard is 7x9 (potrait) but corner detection fails on that, so 6x8 (portrait) is 
    # changed to 8x6 landscape and used
    corners, image_result = detect_chessboard_corners(pattern, (8,6), 15)
    
    cv2.imshow('Detected corners', image_result)

    rows = len(points3D) * 2
    A = np.zeros((rows,12))

    matched_points = list(zip(points3D, corners))
    idx = 0 #used to index into matrix `A`
    for point3D, point2D in matched_points:
        point2D = point2D[0]
        X_Y_Z_1 = (*point3D,1)
        A[idx,0:4] = X_Y_Z_1
        A[idx,8:] = list(map(lambda p: p * -point2D[0], X_Y_Z_1))
        
        A[idx+1,4:8] = X_Y_Z_1
        A[idx+1,8:] = list(map(lambda p: p * -point2D[1], X_Y_Z_1))
        
        idx += 2
        

    # np.set_printoptions(threshold=np.inf, linewidth=200) #debugging only
    # to solve for P s.t AP = 0
    AA = A.T @ A
    P = min_SVD(AA)
    P = P.reshape(3,4)
    print(f"Derived projection matrix:\n{P}\n")

    camera_center = min_SVD(P, full_matrices=False)
    print(f"Camera center:\n{camera_center}\n")
    
    M = P[:,:3]
    R, K = np.linalg.qr(M)

    print(f"Calibration Matrix K:\n{K}\n")
    print(f"Rotation Matrix R:\n{R}\n")
    assert np.allclose(M, R @ K), "QR decomposition"

    t = -(R @ camera_center)
    print(f"Camera Translation t:\n{t}\n")

    
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


    """
    For computing the reprojections, see
        - https://towardsdatascience.com/inverse-projection-transformation-c866ccedef1c
    """