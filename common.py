import cv2
import numpy as np

def chessboard_3DPoints(board_size, square_size):
    """
    - board_size: a tuple (rows,columns) representing the number of rows and columns of the board.
    - square_size: the size of a square (cell) on the board in mm
    """
    
    corners = []
    for col in range(0, board_size[1]):
        for row in range(0, board_size[0]):
            corners.append((row * square_size, col * square_size, 0))
    return corners

def detect_chessboard_corners(image, board_size, square_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    ret, corners = cv2.findChessboardCorners(gray_image, board_size)
    if ret: #found chessboard corners
        print(f"Detected {corners.shape} corners in image")
        cv2.cornerSubPix(gray_image, corners, (3,3), (-1,-1), criteria)
        cv2.drawChessboardCorners(image, board_size, corners, ret)
    else:
        raise ValueError("Failed to detect corners in image")

    return (corners, image)