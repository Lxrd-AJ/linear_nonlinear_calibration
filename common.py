import cv2
import numpy as np

def chessboard_3DPoints(board_size, square_size):
    """
    - board_size: a tuple (rows,columns) representing the number of rows and columns of the board.
    - square_size: the size of a square (cell) on the board in mm
    """
    
    corners = []
    for row in range(0, board_size[0]):
        for col in range(0, board_size[1]):
            corners.append((row * square_size, col * square_size, 0))
    return corners

def detect_chessboard_corners(image, board_size, square_size):
    ret, corners = cv2.findChessboardCorners(image, board_size)
    if ret: #found chessboard corners
        print(corners)
    else:
        print("Failed to detect corners in image")

    return corners