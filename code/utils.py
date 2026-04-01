import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob



def harris_corners(img, max_corners=4,minDistance=60):

    """
    For finding the corners of the board for homography.
    Uses harris corner detector and then uses sort_corners
    before returning it as numpy array.
    Defaults to returning 4 corners.

    """


    # Harris response
    harris = cv2.cornerHarris(img, 15, 15, 0.04)
    harris = cv2.dilate(harris, None)

    # Normalize to 0–255 so goodFeaturesToTrack can use it
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
    harris_norm = np.uint8(harris_norm)

    corners = cv2.goodFeaturesToTrack(
        harris_norm,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=minDistance
    )

    corners = np.array([tuple(pt.ravel()) for pt in corners])
    corners = sort_corners(corners)


    if corners is None:
        Exception("No corners")

    return corners


def plot_corners(img, corners, radius=4, color=(0,0,255), thickness=-1):
    """
    Debugging function

    img: input image (grayscale or BGR)
    corners: list of (x, y) tuples
    """
    # Convert grayscale to BGR so colored dots show up
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for (x, y) in corners:
        cv2.circle(img_color, (int(x), int(y)), radius, color, thickness)

    return img_color




def sort_corners(corners):
    """
    Sorts the corners from harris detector
    corners: list or array of 4 (x, y) points
    returns: array of points ordered as:
             top-left, top-right, bottom-right, bottom-left
    """

    pts = np.array(corners, dtype=np.float32)

    # Step 1: sort by y (top to bottom)
    pts = pts[np.argsort(pts[:,1])]

    # First two are top, last two are bottom
    top = pts[:2]
    bottom = pts[2:]

    # Step 2: sort left-to-right within each row
    top = top[np.argsort(top[:,0])]
    bottom = bottom[np.argsort(bottom[:,0])]

    # Order: TL, TR, BR, BL
    ordered = np.array([top[0],top[1], bottom[1], bottom[0]], dtype=np.float32)
    return ordered


def calibrate_camera(path):

    # === Checkerboard settings ===
    CHECKERBOARD = (8, 6)      # inner corners (width, height)
    SQUARE_SIZE = 25.0         # mm

    # Prepare object points (0,0,0), (25,0,0), (50,0,0) ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    # === Load calibration images ===
    images = glob.glob(f"{path}/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # === Run calibration ===
    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    return camera_matrix, dist_coeffs

def find_correct_rotation(homography, result, idx_board,idx_white):
    """
    This is for finding the correct rotation for the board s.t. 
    white is always at the bottom.
    """

    R90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)


    mid_point_x_white,mid_point_y_white = result[0].boxes.xywh[idx_white][0:2].cpu()
    mid_point_x_board,mid_point_y_board = result[0].boxes.xywh[idx_board][0:2].cpu()

    rotation = np.eye(3)
    while True:
        white_midpoint = cv2.perspectiveTransform(np.array([[[mid_point_x_white,mid_point_y_white]]]), homography)
        board_midpoint = cv2.perspectiveTransform(np.array([[[mid_point_x_board,mid_point_y_board]]]), homography)
        if np.squeeze(white_midpoint)[1]-np.squeeze(board_midpoint)[1] > 50: #Check that whitepoint y-point is sufficiently below board y-point

            return rotation
        else:
            homography = R90 @ homography
            rotation = R90@rotation


    
    

def calculate_homography(result,image_path):
    """
        Calculate and return the homography for BEV
        based on the board mask. Takes in a 
        results object from a YOLO model.
        The corners of a square in an image is points
        that are closest to the edges.
    """
    #This section of the code is to figure out which masks is for which object
    #It assumes only 1 detection for both
    detection_dict = result[0].names
    board_label = [k for k, v in detection_dict.items() if v == "Board"][0]
    white_label = [k for k, v in detection_dict.items() if v == "White_start"][0]
    objects_detected = result[0].boxes.cls
    idx_board = (objects_detected == board_label).nonzero(as_tuple=True)[0][0].item()
    idx_white = (objects_detected == white_label).nonzero(as_tuple=True)[0][0].item()

    #This section finds the corners of the board, based on the board mask.
    board_mask = result[0].masks.xy[idx_board].astype(np.int32) #extract mask

    image = cv2.imread(image_path)
    zeros_img = np.zeros_like(image[:,:,0],dtype=np.uint8)


    mask_boundary = cv2.polylines(zeros_img,     #Create a boundary of the mask
        [board_mask],
        isClosed=True,
        color=255,
        thickness=1
    )


    corners = harris_corners(mask_boundary) #Find the corners of the mask boundary


    h,w = 480,480
    bev_pts = np.array([
        [0, 0],                        
        [w, 0],              
        [w, h],    
        [0, h]              
    ], dtype=np.float32).reshape(-1,1,2)




    H, _ = cv2.findHomography(corners, bev_pts) #Find the homography matrix

    padding = 0  #Padding, only for visualization

    T = np.array([ #Move the image s.t. padding is on all sides
        [1, 0, -padding/2],
        [0, 1, -padding/2],
        [0, 0, 1]
    ], dtype=np.float32)

    T_origin = np.array([  #Move the image to the origin, necessary for rotating
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    T_from_origin = np.array([ #Move the image back from the origin
        [1, 0, w/2],
        [0, 1, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    

    rotation = find_correct_rotation(H,result,idx_board,idx_white) #Find the correct rotation, s.t. white is at the bottom


    return   T_from_origin @ rotation  @ T_origin @ T @ H


def create_fen(result_pieces,idx):
    board_positions = []
    for rank in range(8, 0, -1):
        for file in "ABCDEFGH":
            board_positions.append(f"{file}{rank}")

    # Empty 8×8 board
    board = [["" for _ in range(8)] for _ in range(8)]

    # Your FEN mapping
    fen_dict = {
        "WP": "P", "WR": "R", "WKN": "N", "WB": "B", "WQ": "Q", "WK": "K",
        "BP": "p", "BR": "r", "BKN": "n", "BB": "b", "BQ": "q", "BK": "k",
    }

    # Place pieces
    for i, position_index in enumerate(idx):
        piece_label = result_pieces[0].names[int(result_pieces[0].boxes.cls[i].cpu())]
        fen_letter = fen_dict[piece_label]

        square = board_positions[position_index]
        file = ord(square[0]) - ord("A")      # 0–7
        rank = 8 - int(square[1])             # 0–7

        board[rank][file] = fen_letter

    # Convert board → FEN
    fen_ranks = []
    for row in board:
        fen_row = ""
        empty = 0
        for cell in row:
            if cell == "":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_ranks.append(fen_row)

    fen_board = "/".join(fen_ranks)

    # Final FEN (no castling/en-passant yet)
    fen = f"{fen_board} w - - 0 1"

    return fen
