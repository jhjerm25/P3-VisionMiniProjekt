import cv2
import numpy as np

GRID, CELL = 5, 100
SIZE = GRID * CELL

TERRAIN = ['Grassland', 'Field', 'Forest', 'Water', 'Swamp', 'Mine', 'Castle']

HSV = {
    "Water":     (np.array([100, 150,  80]), np.array([125, 255, 255])),
    "Field":     (np.array([ 20, 150, 150]), np.array([ 35, 255, 255])),
    "Grassland": (np.array([ 35, 100, 100]), np.array([ 60, 255, 255])),
    "Forest":    (np.array([ 40,  80,  20]), np.array([ 85, 255, 130])),
    "Swamp":     (np.array([  8,  25,  90]), np.array([ 28, 120, 220])),
    "Mine":      (np.array([  0,   0,   0]), np.array([ 40, 150,  80])),
}


for terrain_name, (lower_bound, upper_bound) in HSV.items():
    HSV[terrain_name] = (lower_bound.astype(np.float32), upper_bound.astype(np.float32))


def get_all_cell_rois(board):
    rois = []
    for r in range(GRID):
        for c in range(GRID):
            y_start = r * CELL
            y_end = (r + 1) * CELL
            x_start = c * CELL
            x_end = (c + 1) * CELL

            roi = board[y_start:y_end, x_start:x_end]
            rois.append(((r, c), roi))
    return rois


def classify_roi_hsv(roi, color_dist_thresh=5.0, min_pixel_ratio=0.20):
    total_pixels = roi.shape[0] * roi.shape[1]
    best_terrain = "Unknown"
    highest_ratio = 0.0


    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)

    for terrain_name, (lower,upper) in HSV.items():
        
        mask = cv2.inRange(roi_hsv, lower, upper)

        matching_pixels = np.count_nonzero(mask)
        ratio = matching_pixels / total_pixels
    
        if ratio >= min_pixel_ratio and ratio > highest_ratio:
            highest_ratio = ratio
            best_terrain = terrain_name
    return best_terrain, highest_ratio




image_path = r"C:\Users\Jeppe\Downloads\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\8.jpg"
board = cv2.imread(image_path)
board = cv2.resize(board, (SIZE, SIZE))
cell_rois = get_all_cell_rois(board)
visual_board = board.copy()

for (r, c), roi in cell_rois:
    
    terrain, match_pct = classify_roi_hsv(
        roi, color_dist_thresh=40.0, min_pixel_ratio=0.25
    )

    x_start = c * CELL
    y_start = r * CELL
    x_end = (c + 1) * CELL
    y_end = (r + 1) * CELL

    cv2.rectangle(visual_board, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    label = f"{terrain[:5]}"

    cv2.putText(
        visual_board,
        label,
        (x_start + 5, y_start + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

cv2.imshow("Terrain Classification (HSV)", visual_board)
cv2.waitKey(0)
cv2.destroyAllWindows()