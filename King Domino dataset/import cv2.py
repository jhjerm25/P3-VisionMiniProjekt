import cv2
import numpy as np

GRID, CELL = 5, 100
SIZE = GRID * CELL
MARGIN = 12  # ignore the cell border

# Per-pixel reference colours (OpenCV HSV: H 0-179, S 0-255, V 0-255) for the
# "normal" bright terrains. Forest and Mine are dark, so they use rules below.
PROTOTYPES = {
    "Water":     (105, 250, 185),
    "Field":     ( 26, 250, 195),
    "Grassland": ( 43, 215, 155),
    "Swamp":     ( 24, 110, 105),
}
WEIGHTS = np.array([3.0, 1.0, 1.0], dtype=np.float32)  # hue matters most

# Dark-pixel rules
GREEN_H = (33, 85)     # hue range of leaves / grass
GREEN_MIN_S = 50
FOREST_MAX_V = 95      # green and darker than this -> forest
MINE_MAX_V = 60        # dark and NOT green -> mine (black rock / cave)
MIN_S = 40             # ignore grey/white pixels (house walls, glare)
MINE_BLACK_V = 40         # "black" = V below this (and not green)
MINE_MIN_FRACTION = 0.15  # if this much of the cell is black -> Mine

# Castle = grey cobblestone
CASTLE_LOW = np.array([15, 30, 120])
CASTLE_HIGH = np.array([45, 125, 230])
MIN_CASTLE_RATIO = 0.25

NAMES = list(PROTOTYPES) + ["Forest", "Mine"]


def get_all_cell_rois(board):
    rois = []
    for r in range(GRID):
        for c in range(GRID):
            roi = board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            rois.append(((r, c), roi))
    return rois


def castle_ratio(roi):
    inner = roi[MARGIN:-MARGIN, MARGIN:-MARGIN]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CASTLE_LOW, CASTLE_HIGH)
    return np.count_nonzero(mask) / mask.size


def vote_terrain(roi):
    """Every pixel votes for a terrain; the cell gets the majority.

    Houses, crowns, boats and rocks only cover part of a cell, so they can
    only win a minority of the votes. Returns (terrain, share, votes dict).
    """
    inner = roi[MARGIN:-MARGIN, MARGIN:-MARGIN]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    green = (h >= GREEN_H[0]) & (h <= GREEN_H[1]) & (s > GREEN_MIN_S)
    forest = green & (v < FOREST_MAX_V)
    mine = (v < MINE_MAX_V) & ~green
    rest = ~forest & ~mine & (s >= MIN_S)

    votes = {n: 0 for n in NAMES}
    votes["Forest"] = int(forest.sum())
    votes["Mine"] = int(mine.sum())

    if rest.any():
        px = hsv[rest]
        dists = []
        for name in PROTOTYPES:
            p = np.array(PROTOTYPES[name], dtype=np.float32)
            dh = np.abs(px[:, 0] - p[0])
            dh = np.minimum(dh, 180 - dh)  # hue is circular
            d = np.stack([dh, np.abs(px[:, 1] - p[1]), np.abs(px[:, 2] - p[2])], axis=1)
            dists.append(np.linalg.norm(d * WEIGHTS, axis=1))
        winners = np.argmin(np.stack(dists, axis=1), axis=1)
        for i, name in enumerate(PROTOTYPES):
            votes[name] = int((winners == i).sum())

    total = max(sum(votes.values()), 1)
    terrain = max(votes, key=votes.get)

    # Mines are black rock with orange rocks/crowns on top. The bright pixels
    # split the vote, so also check how much of the whole cell is dark.
    # Only count really black pixels: brown shadows in swamp tiles are brighter.
    black_frac = ((v < MINE_BLACK_V) & ~green).sum() / len(hsv)
    if black_frac >= MINE_MIN_FRACTION:
        return "Mine", black_frac, votes, black_frac

    return terrain, votes[terrain] / total, votes, black_frac


image_path = r"C:\Users\Jeppe\Downloads\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\2.jpg"
board = cv2.imread(image_path)
board = cv2.resize(board, (SIZE, SIZE))
cell_rois = get_all_cell_rois(board)
visual_board = board.copy()

# there is exactly one castle: the cell that looks most like cobblestone
ratios = {pos: castle_ratio(roi) for pos, roi in cell_rois}
castle_pos = max(ratios, key=ratios.get)
if ratios[castle_pos] < MIN_CASTLE_RATIO:
    castle_pos = None
print("Top castle ratios:", {k: round(v, 2) for k, v in sorted(ratios.items(), key=lambda kv: -kv[1])[:4]})

for (r, c), roi in cell_rois:
    terrain, share, votes, black_frac = vote_terrain(roi)
    if (r, c) == castle_pos:
        terrain = "Castle"

    top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
    inner_hsv = cv2.cvtColor(roi[MARGIN:-MARGIN, MARGIN:-MARGIN], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    white_frac = np.mean((inner_hsv[:, 1] < 75) & (inner_hsv[:, 2] > 140))
    print(f"({r},{c}) -> {terrain:9s} share={share:.2f} black={black_frac:.2f} "
          f"white={white_frac:.2f} castle_ratio={ratios[(r, c)]:.2f} top votes={top}")

    x, y = c * CELL, r * CELL
    cv2.rectangle(visual_board, (x, y), (x + CELL, y + CELL), (0, 255, 0), 2)
    cv2.putText(visual_board, terrain, (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(visual_board, f"{share:.0%}", (x + 5, y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow("Terrain Classification (pixel voting)", visual_board)
cv2.waitKey(0)
cv2.destroyAllWindows()
