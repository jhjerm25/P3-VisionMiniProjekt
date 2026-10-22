import os
import re
import sys
import cv2
import numpy as np
from collections import deque

GRID, CELL = 5, 100
SIZE = GRID * CELL

TERRAIN = ['Grassland', 'Field', 'Forest', 'Water', 'Swamp', 'Mine', 'Castle', 'Empty']


# Cut the 500x500 board into 25 cells of 100x100 px.
# Returns a list of ((row, col), cell_image), row by row.
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


# Decide the terrain of one cell from its HSV colours.
# Counts how many pixels are water-blue, green, black, yellow and beige,
# then checks the rules in a fixed order (first match wins).
# Returns (terrain, purity), purity = share of pixels that fit the terrain.
def classify_roi_hsv(roi, color_dist_thresh=5.0, min_pixel_ratio=0.20):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.float32) for i in range(3)]
    n = h.size
    med_h, med_s, med_v = np.median(h), np.median(s), np.median(v)

    water = np.count_nonzero((h >= 95) & (h <= 130) & (s >= 100)) / n
    green = np.count_nonzero((h >= 35) & (h <= 85) & (s >= 60)) / n       # any brightness
    black = np.count_nonzero(v < 50) / n
    yellow = np.count_nonzero((h >= 20) & (h <= 34) & (s >= 150) & (v >= 90)) / n
    beige = np.count_nonzero((h >= 10) & (h <= 30) & (s >= 20) & (s <= 130) & (v >= 70)) / n

    # wood grain is smooth (grey std ~4-10), swamp mud/cracks are not (~18-38)
    texture = cv2.cvtColor(roi[10:-10, 10:-10], cv2.COLOR_BGR2GRAY).std()

    if water >= 0.30:
        return "Water", water
    if green >= 0.30:                     # green tile: dark = Forest, bright = Grassland
        return ("Forest", green) if med_v < 80 else ("Grassland", green)
    if black >= 0.30:                     # black background and not green = Mine
        return "Mine", black
    if med_h <= 21 and texture < 14:      # wooden table: more orange than a field and smooth
        return "Empty", 0.9               # (no tile placed here)
    if med_s < 200:                       # swamp is dull beige/brown, a field is saturated yellow
        return "Swamp", beige
    if yellow >= 0.30 or med_s >= 165:
        return "Field", yellow
    return "Unknown", 0.0


CONFIDENT_RATIO = 0.50      # a tile this sure of its terrain can't be the castle


# Measure the cues that make a cell look like the castle:
# many hues, grey stone, cobble lines, a yellow crown, no black, no brown mud.
# Returns the cues as one number array.
def castle_features(roi):
    inner = roi[8:-8, 8:-8]                       # skip the tile border
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.int32) for i in range(3)]

    # 1) hue spread: a castle has many colours, a swamp is mostly one beige
    ang = h.astype(np.float32) * 2 * np.pi / 180
    hue_spread = 1 - np.hypot(np.cos(ang).mean(), np.sin(ang).mean())

    # 2) stone: unsaturated, mid-dark pixels (grey cobbles, tower)
    stone = ((s < 65) & (v > 40) & (v < 150)).mean()

    # 3) mortar: thin dark lines between cobbles (black-hat morphology)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mortar = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) > 10).mean()

    # 4) yellow-green crown (hue is different from the beige swamp)
    crown = ((h >= 26) & (h <= 38) & (s >= 50) & (s <= 210) & (v >= 140)).mean()

    # 5) black background: mine tiles are mostly black, a castle never is
    black = (v < 45).mean()

    # 6) mud: dark brown patches, every swamp has them and a castle does not
    mud = ((h <= 22) & (s >= 110) & (v < 0.8 * np.median(v))).mean()

    return np.array([hue_spread, stone, mortar, crown, black, mud])


# Hue/saturation histogram of a cell (border cut off).
# Used to compare how similar two cells look.
def hs_hist(roi):
    hsv = cv2.cvtColor(roi[8:-8, 8:-8], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


# labels the castle is often mistaken for, so they always stay candidates
CASTLE_LOOKALIKES = ("Swamp", "Mine", "Unknown", "Empty")


# Pick the one castle on the board.
# Candidates are cells with a weak terrain match or a castle look-alike label.
# Each candidate gets a score from the castle cues (compared within this board),
# and the highest score is returned as (row, col).
def find_castle(cells, labels, ratios):
    cand = [rc for rc in cells
            if ratios[rc] < CONFIDENT_RATIO or labels[rc] in CASTLE_LOOKALIKES]
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]

    feats = np.array([castle_features(cells[rc]) for rc in cand])
    # 7) uniqueness: how different the cell is from its most similar cell
    hists = {rc: hs_hist(roi) for rc, roi in cells.items()}
    unique = np.array([min(cv2.compareHist(hists[rc], hists[o], cv2.HISTCMP_BHATTACHARYYA)
                           for o in cells if o != rc) for rc in cand])
    feats = np.column_stack([feats, unique])
    purity = np.array([ratios[rc] for rc in cand])
    # a cell no rule explains well
    unknown = np.array([labels[rc] == "Unknown" or ratios[rc] < 0.11 for rc in cand])
    lo, hi = feats.min(0), feats.max(0)
    z = (feats - lo) / (hi - lo + 1e-9)           # compare tiles within this board
    score = (1.0 * z[:, 0]                        # hue spread
             + 1.5 * z[:, 1]                      # stone
             + 1.5 * z[:, 2]                      # mortar (cobbles)
             - 0.5 * z[:, 4]                      # black = mine
             + 1.5 * (feats[:, 3] >= 0.30)        # big yellow crown
             - 2.0 * purity                       # a tile the terrain rules explain well is no castle
             + 1.0 * unknown                      # a tile no rule explains is a castle suspect
             - 8.0 * feats[:, 5]                  # brown mud = swamp
             + 1.5 * z[:, 6])                     # one of a kind on this board
    return cand[int(np.argmax(score))]


# Find the properties: groups of touching cells (up/down/left/right)
# with the same terrain, using flood fill.
# Castle, Unknown and Empty cells are never part of a group.
def group_properties(terrain_grid):
    seen = set()
    groups = []
    for r in range(GRID):
        for c in range(GRID):
            t = terrain_grid[r][c]
            if (r, c) in seen or t in ("Castle", "Unknown", "Empty"):
                continue
            q, cells = deque([(r, c)]), []
            seen.add((r, c))
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < GRID and 0 <= nx < GRID
                            and (ny, nx) not in seen
                            and terrain_grid[ny][nx] == t):
                        seen.add((ny, nx))
                        q.append((ny, nx))
            groups.append((t, cells))
    return groups


# Run everything on one board image: classify cells, find the castle,
# group the properties and count the score (no crowns yet, 1 point per cell).
# Also draws the labels on a copy of the board for display.
def analyse(image_path, show=True):
    board = cv2.resize(cv2.imread(image_path), (SIZE, SIZE))
    cell_rois = get_all_cell_rois(board)

    cells = {rc: roi for rc, roi in cell_rois}
    labels, ratios = {}, {}
    for rc, roi in cell_rois:
        labels[rc], ratios[rc] = classify_roi_hsv(roi, color_dist_thresh=40.0,
                                                 min_pixel_ratio=0.25)

    castle = find_castle(cells, labels, ratios)
    if castle is not None:
        labels[castle] = "Castle"

    terrain_grid = [[labels[(r, c)] for c in range(GRID)] for r in range(GRID)]
    groups = group_properties(terrain_grid)

    # no crowns yet: every tile is worth 1, so a property = its size * 1
    total = sum(len(cells_) * 1 for _, cells_ in groups)

    visual = board.copy()
    for (r, c), _ in cell_rois:
        x0, y0 = c * CELL, r * CELL
        colour = (0, 0, 255) if (r, c) == castle else (0, 255, 0)
        cv2.rectangle(visual, (x0, y0), (x0 + CELL, y0 + CELL), colour, 2)
        cv2.putText(visual, labels[(r, c)][:5], (x0 + 5, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return castle, terrain_grid, groups, total, visual


FOLDER = (r"C:\Users\Jeppe\Downloads\King Domino dataset\King Domino dataset"
          r"\Cropped and perspective corrected boards")
KEY_SPACE, KEY_ESC = 32, 27


# List the numbered .jpg boards in the folder, sorted as numbers
# so 2.jpg comes before 10.jpg.
def list_boards(folder):
    names = [f for f in os.listdir(folder) if re.fullmatch(r"\d+\.jpg", f, re.IGNORECASE)]
    return sorted(names, key=lambda f: int(os.path.splitext(f)[0]))


# Print one board's result: castle position, the terrain grid,
# each property with its size, and the total score.
def print_result(name, castle, grid, groups, total):
    print(f"\n=== {name} ===")
    print("Castle at (row, col):", castle)
    for row in grid:
        print(" ".join(f"{t[:5]:<9}" for t in row))
    for t, cells_ in groups:
        print(f"{t:<10} size {len(cells_)}  -> {len(cells_)} x 1 = {len(cells_)}")
    print("Total (no crowns):", total)


# Main: go through every board, print the result and show the image.
# SPACE shows the next board, ESC or Q quits.
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER
    files = list_boards(folder)
    print(f"{len(files)} boards found. SPACE = next, ESC or Q = quit.")

    for i, name in enumerate(files):
        castle, grid, groups, total, visual = analyse(os.path.join(folder, name))
        print_result(name, castle, grid, groups, total)

        cv2.imshow("Terrain Classification (HSV)", visual)
        cv2.setWindowTitle("Terrain Classification (HSV)", f"{name}  ({i + 1}/{len(files)})")

        while True:                                   # ignore every key except these
            key = cv2.waitKey(0) & 0xFF
            if key == KEY_SPACE:
                break
            if key in (KEY_ESC, ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                sys.exit()

    cv2.destroyAllWindows()
    print("\nLast board reached.")
