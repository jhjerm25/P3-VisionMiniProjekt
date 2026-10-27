import os
import glob
import cv2
import numpy as np

# Set up paths relative to where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "Full game areas")
output_dir = os.path.join(script_dir, "Cropped and perspective corrected boards")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all matching photos (checking both uppercase and lowercase extensions)
image_files = sorted(
    glob.glob(os.path.join(input_dir, "DSC_*.JPG")) +
    glob.glob(os.path.join(input_dir, "DSC_*.jpg"))
)

if not image_files:
    raise FileNotFoundError(f"No DSC_*.JPG files found in: {input_dir}")

print(f"Loaded {len(image_files)} image(s) from '{input_dir}'.")


class RoiHandler:
    def __init__(self):
        self.points = []

    def grab_click_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 4:
                self.points = []
            self.points.append((x, y))


def load_image(image_file_number, target_size=(1200, 800)):
    filepath = image_files[image_file_number]
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Failed to read image: {filepath}")

    image_small = cv2.resize(image, target_size)

    # Compute scaling ratios so clicks on small window map to full-res original
    scale_x = image.shape[1] / target_size[0]
    scale_y = image.shape[0] / target_size[1]

    return image, image_small, (scale_x, scale_y)


image_file_number = 0
display_size = (1200, 800)
image, image_small, (scale_x, scale_y) = load_image(image_file_number, display_size)

roi = RoiHandler()
dst_points = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)
output_file_number = 1

cv2.namedWindow("Input image")
cv2.setMouseCallback("Input image", roi.grab_click_position)

print("\nControls:")
print(" - Left-click 4 corners of the board (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
print(" - Space: Crop, transform, and save")
print(" - 'd': Previous image")
print(" - 'f': Next image")
print(" - Esc: Exit\n")

while True:
    image_small_lines = image_small.copy()

    # Draw point markers
    for pt in roi.points:
        cv2.circle(image_small_lines, pt, 5, (0, 0, 255), -1)

    # Draw polygon once at least 2 points are selected
    if len(roi.points) > 1:
        pts_array = np.array(roi.points, dtype=np.int32).reshape((-1, 1, 2))
        is_closed = len(roi.points) == 4
        cv2.polylines(image_small_lines, [pts_array], is_closed, (0, 255, 0), 2)

    cv2.imshow("Input image", image_small_lines)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break

    elif len(roi.points) == 4 and key == 32:  # Space
        # Map selected points from display resolution back to full original image resolution
        src_points = np.array(
            [[x * scale_x, y * scale_y] for x, y in roi.points],
            dtype=np.float32
        )

        transform, _ = cv2.findHomography(src_points, dst_points)
        if transform is not None:
            adjusted_image = cv2.warpPerspective(image, transform, (500, 500))
            out_filename = os.path.join(output_dir, f"{output_file_number}.jpg")
            cv2.imwrite(out_filename, adjusted_image)
            print(f"Saved: {out_filename}")
            output_file_number += 1
            roi.points = []

    elif key == ord("d"):  # 'd' for previous
        if image_file_number > 0:
            image_file_number -= 1
            image, image_small, (scale_x, scale_y) = load_image(image_file_number, display_size)
            roi.points = []
            print(f"Viewing: {os.path.basename(image_files[image_file_number])}")

    elif key == ord("f"):  # 'f' for next
        if image_file_number < len(image_files) - 1:
            image_file_number += 1
            image, image_small, (scale_x, scale_y) = load_image(image_file_number, display_size)
            roi.points = []
            print(f"Viewing: {os.path.basename(image_files[image_file_number])}")

    # Break if user closes the window with the 'X' button
    if cv2.getWindowProperty("Input image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()