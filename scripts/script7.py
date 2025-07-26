import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from lib import *
import random

# # Set random seed for reproducible results (optional)
# random.seed(42)

# Create canvas
canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.02

# Generate 10 random boxes with random rotation and combine them as you go
combined_shape = None
boxes = []
for i in range(10):
    # Random position between -0.4 and 0.4
    x = random.uniform(-0.4, 0.4)
    y = random.uniform(-0.4, 0.4)

    # Random width and height between 0.02 and 0.18
    width = random.uniform(0.02, 0.18)
    height = random.uniform(0.02, 0.18)

    # Random rotation angle between 0 and 360 degrees
    angle = random.uniform(0, 360)

    # Create box and then rotate it
    box = Box(Point(x, y), width, height)
    box.rotate(angle)
    boxes.append(box)

    # Combine boxes as they are created
    if combined_shape is None:
        combined_shape = box
    else:
        # combined_shape = round_union(combined_shape, box, 0.02)
        combined_shape = combined_shape | box

# Draw the signed distance field (SDF) of the combined shape
canvas.draw_sdf(combined_shape)

# Draw bounding boxes for visualization
canvas.draw_bounds(combined_shape)

# Save the rendered image
canvas.img.save("output_image.png")

# Display the rendered image
canvas.img.show()