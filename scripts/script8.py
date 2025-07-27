from lib.lib import *

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025

import math

# Wing parameters
root_point = Point(-0.15, 0.35)   # Start point of leading edge (root)
root_chord = 0.7                  # Chord length at root (distance between leading and trailing edge at root, along y)
span = 0.7                     # Total span (distance from root to tip, along x)
sweep_angle = math.radians(0)    # Sweep angle in radians (positive = sweep back)
taper_ratio = 0.25                 # Ratio of tip chord to root chord

# Calculate tip point for leading edge
tip_leading_x = root_point.x + span * math.cos(sweep_angle)
tip_leading_y = root_point.y - span * math.sin(sweep_angle)
tip_leading = Point(tip_leading_x, tip_leading_y)

# Calculate root and tip points for trailing edge
root_trailing = Point(root_point.x, root_point.y - root_chord)
tip_trailing_x = tip_leading_x
tip_trailing_y = tip_leading_y - root_chord * taper_ratio
tip_trailing = Point(tip_trailing_x, tip_trailing_y)

# Create lines for leading and trailing edges
leading_edge = Line(root_point, tip_leading)
trailing_edge = Line(root_trailing, tip_trailing)

final = union(leading_edge, trailing_edge)

final.write_json("output_tree.json")
canvas.draw_sdf(final)
canvas.img.save("output_image.png")
# canvas.img.show()