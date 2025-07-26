from lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025


canvas = Canvas(16 * 96)

box_center = Point(0.0, 0.0)
rotated_box = Box(box_center, 0.3, 0.25)
rotated_box.rotate(30)


left_circle = Circle(Point(-0.25, -0.25), 0.05)
right_circle = Circle(Point(0.1, -0.25), 0.05)
le = Line(Point(-0.35, 0.35), Point(0.2, 0.1))
te = Line(Point(-0.35, -0.35), Point(0.2, -0.1))

combined_shape = union(le, te)


# Save the result shape as a JSON file for later use or inspection
combined_shape.write_json("output_tree.json")

# Draw the signed distance field (SDF) of the result shape onto the canvas
canvas.draw_sdf(combined_shape)

# Draw the bounding box of the result shape for visualization
# canvas.draw_bounds(combined_shape)

# Save the rendered image to a PNG file
canvas.img.save("output_image.png")

# Display the rendered image in a window
# canvas.img.show()