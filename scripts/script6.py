from lib.lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025

# Create a circle and two boxes
origin = Point(0.0, 0.0)
circle = Circle(origin, 0.34)
b1 = Box(origin, 0.2, 0.3)
b2 = Box(origin, 0.75, 0.1)

# Find the closest point on circle and subtract a circle
p_search = Point(0.2, -0.4)
closest_point = p_search - circle.eval_gradient(p_search) * circle.eval_sdf(p_search)
c_closest = Circle(closest_point, 0.02)

# Shapes to draw for search/closest/connector line
c1 = Circle(closest_point, 0.015)
c2 = Circle(p_search, 0.05)
line = Line(p_search, closest_point)

# Final result using booleans
combined_shape = round_difference(circle, b1 | b2, 0.05)
combined_shape = round_difference(combined_shape, c_closest, 0.0075)
combined_shape = combined_shape | c1 | c2 | line
combined_shape = combined_shape | (b1 | b2) + 0.01

# Draw the signed distance field (SDF) of the result shape onto the canvas
canvas.draw_sdf(combined_shape)

# Draw the bounding box of the result shape for visualization
canvas.draw_bounds(combined_shape)
text_point = Point(0.0, 0.0)
canvas.draw_text("TEXT", text_point)

end = time.time()
elapsed = end - start
print(f"Render time: {elapsed:.3f} seconds")

combined_shape.write_json("output_tree.json")
# Save the rendered image to a PNG file
canvas.img.save("output_image.png")
# Display the rendered image in a window
canvas.img.show()