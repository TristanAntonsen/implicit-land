from lib.lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025

center_point = Point(-0.075, 0.0)
main_circle = Circle(center_point, 0.34)
vertical_box = Box(center_point, 0.2, 0.3)
horizontal_box = Box(center_point, 0.75, 0.1)

start_point = Point(0.4, 0.4)
closest_point = start_point - main_circle.eval_gradient(start_point) * main_circle.eval_sdf(start_point)
contact_circle = Circle(closest_point, 0.02)

contact_marker = Circle(closest_point, 0.015)
start_marker = Circle(start_point, 0.015)

base_shape = round_difference(main_circle, vertical_box | horizontal_box, 0.05)
notched_shape = round_difference(base_shape, contact_circle, 0.0075)
final = notched_shape | (horizontal_box | vertical_box) + 0.01
canvas.draw_sdf(final)

## Annotation layer
canvas.set_transparent()
canvas.settings["inner_color"] = Color.RED()

markers  = contact_marker | start_marker
line = Line(start_point, closest_point)
annotated_shape = markers | line

canvas.draw_sdf(annotated_shape)
canvas.draw_bounds(final | annotated_shape)

canvas.img.save("output_image.png")
canvas.img.show()

end = time.time()
elapsed = (end - start) * 1000
print(
    f"Total time (From Python): {elapsed :.2f}ms",
)
