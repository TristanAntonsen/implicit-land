import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.05
canvas.settings["outer_color"] = Color(0, 0, 0, 0)

# Create a circle and two boxes
origin = Point(0.0, 0.0)
a = Plane(origin, 1, 0) - 0.25
b = -Plane(origin, 1, 0) - 0.25
c = Plane(origin, 0, 1) - 0.25
d = -Plane(origin, 0, 1) - 0.25

result = round_intersection(a & c, b & d, 0.)

canvas.draw_sdf(result)
canvas.img.save("output_image.png")
canvas.img.show()
