from lib.lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025


def rc():
    # Generate a random coordinate in the right range
    return random() - 0.5


# Create a circle and two boxes
p1 = Point(0.0, 0.0)
p2 = Point(0.25, 0.25)

a = Circle(p1, 0.18)
b = Circle(p2, 0.13)


sdf = round_union(a, b, 0.15)
# sdf = a | b

# Main SDF
canvas.draw_sdf(sdf)

# rp = Point.random()
rp = Point(0.3, 0.0)
n = sdf.eval_gradient_fd(rp)

# annotation layer (Set background to transparent for overlay)
canvas.settings["outer_color"] = Color.TRANSPARENT()
canvas.settings["normalize_bool"] = "false"

cp = sdf.closest_point(rp)
c1 = Circle(rp, 0.005)
c2 = Circle(cp, 0.005)
l = Line(rp, cp)

canvas.draw_sdf(l | c1 | c2, contours=False, color=Color.GREENMAIN())

canvas.img.save("output_image.png")
canvas.img.show()
