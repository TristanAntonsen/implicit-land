from lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025


def rc():
    # Generate a random coordinate in the right range
    return random() - 0.5


# Create a circle and two boxes
p1 = Point(0.0, 0.0)
p2 = Point(0.35, 0.35)
p3 = p2.rotated(15)

a = Circle(p1, 0.18)
b = Circle(p2, 0.07)
c = Circle(p3, 0.027)


sdf = a | b | c

# Main SDF
canvas.draw_sdf(sdf)

rp = Point.random()
n = sdf.eval_gradient_fd(rp)

# annotation layer
l = Line.from_point_direction(rp, -n, sdf.eval_sdf(rp))
c1 = Circle(rp, 0.005)
c2 = Circle(l.end, 0.005)

canvas.draw_sdf(l | c1 | c2, contours=False, color=Color.GREENMAIN())

canvas.img.save("output_image.png")
canvas.img.show()
