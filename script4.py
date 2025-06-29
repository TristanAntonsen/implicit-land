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


canvas.draw_sdf(sdf)

rp = Point.random()

n = sdf.eval_gradient_fd(rp)

l = Line.from_point_direction(rp, -n, sdf.eval_sdf(rp))
l1 = Line.from_point_direction(rp, -a.eval_gradient_fd(rp), a.eval_sdf(rp))
l2 = Line.from_point_direction(rp, -b.eval_gradient_fd(rp), b.eval_sdf(rp))
l3 = Line.from_point_direction(rp, -c.eval_gradient_fd(rp), c.eval_sdf(rp))

canvas.overlay_primitive(l)
canvas.overlay_primitive(l1)
canvas.overlay_primitive(l2)
canvas.overlay_primitive(l3)
canvas.draw_point(l.start)
canvas.draw_point(l.end)

canvas.img.save("output_image.png")
canvas.img.show()
