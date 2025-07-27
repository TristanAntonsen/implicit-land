from lib.lib import *
import time

start = time.time()

canvas = Canvas(16 * 96)
canvas.settings["contour_spacing"] = 0.025

point_1 = Point(-0.075, 0.0)
body_1 = Circle(point_1, 0.34)
body_2 = Box(point_1, 0.2, 0.3)
body_3 = Box(point_1, 0.75, 0.1)

sp = Point(0.4, 0.4)
cp = sp - body_1.eval_gradient(sp) * body_1.eval_sdf(sp)
body_4 = Circle(cp, 0.02)

body_5 = Circle(cp, 0.015)
body_6 = Circle(sp, 0.015)
body_7 = Line(sp, cp)

body_8 = round_difference(body_1, body_2 | body_3, 0.05)
body_9 = round_difference(body_8, body_4, 0.0075)
body_10 = body_9 | body_5 | body_6
body_11 = body_10 | ((body_2 | body_3) + 0.01)
body_12 = body_11 | body_7

canvas.draw_sdf(body_12)
canvas.draw_bounds(body_12)
canvas.img.save("output_image.png")
canvas.img.show()

end = time.time()
elapsed = (end - start) * 1000
print(
    f"Total time (From Python): {elapsed :.2f}ms",
)
