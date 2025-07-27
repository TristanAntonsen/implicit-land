from lib.lib import *

canvas = Canvas(16 * 96)

c = Circle(Point(0.125, 0.125), 0.125)
b = Box(Point(0, 0), 0.25, 0.25)
result = round_union(b, c, 0.025)

canvas.draw_sdf(result)
canvas.img.save("img/demo_image.png")
canvas.img.show()