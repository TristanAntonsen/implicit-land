from lib.lib import *
import time
from random import random


def rc():
    # Generate a random coordinate in the right range
    return random() - 0.5


start = time.time()

n = 10
r = 0.05
br = 0.025
canvas = Canvas(16 * 96)

origin = Point(rc(), rc())
result = Circle(origin, r)

for i in range(n - 1):
    origin = Point(rc(), rc())
    result = round_union(
        result,
        Circle(
            origin,
            r,
        ),
        br,
    )

canvas.draw_sdf(result)
canvas.img.show()

end = time.time()
elapsed = (end - start) * 1000
print(
    f"Total time (From Python): {elapsed :.2f}ms",
)
