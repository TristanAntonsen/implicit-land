from lib import *

canvas = Canvas(16 * 96)


# Create a circle and two boxes
origin = Point(-0.1, -0.075)
circle = Circle(origin, 0.34)
b1 = Box(origin, 0.2, 0.3)
b2 = Box(origin, 0.75, 0.1)

# find the closest point on c and subtract a circle
p_search = Point(0.4, 0.3)
closest_point =  p_search - circle.eval_gradient(p_search) * circle.eval_sdf(p_search)
c_closest = Circle(closest_point, 0.02)

# shapes to draw for search/closest/connector line
c1 = Circle(closest_point, 0.01)
c2 = Circle(p_search, 0.01)
line = Line(p_search, closest_point)

# final result using booleans
result = round_difference(circle, b1 | b2, 0.05) 
result = round_difference(result, c_closest, 0.0075)
result = result | c1 | c2 | line
result = result | (b1 | b2) + 0.01

canvas.generate_image(result, "output_image.png")