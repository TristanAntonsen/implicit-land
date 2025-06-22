import math


class FormatContext:
    def __init__(self, language="default", float_format="default"):
        self.language = language
        self.float_format = float_format

    def format_number(self, value: float):

        if self.float_format == "explicit_decimal":
            return f"{value:.4f}"

        return str(value)


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x},{self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return Vector(other.x + self.x, other.y + self.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y)

    def __mul__(self, value):
        return Vector(self.x * value, self.y * value)

    def __truediv__(self, value):
        return Vector(self.x / value, self.y / value)

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        return self / self.norm()

    def dot(self, other):
        return self.x * other.x + self.y * other.y


class Point(Vector):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return f"({self.x},{self.y})"

    def min(self, other):
        return

    def max(self, other):
        return Point(max(self.x, other.x), max(self.y, other.y))


class BoundingBox:
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point

    def union(a, b):
        return BoundingBox(
            Point(min(a.min_point.x, b.min_point.x), min(a.min_point.y, b.min_point.y)),
            Point(max(a.max_point.x, b.max_point.x), max(a.max_point.y, b.max_point.y)),
        )


def as_node(value):

    if isinstance(value, SDFNode):
        return value
    elif isinstance(value, Constant):
        return Constant(value)
    else:
        return Field(value)


class SDFNode:
    def to_expr(self) -> str:
        raise NotImplementedError()

    def bounding_box(self) -> BoundingBox:
        raise NotImplementedError()

    def __neg__(self):
        return Operator("multiply", self, as_node(-1))

    def __add__(self, other):
        return Operator("add", self, as_node(other))

    def __radd__(self, other):
        return Operator("add", as_node(other), self)

    def __sub__(self, other):
        return Operator("subtract", self, as_node(other))

    def __rsub__(self, other):
        return Operator("subtract", as_node(other), self)

    def __mul__(self, other):
        return Operator("multiply", self, as_node(other))

    def __rmul__(self, other):
        return Operator("multiply", as_node(other), self)

    def __truediv__(self, other):
        return Operator("divide", self, as_node(other))

    def __rtruediv__(self, other):
        return Operator("divide", as_node(other), self)

    def __or__(self, other):
        return Operator("union", self, as_node(other))

    def __and__(self, other):
        return Operator("intersection", self, as_node(other))


class Field(SDFNode):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def to_expr(self, ctx: FormatContext):
        return str(self.value)


class Constant(SDFNode):
    def __init__(self, value: float):
        self.value: float = value

    def to_expr(self, ctx: FormatContext):
        return str(ctx.format_number(self.value))


class Primitive(SDFNode):
    def __init__(self, center: Point):
        self.center: Point = center


class Circle(Primitive):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

    def eval_sdf(self, p: Point):
        return math.sqrt(p.x * p.x + p.y * p.y) - self.radius

    def eval_gradient(self, p: Point):
        return (p - self.center).normalize()

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        r = ctx.format_number(self.radius)

        return f"circle(p,vec2<f32>({x},{y}),{r})"


class Box(Primitive):
    def __init__(self, center, width, height):
        super().__init__(center)

        self.width = width
        self.height = height

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        w = ctx.format_number(self.width)
        h = ctx.format_number(self.height)

        return f"box(p-vec2<f32>({x},{y}),vec2<f32>({w},{h}))"


def clamp(value, min_value, max_value):

    return min(max(value, min_value), max_value)


class Line(Primitive):
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end
        self.center = (start + end) / 2.0

    def length(self):

        return (self.end - self.start).norm()

    def direction(self):
        return (self.end - self.start).normalize()

    def bary_coords(self, point: Point):

        sp = point - self.start

        return sp.dot(self.direction()) / self.length()

    def eval(self, point: Point):
        ab = self.end - self.start
        t = clamp(ab.dot(point - self.start) / ab.dot(ab), 0.0, 1.0)
        return ((ab * t + self.start) - point).norm()

    def eval_gradient(self, point: Point):
        t = clamp(self.bary_coords(point), 0, 1)
        perp_point: Point = self.direction() * t + self.start
        return (point - perp_point).normalize()

    def to_expr(self, ctx: FormatContext):

        sx = ctx.format_number(self.start.x)
        sy = ctx.format_number(self.start.y)
        ex = ctx.format_number(self.end.x)
        ey = ctx.format_number(self.end.y)

        return f"line(p,vec2<f32>({sx},{sy}),vec2<f32>({ex},{ey}))"


class Operator(SDFNode):
    def __init__(self, op: str, left: SDFNode, right: SDFNode, param=None):
        self.op = op
        self.left = as_node(left)
        self.right = as_node(right)
        self.param = param

    def to_expr(self, ctx: FormatContext):

        left_expr = self.left.to_expr(ctx)
        right_expr = self.right.to_expr(ctx)

        match self.op:
            case "add":
                return f"{left_expr}+{right_expr}"
            case "subtract":
                return f"{left_expr}-{right_expr}"
            case "multiply":
                return f"{left_expr}*{right_expr}"
            case "divide":
                return f"{left_expr}/{right_expr}"
            case "union":
                return f"min({left_expr},{right_expr})"
            case "intersection":
                return f"max({left_expr},{right_expr})"
            case "difference":
                return f"max({left_expr},-{right_expr})"
            case "round_union":
                return f"roundUnion({left_expr},{right_expr},{self.param})"
            case "round_intersection":
                return f"roundIntersection({left_expr},{right_expr},{self.param})"
            case "round_difference":
                return f"roundDifference({left_expr},{right_expr}, {self.param})"
            case _:
                return None


def union(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("union", a, b)


def difference(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("difference", a, b)


def intersection(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("intersection", a, b)


def round_union(a: SDFNode, b: SDFNode, radius: float) -> SDFNode:
    return Operator("round_union", a, b, radius)


def round_difference(a: SDFNode, b: SDFNode, radius: float) -> SDFNode:
    return Operator("round_difference", a, b, radius)


def round_intersection(a: SDFNode, b: SDFNode, radius: float) -> SDFNode:
    return Operator("round_intersection", a, b, radius)


GRID_SIZE = 16
BLUE = (36.0, 138.0, 255.0)
ORANGE = (255.0, 75.0, 0)


class Canvas:

    def __init__(self, resolution):

        if resolution % GRID_SIZE != 0:
            print("Warning: Resolution has been rounded to the nearest multiple of 16")

        self.resolution = round(resolution / GRID_SIZE) * GRID_SIZE
        self.inner_color = "1., 1., 1., 1."
        self.outer_color = ".4, .4, .4, 1."
        self.shader = None

    def init_shader(self, expr):
        if expr is not None:
            template = open("renderer/src/shader_template.wgsl").read()
            shader = template.replace("MAP_FUNCTION", expr)
            shader = shader.replace("INNER_COLOR", self.inner_color)
            shader = shader.replace("OUTER_COLOR", self.outer_color)
            self.shader = shader
        else:
            raise Exception("Empty Expression")

    def set_inner_color(self, r, g, b, normalize=False):

        if normalize:
            self.inner_color = f"{r / 255.},{g / 255.},{b / 255.},{1}"
        else:
            self.inner_color = f"{r},{g},{b},{1}"

    def set_outer_color(self, r, g, b, normalize=False):

        if normalize:
            self.outer_color = f"{r / 255.},{g / 255.},{b / 255.},{1}"
        else:
            self.outer_color = f"{r},{g},{b},{1}"

    def generate_image(self, ctx, path):
        import renderer

        expr = result.to_expr(ctx)

        self.set_inner_color(*BLUE, normalize=True)
        self.set_outer_color(1.0, 1.0, 1.0)
        self.init_shader(expr)

        renderer.render_with_wgsl(self.shader, self.resolution, path)


if __name__ == "__main__":

    ctx = FormatContext(float_format="explicit_decimal")

    canvas = Canvas(16 * 96)
    origin = Point(0.0, 0.0)

    # a = Circle(origin, 0.35)
    # b = Box(origin, 0.2, 10.0)
    # c = Box(origin, 10.0, 0.2)
    # d = Box(origin, 10.0, 0.3)

    # result = round_union(a & -b, c, 0.025)
    # result = result & a & d
    c = Circle(Point(0.2, -0.1), 0.01)
    line = Line(Point(-0.25, 0.0), Point(0.25, 0.25))
    n = line.eval_gradient(c.center)
    d = line.eval(c.center)
    cp = c.center - n * d
    cl = Circle(cp, 0.02)

    result = union(c, line - 0.01) | cl
    canvas.generate_image(ctx, "output_image.png")
