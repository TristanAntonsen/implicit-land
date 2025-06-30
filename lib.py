import renderer
import numpy as np
from PIL import Image, ImageDraw
from random import random


class FormatContext:
    def __init__(self, language="default", float_format="default"):
        self.language = language
        self.float_format = float_format

    def format_number(self, value: float):

        if self.float_format == "explicit_decimal":
            return f"{value:.4f}"

        return str(value)


EPSILON = 0.001


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __str__(self):
        return f"({self.x},{self.y})"

    def wgsl(self):
        return f"vec2<f32>({self.x},{self.y})"

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

    def __iter__(self):
        yield self.x
        yield self.y

    def norm(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        return self / self.norm()

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def rotated(self, angle):
        arad = np.radians(angle)
        ct = np.cos(arad)
        st = np.sin(arad)

        return Vector(self.x * ct - self.y * st, self.x * st + self.y * ct)

    def random():
        return Vector(random() - 0.5, random() - 0.5)


class Point(Vector):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return f"({self.x},{self.y})"

    def min(self, other):
        return

    def max(self, other):
        return Point(max(self.x, other.x), max(self.y, other.y))

    def wgsl(self):
        return f"vec2<f32>({self.x},{self.y})"

    def random():
        return Point(random() - 0.5, random() - 0.5)


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

    def eval_sdf(self, p: Point):
        raise NotImplementedError()

    def eval_gradient_fd(self, p: Point):
        dx = Vector(EPSILON, 0)
        dy = Vector(0, EPSILON)
        ddx = self.eval_sdf(p + dx) - self.eval_sdf(p - dx)
        ddy = self.eval_sdf(p + dy) - self.eval_sdf(p - dy)
        return Vector(ddx, ddy)

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
        pt = p - self.center
        return np.sqrt(pt.x * pt.x + pt.y * pt.y) - self.radius

    def eval_gradient(self, p: Point):
        return (p - self.center).normalize()

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        r = ctx.format_number(self.radius)

        return f"circle(p,vec2<f32>({x},{y}),{r})"


class Plane(Primitive):
    def __init__(self, center, nx, ny):
        super().__init__(center)

        self.normal: Vector = Vector(nx, ny)

    def to_expr(self, ctx: FormatContext):

        return f"plane(p,{self.center.wgsl()},{self.normal.normalize().wgsl()})"


class Box(Primitive):
    def __init__(self, center: Point, width, height):
        super().__init__(center)

        self.width = width
        self.height = height
        self.min_point = center - Vector(width, height) / 2
        self.max_point = center + Vector(width, height) / 2

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        w = ctx.format_number(self.width)
        h = ctx.format_number(self.height)

        return f"box(p-vec2<f32>({x},{y}),vec2<f32>({w},{h}))"

    def eval_gradient(self, point: Point):
        raise NotImplementedError()

    def bounding_box(self):

        pmin = self.center - Vector(self.width, self.height) / 2
        pmax = self.center + Vector(self.width, self.height) / 2

        return BoundingBox(pmin, pmax)


class BoxSharp(Primitive):
    def __init__(self, center, width, height):
        super().__init__(center)

        self.width = width
        self.height = height

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        w = ctx.format_number(self.width)
        h = ctx.format_number(self.height)

        return f"boxSharp(p-vec2<f32>({x},{y}),vec2<f32>({w},{h}))"

    def eval_gradient(self, point: Point):
        raise NotImplementedError()


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
        perp_point: Point = self.direction() * t * self.length() + self.start
        return (point - perp_point).normalize()

    def to_expr(self, ctx: FormatContext):

        sx = ctx.format_number(self.start.x)
        sy = ctx.format_number(self.start.y)
        ex = ctx.format_number(self.end.x)
        ey = ctx.format_number(self.end.y)

        return f"line(p,vec2<f32>({sx},{sy}),vec2<f32>({ex},{ey}))"

    def from_point_direction(point: Point, direction: Vector, length: float):
        return Line(point, point + direction.normalize() * length)


class Operator(SDFNode):
    def __init__(self, op: str, left: SDFNode, right: SDFNode, param=None):
        self.op = op
        self.left = as_node(left)
        self.right = as_node(right)
        self.param = param

    def eval_gradient_fd(self, p):
        return super().eval_gradient_fd(p)

    def eval_sdf(self, p: Point):
        left_sdf = self.left.eval_sdf(p)
        right_sdf = self.right.eval_sdf(p)

        match self.op:
            case "add":
                return left_sdf + right_sdf
            case "subtract":
                return left_sdf - right_sdf
            case "multiply":
                return left_sdf * right_sdf
            case "divide":
                return left_sdf / right_sdf
            case "union":
                return np.minimum(left_sdf, right_sdf)
            case "intersection":
                return np.maximum(left_sdf, right_sdf)
            case "difference":
                return np.maximum(left_sdf, -right_sdf)
            case "round_union":
                raise NotImplementedError
            case "round_intersection":
                raise NotImplementedError
            case "round_difference":
                raise NotImplementedError
            case _:
                return None

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


class Color:
    def __init__(self, r, g, b, alpha=255.0, normalize=True):
        self.r = r
        self.g = g
        self.b = b
        self.a = alpha
        if normalize:
            self.normalize()

    def normalize(self):
        self.r /= 255.0
        self.g /= 255.0
        self.b /= 255.0
        self.a /= 255.0

    def __mul__(self, value):

        return Color(self.r * value, self.g * value, self.b * value, normalize=False)

    def __rmul__(self, value):
        return Color(value * self.r, value * self.g, value * self.b, normalize=False)

    def __str__(self):
        return f"rgba({self.r},{self.g},{self.b},{self.a})"

    def format_str(self):
        return f"{self.r},{self.g},{self.b},{self.a}"

    def format_PIL(self):
        r = round(self.r * 255)
        g = round(self.g * 255)
        b = round(self.b * 255)
        a = round(self.a * 255)

        return f"rgba({r},{g},{b},{a})"

    def ORANGE():
        return Color(255.0, 75.0, 0)

    def RED():
        return Color(255.0, 0.0, 0.0)

    def GREEN():
        return Color(0.0, 255.0, 0.0)

    def BLUE():
        return Color(0.0, 0.0, 255.0)

    def BLUEMAIN():
        return BLUEMAIN

    def GREENMAIN():
        return GREENMAIN

    def WHITE():
        return Color(255.0, 255.0, 255.0)

    def BLACK():
        return Color(0.0, 0.0, 0.0)

    def GRAY():
        return Color(100.0, 100.0, 100.0)

    def TRANSPARENT():
        return Color(0.0, 0.0, 0.0, alpha=0.0)


TILE_SIZE = 16
BLUEMAIN = Color(36.0, 138.0, 255.0)
GREENMAIN = Color(36.0, 255.0, 138.0)

DEFAULT_SETTINGS = {
    "inner_color": BLUEMAIN,
    "outer_color": Color.TRANSPARENT(),
    "border_color": Color.BLACK(),
    "contour_color_inner": Color.BLUEMAIN() * 1.25,
    "contour_color_outer": Color.WHITE(),
    "contour_spacing": 0.015,
    "contour_fade": 0.5,
    "background_color": Color.WHITE(),
}


class Canvas:

    def __init__(self, resolution):

        if resolution % TILE_SIZE != 0:
            print("Warning: Resolution has been rounded to the nearest multiple of 16")

        self.resolution = round(resolution / TILE_SIZE) * TILE_SIZE
        self.shader = None
        self.ctx = FormatContext(float_format="explicit_decimal")
        self.settings = DEFAULT_SETTINGS
        self.img = Image.new(
            "RGBA",
            (self.resolution, self.resolution),
            self.settings["background_color"].format_PIL(),
        )

    def init_shader(self, expr):
        if expr is not None:
            template = open("renderer/src/shader_template.wgsl").read()
            shader = template.replace("MAP_FUNCTION", expr)

            for key, value in self.settings.items():
                if isinstance(value, Color):
                    shader = shader.replace(f"[[{key}]]", value.format_str())
                else:
                    shader = shader.replace(f"[[{key}]]", f"{value}")

            self.shader = shader
        else:
            raise Exception("Empty Expression")

    def draw_sdf(self, sdf, contours=True, color=None):

        if not contours:
            self.settings["contour_color_outer"] = Color.TRANSPARENT()
            self.settings["contour_color_inner"] = Color.TRANSPARENT()

        if color:
            self.settings["inner_color"] = color

        expr = sdf.to_expr(self.ctx)

        self.init_shader(expr)

        data = renderer.render_data(self.shader, self.resolution)

        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.resolution, self.resolution, 4)
        )

        self.img = Image.alpha_composite(self.img, Image.fromarray(arr, mode="RGBA"))

    def draw_point(self, p: Point, color: Color = Color.GREENMAIN(), weight=12):
        draw = ImageDraw.Draw(self.img)
        res = self.resolution
        hres = res / 2
        cx = p.x * res + hres
        cy = -p.y * res + hres
        r = weight
        lw = max(1, round(weight / 3))
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=1,
            width=lw,
            fill=color.format_PIL(),
        )

    def overlay_primitive(
        self, shape: Primitive, color: Color = Color.BLACK(), weight=4
    ):

        draw = ImageDraw.Draw(self.img)
        res = self.resolution
        hres = res / 2
        cx = shape.center.x + hres
        cy = shape.center.y + hres
        c = color.format_PIL()

        if isinstance(shape, Circle):
            r = shape.radius * res
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)

        elif isinstance(shape, Line):
            xy = [
                shape.start.x,
                -shape.start.y,
                shape.end.x,
                -shape.end.y,
            ]
            xy = [i * res + hres for i in xy]
            draw.line(xy, width=weight, fill=c)

        elif isinstance(shape, BoxSharp) or isinstance(shape, Box):

            bbox = shape.bounding_box()
            draw.rectangle(
                [
                    tuple(bbox.min_point * res + Vector(hres, hres)),
                    tuple(bbox.max_point * res + Vector(hres, hres)),
                ],
                width=weight,
                fill=c,
            )

        else:
            raise Exception("Wrong primitive type.")


if __name__ == "__main__":

    canvas = Canvas(16 * 96)

    c = Circle(Point(0.125, 0.125), 0.125)
    b = Box(Point(0, 0), 0.25, 0.25)
    result = c | b
    canvas.draw_sdf(result).show()
