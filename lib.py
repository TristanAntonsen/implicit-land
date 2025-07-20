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


def interpolate(a: float, b: float, t: float):
    return a + (b - a) * t


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
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        elif isinstance(other, float):
            return Vector(self.x + other, self.y + other)
        raise NotImplementedError()

    def __radd__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x + self.x, other.y + self.y)
        elif isinstance(other, float):
            return Vector(other + self.x, other + self.y)
        raise NotImplementedError()

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        elif isinstance(other, float):
            return Vector(self.x - other, self.y - other)
        raise NotImplementedError()

    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x - self.x, other.y - self.y)
        elif isinstance(other, float):
            return Vector(other - self.x, other - self.y)
        return NotImplementedError

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

    @staticmethod
    def random():
        return Vector(random() - 0.5, random() - 0.5)

    def max(self, other):
        return Vector(np.maximum(self.x, other.x), np.maximum(self.y, other.y))

    def min(self, other):
        return Vector(np.minimum(self.x, other.x), np.minimum(self.y, other.y))

    def rotate(self, angle: float):
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        self.x = self.x * cos_a - self.y * sin_a
        self.y = self.x * sin_a + self.y * cos_a

        return self

    def interpolate(self, other, t: float):
        return Vector(interpolate(self.x, other.x, t), interpolate(self.y, other.y, t))


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

    @staticmethod
    def random():
        return Point(random() - 0.5, random() - 0.5)

    @staticmethod
    def origin():
        return Point(0, 0)

    def rotate(self, center, angle: float):
        angle_rad = np.radians(angle)  # degrees
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Translate to origin
        dx = self.x - center.x
        dy = self.y - center.y

        # Rotate and translate back
        self.x = dx * cos_a - dy * sin_a + center.x
        self.y = dx * sin_a + dy * cos_a + center.y

        return self

    def interpolate(self, other, t):
        return super().interpolate(other, t)


class BoundingBox:
    def __init__(self, min_point: Point, max_point: Point):
        self.min_point = min_point
        self.max_point = max_point
        self.center = min_point.interpolate(max_point, 0.5)

    def __str__(self):
        return f"Min: {self.min_point}, Max: {self.max_point})"

    def width(self):
        return self.max_point.x - self.min_point.x

    def height(self):
        return self.max_point.y - self.min_point.y

    def none():
        return BoundingBox(Point(0, 0), Point(0, 0))

    def union(a, b):
        return BoundingBox(
            Point(min(a.min_point.x, b.min_point.x), min(a.min_point.y, b.min_point.y)),
            Point(max(a.max_point.x, b.max_point.x), max(a.max_point.y, b.max_point.y)),
        )

    def corners(self):
        width = self.width()
        height = self.height()

        p1 = self.min_point
        p2 = Point(self.center.x + width / 2, self.center.y - height / 2)
        p3 = self.max_point
        p4 = Point(self.center.x - width / 2, self.center.y + height / 2)

        return [p1, p2, p3, p4]


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

    def to_dict(self) -> str:
        raise NotImplementedError()

    def compute_bounds(self):
        raise NotImplementedError()

    def compute_all_bounds(self):
        return [self.compute_bounds()]

    def eval_sdf(self, p: Point):
        raise NotImplementedError()

    def write_json(self, path: str):
        import json

        json.dump(self.to_dict(), open(path, "w"), indent=2)

    def eval_gradient_fd(self, p: Point):
        dx = Vector(EPSILON, 0)
        dy = Vector(0, EPSILON)
        ddx = self.eval_sdf(p + dx) - self.eval_sdf(p - dx)
        ddy = self.eval_sdf(p + dy) - self.eval_sdf(p - dy)
        return Vector(ddx, ddy)

    def projected_point(self, p: Point):
        n = self.eval_gradient_fd(p)
        return p - n.normalize() * self.eval_sdf(p)

    def closest_point(self, p: Point, iterations: int = 5):

        cp = p
        for i in range(iterations):
            n = self.eval_gradient_fd(cp).normalize()
            cp -= n * self.eval_sdf(cp)
        return cp

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

    def to_dict(self):
        return {"type": "Field", "value": self.value}

    def compute_bounds(self):
        return BoundingBox.none()

    def compute_all_bounds(self):
        return super().compute_all_bounds()


class Constant(SDFNode):
    def __init__(self, value: float):
        self.value: float = value

    def to_expr(self, ctx: FormatContext):
        return str(ctx.format_number(self.value))

    def to_dict(self):
        return {"type": "Constant", "value": self.value}

    def compute_bounds(self):
        return BoundingBox.none()

    def compute_all_bounds(self):
        return super().compute_all_bounds()


class Primitive(SDFNode):
    def __init__(self, center: Point, angle=0):
        self.center: Point = center
        self.rotation: float = 0  # degrees

    def compute_bounds(self):
        raise NotImplementedError()

    def compute_all_bounds(self):
        return super().compute_all_bounds()

    def rotate(self, angle):
        self.rotation = angle


class Circle(Primitive):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

    def compute_bounds(self):
        return BoundingBox(self.center - self.radius, self.center + self.radius)

    def compute_all_bounds(self):
        return super().compute_all_bounds()

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

    def to_dict(self):
        return {
            "type": "Circle",
            "center": str(self.center),
            "radius": self.radius,
        }


class Plane(Primitive):
    def __init__(self, center, nx, ny):
        super().__init__(center)

        self.normal: Vector = Vector(nx, ny)

    def compute_bounds(self):
        return BoundingBox(None, None)

    def compute_all_bounds(self):
        return super().compute_all_bounds()

    def to_expr(self, ctx: FormatContext):

        return f"plane(p,{self.center.wgsl()},{self.normal.normalize().wgsl()})"

    def to_dict(self):
        return {
            "type": "Plane",
            "center": str(self.center),
            "normal": self.normal,
        }


class Box(Primitive):
    def __init__(self, center: Point, width, height):
        super().__init__(center)

        self.width = width
        self.height = height
        self.min_point: Point = Point(center.x - width / 2, center.y - height / 2)
        self.max_point: Point = Point(center.x + width / 2, center.y + height / 2)

    def corners(self):

        p1 = self.min_point
        p2 = Point(self.center.x + self.width / 2, self.center.y - self.height / 2)
        p3 = self.max_point
        p4 = Point(self.center.x - self.width / 2, self.center.y + self.height / 2)

        return [
            p1,
            p2.rotate(self.center, self.rotation),
            p3,
            p4.rotate(self.center, self.rotation),
        ]

    def rotate(self, angle):
        self.rotation = angle
        self.min_point.rotate(self.center, angle)
        self.max_point.rotate(self.center, angle)

    def compute_bounds(self):

        corners = self.corners()
        min_x = min(p.x for p in corners)
        max_x = max(p.x for p in corners)
        min_y = min(p.y for p in corners)
        max_y = max(p.y for p in corners)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def compute_all_bounds(self):
        return super().compute_all_bounds()

    def from_bbox(bbox: BoundingBox):
        center = Point(
            (bbox.min_point.x + bbox.max_point.x) / 2,
            (bbox.min_point.y + bbox.max_point.y) / 2,
        )
        width = bbox.max_point.x - bbox.min_point.x
        height = bbox.max_point.y - bbox.min_point.y
        return Box(center, width, height)

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        w = ctx.format_number(self.width)
        h = ctx.format_number(self.height)
        a = ctx.format_number(np.radians(self.rotation))

        p = f"rotate(p, {a}, {self.center.wgsl()})"

        return f"box({p}-vec2<f32>({x},{y}),vec2<f32>({w},{h}))"

    def to_dict(self):
        return {
            "type": "Box",
            "center": str(self.center),
            "width": self.width,
            "height": self.height,
            "min_point": str((self.min_point.x, self.min_point.y)),
            "max_point": str((self.max_point.x, self.max_point.y)),
        }

    def eval_gradient(self, point: Point):
        raise NotImplementedError()


class BoxSharp(Primitive):
    def __init__(self, center, width, height):
        super().__init__(center)

        self.width = width
        self.height = height
        self.rotation = 0

    def compute_bounds(self):
        return BoundingBox(self.min_point, self.max_point)

    def compute_all_bounds(self):
        return super().compute_all_bounds()

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        w = ctx.format_number(self.width)
        h = ctx.format_number(self.height)
        p = f"rotate(p, {np.radians(self.rotation)}, {self.center.wgsl()})"

        return f"boxSharp({p}-vec2<f32>({x},{y}),vec2<f32>({w},{h}))"

    def to_dict(self):
        return {
            "type": "BoxSharp",
            "center": str(self.center),
            "width": self.width,
            "height": self.height,
            "min_point": str(self.min_point),
            "max_point": str(self.max_point),
        }

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

    def compute_bounds(self):
        max_point = Point(max(self.start.x, self.end.x), max(self.start.y, self.end.y))
        min_point = Point(min(self.start.x, self.end.x), min(self.start.y, self.end.y))
        return BoundingBox(min_point, max_point)

    def compute_all_bounds(self):
        return super().compute_all_bounds()

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

    def to_dict(self):
        return {
            "type": "Line",
            "center": str(self.center),
            "start": str(self.start),
            "end": str(self.end),
        }

    def from_point_direction(point: Point, direction: Vector, length: float):
        return Line(point, point + direction.normalize() * length)


class Capsule(Primitive):
    def __init__(self, start: Point, end: Point, width: float):
        self.start = start
        self.end = end
        self.center = (start + end) / 2.0
        self.width = width

    def direction(self):
        return (self.end - self.start).normalize()

    def compute_bounds(self):
        max_point = Point(max(self.start.x, self.end.x), max(self.start.y, self.end.y))
        min_point = Point(min(self.start.x, self.end.x), min(self.start.y, self.end.y))
        hw = self.width / 2
        min_point.x -= hw
        min_point.y -= hw
        max_point.x += hw
        max_point.y += hw

        return BoundingBox(min_point, max_point)

    def compute_all_bounds(self):
        return super().compute_all_bounds()

    def eval(self, point: Point):
        ab = self.end - self.start
        t = clamp(ab.dot(point - self.start) / ab.dot(ab), 0.0, 1.0)
        return ((ab * t + self.start) - point).norm() - self.width / 2

    def eval_gradient(self, point: Point):
        t = clamp(self.bary_coords(point), 0, 1)
        perp_point: Point = self.direction() * t * self.length() + self.start
        return (point - perp_point).normalize()

    def to_expr(self, ctx: FormatContext):

        sx = ctx.format_number(self.start.x)
        sy = ctx.format_number(self.start.y)
        ex = ctx.format_number(self.end.x)
        ey = ctx.format_number(self.end.y)
        w = ctx.format_number(self.width)
        return f"capsule(p,vec2<f32>({sx},{sy}),vec2<f32>({ex},{ey}),{w})"

    def to_dict(self):
        return {
            "type": "Capsule",
            "center": str(self.center),
            "start": str(self.start),
            "end": str(self.end),
            "width": str(self.width),
        }


def round_union_eval(a: float, b: float, r: float):
    u = Vector(r - a, r - b).max(Vector(0.0, 0.0))
    return np.maximum(r, np.minimum(a, b)) - u.norm()


class Operator(SDFNode):
    def __init__(self, op: str, left: SDFNode, right: SDFNode, param=None):
        self.op = op
        self.left = as_node(left)
        self.right = as_node(right)
        self.param = param

    def eval_gradient_fd(self, p):
        return super().eval_gradient_fd(p)

    def compute_bounds(self):
        left_bounds: BoundingBox = self.left.compute_bounds()
        right_bounds: BoundingBox = self.right.compute_bounds()
        return left_bounds.union(right_bounds)

    def compute_all_bounds(self):
        all_bounds = []

        all_bounds.extend(self.left.compute_all_bounds())
        all_bounds.extend(self.right.compute_all_bounds())
        all_bounds.append(self.compute_bounds())

        return all_bounds

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
                return round_union_eval(left_sdf, right_sdf, self.param)
            case "round_intersection":
                raise NotImplementedError
            case "round_difference":
                raise NotImplementedError
            case _:
                return None

    def to_dict(self):

        return {
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

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
        return Color(36.0, 138.0, 255.0)

    def GREENMAIN():
        return Color(36.0, 255.0, 138.0)

    def WHITE():
        return Color(255.0, 255.0, 255.0)

    def BLACK():
        return Color(0.0, 0.0, 0.0)

    def GRAY():
        return Color(100.0, 100.0, 100.0)

    def TRANSPARENT():
        return Color(0.0, 0.0, 0.0, alpha=0.0)


TILE_SIZE = 16


DEFAULT_SETTINGS = {
    "inner_color": Color.BLUEMAIN(),
    "outer_color": Color.WHITE(),
    "border_color": Color.BLACK(),
    "contour_color_inner": Color.BLUEMAIN() * 1.25,
    "contour_color_outer": Color.WHITE() * 0.8,
    "contour_spacing": 0.015,
    "contour_fade": 0.5,
    "background_color": Color.WHITE(),
    "normalize_bool": "true",
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

    def init_shader(self, expr, debug=False):
        if expr is not None:
            template = open("renderer/src/shader_template.wgsl").read()
            shader = template.replace("MAP_FUNCTION", expr)

            for key, value in self.settings.items():
                if isinstance(value, Color):
                    shader = shader.replace(f"[[{key}]]", value.format_str())
                else:
                    shader = shader.replace(f"[[{key}]]", f"{value}")
            self.shader = shader

            if debug:
                open("shader_debug.wgsl", "w").write(shader)
        else:
            raise Exception("Empty Expression")

    def draw_bounds(self, sdf: SDFNode):

        for bs in sdf.compute_all_bounds():
            self.overlay_primitive(Box.from_bbox(bs))

    def draw_sdf(self, sdf: SDFNode, contours=True, color=None):

        if not contours:
            self.settings["contour_color_outer"] = Color.TRANSPARENT()
            self.settings["contour_color_inner"] = Color.TRANSPARENT()

        if color:
            self.settings["inner_color"] = color

        expr = sdf.to_expr(self.ctx)

        self.init_shader(expr, debug=True)
        data = renderer.render_data(self.shader, (self.resolution, self.resolution))

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

    def overlay_primitive(self, shape: Primitive, color: Color = Color.RED(), weight=4):

        draw = ImageDraw.Draw(self.img)
        res = self.resolution
        hres = res / 2
        c = color.format_PIL()

        def to_pil_coords(p: Point):
            px = p.x * self.resolution + hres
            py = -p.y * self.resolution + hres
            return (px, py)

        if isinstance(shape, Circle):
            cx, cy = to_pil_coords(shape.center)
            r = shape.radius * res
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=None, outline=c)

        elif isinstance(shape, Box) or isinstance(shape, BoxSharp):

            minx, miny = to_pil_coords(shape.min_point)
            maxx, maxy = to_pil_coords(shape.max_point)
            draw.rectangle(
                [
                    # PIL has top left bottom right convention
                    (minx, maxy),
                    (maxx, miny),
                ],
                width=weight,
                fill=None,
                outline=c,
            )

        else:
            raise Exception("Wrong primitive type.")


if __name__ == "__main__":

    canvas = Canvas(16 * 96)

    c = Circle(Point(0.125, 0.125), 0.125)
    b = Box(Point(0, 0), 0.25, 0.25)
    result = c | b
    canvas.draw_sdf(result)
    canvas.img.save("output_image.png")
    canvas.img.show()
