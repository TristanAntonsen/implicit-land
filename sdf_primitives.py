import numpy as np
from geometry import Point, Vector, BoundingBox
from sdf_core import Primitive
from utils import clamp


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

    def to_expr(self, ctx):
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

    def to_expr(self, ctx):
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

    def to_expr(self, ctx):
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

    def to_expr(self, ctx):
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

    def to_expr(self, ctx):
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

    def to_expr(self, ctx):
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