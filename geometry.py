import numpy as np
from random import random


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

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

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

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return self.min_point == other.min_point and self.max_point == other.max_point

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
