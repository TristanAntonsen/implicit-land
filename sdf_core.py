import numpy as np
from geometry import Point, Vector
from utils import clamp

EPSILON = 0.001


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
        from sdf_operations import Operator
        return Operator("multiply", self, as_node(-1))

    def __add__(self, other):
        from sdf_operations import Operator
        return Operator("add", self, as_node(other))

    def __radd__(self, other):
        from sdf_operations import Operator
        return Operator("add", as_node(other), self)

    def __sub__(self, other):
        from sdf_operations import Operator
        return Operator("subtract", self, as_node(other))

    def __rsub__(self, other):
        from sdf_operations import Operator
        return Operator("subtract", as_node(other), self)

    def __mul__(self, other):
        from sdf_operations import Operator
        return Operator("multiply", self, as_node(other))

    def __rmul__(self, other):
        from sdf_operations import Operator
        return Operator("multiply", as_node(other), self)

    def __truediv__(self, other):
        from sdf_operations import Operator
        return Operator("divide", self, as_node(other))

    def __rtruediv__(self, other):
        from sdf_operations import Operator
        return Operator("divide", as_node(other), self)

    def __or__(self, other):
        from sdf_operations import Operator
        return Operator("union", self, as_node(other))

    def __and__(self, other):
        from sdf_operations import Operator
        return Operator("intersection", self, as_node(other))


class Field(SDFNode):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def to_expr(self, ctx):
        return str(self.value)

    def to_dict(self):
        return {"type": "Field", "value": self.value}

    def compute_bounds(self):
        from geometry import BoundingBox
        return BoundingBox.none()

    def compute_all_bounds(self):
        return super().compute_all_bounds()


class Constant(SDFNode):
    def __init__(self, value: float):
        self.value: float = value

    def to_expr(self, ctx):
        return str(ctx.format_number(self.value))

    def to_dict(self):
        return {"type": "Constant", "value": self.value}

    def compute_bounds(self):
        from geometry import BoundingBox
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