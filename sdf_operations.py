import numpy as np
from geometry import Vector
from sdf_core import SDFNode, as_node


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
        left_bounds = self.left.compute_bounds()
        right_bounds = self.right.compute_bounds()
        return left_bounds.union(right_bounds)

    def compute_all_bounds(self):
        all_bounds = []

        all_bounds.extend(self.left.compute_all_bounds())
        all_bounds.extend(self.right.compute_all_bounds())
        all_bounds.append(self.compute_bounds())

        return all_bounds

    def eval_sdf(self, p):
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

    def to_expr(self, ctx):
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