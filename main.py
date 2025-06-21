class FormatContext:
    def __init__(self, language="default", float_format="default"):
        self.language = language
        self.float_format = float_format

    def format_number(self, value: float):

        if self.float_format == "explicit_decimal":
            return f"{value:.1f}"

        return str(value)


class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

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

    @staticmethod
    def union(a, b):
        return BoundingBox(
            Point(min(a.min_point.x, b.min_point.x), min(a.min_point.y, b.min_point.y)),
            Point(max(a.max_point.x, b.max_point.x), max(a.max_point.y, b.max_point.y)),
        )


def as_node(value):
    return value if isinstance(value, SDFNode) else Constant(value)


class SDFNode:
    def to_expr(self) -> str:
        raise NotImplementedError()

    def bounding_box(self) -> BoundingBox:
        raise NotImplementedError()

    def __add__(self, other):
        return Operator("add", self, as_node(other))

    def __radd__(self, other):
        return Operator("add", self, as_node(other))

    def __sub__(self, other):
        return Operator("sub", self, as_node(other))

    def __rsub__(self, other):
        return Operator("sub", self, as_node(other))

    def __mul__(self, other):
        return Operator("multiply", self, as_node(other))

    def __rmul__(self, other):
        return Operator("multiply", self, as_node(other))

    def __truediv__(self, other):
        return Operator("divide", self, as_node(other))

    def __rtruediv__(self, other):
        return Operator("divide", self, as_node(other))

    def __or__(self, other):
        return Operator("union", self, as_node(other))

    def __and__(self, other):
        return Operator("intersection", self, as_node(other))


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

    def to_expr(self, ctx: FormatContext):

        x = ctx.format_number(self.center.x)
        y = ctx.format_number(self.center.y)
        r = ctx.format_number(self.radius)

        return f"circle(p,vec2({x},{y}),{r})"


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

        return f"box(p,vec2({x},{y}),vec2({w},{h}))"


class Operator(SDFNode):
    def __init__(self, op: str, left: SDFNode, right: SDFNode):
        self.op = op
        self.left = as_node(left)
        self.right = as_node(right)

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
            case "subtraction":
                return f"max({left_expr},-{right_expr})"
            case _:
                return None


def union(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("union", a, b)


def subtraction(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("subtraction", a, b)


def intersection(a: SDFNode, b: SDFNode) -> SDFNode:
    return Operator("intersection", a, b)


if __name__ == "__main__":

    import renderer

    ctx = FormatContext(float_format="explicit_decimal")

    a = Circle(Point(0.0, 0.0), 0.5)
    b = Box(Point(-0.5, -0.5), 1.0, 1.0)

    c = union(a, b)
    final = c
    expr = final.to_expr(ctx)

    # print(expr)
    shader = open("renderer/src/compute_shader.wgsl").read()
    renderer.render_with_wgsl(shader)

    # import subprocess
    # subprocess.run("pbcopy", text=True, input=expr)
