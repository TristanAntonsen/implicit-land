import renderer
import numpy as np
from PIL import Image, ImageDraw
from geometry import BoundingBox, Point
from sdf_core import SDFNode
from sdf_primitives import Circle, Box, BoxSharp, Primitive
from utils import Color, FormatContext

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
    "normalize_bool": "false",
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

    def draw_bounds(self, sdf: SDFNode, show_none=False):
        none_bbox = BoundingBox.none()
        for bs in sdf.compute_all_bounds():
            if bs != none_bbox or show_none:
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

    def draw_text(
        self, text: str, p: Point, color: Color = Color.BLACK(), size: int = 36
    ):
        from PIL import ImageFont

        draw = ImageDraw.Draw(self.img)
        res = self.resolution
        hres = res / 2
        cx = p.x * res + hres
        cy = -p.y * res + hres
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size=size)  # macOS
        draw.text((cx, cy), text, fill=color.format_PIL(), font=font)

    def overlay_primitive(self, shape: Primitive, color: Color = Color.RED(), weight=2):
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
