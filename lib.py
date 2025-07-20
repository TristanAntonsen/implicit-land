# Import all functionality from separated modules for backward compatibility

# Geometry classes
from geometry import Vector, Point, BoundingBox, interpolate

# Utility classes and functions
from utils import FormatContext, Color, clamp

# SDF core functionality
from sdf_core import SDFNode, Field, Constant, Primitive, as_node, EPSILON

# SDF primitive shapes
from sdf_primitives import Circle, Box, BoxSharp, Line, Capsule, Plane

# SDF operations
from sdf_operations import Operator, union, difference, intersection, round_union, round_difference, round_intersection, round_union_eval

# Canvas and rendering
from canvas import Canvas, TILE_SIZE, DEFAULT_SETTINGS

# Re-export everything for backward compatibility
__all__ = [
    'Vector', 'Point', 'BoundingBox', 'interpolate',
    'FormatContext', 'Color', 'clamp',
    'SDFNode', 'Field', 'Constant', 'Primitive', 'as_node', 'EPSILON',
    'Circle', 'Box', 'BoxSharp', 'Line', 'Capsule', 'Plane',
    'Operator', 'union', 'difference', 'intersection', 'round_union', 'round_difference', 'round_intersection', 'round_union_eval',
    'Canvas', 'TILE_SIZE', 'DEFAULT_SETTINGS'
]


if __name__ == "__main__":
    canvas = Canvas(16 * 96)

    c = Circle(Point(0.125, 0.125), 0.125)
    b = Box(Point(0, 0), 0.25, 0.25)
    result = c | b
    canvas.draw_sdf(result)
    canvas.img.save("output_image.png")
    canvas.img.show()