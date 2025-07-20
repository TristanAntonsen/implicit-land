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