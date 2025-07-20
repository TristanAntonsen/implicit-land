class FormatContext:
    def __init__(self, language="default", float_format="default"):
        self.language = language
        self.float_format = float_format

    def format_number(self, value: float):
        if self.float_format == "explicit_decimal":
            return f"{value:.4f}"
        return str(value)


def clamp(value, min_value, max_value):
    return min(max(value, min_value), max_value)


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