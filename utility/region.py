import copy
import itertools


class BoundingBox:
    """BoundingBox类, 区域遵循前闭后开的惯例."""

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @staticmethod
    def create_from_image(image):
        height, width = image.shape[:2]
        return BoundingBox(0, 0, width, height)

    @staticmethod
    def create_from_rect(x, y, width, height):
        return BoundingBox(x, y, x + width, y + height)

    def contains(self, *, point=None, bbox=None):
        x1, y1, x2, y2 = self.bbox
        if point is not None:
            return x1 <= point[0] < x2 and y1 <= point[1] < y2
        if isinstance(bbox, (tuple, list)):
            bbox = BoundingBox(*bbox)
        a1, b1, a2, b2 = bbox.bbox
        return x1 <= a1 <= a2 <= x2 and y1 <= b1 <= b2 <= y2

    def intersect(self, other):
        result = copy.copy(self)
        sx1, sy1, sx2, sy2 = self.bbox
        ox1, oy1, ox2, oy2 = other.bbox
        result.x1 = max(sx1, ox1)
        result.y1 = max(sy1, oy1)
        result.x2 = min(sx2, ox2)
        result.y2 = min(sy2, oy2)
        valid = result.x1 < result.x2 and result.y1 < result.y2
        return result if valid else None

    def iou(self, other):
        intersect = self.intersect(other)
        if intersect is None: return 0
        area = intersect.area
        return area / (self.area + other.area - area)

    def iomin(self, other):
        min_area = min(self.area, other.area)
        if min_area <= 0: return 0
        intersect = self.intersect(other)
        if intersect is None: return 0
        return intersect.area / min_area

    def scale(self, scale_x, scale_y=None):
        if scale_y is None: scale_y = scale_x
        result = copy.copy(self)
        result.x1 = self.x1 * scale_x
        result.x2 = self.x2 * scale_x
        result.y1 = self.y1 * scale_y
        result.y2 = self.y2 * scale_y
        return result

    def translate(self, offset_x, offset_y=None):
        if offset_y is None: offset_y = offset_x
        result = copy.copy(self)
        result.x1 = self.x1 + offset_x
        result.x2 = self.x2 + offset_x
        result.y1 = self.y1 + offset_y
        result.y2 = self.y2 + offset_y
        return result

    def expand(self, ratio_w, ratio_h=None):
        if ratio_h is None: ratio_h = ratio_w
        w = self.width * ratio_w
        h = self.height * ratio_h
        return self.pad(w, h)

    def pad(self, pad_left, pad_top=None, pad_right=None, pad_bottom=None):
        if pad_top is None: pad_top = pad_left
        if pad_right is None: pad_right = pad_left
        if pad_bottom is None: pad_bottom = pad_top
        x1, y1, x2, y2 = self.bbox
        result = copy.copy(self)
        result.x1 = x1 - pad_left
        result.x2 = x2 + pad_right
        result.y1 = y1 - pad_top
        result.y2 = y2 + pad_bottom
        return result

    def toint(self):
        result = copy.copy(self)
        result.x1 = int(round(self.x1))
        result.y1 = int(round(self.y1))
        result.x2 = int(round(self.x2))
        result.y2 = int(round(self.y2))
        return result

    @property
    def top_left(self):
        return min(self.x1, self.x2), min(self.y1, self.y2)

    @property
    def bottom_right(self):
        return max(self.x1, self.x2), max(self.y1, self.y2)

    @property
    def bbox(self):
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        return x1, y1, x2, y2

    @property
    def width(self):
        return abs(self.x1 - self.x2)

    @property
    def height(self):
        return abs(self.y1 - self.y2)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def aspect_ratio(self):
        return self.width / self.height if self.height else 0

    def __eq__(self, other):
        return self.bbox == other.bbox

    def __hash__(self):
        return hash(self.bbox)

    def __str__(self):
        x1, y1, x2, y2 = self.bbox
        return f"bbox: [{x1}, {y1}, {x2}, {y2}]"


class Region(BoundingBox):
    "Region类代表一个检测框, 保存文件时用json格式."

    def __init__(self, bbox=None, label=None, score=1.0):
        super().__init__(*(bbox or (0, 0, 0, 0)))
        self.label = label
        self.score = score

    def fromdict(self, value):
        self.x1, self.y1, self.x2, self.y2 = value["bbox"]
        self.label = value["label"]
        self.score = value["score"]
        return self

    def todict(self):
        return {"bbox": self.bbox, "label": self.label, "score": self.score}

    def __str__(self):
        x1, y1, x2, y2 = self.bbox
        bbox = f"[{x1}, {y1}, {x2}, {y2}]"
        return f"bbox: {bbox}, label: {self.label}, score: {self.score:.4f}"


def remove_overlap_regions(regions, max_iou=0.4):
    """移除overlap的区域."""

    if max_iou >= 1.0: return regions
    calc_iou = lambda pair: pair[0].iou(pair[1])
    while len(regions) >= 2:
        comb = itertools.combinations(regions, 2)
        best = max(comb, key=calc_iou)
        if calc_iou(best) < max_iou: break
        regions.remove(min(best, key=lambda x: x.score))
    return regions


class LabeledBoundingBox(BoundingBox):

    def __init__(self, bbox=None, label=None, score=1.0, mask=None, polygon=None):
        super().__init__(*(bbox or (0, 0, 0, 0)))
        self.label = label
        self.score = score
        self.mask = mask
        self.polygon = polygon

    def __str__(self):
        x1, y1, x2, y2 = self.bbox
        bbox = f"[{x1}, {y1}, {x2}, {y2}]"
        return f"bbox: {bbox}, label: {self.label}, score: {self.score:.4f}"

    def roi(self, input):
        x1, y1, x2, y2 = self.toint().bbox
        roi = input[y1:y2, x1:x2]
        return roi

    def mask_value(self,x,y):
        return self.mask[y][x]





if __name__ == "__main__":
    pass