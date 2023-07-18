import copy
from typing import Literal

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib.patches import Circle, Rectangle
from pydantic import BaseModel

from .auxiliary_methods import set2Decimal
from .constants import Axis, RotationType

DEFAULT_NUMBER_OF_DECIMALS = 0
START_POSITION = [0, 0, 0]


class Item(BaseModel):
    partno: str
    name: str
    typeof: Literal["cube", "cylinder"]
    width: float
    height: float
    depth: float
    weight: float
    level: int  # Packing Priority level, choose 1-3
    loadbear: int
    # Upside down?
    upside_down_: bool
    color: str
    rotation_type: int = 0
    position: list[int] = START_POSITION
    number_of_decimals: int = DEFAULT_NUMBER_OF_DECIMALS

    @property
    def upside_down(self) -> bool:
        return self.upside_down_ if self.typeof == "cube" else False

    @upside_down.setter
    def upside_down(self, value: bool) -> None:
        self.upside_down_ = value

    def string(self):
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.weight,
            self.position, self.rotation_type, self.get_volume()
        )

    def get_volume(self):
        return self.width * self.height * self.depth

    def get_max_area(self):
        a = sorted([self.width, self.height, self.depth], reverse=True) if self.upside_down else [
            self.width, self.height, self.depth]
        return a[0] * a[1]

    def get_dimension(self):
        rotation_types = {
            RotationType.RT_WHD: [self.width, self.height, self.depth],
            RotationType.RT_HWD: [self.height, self.width, self.depth],
            RotationType.RT_HDW: [self.height, self.depth, self.width],
            RotationType.RT_DHW: [self.depth, self.height, self.width],
            RotationType.RT_DWH: [self.depth, self.width, self.height],
            RotationType.RT_WDH: [self.width, self.depth, self.height]
        }

        return rotation_types.get(self.rotation_type, [])


def rectIntersect(item1: Item, item2: Item, x: int, y: int) -> bool:
    """
    Check if two rectangles represented by `item1` and `item2` intersect.

    Args:
        item1 (Item): The first rectangle item.
        item2 (Item): The second rectangle item.
        x (int): The index representing the x-axis dimension in the `position` attribute of the items.
        y (int): The index representing the y-axis dimension in the `position` attribute of the items.

    Returns:
        bool: True if the rectangles intersect, False otherwise.
    """
    # get item dimensions
    dim1 = item1.get_dimension()
    dim2 = item2.get_dimension()

    # get the center of the x and the y edge of the rectangle
    center_x1 = item1.position[x] + dim1[x]/2
    center_y1 = item1.position[y] + dim1[y]/2
    center_x2 = item2.position[x] + dim2[x]/2
    center_y2 = item2.position[y] + dim2[y]/2

    # calculate the distance
    intersect_x = abs(center_x1 - center_x2)
    intersect_y = abs(center_y1 - center_y2)

    # check if the distance is greater than the x or y dimensions of the items added and divided by 2
    return intersect_x < (dim1[x]+dim2[x]) / 2 and intersect_y < (dim1[y]+dim2[y]) / 2


def intersect(item1: Item, item2: Item) -> bool:
    """Check if two cuboids intersect.

    Args:
        item1 (Item): Cuboid 1.
        item2 (Item): Cuboid 2.

    Returns:
        bool: True if the cuboids intersect, False otherwise.
    """
    return (
        rectIntersect(item1, item2, Axis.WIDTH, Axis.HEIGHT) and  # noqa
        rectIntersect(item1, item2, Axis.HEIGHT, Axis.DEPTH) and  # noqa
        rectIntersect(item1, item2, Axis.WIDTH, Axis.DEPTH)
    )


class Bin(BaseModel):
    partno: str
    width: float
    height: float
    depth: float
    max_weight: float
    corner: int = 0
    items: list[Item] = []
    fit_items: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    unfitted_items: list = []
    number_of_decimals: int = DEFAULT_NUMBER_OF_DECIMALS
    fix_point: bool = False
    check_stable: bool = False
    support_surface_ratio: float = 0
    bin_type: int = 1  # 1 for a general container, 2 for an top open container
    gravity: list = []  # used to put gravity distribution

    # TODO
    # @property
    # def fit_items(self):
    #     return np.array([[0.0, float(self.width), 0.0, float(self.height), 0.0, 0.0]])

    def string(self) -> str:
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.max_weight,
            self.get_volume()
        )

    def get_volume(self) -> float:
        return self.width * self.height * self.depth

    def get_total_weight(self) -> float:
        return sum([item.weight for item in self.items])

    def put_item(self, item: Item, pivot: list[int]) -> bool:
        """ put item in bin TODO"""
        fit = False
        # save the items position and update its position to the pivot point
        valid_item_position = item.position
        item.position = pivot
        # get all relevant rotation options
        rotations = RotationType.ALL if item.upside_down else RotationType.Notupdown
        # check each rotation
        for i in rotations:
            item.rotation_type = i
            item_dimension = item.get_dimension()

            # rotate
            if (
                self.width < pivot[0] + item_dimension[0] or
                self.height < pivot[1] + item_dimension[1] or
                self.depth < pivot[2] + item_dimension[2]
            ):
                continue

            fit = True
            for current_item_in_bin in self.items:
                # if any item intersects with the newly placed item at the pivot point, the item does not fit
                if intersect(current_item_in_bin, item):
                    fit = False
                    break

            if not fit:
                item.position = valid_item_position
                return False

            # TODO position reset?
            # check if adding the item does not exceed the maximum allowed weight
            if self.get_total_weight() + item.weight > self.max_weight:
                return False

            # fix point float prob
            if self.fix_point:
                fit = self.check_fixpoint(item, item_dimension, pivot)

            if fit:
                self.items.append(copy.deepcopy(item))
            else:
                item.position = valid_item_position
            return fit
        return fit

    def check_fixpoint(self, item, item_dimension, pivot) -> bool:
        [w, h, d] = item_dimension
        [x, y, z] = [float(pivot[0]), float(pivot[1]), float(pivot[2])]

        # TODO why three times?
        for i in range(3):
            # fix height
            y = self.check_height(
                [x, (x + float(w)), y, (y + float(h)), z, (z + float(d))])
            # fix width
            x = self.check_width(
                [x, (x + float(w)), y, (y + float(h)), z, (z + float(d))])
            # fix depth
            z = self.check_depth(
                [x, (x + float(w)), y, (y + float(h)), z, (z + float(d))])

        # check stability on item
        # The support ratio is the ratio of the underlying items touching the newly placed item to the lower surface of the newly placed item
        # rule:
        # 1. Define a support ratio, if the ratio below the support surface does not exceed this ratio, compare the second rule.
        # 2. If there is no support under any vertices of the bottom of the item, then fit = False.
        if self.check_stable:
            # Calculate the surface area of ​​item (W * D).
            item_area_lower = int(item_dimension[0] * item_dimension[2])
            # Calculate the surface area of ​​the underlying items, supporting the new one.
            support_area_upper = 0
            # check all items that have such a height, that is the same height as the pivot point
            for i in [i for i in self.fit_items if y == i[3]]:
                # calculate the intersecting area of boxes with the right dimension and the bottom surface of the new item
                support_area_upper += self.calculate_rect_intersect(
                    [x, x + w, z, z + d], [i[0], i[1], i[4], i[5]])

            # Verify that the lower support surface area is greater than the upper support surface area * support_surface_ratio.
            if support_area_upper / item_area_lower < self.support_surface_ratio:
                # If not, get four vertices of the bottom of the item and check whether the vertices are supported
                bottom_vertices = [
                    [x, z], [x+float(w), z], [x, z+float(d)], [x+float(w), z+float(d)]]
                # If any vertex is not supported, fit = False.
                c = [False, False, False, False]
                for i in [i for i in self.fit_items if y == i[3]]:
                    for idx, vertex in enumerate(bottom_vertices):
                        if (i[0] <= vertex[0] <= i[1]) and (i[2] <= vertex[1] <= i[3]):
                            c[idx] = True
                if False in c:
                    return False

        # add the item with the correct dimensions to the fitted items
        self.fit_items.append([x, x+float(w), y, y+float(h), z, z+float(d)])
        item.position = [x, y, z]
        return True

    def calculate_rect_intersect(self, rect1: list[float], rect2: list[float]) -> float:
        """
        Calculate the intersecting area of two rectangles.

        Args:
            rect1 (list): Coordinates of the first rectangle in the format [x1, y1, x2, y2].
            rect2 (list): Coordinates of the second rectangle in the format [x1, y1, x2, y2].

        Returns:
            float: The intersecting area of the two rectangles. Returns 0 if there is no intersection.
        """

        # Get the coordinates of the rectangles
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2

        # Calculate the coordinates of the intersection rectangle
        x_left: float = max(x1, x3)
        y_top: float = max(y1, y3)
        x_right: float = min(x2, x4)
        y_bottom: float = min(y2, y4)

        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0

        # Calculate the intersecting area
        intersecting_area: float = (x_right - x_left) * (y_bottom - y_top)
        return intersecting_area

    def check_depth(self, not_fixed_item: list[float]) -> float:
        """Takes an item and tries to find the point minimizing the gap in the z-dimension (depth). The method checks if the item can be placed deeper than the current pivot point suggests.

        Args:
            not_fixed_item (list[float]): The item, if placed at the current pivot point.

        Returns:
            float: Fixed z dimension, to updated the pivot point
        """

        # Initialize z-axis ranges with bottom and top limits
        z_dim_points = [[0, 0], [float(self.depth), float(self.depth)]]

        # Iterate over each item in fit_items
        for item in self.fit_items:
            # check if there is an intersection in the rectangle if not considering depth
            item_rect = [item[0], item[1], item[2], item[3]]
            not_fixed_rect = [not_fixed_item[0], not_fixed_item[1],
                              not_fixed_item[2], not_fixed_item[3]]
            area = self.calculate_rect_intersect(item_rect, not_fixed_rect)

            # Append z-axis points of the item to the list if there is an intersection
            if area > 0:
                z_dim_points.append([float(item[4]), float(item[5])])

        # Calculate the max depth of the not_fixed_item range
        max_depth = not_fixed_item[5] - not_fixed_item[4]

        # Sort z-axis ranges based on the upper limit
        sorted_z_dim_points = sorted(
            z_dim_points, key=lambda z_range: z_range[1])

        # Iterate over sorted z-axis ranges to find suitable space for fixing the item
        for i in range(len(sorted_z_dim_points) - 1):
            current_range = sorted_z_dim_points[i]
            next_range = sorted_z_dim_points[i + 1]

            # Check if the space is not too big to push the pivot point to the edge in the z-dimension
            if next_range[0] - current_range[1] >= max_depth:
                # TODO why not current_range[0]?
                return current_range[1]

        # If no suitable space is found, fix the item at the initial lower limit of not_fixed_item range
        return not_fixed_item[4]

    def check_width(self, not_fixed_item: list[float]) -> float:
        """Takes an item and tries to find the point minimizing the gap in the x-dimension (width). The method checks if the item can be placed deeper than the current pivot point suggests.

        Args:
            not_fixed_item (list[float]): The item, if placed at the current pivot point.

        Returns:
            float: Fixed x dimension, to updated the pivot point
        """

        # Initialize x-axis ranges with bottom and top limits
        x_dim_points = [[0, 0], [float(self.width), float(self.width)]]

        # Iterate over each item in fit_items
        for item in self.fit_items:
            # check if there is an intersection in the rectangle if not considering width
            item_rect = [item[4], item[5], item[2], item[3]]
            not_fixed_rect = [not_fixed_item[4], not_fixed_item[5],
                              not_fixed_item[2], not_fixed_item[3]]
            area = self.calculate_rect_intersect(item_rect, not_fixed_rect)

            # Append x-axis points of the item to the list if there is an intersection
            if area > 0:
                x_dim_points.append([float(item[0]), float(item[1])])

        # Calculate the max width of the not_fixed_item range
        max_width = not_fixed_item[1] - not_fixed_item[0]

        # Sort x-axis ranges based on the upper limit
        sorted_x_dim_points = sorted(
            x_dim_points, key=lambda x_range: x_range[1])

        # Iterate over sorted x-axis ranges to find suitable space for fixing the item
        for i in range(len(sorted_x_dim_points) - 1):
            current_range = sorted_x_dim_points[i]
            next_range = sorted_x_dim_points[i + 1]

            # Check if the space is not too big to push the pivot point to the edge in the x-dimension
            if next_range[0] - current_range[1] >= max_width:
                # TODO why not current_range[0]?
                return current_range[1]

        # If no suitable space is found, fix the item at the initial lower limit of not_fixed_item range
        return not_fixed_item[0]

    def check_height(self, not_fixed_item: list[float]) -> float:
        """Takes an item and tries to find the point minimizing the gap in the y-dimension (height). The method checks if the item can be placed deeper than the current pivot point suggests.

        Args:
            not_fixed_item (list[float]): The item, if placed at the current pivot point.

        Returns:
            float: Fixed z dimension, to updated the pivot point
        """

        # Initialize y-axis ranges with bottom and top limits
        y_dim_points = [[0, 0], [float(self.height), float(self.height)]]

        # Iterate over each item in fit_items
        for item in self.fit_items:
            # check if there is an intersection in the rectangle if not considering height
            item_rect = [item[4], item[5], item[0], item[1]]
            not_fixed_rect = [not_fixed_item[4], not_fixed_item[5],
                              not_fixed_item[0], not_fixed_item[1]]
            area = self.calculate_rect_intersect(item_rect, not_fixed_rect)

            # Append y-axis points of the item to the list if there is an intersection
            if area > 0:
                y_dim_points.append([float(item[2]), float(item[3])])

        # Calculate the max height of the not_fixed_item range
        max_height = not_fixed_item[1] - not_fixed_item[0]

        # Sort y-axis ranges based on the upper limit
        sorted_y_dim_points = sorted(
            y_dim_points, key=lambda y_range: y_range[1])

        # Iterate over sorted y-axis ranges to find suitable space for fixing the item
        for i in range(len(sorted_y_dim_points) - 1):
            current_range = sorted_y_dim_points[i]
            next_range = sorted_y_dim_points[i + 1]

            # Check if the space is not too big to push the pivot point to the edge in the y-dimension
            if next_range[0] - current_range[1] >= max_height:
                # TODO why not current_range[0]?
                return current_range[1]

        # If no suitable space is found, fix the item at the initial lower limit of not_fixed_item range
        return not_fixed_item[2]

    def create_corners(self) -> list[Item]:
        """Add the corners of a container"""
        if self.corner != 0:
            corner = self.corner
            corner_list: list[Item] = list()
            for i in range(8):
                corner_list.append(Item(
                    partno="Corner{}".format(i),
                    name="Corner",
                    typeof="cube",
                    width=corner,
                    height=corner,
                    depth=corner,
                    weight=0,
                    level=0,
                    loadbear=0,
                    upside_down_=True,
                    color="#000000"))
            return corner_list
        return []

    # TODO check
    def put_corner(self, pos_idx: int, corner: Item) -> None:
        """put corner in bin"""
        x = self.width - self.corner
        y = self.height - self.corner
        z = self.depth - self.corner
        pos = [[0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0],
               [x, y, 0], [x, 0, 0], [x, 0, z], [x, y, z]]
        corner.position = pos[pos_idx]
        self.items.append(corner)

        corner_pos = [float(corner.position[0]), float(corner.position[0])+float(self.corner), float(corner.position[1]), float(
            corner.position[1])+float(self.corner), float(corner.position[2]), float(corner.position[2])+float(self.corner)]

        self.fit_items.append(corner_pos)

    def clear(self) -> None:
        """Clear the bin from all items"""
        self.items = []
        self.fit_items = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


class Packer(BaseModel):
    bins: list[Bin] = []
    items: list[Item] = []
    unfit_items: list[Item] = []
    total_items: int = 0
    binding: list[tuple] = []

    def add_bin(self, bin: Bin) -> None:
        self.bins.append(bin)

    def add_item(self, item: Item) -> None:
        self.total_items = len(self.items) + 1
        self.items.append(item)

    def pack_item_to_bin(self, bin: Bin, item: Item, fix_point: bool, check_stable: bool, support_surface_ratio: float) -> None:
        """Pack a given item into a given bin TODO docstring"""
        bin.fix_point = fix_point
        bin.check_stable = check_stable
        bin.support_surface_ratio = support_surface_ratio

        # if corner exists, first add corner in box.
        # put item on (0,0,0)
        if bin.corner != 0 and not bin.items:
            corner_lst = bin.create_corners()
            for i, corner in enumerate(corner_lst):
                bin.put_corner(i, corner)

        # add item at the start position if bin is empty
        elif not bin.items:
            success = bin.put_item(item, item.position)
            if not success:
                bin.unfitted_items.append(item)
            return

        # try each axis and according pivot point until the first one fits
        for axis in range(0, 3):
            for bin_item in bin.items:
                pivot = [0, 0, 0]
                item_x, item_y, item_z = bin_item.position
                w, h, d = bin_item.get_dimension()
                if axis == Axis.WIDTH:
                    pivot = [item_x + w,
                             item_y,
                             item_z]
                elif axis == Axis.HEIGHT:
                    pivot = [item_x,
                             item_y + h,
                             item_z]
                elif axis == Axis.DEPTH:
                    pivot = [item_x,
                             item_y,
                             item_z + d]

                if bin.put_item(item, pivot):
                    return
        # if the return has not been reached, the item does not fit in the bin and ist stored accordingly
        bin.unfitted_items.append(item)

    def sort_binding(self) -> None:
        """ sorted by binding """
        bindings: list[list[Item]] = []
        other: list[Item] = []
        for i in range(len(self.binding)):
            bindings.append([])
            for item in self.items:
                if item.name in self.binding[i]:
                    bindings[i].append(item)
                else:
                    other.append(item)

        bindings.sort(key=len)
        self.items = [b for binding in bindings for b in binding] + other

    def packing_order(self) -> None:
        """Sorts the items of each bin based on the bin type.

        Raises:
            RuntimeError: If the bin type provided is not specified.
        """
        for bin in self.bins:
            # general container
            if bin.bin_type == 1:
                bin.items.sort(key=lambda item: (
                    item.position[1], item.position[2], item.position[0]))
            # open top container
            elif bin.bin_type == 2:
                bin.items.sort(key=lambda item: (
                    item.position[0], item.position[1], item.position[2]))
            else:
                raise RuntimeError(
                    f"Item order cannot be determined for unspecified container type (Type {bin.bin_type})")

    def gravityCenter_old(self, bin):
        """Deviation Of Cargo gravity distribution"""
        w = int(bin.width)
        h = int(bin.height)
        d = int(bin.depth)

        area1 = [set(range(0, w//2+1)), set(range(0, h//2+1)), 0]
        area2 = [set(range(w//2+1, w+1)), set(range(0, h//2+1)), 0]
        area3 = [set(range(0, w//2+1)), set(range(h//2+1, h+1)), 0]
        area4 = [set(range(w//2+1, w+1)), set(range(h//2+1, h+1)), 0]
        area = [area1, area2, area3, area4]

        for i in bin.items:

            x_st = int(i.position[0])
            y_st = int(i.position[1])
            if i.rotation_type == 0:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 1:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 2:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.depth)
            elif i.rotation_type == 3:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 4:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 5:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.depth)

            x_set = set(range(x_st, int(x_ed)+1))
            y_set = set(range(y_st, y_ed+1))

            # cal gravity distribution
            for j in range(len(area)):
                if x_set.issubset(area[j][0]) and y_set.issubset(area[j][1]):
                    area[j][2] += int(i.weight)
                    break
                # include x and !include y
                elif x_set.issubset(area[j][0]) == True and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0:
                    y = len(y_set & area[j][1]) / (y_ed - y_st) * int(i.weight)
                    area[j][2] += y
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - x)
                    else:
                        area[j+2][2] += (int(i.weight) - y)
                    break
                # include y and !include x
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == True and len(x_set & area[j][0]) != 0:
                    x = len(x_set & area[j][0]) / (x_ed - x_st) * int(i.weight)
                    area[j][2] += x
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - x)
                    else:
                        area[j+2][2] += (int(i.weight) - x)
                    break
                # !include x and !include y
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0 and len(x_set & area[j][0]) != 0:
                    all = (y_ed - y_st) * (x_ed - x_st)
                    y = len(y_set & area[0][1])
                    y_2 = y_ed - y_st - y
                    x = len(x_set & area[0][0])
                    x_2 = x_ed - x_st - x
                    area[0][2] += x * y / all * int(i.weight)
                    area[1][2] += x_2 * y / all * int(i.weight)
                    area[2][2] += x * y_2 / all * int(i.weight)
                    area[3][2] += x_2 * y_2 / all * int(i.weight)
                    break

        r = [area[0][2], area[1][2], area[2][2], area[3][2]]
        result = []
        for i in r:
            result.append(round(i / sum(r) * 100, 2))
        return result

    # TODO refactor
    def gravityCenter(self, bin: Bin):
        """Deviation Of Cargo gravity distribution"""

        # get the sizes TODO change to self
        w = int(bin.width)
        h = int(bin.height)
        d = int(bin.depth)

        # divide the area into 4 chunks, TODO better way
        area = [[set(range(0, w // 2 + 1)), set(range(0, h // 2 + 1)), 0],
                [set(range(w // 2 + 1, w + 1)), set(range(0, h // 2 + 1)), 0],
                [set(range(0, w // 2 + 1)), set(range(h // 2 + 1, h + 1)), 0],
                [set(range(w // 2 + 1, w + 1)), set(range(h // 2 + 1, h + 1)), 0]]

        def calculate_weight(x_set, y_set, weight):
            for j in range(len(area)):
                if x_set.issubset(area[j][0]) and y_set.issubset(area[j][1]):
                    area[j][2] += weight
                    break
                elif x_set.issubset(area[j][0]) and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0:
                    y = len(y_set & area[j][1]) / (y_end - y_start) * weight
                    area[j][2] += y
                    if j >= 2:
                        area[j - 2][2] += (weight - x)
                    else:
                        area[j + 2][2] += (weight - y)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) and len(x_set & area[j][0]) != 0:
                    x = len(x_set & area[j][0]) / (x_end - x_start) * weight
                    area[j][2] += x
                    if j >= 2:
                        area[j - 2][2] += (weight - x)
                    else:
                        area[j + 2][2] += (weight - x)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == False and len(
                        y_set & area[j][1]) != 0 and len(x_set & area[j][0]) != 0:
                    all = (y_end - y_start) * (x_end - x_start)
                    y = len(y_set & area[0][1])
                    y_2 = y_end - y_start - y
                    x = len(x_set & area[0][0])
                    x_2 = x_end - x_start - x
                    area[0][2] += x * y / all * weight
                    area[1][2] += x_2 * y / all * weight
                    area[2][2] += x * y_2 / all * weight
                    area[3][2] += x_2 * y_2 / all * weight
                    break

        for i in bin.items:
            # get start and end positions, based on rotation
            x_start = int(i.position[0])
            y_start = int(i.position[1])
            if i.rotation_type == 0:
                x_end = int(i.position[0] + i.width)
                y_end = int(i.position[1] + i.height)
            elif i.rotation_type == 1:
                x_end = int(i.position[0] + i.height)
                y_end = int(i.position[1] + i.width)
            elif i.rotation_type == 2:
                x_end = int(i.position[0] + i.height)
                y_end = int(i.position[1] + i.depth)
            elif i.rotation_type == 3:
                x_end = int(i.position[0] + i.depth)
                y_end = int(i.position[1] + i.height)
            elif i.rotation_type == 4:
                x_end = int(i.position[0] + i.depth)
                y_end = int(i.position[1] + i.width)
            elif i.rotation_type == 5:
                x_end = int(i.position[0] + i.width)
                y_end = int(i.position[1] + i.depth)

            x_set = set(range(x_start, x_end + 1))
            y_set = set(range(y_start, y_end + 1))

            calculate_weight(x_set, y_set, int(i.weight))

        r = [area[0][2], area[1][2], area[2][2], area[3][2]]
        result = [round(i / sum(r) * 100, 2) for i in r]
        return result

    def pack(self, bigger_first: bool = False, distribute_items: bool = True, fix_point: bool = True, check_stable: bool = True, support_surface_ratio: float = 0.75, binding: list[tuple] = [], number_of_decimals: int = DEFAULT_NUMBER_OF_DECIMALS):
        """pack master func TODO docstring"""

        self.binding = binding
        # Bin : sorted by volume
        self.bins.sort(key=lambda bin: bin.get_volume(), reverse=bigger_first)

        # Item (TODO) : sorted by volume -> sorted by loadbear -> sorted by level -> binding
        self.items.sort(key=lambda item: item.get_volume(),
                        reverse=bigger_first)
        self.items.sort(key=lambda item: item.loadbear, reverse=True)
        self.items.sort(key=lambda item: item.level, reverse=False)

        # packing order of items
        self.packing_order()

        # sorted by binding
        if binding != []:
            self.sort_binding()

        for bin in self.bins:
            # TODO enforce binding, do not just sort
            # pack items to bin
            for item in self.items:
                self.pack_item_to_bin(bin=bin,
                                      item=item,
                                      fix_point=fix_point,
                                      check_stable=check_stable,
                                      support_surface_ratio=support_surface_ratio)

            # Deviation Of Cargo Gravity Center TODO
            # bin.gravity = self.gravityCenter(bin)

            if distribute_items:
                for item in bin.items:
                    self.items.remove(item)

        if self.items != []:
            self.unfit_items = copy.deepcopy(self.items)
            self.items = []


class Painter:

    def __init__(self, bins):
        self.items = bins.items
        self.width = bins.width
        self.height = bins.height
        self.depth = bins.depth

    def _plotCube(self, ax, x, y, z, dx, dy, dz, color="red", mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        """ Auxiliary function to plot a cube. code taken somewhere from the web.  """
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]

        kwargs = {"alpha": 1, "color": color, "linewidth": linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec="black", alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec="black", alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec="black", alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec="black", alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec="black", alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec="black", alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text != "":
                ax.text((x + dx/2), (y + dy/2), (z + dz/2), str(text),
                        color="black", fontsize=fontsize, ha="center", va="center")

            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")

    def _plotCylinder(self, ax, x, y, z, dx, dy, dz, color="red", mode=2, text="", fontsize=10, alpha=0.2):
        """ Auxiliary function to plot a Cylinder  """
        # plot the two circles above and below the cylinder
        p = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        p2 = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
        # plot a circle in the middle of the cylinder
        center_z = np.linspace(0, dz, 10)
        theta = np.linspace(0, 2*np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, center_z)
        x_grid = dx / 2 * np.cos(theta_grid) + x + dx / 2
        y_grid = dy / 2 * np.sin(theta_grid) + y + dy / 2
        z_grid = z_grid + z
        ax.plot_surface(x_grid, y_grid, z_grid, shade=False,
                        fc=color, alpha=alpha, color=color)
        if text != "":
            ax.text((x + dx/2), (y + dy/2), (z + dz/2), str(text),
                    color="black", fontsize=fontsize, ha="center", va="center")

    def plotBoxAndItems(self, title="", alpha=0.2, write_num=False, fontsize=10):
        """ side effective. Plot the Bin and the items it contains. """
        fig = plt.figure()
        axGlob = plt.axes(projection="3d")

        # plot bin
        self._plotCube(axGlob, 0, 0, 0, float(self.width), float(self.height), float(
            self.depth), color="black", mode=1, linewidth=2, text="")

        counter = 0
        # fit rotation type
        for item in self.items:
            rt = item.rotation_type
            x, y, z = item.position
            [w, h, d] = item.get_dimension()
            color = item.color
            text = item.partno if write_num else ""

            if item.typeof == "cube":
                # plot item of cube
                self._plotCube(axGlob, float(x), float(y), float(z), float(w), float(h), float(
                    d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.typeof == "cylinder":
                # plot item of cylinder
                self._plotCylinder(axGlob, float(x), float(y), float(z), float(w), float(
                    h), float(d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)

            counter = counter + 1

        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt

    def setAxesEqual(self, ax):
        """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib"s
        ax.set_aspect("equal") and ax.axis("equal") not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca()."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
