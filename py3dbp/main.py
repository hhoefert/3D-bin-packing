import copy
from decimal import Decimal
from typing import Literal

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
# required to plot a representation of Bin and contained items
from matplotlib.patches import Circle, Rectangle
from pydantic import BaseModel

from .auxiliary_methods import intersect, set2Decimal
from .constants import Axis, RotationType

DEFAULT_NUMBER_OF_DECIMALS = 0
START_POSITION = [0, 0, 0]


class Item(BaseModel):
    partno: str
    name: str
    typeof: Literal["cube", "cylinder"]
    width: Decimal
    height: Decimal
    depth: Decimal
    weight: Decimal
    level: Literal[1, 2, 3]  # Packing Priority level, choose 1-3
    loadbear: int
    # Upside down?
    _upside_down: bool
    color: str
    rotation_type: int = 0
    position: list[int] = START_POSITION
    number_of_decimals: int = DEFAULT_NUMBER_OF_DECIMALS

    @property
    def upside_down(self) -> bool:
        return self._upside_down if self.typeof == 'cube' else False

    def formatNumbers(self, number_of_decimals):
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.weight = set2Decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.weight,
            self.position, self.rotation_type, self.getVolume()
        )

    def getVolume(self):
        return set2Decimal(self.width * self.height * self.depth, self.number_of_decimals)

    def getMaxArea(self):
        a = sorted([self.width, self.height, self.depth], reverse=True) if self.upside_down else [
            self.width, self.height, self.depth]

        return set2Decimal(a[0] * a[1], self.number_of_decimals)

    def getDimension(self):
        rotation_types = {
            RotationType.RT_WHD: [self.width, self.height, self.depth],
            RotationType.RT_HWD: [self.height, self.width, self.depth],
            RotationType.RT_HDW: [self.height, self.depth, self.width],
            RotationType.RT_DHW: [self.depth, self.height, self.width],
            RotationType.RT_DWH: [self.depth, self.width, self.height],
            RotationType.RT_WDH: [self.width, self.depth, self.height]
        }

        return rotation_types.get(self.rotation_type, [])


class Bin(BaseModel):
    partno: str
    width: Decimal
    height: Decimal
    depth: Decimal
    max_weight: Decimal
    corner: int = 0
    items: list = []
    unfitted_items: list = []
    number_of_decimals: int = DEFAULT_NUMBER_OF_DECIMALS
    fix_point: bool = False
    check_stable: bool = False
    support_surface_ratio: float = 0
    put_type: int = 1
    gravity: list = []  # used to put gravity distribution

    @property
    def fit_items(self):
        return np.array([[0.0, float(self.width), 0.0, float(self.height), 0.0, 0.0]])

    def formatNumbers(self, number_of_decimals) -> None:
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.max_weight = set2Decimal(self.max_weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self) -> str:
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.max_weight,
            self.getVolume()
        )

    def getVolume(self) -> Decimal:
        return set2Decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )

    def getTotalWeight(self) -> Decimal:
        total_weight = sum([item.weight for item in self.items])
        return set2Decimal(total_weight, self.number_of_decimals)

    def putItem(self, item: Item, pivot: list[int], axis: Axis | None = None):
        ''' put item in bin TODO'''
        fit = False
        # save the items position and update its position to the pivot point
        valid_item_position = item.position
        item.position = pivot
        # get all relevant rotation options
        rotations = RotationType.ALL if item.upside_down else RotationType.Notupdown
        # check each rotation
        for i in rotations:
            item.rotation_type = i
            item_dimension = item.getDimension()

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

            # check if adding the item does not exceed the maximum allowed weight
            if self.getTotalWeight() + item.weight > self.max_weight:
                return False

            # fix point float prob
            if self.fix_point:
                [w, h, d] = item_dimension
                [x, y, z] = [float(pivot[0]), float(
                    pivot[1]), float(pivot[2])]

                # TODO why three times?
                for i in range(3):
                    # fix height
                    y = self.checkHeight(
                        [x, x+float(w), y, y+float(h), z, z+float(d)])
                    # fix width
                    x = self.checkWidth(
                        [x, x+float(w), y, y+float(h), z, z+float(d)])
                    # fix depth
                    z = self.checkDepth(
                        [x, x+float(w), y, y+float(h), z, z+float(d)])

                # check stability on item
                # The support ration is the ratio of the underlying items touching the newly placed item to the lower surface of the newly placed item
                # rule :
                # 1. Define a support ratio, if the ratio below the support surface does not exceed this ratio, compare the second rule.
                # 2. If there is no support under any vertices of the bottom of the item, then fit = False.
                if self.check_stable:
                    # Calculate the surface area of ​​item.
                    item_area_lower = int(
                        item_dimension[0] * item_dimension[1])
                    # Calculate the surface area of ​​the underlying support.
                    support_area_upper = 0
                    for item in [item for item in self.fit_items if z == item[5]]:
                        # Verify that the lower support surface area is greater than the upper support surface area * support_surface_ratio.
                        # TODO better way to calculate the intersecting area
                        area = len(set(range(int(x), int(x+int(w)))) & set(range(int(item[0]), int(item[1])))) * len(
                            set(range(int(y), int(y+int(h)))) & set(range(int(item[2]), int(item[3]))))
                        support_area_upper += area

                    # If not , get four vertices of the bottom of the item.
                    if support_area_upper / item_area_lower < self.support_surface_ratio:
                        four_vertices = [
                            [x, y], [x+float(w), y], [x, y+float(h)], [x+float(w), y+float(h)]]
                        #  If any vertices is not supported, fit = False.
                        c = [False, False, False, False]
                        for item in [item for item in self.fit_items if z == item[5]]:
                            for idx, vertex in enumerate(four_vertices):
                                if (item[0] <= vertex[0] <= item[1]) and (item[2] <= vertex[1] <= item[3]):
                                    c[idx] = True
                        if False in c:
                            item.position = valid_item_position
                            return False

                self.fit_items = np.append(self.fit_items, np.array(
                    [[x, x+float(w), y, y+float(h), z, z+float(d)]]), axis=0)
                item.position = [set2Decimal(
                    x), set2Decimal(y), set2Decimal(z)]

            if fit:
                self.items.append(copy.deepcopy(item))
            return fit

        else:
            item.position = valid_item_position

        return fit

    def checkDepth(self, unfix_point):
        ''' fix item position z '''
        z_ = [[0, 0], [float(self.depth), float(self.depth)]]
        for j in self.fit_items:
            # creat x set
            x_bottom = set([i for i in range(int(j[0]), int(j[1]))])
            x_top = set(
                [i for i in range(int(unfix_point[0]), int(unfix_point[1]))])
            # creat y set
            y_bottom = set([i for i in range(int(j[2]), int(j[3]))])
            y_top = set(
                [i for i in range(int(unfix_point[2]), int(unfix_point[3]))])
            # find intersection on x set and y set.
            if len(x_bottom & x_top) != 0 and len(y_bottom & y_top) != 0:
                z_.append([float(j[4]), float(j[5])])
        top_depth = unfix_point[5] - unfix_point[4]
        # find diff set on z_.
        z_ = sorted(z_, key=lambda z_: z_[1])
        for j in range(len(z_)-1):
            if z_[j+1][0] - z_[j][1] >= top_depth:
                return z_[j][1]
        return unfix_point[4]

    def checkWidth(self, unfix_point):
        ''' fix item position x '''
        x_ = [[0, 0], [float(self.width), float(self.width)]]
        for j in self.fit_items:
            # creat z set
            z_bottom = set([i for i in range(int(j[4]), int(j[5]))])
            z_top = set(
                [i for i in range(int(unfix_point[4]), int(unfix_point[5]))])
            # creat y set
            y_bottom = set([i for i in range(int(j[2]), int(j[3]))])
            y_top = set(
                [i for i in range(int(unfix_point[2]), int(unfix_point[3]))])
            # find intersection on z set and y set.
            if len(z_bottom & z_top) != 0 and len(y_bottom & y_top) != 0:
                x_.append([float(j[0]), float(j[1])])
        top_width = unfix_point[1] - unfix_point[0]
        # find diff set on x_bottom and x_top.
        x_ = sorted(x_, key=lambda x_: x_[1])
        for j in range(len(x_)-1):
            if x_[j+1][0] - x_[j][1] >= top_width:
                return x_[j][1]
        return unfix_point[0]

    def checkHeight(self, unfix_point):
        '''fix item position y '''
        y_ = [[0, 0], [float(self.height), float(self.height)]]
        for j in self.fit_items:
            # creat x set
            x_bottom = set([i for i in range(int(j[0]), int(j[1]))])
            x_top = set(
                [i for i in range(int(unfix_point[0]), int(unfix_point[1]))])
            # creat z set
            z_bottom = set([i for i in range(int(j[4]), int(j[5]))])
            z_top = set(
                [i for i in range(int(unfix_point[4]), int(unfix_point[5]))])
            # find intersection on x set and z set.
            if len(x_bottom & x_top) != 0 and len(z_bottom & z_top) != 0:
                y_.append([float(j[2]), float(j[3])])
        top_height = unfix_point[3] - unfix_point[2]
        # find diff set on y_bottom and y_top.
        y_ = sorted(y_, key=lambda y_: y_[1])
        for j in range(len(y_)-1):
            if y_[j+1][0] - y_[j][1] >= top_height:
                return y_[j][1]

        return unfix_point[2]

    def addCorner(self):
        '''add container coner '''
        if self.corner != 0:
            corner = set2Decimal(self.corner)
            corner_list = []
            for i in range(8):
                a = Item(
                    partno='corner{}'.format(i),
                    name='corner',
                    typeof='cube',
                    width=corner,
                    height=corner,
                    depth=corner,
                    weight=0,
                    level=0,
                    loadbear=0,
                    _upside_down=True,
                    color='#000000')

                corner_list.append(a)
            return corner_list

    def putCorner(self, info, item):
        '''put coner in bin '''
        fit = False
        x = set2Decimal(self.width - self.corner)
        y = set2Decimal(self.height - self.corner)
        z = set2Decimal(self.depth - self.corner)
        pos = [[0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0],
               [x, y, 0], [x, 0, 0], [x, 0, z], [x, y, z]]
        item.position = pos[info]
        self.items.append(item)

        corner = [float(item.position[0]), float(item.position[0])+float(self.corner), float(item.position[1]), float(
            item.position[1])+float(self.corner), float(item.position[2]), float(item.position[2])+float(self.corner)]

        self.fit_items = np.append(self.fit_items, np.array([corner]), axis=0)
        return

    def clearBin(self):
        ''' clear item which in bin '''
        self.items = []
        self.fit_items = np.array([[0, self.width, 0, self.height, 0, 0]])
        return


class Packer:

    def __init__(self):
        self.bins = []
        self.items = []
        self.unfit_items = []
        self.total_items = 0
        self.binding = []
        # self.apex = []

    def addBin(self, bin):
        return self.bins.append(bin)

    def addItem(self, item):
        self.total_items = len(self.items) + 1

        return self.items.append(item)

    def pack2Bin(self, bin, item, fix_point, check_stable, support_surface_ratio):
        ''' pack item to bin '''
        fitted = False
        bin.fix_point = fix_point
        bin.check_stable = check_stable
        bin.support_surface_ratio = support_surface_ratio

        # first put item on (0,0,0) , if corner exist ,first add corner in box.
        if bin.corner != 0 and not bin.items:
            corner_lst = bin.addCorner()
            for i in range(len(corner_lst)):
                bin.putCorner(i, corner_lst[i])

        elif not bin.items:
            response = bin.putItem(item, item.position)

            if not response:
                bin.unfitted_items.append(item)
            return

        for axis in range(0, 3):
            items_in_bin = bin.items
            for ib in items_in_bin:
                pivot = [0, 0, 0]
                w, h, d = ib.getDimension()
                if axis == Axis.WIDTH:
                    pivot = [ib.position[0] + w,
                             ib.position[1], ib.position[2]]
                elif axis == Axis.HEIGHT:
                    pivot = [ib.position[0],
                             ib.position[1] + h, ib.position[2]]
                elif axis == Axis.DEPTH:
                    pivot = [ib.position[0],
                             ib.position[1], ib.position[2] + d]

                if bin.putItem(item, pivot, axis):
                    fitted = True
                    break
            if fitted:
                break
        if not fitted:
            bin.unfitted_items.append(item)

    def sortBinding(self, bin):
        ''' sorted by binding '''
        b, front, back = [], [], []
        for i in range(len(self.binding)):
            b.append([])
            for item in self.items:
                if item.name in self.binding[i]:
                    b[i].append(item)
                elif item.name not in self.binding:
                    if len(b[0]) == 0 and item not in front:
                        front.append(item)
                    elif item not in back and item not in front:
                        back.append(item)

        min_c = min([len(i) for i in b])

        sort_bind = []
        for i in range(min_c):
            for j in range(len(b)):
                sort_bind.append(b[j][i])

        for i in b:
            for j in i:
                if j not in sort_bind:
                    self.unfit_items.append(j)

        self.items = front + sort_bind + back
        return

    def putOrder(self):
        '''Arrange the order of items '''
        r = []
        for i in self.bins:
            # open top container
            if i.put_type == 2:
                i.items.sort(key=lambda item: item.position[0], reverse=False)
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
            # general container
            elif i.put_type == 1:
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
                i.items.sort(key=lambda item: item.position[0], reverse=False)
            else:
                pass
        return

    def gravityCenter_old(self, bin):
        ''' 
        Deviation Of Cargo gravity distribution
        '''
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

    def gravityCenter(self, bin):
        ''' 
        Deviation Of Cargo gravity distribution
        '''
        w = int(bin.width)
        h = int(bin.height)
        d = int(bin.depth)

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
                    y = len(y_set & area[j][1]) / (y_ed - y_st) * weight
                    area[j][2] += y
                    if j >= 2:
                        area[j - 2][2] += (weight - x)
                    else:
                        area[j + 2][2] += (weight - y)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) and len(x_set & area[j][0]) != 0:
                    x = len(x_set & area[j][0]) / (x_ed - x_st) * weight
                    area[j][2] += x
                    if j >= 2:
                        area[j - 2][2] += (weight - x)
                    else:
                        area[j + 2][2] += (weight - x)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == False and len(
                        y_set & area[j][1]) != 0 and len(x_set & area[j][0]) != 0:
                    all = (y_ed - y_st) * (x_ed - x_st)
                    y = len(y_set & area[0][1])
                    y_2 = y_ed - y_st - y
                    x = len(x_set & area[0][0])
                    x_2 = x_ed - x_st - x
                    area[0][2] += x * y / all * weight
                    area[1][2] += x_2 * y / all * weight
                    area[2][2] += x * y_2 / all * weight
                    area[3][2] += x_2 * y_2 / all * weight
                    break

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

            x_set = set(range(x_st, int(x_ed) + 1))
            y_set = set(range(y_st, y_ed + 1))

            calculate_weight(x_set, y_set, int(i.weight))

        r = [area[0][2], area[1][2], area[2][2], area[3][2]]
        result = [round(i / sum(r) * 100, 2) for i in r]
        return result

    def pack(self, bigger_first=False, distribute_items=True, fix_point=True, check_stable=True, support_surface_ratio=0.75, binding=[], number_of_decimals=DEFAULT_NUMBER_OF_DECIMALS):
        '''pack master func '''
        # set decimals
        for bin in self.bins:
            bin.formatNumbers(number_of_decimals)

        for item in self.items:
            item.formatNumbers(number_of_decimals)
        # add binding attribute
        self.binding = binding
        # Bin : sorted by volumn
        self.bins.sort(key=lambda bin: bin.getVolume(), reverse=bigger_first)
        # Item : sorted by volumn -> sorted by loadbear -> sorted by level -> binding
        self.items.sort(key=lambda item: item.getVolume(),
                        reverse=bigger_first)
        # self.items.sort(key=lambda item: item.getMaxArea(), reverse=bigger_first)
        self.items.sort(key=lambda item: item.loadbear, reverse=True)
        self.items.sort(key=lambda item: item.level, reverse=False)
        # sorted by binding
        if binding != []:
            self.sortBinding(bin)

        for idx, bin in enumerate(self.bins):
            # pack item to bin
            for item in self.items:
                self.pack2Bin(bin, item, fix_point, check_stable,
                              support_surface_ratio)

            if binding != []:
                # resorted
                self.items.sort(
                    key=lambda item: item.getVolume(), reverse=bigger_first)
                self.items.sort(key=lambda item: item.loadbear, reverse=True)
                self.items.sort(key=lambda item: item.level, reverse=False)
                # clear bin
                bin.items = []
                bin.unfitted_items = self.unfit_items
                bin.fit_items = np.array([[0, bin.width, 0, bin.height, 0, 0]])
                # repacking
                for item in self.items:
                    self.pack2Bin(bin, item, fix_point,
                                  check_stable, support_surface_ratio)

            # Deviation Of Cargo Gravity Center
            self.bins[idx].gravity = self.gravityCenter(bin)

            if distribute_items:
                for bitem in bin.items:
                    no = bitem.partno
                    for item in self.items:
                        if item.partno == no:
                            self.items.remove(item)
                            break

        # put order of items
        self.putOrder()

        if self.items != []:
            self.unfit_items = copy.deepcopy(self.items)
            self.items = []
        # for item in self.items.copy():
        #     if item in bin.unfitted_items:
        #         self.items.remove(item)


class Painter:

    def __init__(self, bins):
        self.items = bins.items
        self.width = bins.width
        self.height = bins.height
        self.depth = bins.depth

    def _plotCube(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        """ Auxiliary function to plot a cube. code taken somewhere from the web.  """
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]

        kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text != "":
                ax.text((x + dx/2), (y + dy/2), (z + dz/2), str(text),
                        color='black', fontsize=fontsize, ha='center', va='center')

            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")

    def _plotCylinder(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, text="", fontsize=10, alpha=0.2):
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
                    color='black', fontsize=fontsize, ha='center', va='center')

    def plotBoxAndItems(self, title="", alpha=0.2, write_num=False, fontsize=10):
        """ side effective. Plot the Bin and the items it contains. """
        fig = plt.figure()
        axGlob = plt.axes(projection='3d')

        # plot bin
        self._plotCube(axGlob, 0, 0, 0, float(self.width), float(self.height), float(
            self.depth), color='black', mode=1, linewidth=2, text="")

        counter = 0
        # fit rotation type
        for item in self.items:
            rt = item.rotation_type
            x, y, z = item.position
            [w, h, d] = item.getDimension()
            color = item.color
            text = item.partno if write_num else ""

            if item.typeof == 'cube':
                # plot item of cube
                self._plotCube(axGlob, float(x), float(y), float(z), float(w), float(h), float(
                    d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.typeof == 'cylinder':
                # plot item of cylinder
                self._plotCylinder(axGlob, float(x), float(y), float(z), float(w), float(
                    h), float(d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)

            counter = counter + 1

        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt

    def setAxesEqual(self, ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().'''
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
