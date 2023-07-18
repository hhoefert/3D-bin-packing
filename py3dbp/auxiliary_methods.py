from decimal import Decimal

from packer.py3dbp.main import Item
from .constants import Axis


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


def getLimitNumberOfDecimals(number_of_decimals: int) -> Decimal:
    """Returns a Decimal object 1.0000... with number_of_decimals decimals.

    Args:
        number_of_decimals (int): The number of decimals.

    Returns:
        Decimal: The resulting decimal object
    """
    return Decimal('1.{}'.format('0' * number_of_decimals))


def set2Decimal(value, number_of_decimals: int = 0) -> Decimal:
    """Turns the value into a decimals with number_of_decimals decimals.

    Args:
        value (_type_): Any value that can be represented as decimal.
        number_of_decimals (int, optional): The number of decimals. Defaults to 0.

    Returns:
        Decimal: The resulting decimal object
    """
    lim_decimals = getLimitNumberOfDecimals(number_of_decimals)
    return Decimal(value).quantize(lim_decimals)
