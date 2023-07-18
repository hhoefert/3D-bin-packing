from decimal import Decimal


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
