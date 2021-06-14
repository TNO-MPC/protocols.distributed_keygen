"""
Useful functions for the distributed keygen module.
"""

from typing import List, Optional


def mult_list(list_: List[int], modulus: Optional[int] = None) -> int:
    """
    Utility function to multiply a list of numbers in a modular group

    :param list_: list of elements
    :param modulus: modulus to be applied
    :return: product of the elements in the list modulo the modulus
    """
    out = 1
    if modulus is None:
        for element in list_:
            out = out * element
    else:
        for element in list_:
            out = out * element % modulus
    return out
