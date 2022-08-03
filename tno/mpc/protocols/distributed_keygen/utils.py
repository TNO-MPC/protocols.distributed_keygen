"""
Useful functions for the distributed keygen module.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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


@dataclass
class Shares:
    r"""
    Shares contains all shares of this party.
    Every subclass contains an object for that element, such as $p$ or $q$.
    These objects contain up to two entries: "additive" and "shares",
    in "additive", the local additive share of that element is stored,
    in "shares", the shamir shares of the local additive share are stored.
    """

    @dataclass
    class P:
        r"""
        Shares of $p$.
        """
        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class Q:
        r"""
        Shares of $q$.
        """
        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class N:
        r"""
        Shares of $n$.
        """
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class Biprime:
        """
        Shares of the used biprime.
        """

        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class V:
        r"""
        Shares of $v$.
        """
        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class Lambda:
        r"""
        Shares of $\lambda$.
        """
        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class Beta:
        r"""
        Shares of $\beta$.
        """
        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class SecretKey:
        """
        Shares of the secret key.
        """

        additive: int = 0
        shares: Dict[int, int] = field(default_factory=dict)

    p: "Shares.P" = field(default_factory=P)
    q: "Shares.Q" = field(default_factory=Q)
    n: "Shares.N" = field(default_factory=N)
    biprime: "Shares.Biprime" = field(default_factory=Biprime)
    v: "Shares.V" = field(default_factory=V)
    lambda_: "Shares.Lambda" = field(default_factory=Lambda)
    beta: "Shares.Beta" = field(default_factory=Beta)
    secret_key: "Shares.SecretKey" = field(default_factory=SecretKey)
