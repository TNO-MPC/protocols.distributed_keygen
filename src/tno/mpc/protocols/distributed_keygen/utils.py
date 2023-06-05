"""
Useful functions for the distributed keygen module.
"""

import operator
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.shamir import ShamirSecretSharingScheme as Shamir
from tno.mpc.encryption_schemes.shamir import ShamirShares

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


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


class Variable:
    """
    This class represents a variable that is secret shared between
    parties and eases tracking the shares.

    A variable has a label and an owner. Each party needs to create an instance
    of Variable with the same label for each variable used in the computation.
    The label should uniquely identify the variable in the computation. The
    owner is the index of the party that owns the variable. The owner is the
    only party that can set the plaintext value of the variable (the party
    providing the input).

    It is possible for the variable to not have an owner (owner=-1) if no party
    knows the secret value (e.g. after a multiplication with another Variable).

    A Variable stores its value in two fields, _input and _sharing. The _input
    field stores the original plaintext value of this variable. This field can
    only be read and written if the party is the owner of the variable. The
    _sharing is the object used to store the shares this party has of the
    variable.
    """

    def __init__(self, label: str, owner: int = -1) -> None:
        """
        Create a new Variable with the given label and owner.

        :param label: label of this variable
        :param owner: index of the party that owns this variable
        """
        self.label = label
        self.owner = owner

        self._input: int | None = None
        self._sharing: Any = None

    def get_plaintext(self) -> int:
        """
        Return the original plaintext used as input to set this variable.
        Only the owner of the variable can know this value.

        :raises ValueError: if the plaintext value of this variable is not known
        :return: the plaintext value of this variable
        """
        if self._input is None:
            raise ValueError(
                "The plaintext input value of this variable not known. Either \
                the variable has not been set or you are not the owner of the \
                variable (you cannot know a secret you did not create without \
                reconstructing it)."
            )
        return self._input

    def set_plaintext(self, value: int) -> None:
        """
        Set the value of this variable to the given value. Only the owner of
        the variable can set the plaintext value.

        :param value: value to set this variable to
        """
        self._input = value

    def clone(self) -> Any:
        """
        Return a new instance of this class using the given instance.
        Essentially a shallow copy.

        :return: new instance of this class with the same label and owner, but
            no shares or plaintext value
        """
        return self.__class__(label=self.label, owner=self.owner)

    def __add__(self, other: Any) -> Any:
        """
        Add this variable with another variable. The other variable must be
        of the same type as this variable.

        :param other: variable to add with this variable
        :return: a new variable storing the result of adding the two variables
        """
        raise NotImplementedError

    def __mul__(self, other: Any) -> Any:
        """
        Multiply this variable with another variable. The other variable
        must be of the same type as this variable.

        :param other: variable to multiply with this variable
        :return: a new variable storing the result of multiplying the two variables
        """
        raise NotImplementedError

    def share(self, index: int) -> None:
        """
        Create and store a sharing of this variable. Only the owner of the
        variable can share if the _input field is set.

        :param index: index of this party
        """
        raise NotImplementedError

    def reconstruct(self) -> Any:
        """
        Reconstruct the secret value of this variable.

        :return: the reconstructed secret value stored in this variable
        """
        raise NotImplementedError

    def get_share(self, index: int) -> Any:
        """
        Return the share of party index.

        :param index: index of the party to get the share of. index must be in
            the range $[0, n-1]$ where $n$ is the number of parties.
        :return: the share of party index
        """
        raise NotImplementedError

    def set_share(self, index: int, share: Any) -> None:
        """
        Set the share of party index with the value share.

        :param index: index of the party to set the share of. index must be in
            the range $[0, n-1]$ where $n$ is the number of parties.
        :param share: the share to set for party index
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"Variable(label={self.label}, owner={self.owner}, shares={self._sharing})"
        )


class ShamirVariable(Variable):
    """
    Implementation of a secret-shared Variable using the Shamir Secret
    Sharing scheme.
    """

    def __init__(self, shamir: Shamir, label: str, owner: int = -1) -> None:
        super().__init__(label, owner)

        self.shamir_scheme: Shamir = shamir

        self._input: int | None = None
        self._sharing: ShamirShares = ShamirShares(shamir, {})

        self._index: int = -1  # Stored on .share(), used in __add__ and __mul__

    @override
    def clone(self) -> Any:
        """
        Return a new instance of this class using the given instance.
        Essentially a shallow copy.

        :return: new instance of this class with the same label and owner, but
            no shares or plaintext value
        """
        return self.__class__(
            shamir=self.shamir_scheme, label=self.label, owner=self.owner
        )

    @override
    def __add__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            raise ValueError(
                "Can only add a ShamirVariable with another ShamirVariable"
            )
        if len(self._sharing.shares.keys()) == 0:
            raise ValueError("Cannot add a variable that has not been shared")

        # In case this party is the owner of the variable, it has all shares.
        # We need to ensure that the owner of a variable only calculates with
        # its own share (otherwise ShamirShares class gets confused.)
        self_sharing = self._sharing
        if len(self._sharing.shares.keys()) > 1:
            self_sharing = ShamirShares(
                self.shamir_scheme, {self._index: self._sharing.shares[self._index]}
            )

        result = self.clone()
        result._sharing = self_sharing + other._sharing
        return result

    @override
    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            raise ValueError(
                "Can only multiply a ShamirVariable with another ShamirVariable"
            )
        if len(self._sharing.shares.keys()) == 0:
            raise ValueError("Cannot multiply a variable that has not been shared")

        # In case this party is the owner of the variable, it has all shares
        # We need to ensure that the owner of a variable only calculates with
        # its own share
        self_sharing = self._sharing
        if len(self._sharing.shares.keys()) > 1:
            self_sharing = ShamirShares(
                self.shamir_scheme, {self._index: self._sharing.shares[self._index]}
            )

        result_sharing = self_sharing * other._sharing

        result = self.clone()
        result._sharing = result_sharing
        result.shamir_scheme = result_sharing.scheme

        return result

    @override
    def share(self, index: int) -> None:
        if self.owner != index:
            raise ValueError("Only the owner of a variable can share it")
        if self._input is None:
            raise ValueError("Set the value of the variable before sharing it")

        self._sharing = self.shamir_scheme.share_secret(self._input)
        self._index = index

    @override
    def reconstruct(self) -> Any:
        """
        Reconstruct the secret value of this variable.
        See :meth:`ShamirShares.reconstruct_secret` for more details.

        :return: the reconstructed secret value stored in this variable
        """
        return self._sharing.reconstruct_secret()

    @override
    def get_share(self, index: int) -> Any:
        """
        Return the share of party index.

        :param index: index of the party to get the share of. index must be in
            the range $[0, n-1]$ where $n$ is the number of parties.
        :raise ValueError: if there is no share for party index
        :return: the share of party index
        """
        share = self._sharing.shares.get(index)
        if share is None:
            raise ValueError(
                f"There is no share for party {index} of variable {self.label}"
            )
        return share

    @override
    def set_share(self, index: int, share: Any) -> None:
        self._sharing.shares[index] = share

    def get_shares(self) -> Dict[int, int]:
        """
        Return all shares of this variable.

        :return: a dictionary mapping party indices to shares
        """
        return self._sharing.shares


class AdditiveVariable(Variable):
    """
    Simple additive secret sharing scheme.
    """

    def __init__(self, label: str, modulus: int, owner: int = -1) -> None:
        super().__init__(label, owner)
        self._modulus = modulus
        self._sharing: Dict[int, int] = {}

        self._index: int = -1  # Stored on .share(), used in __add__ and __mul__

    @override
    def clone(self) -> Any:
        """
        Return a new instance of this class using the given instance.
        Essentially a shallow copy.

        :return: new instance of this class with the same label and owner, but
            no shares or plaintext value
        """
        return self.__class__(label=self.label, owner=self.owner, modulus=self._modulus)

    @override
    def __add__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            raise ValueError(
                "Can only add a AdditiveVariable with another AdditiveVariable"
            )

        # In case this party is the owner of the variable, it has all shares.
        # We need to ensure that the owner of a variable only calculates with
        # its own share (otherwise ShamirShares class gets confused.)
        self_sharing = self._sharing
        if len(self._sharing.keys()) > 1:
            self_sharing = {self._index: self._sharing[self._index]}

        if self_sharing.keys() != other._sharing.keys():
            raise ValueError(
                "Can only add variables that have both been shared to the same parties"
            )

        result = self.clone()
        # Add the shares for each key
        result._sharing = Counter(self_sharing) + Counter(other._sharing)

        return result

    @override
    def __mul__(self, other: Any) -> Any:
        raise NotImplementedError("This scheme only supports addition.")

    @override
    def reconstruct(self) -> Any:
        """
        Reconstruct the secret value of this variable.
        See :meth:`ShamirShares.reconstruct_secret` for more details.

        :return: the reconstructed secret value stored in this variable
        """
        return sum(self._sharing.values()) % self._modulus

    @override
    def share(self, index: int) -> None:
        raise NotImplementedError

    @override
    def get_share(self, index: int) -> Any:
        """
        Return the share of party index.

        :param index: index of the party to get the share of. index must be in
            the range $[0, n-1]$ where $n$ is the number of parties.
        :return: the share of party index
        """
        return self._sharing[index]

    @override
    def set_share(self, index: int, share: Any) -> None:
        self._sharing[index] = share


V = TypeVar("V", bound=Variable)


class Batched(Generic[V], Variable):
    """
    A Batched Variable is a list of variables all representing copies of the
    same variable (with different values).

    This class allows one to easily set and get the shares of all variables in
    the batch, useful when sending and receiving messages about this batch.
    Furthermore, the class implements arithmetic operations for operating on
    batches.

    This class is useful if a calculation needs to be performed many times (e.g.
    try until succeed). The batch allows one to describe the computation as if
    one were using a single variable.
    """

    variables: List[V]
    batch_size: int

    def __init__(self, var: V, batch_size: int) -> None:
        """
        Create a Batched Variable from a single Variable.

        A list of 'batch_size' Variables is created, where each variable is
        instantiated as a shallow copy of the given 'var', which can be seen as
        the blueprint. Note this will only copy over the meta-data of the
        variable, not the value.

        :param var: Variable to batch
        :param batch_size: number of copies of the Variable
        """
        Variable.__init__(self, var.label, var.owner)

        self.variables: List[V] = [var.clone() for _ in range(batch_size)]
        self.batch_size = batch_size

    def set_plaintext(self, value: int) -> None:
        raise NotImplementedError("Please use set_plaintexts instead.")

    def set_plaintexts(self, values: List[int]) -> None:
        """
        Set the value of all variables in the batch. Each variable is set individually.

        :param values: list of values to set
        """
        for _, val in enumerate(values):
            self.variables[_].set_plaintext(val)

    @override
    def clone(self) -> Any:
        """
        Return a new instance of this class using the given instance.
        Essentially a shallow copy.

        :return: new instance of this class with the same label and owner, but
            no shares or plaintext value
        """
        return self.__class__(self.variables[0], self.batch_size)

    @override
    def __add__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            raise ValueError("Can only add a Batched with another Batched")

        result = self.clone()
        result.variables = list(map(operator.add, self.variables, other.variables))

        return result

    @override
    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            raise ValueError("Can only multiply a Batched with another Batched")

        result = self.clone()
        result.variables = list(map(operator.mul, self.variables, other.variables))

        return result

    @override
    def reconstruct(self) -> Any:
        """
        Reconstruct the secret value of this variable.

        :return: the reconstructed secret value stored in this variable
        """
        return [v.reconstruct() for v in self.variables]

    @override
    def share(self, index: int) -> None:
        for i in range(self.batch_size):
            self.variables[i].share(index)

    @override
    def get_share(self, index: int) -> List[Any]:
        """
        Return the share of party index.

        :param index: index of the party to get the share of. index must be in
            the range $[0, n-1]$ where $n$ is the number of parties.
        :return: the share of party index
        """
        return [var.get_share(index) for var in self.variables]

    @override
    def set_share(self, index: int, share: Any) -> None:
        for i, share_ in enumerate(share):
            self.variables[i].set_share(index, share_)

    def __getitem__(self, index: int) -> V:
        """
        Return the Variable at the given index.

        :param index: index of the Variable to return
        :return: Variable at the given index
        """
        return self.variables[index]


async def exchange_shares(
    group: List[V], index: int, pool: Pool, party_indices: Dict[str, int], msg_id: str
) -> None:
    """
    All parties send, for the variables in the group they own, to the other
    parties the shares intended for said party. Each party receives the shares
    intended for them.

    Note: This mutates the variables in the group.

    :param group: a list of variables to consider in this exchange
    :param index: index of this party
    :param pool: network of involved parties
    :param party_indices: mapping from party names to indices
    :param msg_id: Optional message id.
    :raises ValueError: if a variable is received with an unknown label
    """
    group_dict: Dict[str, V] = {v.label: v for v in group}

    # Send shares to other parties for the variables we own within the group
    other_parties = pool.pool_handlers.keys()
    for party in other_parties:
        message: Dict[str, List[Dict[str, str]]] = {"value": []}

        # Add all variables in the group that are owned by this party to the message
        for label, variable in group_dict.items():
            if variable.owner == index:
                message["value"].append(
                    {
                        "label": label,
                        "value": variable.get_share(party_indices[party]),
                    }
                )

        # Send the message to the party
        pool.asend(party, message, msg_id=msg_id)

    # Receive shares from other parties for the variables we don't own within the group
    messages = await pool.recv_all(msg_id=msg_id)
    for party, message in messages:
        for received_var in message["value"]:
            if received_var["label"] not in group_dict:
                raise ValueError(
                    f"Received a variable with unknown label {received_var['label']}"
                )

            group_dict[received_var["label"]].set_share(index, received_var["value"])


async def exchange_reconstruct(
    var: Variable, index: int, pool: Pool, party_indices: Dict[str, int], msg_id: str
) -> None:
    """
    Exchange shares of this variable with the other parties in the pool to
    allow all parties to locally reconstruct the variable.

    :param var: variable to exchange shares for
    :param index: index of this party
    :param pool: network of involved parties
    :param party_indices: mapping from party names to indices
    :param msg_id: Optional message id.
    """

    # Message containing our share of the secret
    message = {
        "label": var.label,
        "value": var.get_share(index),
    }

    # Send our share to all other parties
    other_parties = pool.pool_handlers.keys()
    for party in other_parties:
        pool.asend(party, message, msg_id=msg_id)

    # Gather the shares of the other parties
    messages = await pool.recv_all(msg_id=msg_id)
    for party, message in messages:
        var.set_share(party_indices[party], message["value"])


@dataclass
class Shares:
    r"""
    Shares contains all shares of this party.
    Every subclass contains an object for that element, such as $p$ or $q$.
    These objects contain up to two entries: "additive" and "shares",
    in "additive", the local additive share of that element is stored,
    in "shares", the shamir shares of the local additive share are stored.

    To support the batching of messages for compute_modulus, we store lists of
    $P$'s and $Q$'s.
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

    lambda_: "Shares.Lambda" = field(default_factory=Lambda)
    beta: "Shares.Beta" = field(default_factory=Beta)
    secret_key: "Shares.SecretKey" = field(default_factory=SecretKey)
