"""
Paillier secret key that is shared amongst several parties.
"""

from __future__ import annotations

import sys
from typing import Any

# Check to see if the communication module is available
try:
    from tno.mpc.communication import RepetitionError, Serialization

    COMMUNICATION_INSTALLED = True
except ModuleNotFoundError:
    COMMUNICATION_INSTALLED = False
from tno.mpc.encryption_schemes.paillier.paillier import PaillierCiphertext
from tno.mpc.encryption_schemes.shamir import IntegerShares
from tno.mpc.encryption_schemes.templates import SecretKey, SerializationError
from tno.mpc.encryption_schemes.utils import mod_inv, pow_mod

from .utils import mult_list

if sys.version_info < (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class PaillierSharedKey(SecretKey):
    """
    Class containing relevant attributes and methods of a shared paillier key.
    """

    def __init__(
        self, n: int, t: int, player_id: int, share: IntegerShares, theta: int
    ) -> None:
        """
        Initializes a Paillier shared key.

        :param n: modulus of the DistributedPaillier scheme this secret key belongs to
        :param t: corruption_threshold of the secret sharing
        :param player_id: the index of the player to whom the key belongs
        :param share: secret sharing of the exponent used during decryption
        :param theta: Value used in the computation of a full decryption after partial decryptions
            have been obtained. We refer to the paper for more details
        """
        super().__init__()
        self.share = share
        self.n = n
        self.n_square = n * n
        self.t = t
        self.player_id = player_id
        self.theta = theta
        self.theta_inv = mod_inv(self.theta, self.n)

    def partial_decrypt(self, ciphertext: PaillierCiphertext) -> int:
        """
        Function that does local computations to get a partial decryption of a ciphertext.

        :param ciphertext: ciphertext to be partially decrypted
        :raise TypeError: If the given ciphertext is not of type PaillierCiphertext.
        :raise ValueError: If the ciphertext is encrypted against a different key.
        :return: partial decryption of ciphertext
        """

        if not isinstance(ciphertext, PaillierCiphertext):
            raise TypeError(
                f"Expected ciphertext to be a PaillierCiphertext not: {type(ciphertext)}"
            )

        if self.n != ciphertext.scheme.public_key.n:
            raise ValueError("encrypted against a different key!")
        ciphertext_value = ciphertext.get_value()
        n_fac = self.share.n_fac
        other_honest_players = [
            i + 1 for i in range(self.share.degree + 1) if i + 1 != self.player_id
        ]

        # NB: Here the reconstruction set is implicit defined, but any
        # large enough subset of shares will do.
        # reconstruction_shares = {key: shares[key] for key in list(shares.keys())[:degree + 1]}

        lagrange_interpol_enumerator = mult_list(other_honest_players)
        lagrange_interpol_denominator = mult_list(
            [(j - self.player_id) for j in other_honest_players]
        )
        exp = (
            n_fac * lagrange_interpol_enumerator * self.share.shares[self.player_id]
        ) // lagrange_interpol_denominator

        # Notice that the partial decryption is already raised to the power given
        # by the Lagrange interpolation coefficient
        if exp < 0:
            ciphertext_value = mod_inv(ciphertext_value, self.n_square)
            exp = -exp
        partial_decryption = pow_mod(ciphertext_value, exp, self.n_square)
        return partial_decryption

    def decrypt(self, partial_dict: dict[int, int]) -> int:
        r"""
        Function that uses partial decryptions of other parties to reconstruct a
        full decryption of the initial ciphertext.

        :param partial_dict: dictionary containing the partial decryptions of each party
        :raise ValueError: Either in case not enough shares are known in order to decrypt.
            Or when the combined decryption minus one is not divisible by $N$. This last case is
            most likely caused by the fact the ciphertext that is being decrypted,
            differs between parties.
        :return: full decryption
        """

        partial_decryptions = [
            partial_dict[i + 1] for i in range(self.share.degree + 1)
        ]

        if len(partial_decryptions) < self.share.degree + 1:
            raise ValueError("Not enough shares.")

        combined_decryption = (
            mult_list(partial_decryptions[: self.share.degree + 1]) % self.n_square
        )

        if (combined_decryption - 1) % self.n != 0:
            raise ValueError(
                "Combined decryption minus one is not divisible by N. This might be caused by the "
                "fact that the ciphertext that is being decrypted, differs between the parties."
            )

        message = ((combined_decryption - 1) // self.n * self.theta_inv) % self.n

        return message

    # region Serialization logic

    class SerializedPaillierSharedKey(TypedDict):
        """
        Serialized PaillierSharedKey for e.g. storing the key to disk.
        """

        n: int
        t: int
        player_id: int
        share: IntegerShares
        theta: int

    def serialize(
        self, **_kwargs: Any
    ) -> PaillierSharedKey.SerializedPaillierSharedKey:
        r"""
        Serialization function for public keys, which will be passed to the communication module.

        :param \**_kwargs: optional extra keyword arguments
        :raise SerializationError: When communication library is not installed.
        :return: serialized version of this PaillierSharedKey.
        """
        if not COMMUNICATION_INSTALLED:
            raise SerializationError()
        return {
            "n": self.n,
            "t": self.t,
            "player_id": self.player_id,
            "share": self.share,
            "theta": self.theta,
        }

    @staticmethod
    def deserialize(
        obj: PaillierSharedKey.SerializedPaillierSharedKey, **_kwargs: Any
    ) -> PaillierSharedKey:
        r"""
        Deserialization function for public keys, which will be passed to the communication module.

        :param obj: serialized version of a PaillierSharedKey.
        :param \**_kwargs: optional extra keyword arguments
        :raise SerializationError: When communication library is not installed.
        :return: Deserialized PaillierSharedKey from the given dict.
        """
        if not COMMUNICATION_INSTALLED:
            raise SerializationError()
        return PaillierSharedKey(
            n=obj["n"],
            t=obj["t"],
            player_id=obj["player_id"],
            share=obj["share"],
            theta=obj["theta"],
        )

    # endregion

    def __eq__(self, other: object) -> bool:
        """
        Compare this PaillierSharedKey with another to determine (in)equality.

        :param other: Object to compare this PaillierSharedKey with.
        :raise TypeError: When other object is not a PaillierSharedKey.
        :return: Boolean value representing (in)equality of both objects.
        """
        if not isinstance(other, PaillierSharedKey):
            raise TypeError(
                f"Expected comparison with another PaillierSharedKey, not {type(other)}"
            )
        return (
            self.share == other.share
            and self.n == other.n
            and self.t == other.t
            and self.player_id == other.player_id
            and self.theta == other.theta
        )

    def __str__(self) -> str:
        """
        Utility function to represent the local share of the private key as a string.

        :return: String representation of this private key part.
        """
        return str(
            {
                "priv_shared_key": {
                    "n": self.n,
                    "t": self.t,
                    "player_id": self.player_id,
                    "theta": self.theta,
                    "share": self.share,
                }
            }
        )


if COMMUNICATION_INSTALLED:
    try:
        Serialization.register_class(PaillierSharedKey)
    except RepetitionError:
        pass
