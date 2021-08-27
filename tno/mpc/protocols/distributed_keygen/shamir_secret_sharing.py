"""
Utility for Shamir secret sharing.
"""

from __future__ import annotations

import secrets
import warnings
from typing import Any, Dict, Union

from tno.mpc.encryption_schemes.utils import mod_inv

from .shamir_secret_sharing_integers import IntegerShares
from .utils import mult_list


class ShamirSecretSharingScheme:
    """
    Class with Shamir Secret sharing functionality.
    """

    def __init__(
        self, modulus: int, number_of_parties: int, polynomial_degree: int
    ) -> None:
        r"""
        Initialize a $t$-out-of-$n$ secret sharing scheme where

        - $t$ = \text{polynomial_degree} + 1$
        - $n$ = \text{number_of_parties}$

        Note that polynomial_degree is the same as the corruption threshold.

        :param modulus: prime modulus of the coefficients in the polynomials used to create shares
        :param number_of_parties: number of shares that need to be created for each sharing
        :param polynomial_degree: degree of the polynomials used to create shares
        """
        self.modulus = modulus
        self.number_of_parties = number_of_parties
        self.polynomial_degree = polynomial_degree

        # Vandermonde matrix for evaluation of polynomials at points [1,..,n].
        # This essentialy creates a matrix that precomputes i**j for all possible i**j that are
        # needed for the evaluation of sharing polynomials. We now have that i**j = Vm[i][j].
        # To evaluate a polynomial p(x) = a0 + a1 * x + ... + ad * x**d we can simply compute
        # a0 * Vm[x][0] + a1 * Vm[x][1] + ... + ad * Vm[x][d].
        self.van_der_monde = [
            [pow(i + 1, j, modulus) for j in range(polynomial_degree + 1)]
            for i in range(number_of_parties)
        ]

    def share_secret(self, secret: int) -> ShamirShares:
        """
        Function that creates shares of a value for each party.

        :param secret: secret to be shared
        :return: sharing of the secret
        """
        # Sample random polynomial of degree t with constant coefficient

        secret_poly = [secret] + [
            secrets.randbelow(self.modulus) for _ in range(self.polynomial_degree)
        ]
        # Create an array of all the shares
        # Player IDs are equal to the points of evaluation.
        shares = {
            ind
            + 1: sum(
                [
                    self.van_der_monde[ind][i] * secret_poly[i] % self.modulus
                    for i in range(self.polynomial_degree + 1)
                ]
            )
            for ind in range(self.number_of_parties)
        }
        sharing = ShamirShares(self, shares)
        return sharing

    def __eq__(self, other: object) -> bool:
        """
        Compare equality between this ShamirSecretSharingScheme and the other object.

        :param other: Object to compare with.
        :return: Boolean stating (in)equality
        """
        if isinstance(other, ShamirSecretSharingScheme):
            return (
                self.modulus == other.modulus
                and self.number_of_parties == other.number_of_parties
                and self.polynomial_degree == other.polynomial_degree
            )
        # else
        return False

    def serialize(self) -> Dict[str, Dict[str, int]]:
        """
        Serialization function

        :return: json object containing the necessary information to deserialize
        """

        return {
            "ShamirSecretSharingScheme": {
                "P": self.modulus,
                "n": self.number_of_parties,
                "t": self.polynomial_degree,
            }
        }


class ShamirShares:
    """
    Class that keeps track of the shares for a certain value
    """

    def __init__(
        self, shamir_sss: ShamirSecretSharingScheme, shares: Dict[int, int]
    ) -> None:
        self.scheme = shamir_sss
        self.shares = shares
        # The degree of the polynomial used for sharing the secret, i.e. at least degree+1 shares
        # are required to reconstruct.

        self.degree = self.scheme.polynomial_degree

    def __str__(self) -> str:
        """
        String formatted version of this ShamirShares object.

        :return: Pretty string.
        """
        if self.shares:
            text = "shares: "
            for share in self.shares.keys():
                text += str(self.shares[share]) + " "
            text += "degree: " + str(self.degree)
            return text
        # else
        return "None"

    def serialize(
        self,
    ) -> Dict[str, Dict[str, Union[int, Dict[int, int], Dict[str, Dict[str, int]]]]]:
        """
        Serialization function

        :return: json object containing the necessary information to deserialize
        """
        return {
            "ShamirShares": {
                "scheme": self.scheme.serialize(),
                "shares": self.shares,
                "degree": self.degree,
            }
        }

    def reconstruct_secret(self) -> int:
        """
        Function that uses the shares from other parties to reconstruct the secret

        :raise ValueError: In case not enough shares are known to reconstruct the secret.
        :return: original secret
        """
        if len(self.shares) < self.degree + 1:
            raise ValueError("Too little shares to reconstruct.")

        # We will use the first self.degree+1 shares to reconstruct. This can be any subset.
        # Hence, here the reconstruction set is implicitly defined.
        reconstruction_shares = {
            key: self.shares[key] for key in list(self.shares.keys())[: self.degree + 1]
        }

        # We precomputed some values so that the Lagrange interpolation.
        # We assume that we can always use shares f(1), ... f(self.degree+1)

        lagrange_interpol_enum = {
            i: mult_list(
                [j for j in reconstruction_shares.keys() if i != j], self.scheme.modulus
            )
            for i in reconstruction_shares.keys()
        }
        lagrange_interpol_denom = {
            i: mult_list(
                [(j - i) for j in reconstruction_shares.keys() if i != j],
                self.scheme.modulus,
            )
            for i in reconstruction_shares.keys()
        }

        secret = int(
            sum(
                lagrange_interpol_enum[i]
                * mod_inv(lagrange_interpol_denom[i], self.scheme.modulus)
                * reconstruction_shares[i]
                % self.scheme.modulus
                for i in reconstruction_shares.keys()
            )
            % self.scheme.modulus
        )
        return secret

    def __add__(self, other: ShamirShares) -> ShamirShares:
        """
        Add the shares belonging to the two given ShamirShares values together.

        :param other: Shares to be added to these shares.
        :raise ValueError: In case a different secret sharing scheme was used.
        :return: New ShamirShares object where the shares have been added together.
        """
        if self.scheme != other.scheme:
            raise ValueError(
                "Different secret sharing schemes have been used, i.e. shares are incompatible."
            )

        shares = {
            i: (self.shares[i] + other.shares[i]) % self.scheme.modulus
            for i in self.shares.keys()
        }
        return ShamirShares(self.scheme, shares)

    def __mul__(self, other: ShamirShares) -> ShamirShares:
        """
        Multiply the shares belonging to the two given ShamirShares values together. Only
        possible when both schemes are the same.

        :param other: Shares to be multiplied with these shares.
        :return: New ShamirShares object where the shares have been multiplied together.
        """
        if self.scheme != other.scheme:
            # If self is multiplied (from the right) by another object, we redirect to the __rmul__
            # method of that object. Only implemented when other is a shamir secret sharing over
            # the integers. In this case all shares are reduced modulo the Shamir modulus and a
            # shamir sharing is returned.
            return NotImplemented

        shares = {
            i: (self.shares[i] * other.shares[i]) % self.scheme.modulus
            for i in self.shares.keys()
        }
        mult_scheme = ShamirSecretSharingScheme(
            self.scheme.modulus,
            self.scheme.number_of_parties,
            self.scheme.polynomial_degree + other.scheme.polynomial_degree,
        )
        return ShamirShares(mult_scheme, shares)

    def __rmul__(self, other: Any) -> ShamirShares:
        """
        Multiply the shares belonging to this value with a given scalar integer or IntegerShares
        object.
        Note: This operation returns a Shamir sharing which inherits the statistical security
        of the integer sharing and should therefore only be used with caution.

        :param other: IntegerShares or scalar to be multiplied with these shares.
        :raise ValueError: raised when shares are incompatible.
        :return: New ShamirShares object where the shares have been multiplied together.
        """
        if isinstance(other, int):
            # Scalar multiplication from the left by an integer
            shares = {
                i: (other * self.shares[i]) % self.scheme.modulus
                for i in self.shares.keys()
            }
            return ShamirShares(self.scheme, shares)
        if isinstance(other, IntegerShares):
            # Multiply by a sharing over the integers and return a Shamir Sharing
            # NB: This operation returns a Shamir sharing which inherits the statistical security
            # of the integer sharing and should therefore only be used with caution.
            warnings.warn("Caution multiplying integer shares by shamir shares.")

            shares = {
                i: (
                    self.shares[i]
                    * other.shares[i]
                    * mod_inv(other.scaling, self.scheme.modulus)
                )
                % self.scheme.modulus
                for i in self.shares.keys()
            }
            mult_scheme = ShamirSecretSharingScheme(
                self.scheme.modulus,
                self.scheme.number_of_parties,
                self.scheme.polynomial_degree + other.scheme.polynomial_degree,
            )
            return ShamirShares(mult_scheme, shares)
        # else
        raise ValueError(
            "Different secret sharing schemes have been used, i.e. shares are incompatible."
        )
