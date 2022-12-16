"""
Code for a single player in the Paillier distributed key-generation protocol.
"""

from __future__ import annotations

import copy
import logging
import math
import secrets
import warnings
from dataclasses import asdict
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast, overload

import sympy
from typing_extensions import TypedDict

from tno.mpc.communication import RepetitionError, Serialization, SupportsSerialization
from tno.mpc.communication.httphandlers import HTTPClient
from tno.mpc.communication.pool import Pool
from tno.mpc.encryption_schemes.paillier import (
    Paillier,
    PaillierCiphertext,
    PaillierPublicKey,
    PaillierSecretKey,
    paillier,
)
from tno.mpc.encryption_schemes.shamir import IntegerShares
from tno.mpc.encryption_schemes.shamir import (
    ShamirSecretSharingIntegers as IntegerShamir,
)
from tno.mpc.encryption_schemes.shamir import ShamirSecretSharingScheme as Shamir
from tno.mpc.encryption_schemes.shamir import ShamirShares
from tno.mpc.encryption_schemes.templates.encryption_scheme import EncodedPlaintext
from tno.mpc.encryption_schemes.utils import pow_mod

from .paillier_shared_key import PaillierSharedKey
from .utils import Shares


class DistributedPaillier(Paillier, SupportsSerialization):
    """
    Class that acts as one of the parties involved in distributed Paillier secret key generation.
    The pool represents the network of parties involved in the key generation protocol.
    """

    default_key_length = 2048
    default_prime_threshold = 2000
    default_biprime_param = 40
    default_sec_shamir = 40
    default_corruption_threshold = 1
    _global_instances: Dict[int, Dict[int, "DistributedPaillier"]] = {}
    _local_instances: Dict[int, "DistributedPaillier"] = {}

    @classmethod
    async def from_security_parameter(  # type: ignore[override]
        cls,
        pool: Pool,
        corruption_threshold: int = default_corruption_threshold,
        key_length: int = default_key_length,
        prime_threshold: int = default_prime_threshold,
        correct_param_biprime: int = default_biprime_param,
        stat_sec_shamir: int = default_sec_shamir,
        distributed: bool = True,
        precision: int = 0,
    ) -> DistributedPaillier:
        r"""
        Function that takes security parameters related to secret sharing and Paillier and
        initiates a protocol to create a shared secret key between the parties in the provided
        pool.

        :param precision: precision of the fixed point encoding in Paillier
        :param pool: The network of involved parties
        :param corruption_threshold: Maximum number of allowed corruptions. We require for the
            number of parties in the pool and the corruption threshold that
            $$\text{number_of_parties} >= 2 * \text{corruption_threshold} + 1$$.
            This is because we need to multiply secret sharings that both use polynomials of
            degree corruption_threshold. The resulting secret sharing then becomes a polynomial
            of degree $2*\text{corruption_threshold}$ and it requires at least $2*text{corruption_threshold}+1$
            evaluation points to reconstruct the secret in that sharing.
        :param key_length: desired bit length of the modulus $N$
        :param prime_threshold: Upper bound on the number of prime numbers to check during
            primality tests
        :param correct_param_biprime: parameter that affects the certainty of the generated $N$
            to be the product of two primes
        :param stat_sec_shamir: security parameter for the Shamir secret sharing over the integers
        :param distributed: Whether the different parties are run on different python instances
        :param precision: precision (number of decimals) to ensure
        :raise ValueError: In case the number of parties $n$ and the corruption threshold $t$ do
            not satisfy that $n \geq 2*t + 1$
        :raise Exception: In case the parties agree on a session id that is already being used.
        :return: DistributedPaillier scheme containing a regular Paillier public key and a shared
            secret key.
        """
        (
            number_of_players,
            prime_length,
            prime_list,
            shamir_scheme,
            shares,
            other_parties,
        ) = cls.setup_input(pool, key_length, prime_threshold, corruption_threshold)
        index, party_indices, zero_share, session_id = await cls.setup_protocol(
            shamir_scheme, other_parties, pool
        )

        # check if number_of_parties >= 2 * corruption_threshold + 1
        if number_of_players < 2 * corruption_threshold + 1:
            raise ValueError(
                "For a secret sharing scheme that needs to do a homomorphic "
                f"multiplication, \nwhich is the case during distributed key generation "
                f"with Paillier,\nwe require for the number of parties n and the corruption "
                f"threshold t that n >= 2*t + 1.\n"
                f"The given pool contains {number_of_players} parties (n) and the given corruption "
                f"threshold (t) is {corruption_threshold}."
            )

        # generate keypair
        public_key, secret_key = await cls.generate_keypair(
            stat_sec_shamir,
            number_of_players,
            corruption_threshold,
            shares,
            index,
            zero_share,
            pool,
            prime_list,
            prime_length,
            party_indices,
            correct_param_biprime,
            shamir_scheme,
            session_id,
        )

        scheme = cls(
            public_key=public_key,
            secret_key=secret_key,
            precision=precision,
            pool=pool,
            index=index,
            party_indices=party_indices,
            shares=shares,
            session_id=session_id,
            distributed=distributed,
        )
        # We need to distinguish the case where the parties share a python instance and where they
        # are run in different python instances. If the same python instance is used, then we need
        # to save a different DistributedPaillier instance for each party. If different python
        # instances are used, then we have exactly one DistributedPaillier instance in the python
        # instance for that session.
        if distributed:
            if session_id in cls._local_instances:
                raise Exception(
                    "An already existing session ID is about to be overwritten. "
                    "This can only happen if multiple sessions are run within the same python "
                    "instance and one of those session has the same ID"
                )
            cls._local_instances[session_id] = scheme
        else:
            if index in cls._global_instances:
                if session_id in cls._global_instances[index]:
                    raise Exception(
                        "An already existing session ID is about to be overwritten. "
                        "This can only happen if multiple sessions are run within the same python "
                        "instance and one of those session has the same ID"
                    )
                cls._global_instances[index][session_id] = scheme
            else:
                cls._global_instances[index] = {session_id: scheme}

        if key_length < 1024:
            warnings.warn(
                f"The key length={key_length} is lower than the advised minimum of 1024."
            )

        return scheme

    def __init__(
        self,
        public_key: PaillierPublicKey,
        secret_key: PaillierSharedKey,
        precision: int,
        pool: Pool,
        index: int,
        party_indices: Dict[str, int],
        shares: Shares,
        session_id: int,
        distributed: bool,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a DistributedPaillier instance with a public Paillier key and a shared
        secret Paillier key.

        :param public_key: The Paillier public key
        :param secret_key: The shared secret Paillier key
        :param precision: The precision of the resulting scheme
        :param pool: The pool with connections of parties involved in the shared secret key
        :param index: The index of the party who owns this instance within the pool
        :param party_indices: Dictionary mapping parties in the pool to their indices
        :param shares: Data class that stores and keeps track of shares during decryption
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param distributed: Boolean value indicating whether the protocol that generated the keys
            for this DistributedPaillier scheme was run in different Python instances (True) or in a
            single python instance (False)
        :param kwargs: Any keyword arguments that are passed to the super __init__ function
        """
        super().__init__(
            public_key, cast(PaillierSecretKey, secret_key), precision, False, **kwargs
        )

        # these variables are necessary during decryption
        self.pool = pool
        self.index = index
        self.shares = shares
        self.party_indices = party_indices
        self.session_id = session_id
        self.distributed = distributed

    def __eq__(self, other: object) -> bool:
        """
        Compare this Distributed Paillier scheme with another to determine (in)equality. Does not
        take the secret key into account as it might not be known and the public key combined
        with the precision and the session id.

        :param other: Object to compare this Paillier scheme with.
        :return: Boolean value representing (in)equality of both objects.
        """
        # Equality should still hold if the secret key is not available
        return (
            isinstance(other, DistributedPaillier)
            and self.precision == other.precision
            and self.public_key == other.public_key
            and self.session_id == other.session_id
        )

    # region Decryption
    async def decrypt(  # type: ignore[override]
        self,
        ciphertext: PaillierCiphertext,
        apply_encoding: bool = True,
        receivers: Optional[List[str]] = None,
    ) -> Optional[paillier.Plaintext]:
        """
        Decrypts the input ciphertext. Starts a protocol between the parties involved to create
        local decryptions, send them to the other parties and combine them into full decryptions
        for each party.

        :param ciphertext: Ciphertext to be decrypted.
        :param apply_encoding: Boolean indicating whether the decrypted ciphertext is decoded
            before it is returned. Defaults to True.
        :param receivers: An optional list specifying the names of the receivers, your own 'name'
            is "self".
        :return: Plaintext decrypted value.
        """
        decrypted_ciphertext = await self._decrypt_raw(ciphertext, receivers)
        return (
            self.apply_encoding(decrypted_ciphertext, apply_encoding)
            if decrypted_ciphertext is not None
            else None
        )

    async def _decrypt_raw(  # type: ignore[override]
        self,
        ciphertext: PaillierCiphertext,
        receivers: Optional[List[str]] = None,
    ) -> Optional[EncodedPlaintext[int]]:
        """
        Function that starts a protocol between the parties involved to create local decryptions,
        send them to the other parties and combine them into full decryptions for each party.

        :param ciphertext: The ciphertext to be decrypted.
        :param receivers: An optional list specifying the names of the receivers, your own 'name'
            is "self". If none is provided it is sent to all parties in the pool.
        :return: The encoded plaintext corresponding to the ciphertext.
        """
        receivers_without_self: Optional[List[str]]
        if receivers is not None:
            # If we are part of the receivers, we expect the other parties to send us partial
            # decryptions
            self_receive = "self" in receivers
            # We will broadcast our partial decryption to all receivers, but we do not need to send
            # anything to ourselves.
            if self_receive:
                receivers_without_self = [recv for recv in receivers if recv != "self"]
            else:
                receivers_without_self = receivers
        else:
            # If no receivers are specified, we assume everyone will receive the partial decryptions
            self_receive = True
            receivers_without_self = receivers

        # generate the local partial decryption
        partial_decryption_shares = {
            self.index: cast(PaillierSharedKey, self.secret_key).partial_decrypt(
                ciphertext
            )
        }

        # send the partial decryption to all other parties in the provided network
        encryption_hash = bin(ciphertext.peek_value()).zfill(32)[2:34]
        message_id = (
            f"distributed_decryption_session#{self.session_id}_hash#{encryption_hash}"
        )
        if receivers_without_self is None or len(receivers_without_self) != 0:
            self.pool.async_broadcast(
                {
                    "content": "partial_decryption",
                    "value": partial_decryption_shares[self.index],
                },
                msg_id=message_id,
                handler_names=receivers_without_self,
            )

        if self_receive:
            # receive the partial decryption from the other parties
            other_partial_decryption_shares = await self.pool.recv_all(
                msg_id=message_id
            )
            for party, message in other_partial_decryption_shares:
                msg_content = message["content"]
                err_msg = f"received a share for {msg_content}, but expected partial_decryption"
                assert msg_content == "partial_decryption", err_msg
                partial_decryption_shares[self.party_indices[party]] = message["value"]

            # combine all partial decryption to obtain the full decryption
            decryption = cast(PaillierSharedKey, self.secret_key).decrypt(
                partial_decryption_shares
            )
            return EncodedPlaintext(decryption, scheme=self)
        return None

    def apply_encoding(
        self, decrypted_ciphertext: EncodedPlaintext[int], apply_encoding: bool
    ) -> paillier.Plaintext:
        """
        Function which decodes a decrypted ciphertext

        :param decrypted_ciphertext: ciphertext to decode
        :param apply_encoding: Boolean indicating if `decrypted_ciphertext` needs to be decoded.
        :return: The decoded plaintext
        """
        return (
            self.decode(decrypted_ciphertext)
            if apply_encoding
            else decrypted_ciphertext.value
        )

    async def decrypt_sequence(  # type: ignore[override]
        self,
        ciphertext_sequence: Iterable[PaillierCiphertext],
        apply_encoding: bool = True,
        receivers: Optional[List[str]] = None,
    ) -> Optional[List[paillier.Plaintext]]:
        """
        Decrypts the list of ciphertexts

        :param ciphertext_sequence: Sequence of Ciphertext to be decrypted
        :param apply_encoding: Boolean indicating whether the decrypted ciphertext is decoded before it is returned.
            Defaults to True.
        :param receivers: The receivers of all (partially) decrypted ciphertexts. If None is given it is sent to all
            parties. If a list is provided it is sent to those receivers.
        :return: The list of encoded plaintext corresponding to the ciphertext, or None if 'self' is not in the
            receivers list.
        """

        decrypted_ciphertext_list = await self._decrypt_sequence_raw(
            ciphertext_sequence, receivers
        )
        return (
            None
            if decrypted_ciphertext_list is None
            else [
                self.apply_encoding(decryption, apply_encoding)
                for decryption in decrypted_ciphertext_list
            ]
        )

    async def _decrypt_sequence_raw(
        self,
        ciphertext_sequence: Iterable[PaillierCiphertext],
        receivers: Optional[List[str]] = None,
    ) -> Optional[List[EncodedPlaintext[int]]]:
        """
        Function that starts a protocol between the parties involved to create local decryptions,
        send them to the other parties and combine them into full decryptions for each party.

        :param ciphertext_sequence: The sequence of ciphertext to be decrypted.
        :param receivers: An optional list specifying the names of the receivers, your own 'name'
            is "self". If None is provided it is sent to all parties.
        :return: The list of encoded plaintext corresponding to the ciphertext, or None if 'self' is not in the
            receivers list.
        """

        receivers_without_self: Optional[List[str]]
        if receivers is not None:
            # If we are part of the receivers, we expect the other parties to send us partial
            # decryptions
            self_receive = "self" in receivers
            # We will broadcast our partial decryption to all receivers, but we do not need to send
            # anything to ourselves.
            if self_receive:
                receivers_without_self = [recv for recv in receivers if recv != "self"]
            else:
                receivers_without_self = receivers
        else:
            # If no receivers are specified, we assume everyone will receive the partial decryptions
            self_receive = True
            receivers_without_self = receivers

        # partially decrypt the received cipher texts
        partially_decrypted_shares = [
            cast(PaillierSharedKey, self.secret_key).partial_decrypt(ciphertext)
            for ciphertext in ciphertext_sequence
        ]

        # send the partial decryption to all other parties in the provided network
        encryption_hash = (
            bin(next(iter(ciphertext_sequence)).peek_value()).zfill(32)[2:34]
            + f"{len(partially_decrypted_shares)}"
        )
        message_id = (
            f"distributed_decryption_session#{self.session_id}_hash#{encryption_hash}"
        )
        if receivers_without_self is None or len(receivers_without_self) != 0:
            self.pool.async_broadcast(
                {
                    "content": "partial_decryption_sequence",
                    "value": partially_decrypted_shares,
                },
                msg_id=message_id,
                handler_names=receivers_without_self,
            )

        if self_receive:

            # store the partial decryptions per party
            shares_dict_per_decryption: List[Dict[int, int]] = [
                {self.index: partially_decrypted_share}
                for partially_decrypted_share in partially_decrypted_shares
            ]

            # receive the partial decryption from the other parties
            partial_decryptions_other_parties = await self.pool.recv_all(
                msg_id=message_id,
            )
            for party, message in partial_decryptions_other_parties:
                msg_content = message["content"]
                err_msg = f"received a share for {msg_content}, but expected partial_decryption_sequence"
                assert msg_content == "partial_decryption_sequence", err_msg
                partial_decryptions_party = message["value"]
                for shares_dict, partial_decryption in zip(
                    shares_dict_per_decryption, partial_decryptions_party
                ):
                    shares_dict[self.party_indices[party]] = partial_decryption

            # decrypt all the shares
            decryption_results = []

            for shares_dict in shares_dict_per_decryption:
                # combine all partial decryption to obtain the full decryption
                decryption = cast(PaillierSharedKey, self.secret_key).decrypt(
                    shares_dict
                )
                decryption_results.append(EncodedPlaintext(decryption, scheme=self))
            return decryption_results
        return None

    # endregion

    # region Setup functions

    @classmethod
    def setup_input(
        cls,
        pool: Pool,
        key_length: int,
        prime_threshold: int,
        corruption_threshold: int,
    ) -> Tuple[int, int, List[int], Shamir, Shares, List[str]]:
        r"""
        Function that sets initial variables for the process of creating a shared secret key

        :param pool: network of involved parties
        :param key_length: desired bit length of the modulus $N = p \cdot q$
        :param prime_threshold: Bound on the number of prime numbers to be checked for primality
            tests
        :param corruption_threshold: Number of parties that are allowed to be corrupted
        :return: A tuple of initiated variables, containing first the number_of_players,
            second the length of the primes $p$ and $q$, third a list of small primes for the
            small_prime test (empty if the length of $p$ and $q$ is smaller than the
            prime_threshold), fourth a regular Shamir Sharing scheme, fifth a Shares data structure
            for holding relevant shares, and last a list of the names of other parties.
        """
        number_of_players = len(pool.pool_handlers) + 1

        # key length of primes p and q
        prime_length = key_length // 2

        # if the primes are smaller than the small prime threshold,
        # there's no point in doing a small prime test
        if prime_length < math.log(prime_threshold):
            prime_threshold = 1
        prime_list = list(sympy.primerange(3, prime_threshold + 1))
        shamir_scheme = cls.__init_shamir_scheme(
            prime_length, number_of_players, corruption_threshold
        )

        shares = Shares()

        other_parties = list(pool.pool_handlers.keys())
        return (
            number_of_players,
            prime_length,
            prime_list,
            shamir_scheme,
            shares,
            other_parties,
        )

    @classmethod
    async def setup_protocol(
        cls, shamir_scheme: Shamir, other_parties: List[str], pool: Pool
    ) -> Tuple[int, Dict[str, int], ShamirShares, int]:
        """
        Function that initiates a protocol to determine IDs and sets own ID
        Additionally, the protocol prepares a secret sharing of 0 under a 2t-out-of-n
        threshold scheme to be used later on.

        :param shamir_scheme: Shamir secret sharing scheme to be used for p and q
        :param other_parties: Names of the other parties in the pool
        :param pool: network of involved parties
        :return: This party's index, a dictionary with indices for the other parties, and a
            zero-sharing in a 2t-out-of-n thresholds scheme to be used later on, the session id
        """

        # start indices protocol
        party_indices, session_id = await cls.get_indices(pool)

        # prepare zero sharing
        zero_sharing_scheme = Shamir(
            shamir_scheme.modulus,
            shamir_scheme.number_of_parties,
            shamir_scheme.polynomial_degree * 2,
        )
        zero_sharing = zero_sharing_scheme.share_secret(0)

        index = party_indices["self"]

        # send zero shares to other parties
        zero_share_msg_id = f"distributed_keygen_session#{session_id}_zero"
        for party in other_parties:
            party_share = zero_sharing.shares[party_indices[party]]
            pool.asend(
                party, {"content": "zero", "value": party_share}, zero_share_msg_id
            )

        # receive all zero shares of others
        responses = await pool.recv_all(msg_id=zero_share_msg_id)
        assert all(d["content"] == "zero" for _, d in responses)
        shares = [d["value"] for _, d in responses]

        # local share of the final zero sharing
        final_zero_share = zero_sharing.shares[index] + sum(shares)
        zero_share = ShamirShares(zero_sharing_scheme, {index: final_zero_share})
        return index, party_indices, zero_share, session_id

    @classmethod
    async def get_indices(cls, pool: Pool) -> Tuple[Dict[str, int], int]:
        """
        Function that initiates a protocol to determine IDs (indices) for each party

        :param pool: network of involved parties
        :return: dictionary from party name to index, where the entry "self" contains this party's
            index
        """
        success = False
        list_to_sort = []
        attempt = 0
        while not success:
            success = True
            attempt += 1

            # generate random number
            random_number_self = randint(0, 1000000)

            # send random number to all other parties
            pool.async_broadcast(
                random_number_self, msg_id=f"distributed_keygen_random_number#{attempt}"
            )

            # receive random numbers from the other parties
            responses = await pool.recv_all(
                msg_id=f"distributed_keygen_random_number#{attempt}"
            )

            list_to_sort = [("self", random_number_self)]
            for party, random_number_party in responses:
                if random_number_party not in [rn for _, rn in list_to_sort]:
                    list_to_sort.append((party, random_number_party))
                else:
                    success = False

        # sort the list based on the random numbers
        sorted_list = sorted(list_to_sort, key=lambda j: j[1])
        party_indices = {}

        # extract the party names from the sorted list and assign an index based on the position.
        # this dictionary should be the same for each party
        for index, party in enumerate([party_name for party_name, _ in sorted_list]):
            party_indices[party] = index + 1

        session_id = sum(i[1] for i in sorted_list) % 1000000

        return party_indices, session_id

    @classmethod
    def __init_shamir_scheme(
        cls, prime_length: int, number_of_players: int, corruption_threshold: int
    ) -> Shamir:
        """
        Function to initialize the regular Shamir scheme

        :param prime_length: bit length of the shamir prime
        :param number_of_players: number of parties involved in total (n)
        :param corruption_threshold: number of parties allowed to be corrupted
        :return: Shamir secret sharing scheme
        """
        shamir_length = 2 * (prime_length + math.ceil((math.log2(number_of_players))))
        shamir_scheme = Shamir(
            sympy.nextprime(2**shamir_length),
            number_of_players,
            corruption_threshold,
        )
        return shamir_scheme

    @classmethod
    async def generate_keypair(
        cls,
        stat_sec_shamir: int,
        number_of_players: int,
        corruption_threshold: int,
        shares: Shares,
        index: int,
        zero_share: ShamirShares,
        pool: Pool,
        prime_list: List[int],
        prime_length: int,
        party_indices: Dict[str, int],
        correct_param_biprime: int,
        shamir_scheme: Shamir,
        session_id: int,
    ) -> Tuple[PaillierPublicKey, PaillierSharedKey]:
        """
        Function to distributively generate a shared secret key and a corresponding public key

        :param stat_sec_shamir: security parameter for Shamir secret sharing over the integers
        :param number_of_players: number of parties involved in the protocol
        :param corruption_threshold: number of parties that are allowed to be corrupted
        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param index: index of this party
        :param zero_share: A secret sharing of $0$ in a $2t$-out-of-$n$ shamir secret sharing scheme
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :return: regular Paillier public key and a shared secret key
        """
        secret_key = await cls.generate_secret_key(
            stat_sec_shamir,
            number_of_players,
            corruption_threshold,
            shares,
            index,
            zero_share,
            pool,
            prime_list,
            prime_length,
            party_indices,
            correct_param_biprime,
            shamir_scheme,
            session_id,
        )
        modulus = secret_key.n
        public_key = PaillierPublicKey(modulus, modulus + 1)

        logging.info("Key generation complete")
        return public_key, secret_key

    @classmethod
    async def generate_pq(
        cls,
        shares: Shares,
        pool: Pool,
        index: int,
        prime_length: int,
        party_indices: Dict[str, int],
        shamir_scheme: Shamir,
        session_id: int,
    ) -> Tuple[ShamirShares, ShamirShares]:
        """
        Function to generate primes $p$ and $q$

        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param pool: network of involved parties
        :param index: index of this party
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :return: sharings of $p$ and $q$
        """
        shares.p.additive = cls.generate_prime_additive_share(index, prime_length)
        shamir_msg_id = f"distributed_keygen_session#{session_id}_shamir"
        cls.shamir_share_and_send(
            "p", shares, shamir_scheme, index, pool, party_indices, shamir_msg_id + "p"
        )
        await cls.gather_shares("p", pool, shares, party_indices, shamir_msg_id + "p")
        p_sharing = cls.__add_received_shamir_shares("p", shares, index, shamir_scheme)
        shares.q.additive = cls.generate_prime_additive_share(index, prime_length)
        cls.shamir_share_and_send(
            "q", shares, shamir_scheme, index, pool, party_indices, shamir_msg_id + "q"
        )
        await cls.gather_shares("q", pool, shares, party_indices, shamir_msg_id + "q")
        q_sharing = cls.__add_received_shamir_shares("q", shares, index, shamir_scheme)
        return p_sharing, q_sharing

    @classmethod
    def generate_prime_additive_share(cls, index: int, prime_length: int) -> int:
        r"""
        Generate a random value between $2^(\text{length}-1)$ and 2^\text{length}.
        the function will ensure that the random
        value is equal to $3 \mod 4$ for the fist player, and to $0 \mod 4$ for all
        other players.
        This is necessary to generate additive shares of $p$ and $q$, or the
        bi-primality test will not work.

        :param index: index of this party
        :param prime_length: desired bit length of primes $p$ and $q$
        :return: a random integer of the desired bit length and value modulo $4$
        """
        if index == 1:
            mod4 = 3
        else:
            mod4 = 0

        random_number = secrets.randbits(prime_length - 3) << 2
        additive_share: int = 2 ** (prime_length - 1) + random_number + mod4
        return additive_share

    @classmethod
    def shamir_share_and_send(
        cls,
        content: str,
        shares: Shares,
        shamir_scheme: Shamir,
        index: int,
        pool: Pool,
        party_indices: Dict[str, int],
        msg_id: Optional[str] = None,
    ) -> None:
        """
        Create a secret-sharing of the input value, and send each share to
        the corresponding player, together with the label content

        :param content: string identifying the number to be shared and sent
        :param shares: dictionary keeping track of shares for different parties and numbers
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing scheme
        :param index: index of this party
        :param pool: network of involved parties
        :param party_indices: mapping from party names to indices
        :param msg_id: Optional message id.
        :raise NotImplementedError: In case the given content is not "p" or "q".
        """

        # retrieve the local additive share for content
        value = asdict(shares)[content]["additive"]

        # create a shamir sharing of this value
        value_sharing = shamir_scheme.share_secret(value)

        # Save this player's shamir share of the local additive share
        if content == "p":
            shares.p.shares[index] = value_sharing.shares[index]
        elif content == "q":
            shares.q.shares[index] = value_sharing.shares[index]
        else:
            raise NotImplementedError(
                f"Don't know what to do with this content: {content}"
            )

        # Send the other players' shares of the local additive share
        other_parties = pool.pool_handlers.keys()
        for party in other_parties:
            party_share = value_sharing.shares[party_indices[party]]
            pool.asend(party, {"content": content, "value": party_share}, msg_id=msg_id)

    @classmethod
    def int_shamir_share_and_send(
        cls,
        content: str,
        shares: Shares,
        int_shamir_scheme: IntegerShamir,
        index: int,
        pool: Pool,
        party_indices: Dict[str, int],
        msg_id: Optional[str] = None,
    ) -> None:
        r"""
        Create a secret-sharing of the input value, and send each share to
        the corresponding player, together with the label content

        :param content: string identifying the number to be shared and sent
        :param shares: dictionary keeping track of shares for different parties and numbers
        :param int_shamir_scheme: Shamir secret sharing scheme over the integers
        :param index: index of this party
        :param pool: network of involved parties
        :param party_indices: mapping from party names to indices
        :param msg_id: Optional message id.
        :raise NotImplementedError: In case the given content is not "lambda\_" or "beta".
        """
        # retrieve the local additive share for content
        value = asdict(shares)[content]["additive"]

        # create a shamir sharing of this value
        value_sharing = int_shamir_scheme.share_secret(value)

        # Save this player's shamir share of the local additive share
        if content == "lambda_":
            shares.lambda_.shares[index] = value_sharing.shares[index]
        elif content == "beta":
            shares.beta.shares[index] = value_sharing.shares[index]
        else:
            raise NotImplementedError(
                f"Don't know what to do with this content: {content}"
            )

        # Send the other players' shares of the local additive share
        other_parties = pool.pool_handlers.keys()
        for party in other_parties:
            party_share = value_sharing.shares[party_indices[party]]
            pool.asend(party, {"content": content, "value": party_share}, msg_id=msg_id)

    @classmethod
    def __add_received_shamir_shares(
        cls, content: str, shares: Shares, index: int, shamir_scheme: Shamir
    ) -> ShamirShares:
        """
        Fetch shares labeled with content and add them to
        own_share_value.

        :param content: string identifying the number to be retrieved
        :param shares: dictionary keeping track of shares for different parties and numbers
        :param index: index of this party
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing
        :return: sum of all the shares for the number identified by content
        """

        shamir_shares = [
            ShamirShares(shamir_scheme, {index: v})
            for k, v in asdict(shares)[content]["shares"].items()
        ]
        for i in range(1, len(shamir_shares)):
            shamir_shares[0] += shamir_shares[i]
        return shamir_shares[0]

    @classmethod
    def __int_add_received_shares(
        cls,
        content: str,
        int_shamir_scheme: IntegerShamir,
        shares: Shares,
        index: int,
        corruption_threshold: int,
    ) -> IntegerShares:
        """
        Fetch shares labeled with content and add them to own_share_value.

        :param content: string identifying the number to be retrieved
        :param int_shamir_scheme: Shamir secret sharing scheme over the integers
        :param shares: dictionary keeping track of shares for different parties and numbers
        :param index: index of this party
        :param corruption_threshold: number of parties that are allowed to be corrupted
        :return: sum of the integer sharing of the number identified by content
        """
        integer_shares = [
            IntegerShares(
                int_shamir_scheme,
                {index: v},
                corruption_threshold,
                scaling=math.factorial(int_shamir_scheme.number_of_parties),
            )
            for k, v in asdict(shares)[content]["shares"].items()
        ]
        for i in range(1, len(integer_shares)):
            integer_shares[0] += integer_shares[i]
        return integer_shares[0]

    @classmethod
    def __mul_received_v_and_check(cls, shares: Shares, modulus: int) -> bool:
        """
        Function to test whether a certain primality check holds

        :param shares: dictionary keeping track of shares for a certain value
        :param modulus: value of $N$
        :return: true if the biprimality tests succeeds and false if it fails
        """
        product = 1
        for key, value in shares.v.shares.items():
            if key != 1:
                product *= value
        value1 = shares.v.shares[1]

        # The below test determines if N is "probably" the product of two primes (if the
        # statement is True). Otherwise, N is definitely not the product of two primes.
        return ((value1 % modulus) == (product % modulus)) or (
            (value1 % modulus) == (-product % modulus)
        )

    @classmethod
    async def gather_shares(
        cls,
        content: str,
        pool: Pool,
        shares: Shares,
        party_indices: Dict[str, int],
        msg_id: Optional[str] = None,
    ) -> None:
        r"""
        Gather all shares with label content

        :param content: string identifying a number
        :param pool: network of involved parties
        :param shares: dictionary keeping track of shares of different parties for certain numbers
        :param party_indices: mapping from party names to indices
        :param msg_id: Optional message id.
        :raise NotImplementedError: In case the given content is not any of the possible values
            for which we store shares ("p", "q", "n", "biprime", "lambda\_", "beta", "secret_key").
        """
        shares_from_other_parties = await pool.recv_all(msg_id=msg_id)
        for party, message in shares_from_other_parties:
            msg_content = message["content"]
            err_msg = f"received a share for {msg_content}, but expected {content}"
            assert msg_content == content, err_msg
            if content == "p":
                shares.p.shares[party_indices[party]] = message["value"]
            elif content == "q":
                shares.q.shares[party_indices[party]] = message["value"]
            elif content == "n":
                shares.n.shares[party_indices[party]] = message["value"]
            elif content == "biprime":
                shares.biprime.shares[party_indices[party]] = message["value"]
            elif content == "v":
                shares.v.shares[party_indices[party]] = message["value"]
            elif content == "lambda_":
                shares.lambda_.shares[party_indices[party]] = message["value"]
            elif content == "beta":
                shares.beta.shares[party_indices[party]] = message["value"]
            elif content == "secret_key":
                shares.secret_key.shares[party_indices[party]] = message["value"]
            else:
                raise NotImplementedError(
                    f"Don't know what to do with this content: {content}"
                )

    @classmethod
    async def __biprime_test(
        cls,
        correct_param_biprime: int,
        shares: Shares,
        modulus: int,
        pool: Pool,
        index: int,
        party_indices: Dict[str, int],
        session_id: int,
    ) -> bool:
        """
        Function to test for biprimality of $N$

        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated modulus is biprime
        :param shares: dictionary keeping track of shares for different parties for certain numbers
        :param modulus: the modulus $N$
        :param pool: network of involved parties
        :param index: index of this party
        :param party_indices: mapping from party name to indices
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :return: true if the test succeeds and false if it fails
        """
        successfull_biprime_tests = 0
        biprime_test_attempts = 0
        while successfull_biprime_tests < correct_param_biprime:
            test_value = secrets.randbelow(modulus)
            pool.async_broadcast(
                {"content": "biprime", "value": test_value},
                msg_id=f"distributed_keygen_session#{session_id}_{biprime_test_attempts}_biprime_test",
            )
            shares.biprime.shares[index] = test_value
            await cls.gather_shares(
                "biprime",
                pool,
                shares,
                party_indices,
                msg_id=f"distributed_keygen_session#{session_id}_{biprime_test_attempts}_biprime_test",
            )
            biprime_test_attempts += 1

            test_value = 0
            for value in shares.biprime.shares.values():
                test_value += value
            test_value = test_value % modulus

            if sympy.jacobi_symbol(test_value, modulus) == 1:
                if index == 1:
                    v_value = int(
                        pow_mod(
                            test_value,
                            (modulus - shares.p.additive - shares.q.additive + 1) // 4,
                            modulus,
                        )
                    )
                else:
                    v_value = int(
                        pow_mod(
                            test_value,
                            (shares.p.additive + shares.q.additive) // 4,
                            modulus,
                        )
                    )
                shares.v.shares[index] = v_value
                pool.async_broadcast(
                    {"content": "v", "value": v_value},
                    msg_id=f"distributed_keygen_session#{session_id}_biprime_v",
                )
                await cls.gather_shares(
                    "v",
                    pool,
                    shares,
                    party_indices,
                    msg_id=f"distributed_keygen_session#{session_id}_biprime_v",
                )

                if cls.__mul_received_v_and_check(shares, modulus):
                    successfull_biprime_tests += 1
                else:
                    return False
        return True

    @classmethod
    def __generate_lambda_addit_share(
        cls,
        index: int,
        modulus: int,
        shares: Shares,
    ) -> int:
        """
        Function to generate an additive share of lambda

        :param index: index of this party
        :param modulus: modulus $N$
        :param shares: dictionary keeping track of shares for different parties for certain numbers
        :return: additive share of lambda
        """
        if index == 1:
            return modulus - shares.p.additive - shares.q.additive + 1
        # else
        return 0 - shares.p.additive - shares.q.additive

    @classmethod
    def __small_prime_divisors_test(cls, prime_list: List[int], modulus: int) -> bool:
        """
        Function to test $N$ for small prime divisors

        :param prime_list: list of prime numbers
        :param modulus: modulus $N$
        :return: true if $N$ has small divisors and false otherwise
        """
        for prime in prime_list:
            if modulus % prime == 0:
                return True
        return False

    @classmethod
    async def compute_modulus(
        cls,
        shares: Shares,
        zero_share: ShamirShares,
        index: int,
        pool: Pool,
        prime_list: List[int],
        party_indices: Dict[str, int],
        prime_length: int,
        shamir_scheme: Shamir,
        correct_param_biprime: int,
        session_id: int,
    ) -> int:
        r"""
        Function that starts a protocol to generate candidates for $p$ and $q$
        the multiplication of the two is then checked for biprimality to ensure it is a valid
        modulus. This is run until it succeeds.

        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param zero_share: A secret sharing of $0$ in a $2t$-out-of-$n$ shamir secret sharing scheme
        :param index: index of this party
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param party_indices: mapping from party names to indices
        :param prime_length: desired bit length of $p$ and $q$
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing scheme
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :return: modulus $N$
        """

        sp_err_counter = 0
        bip_err_counter = 0
        bip = False
        logging.info("Computing N")
        modulus = 0
        counter = 0
        while not bip:
            counter += 1

            shares.biprime = Shares.Biprime()
            shares.v = Shares.V()

            # secreting sharings of p and q
            p_sharing, q_sharing = await cls.generate_pq(
                shares,
                pool,
                index,
                prime_length,
                party_indices,
                shamir_scheme,
                session_id,
            )

            # secret sharing of the modulus
            modulus_sharing = p_sharing * q_sharing

            # Add 0-share to fix distribution
            modulus_sharing += zero_share

            shares.n.shares[index] = modulus_sharing.shares[index]
            pool.async_broadcast(
                {"content": "n", "value": modulus_sharing.shares[index]},
                msg_id=f"distributed_keygen_session#{session_id}_modulus",
            )
            await cls.gather_shares(
                "n",
                pool,
                shares,
                party_indices,
                msg_id=f"distributed_keygen_session#{session_id}_modulus",
            )
            modulus_sharing.shares = shares.n.shares
            modulus = modulus_sharing.reconstruct_secret()
            if not cls.__small_prime_divisors_test(prime_list, modulus):
                bip = await cls.__biprime_test(
                    correct_param_biprime,
                    shares,
                    modulus,
                    pool,
                    index,
                    party_indices,
                    session_id,
                )
                if not bip:
                    bip_err_counter += 1
            else:
                sp_err_counter += 1

        logging.info(f"N = {modulus}")
        logging.info(f"Failures counter: sp={sp_err_counter} biprime={bip_err_counter}")
        return modulus

    @classmethod
    async def generate_secret_key(
        cls,
        stat_sec_shamir: int,
        number_of_players: int,
        corruption_threshold: int,
        shares: Shares,
        index: int,
        zero_share: ShamirShares,
        pool: Pool,
        prime_list: List[int],
        prime_length: int,
        party_indices: Dict[str, int],
        correct_param_biprime: int,
        shamir_scheme: Shamir,
        session_id: int,
    ) -> PaillierSharedKey:
        """
        Functions that generates the modulus and sets up the sharing of the private key

        :param stat_sec_shamir: security parameter for the Shamir secret sharing over the integers
        :param number_of_players: total number of participants in this session (including self)
        :param corruption_threshold: Maximum number of allowed corruptions
        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param index: index of this party
        :param zero_share: A secret sharing of $0$ in a $2t$-out-of-$n$ shamir secret sharing scheme
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param shamir_scheme: $t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :return: shared secret key
        """
        modulus = await cls.compute_modulus(
            shares,
            zero_share,
            index,
            pool,
            prime_list,
            party_indices,
            prime_length,
            shamir_scheme,
            correct_param_biprime,
            session_id,
        )
        int_shamir_scheme = IntegerShamir(
            stat_sec_shamir,
            modulus,
            number_of_players,
            corruption_threshold,
        )

        shares.lambda_.additive = cls.__generate_lambda_addit_share(
            index, modulus, shares
        )
        shamir_msg_id = f"distributed_keygen_session#{session_id}_int_shamir"
        cls.int_shamir_share_and_send(
            "lambda_",
            shares,
            int_shamir_scheme,
            index,
            pool,
            party_indices,
            shamir_msg_id + "lambda",
        )
        await cls.gather_shares(
            "lambda_", pool, shares, party_indices, shamir_msg_id + "lambda"
        )
        lambda_ = cls.__int_add_received_shares(
            "lambda_", int_shamir_scheme, shares, index, corruption_threshold
        )

        secret_key_sharing: IntegerShares
        while True:
            shares.secret_key = Shares.SecretKey()
            shares.beta = Shares.Beta()
            shares.beta.additive = secrets.randbelow(modulus)
            cls.int_shamir_share_and_send(
                "beta",
                shares,
                int_shamir_scheme,
                index,
                pool,
                party_indices,
                shamir_msg_id + "beta",
            )
            await cls.gather_shares(
                "beta", pool, shares, party_indices, shamir_msg_id + "beta"
            )
            beta = cls.__int_add_received_shares(
                "beta", int_shamir_scheme, shares, index, corruption_threshold
            )
            secret_key_sharing = lambda_ * beta
            temp_secret_key = copy.deepcopy(secret_key_sharing)
            temp_secret_key.shares = {
                key: (value % modulus) for key, value in temp_secret_key.shares.items()
            }
            shares.secret_key.shares = temp_secret_key.shares

            pool.async_broadcast(
                {"content": "secret_key", "value": temp_secret_key.shares[index]},
                msg_id=f"distributed_keygen_session#{session_id}_sk",
            )
            await cls.gather_shares(
                "secret_key",
                pool,
                shares,
                party_indices,
                msg_id=f"distributed_keygen_session#{session_id}_sk",
            )
            reconstructed_secret_key = temp_secret_key.reconstruct_secret(
                modulus=modulus
            )
            theta = (
                reconstructed_secret_key
                * math.factorial(int_shamir_scheme.number_of_parties) ** 3
            ) % modulus
            if math.gcd(theta, modulus) != 0:
                break

        secret_key = PaillierSharedKey(
            n=modulus,
            t=corruption_threshold,
            player_id=index,
            theta=theta,
            share=secret_key_sharing,
        )
        return secret_key

    class SerializedDistributedPaillier(Paillier.SerializedPaillier, TypedDict):
        session_id: int
        distributed: bool
        index: int

    def serialize(
        self, **_kwargs: Any
    ) -> DistributedPaillier.SerializedDistributedPaillier:
        r"""
        Serialization function for Distributed Paillier schemes, which will be passed to
        the communication module

        :param \**_kwargs: optional extra keyword arguments
        :return: Dictionary containing the serialization of this DistributedPaillier scheme.
        """
        return {
            "session_id": self.session_id,
            "distributed": self.distributed,
            "index": self.index,
            "prec": self.precision,
            "pubkey": self.public_key,
        }

    @overload
    @staticmethod
    def deserialize(
        obj: DistributedPaillier.SerializedDistributedPaillier,
        *,
        origin: Optional[HTTPClient] = ...,
        **kwargs: Any,
    ) -> "DistributedPaillier":
        ...

    @overload
    @staticmethod
    def deserialize(
        obj: Paillier.SerializedPaillier,
        *,
        origin: Optional[HTTPClient] = ...,
        **kwargs: Any,
    ) -> "Paillier":
        ...

    @staticmethod
    def deserialize(
        obj: Union[
            DistributedPaillier.SerializedDistributedPaillier,
            Paillier.SerializedPaillier,
        ],
        *,
        origin: Optional[HTTPClient] = None,
        **kwargs: Any,
    ) -> Union["DistributedPaillier", "Paillier"]:
        r"""
        Deserialization function for Distributed Paillier schemes, which will be passed to
        the communication module

        :param obj: serialization of a distributed paillier scheme.
        :param origin: HTTPClient representing where the message came from if applicable
        :param \**kwargs: optional extra keyword arguments
        :return: Deserialized DistributedPaillier scheme, local instance thereof, or a regular
            Paillier scheme in case this party is not part of the distributed session.
        """
        session_id = obj.get("session_id", None)
        if isinstance(session_id, int):
            if obj.get("distributed", False):
                # The scheme should be stored in the local instances through the session ID
                # If it is not, then this party was not part of the initial protocol
                if session_id in DistributedPaillier._local_instances:
                    return DistributedPaillier._local_instances[session_id]
            else:
                # The scheme should be stored in the global instances through the session ID
                # If it is not, then this party was not part of the initial protocol
                index = obj.get("index", None)
                if (
                    isinstance(index, int)
                    and session_id in DistributedPaillier._global_instances[index]
                ):
                    return DistributedPaillier._global_instances[index][session_id]
        # This party is not part of the distributed session, so we parse it as a Paillier scheme
        paillier_obj: Paillier.SerializedPaillier = {
            "prec": obj["prec"],
            "pubkey": obj["pubkey"],
        }
        return Paillier.deserialize(paillier_obj, origin=origin, **kwargs)

    # endregion


# Load the serialization logic into the communication module
try:
    Serialization.register_class(DistributedPaillier, check_annotations=False)
except RepetitionError:
    pass
