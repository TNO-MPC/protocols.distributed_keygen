"""
Code for a single player in the Paillier distributed key-generation protocol.
"""

from __future__ import annotations

import copy
import logging
import math
import secrets
import warnings
from collections.abc import Iterable
from dataclasses import asdict
from random import randint
from typing import Any, TypedDict, cast, overload

# ormsgpack dependency already included by the communication package
import ormsgpack as ormsgpack
import sympy

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
from tno.mpc.encryption_schemes.templates.encryption_scheme import EncodedPlaintext
from tno.mpc.encryption_schemes.utils import pow_mod

from tno.mpc.protocols.distributed_keygen.paillier_shared_key import PaillierSharedKey
from tno.mpc.protocols.distributed_keygen.utils import (
    AdditiveVariable,
    Batched,
    ShamirVariable,
    Shares,
    exchange_reconstruct,
    exchange_shares,
)

try:
    from tno.mpc.communication import (
        RepetitionError,
        Serialization,
        SupportsSerialization,
    )

    COMMUNICATION_INSTALLED = True
except ModuleNotFoundError:
    COMMUNICATION_INSTALLED = False

logger = logging.getLogger(__name__)
# the generators must have a jacobi symbol of 1. To ensure we have sufficient number of generators with a jacobi symbol of 1 we generate four times as many as we need (we need `correct_biprime_param` amount)
JACOBI_CORRECTION_FACTOR = 4

DIST_KEY_STORAGE_PACK_OPTIONS = (
    ormsgpack.OPT_PASSTHROUGH_BIG_INT
    | ormsgpack.OPT_PASSTHROUGH_TUPLE
    | ormsgpack.OPT_PASSTHROUGH_DATACLASS
    | ormsgpack.OPT_SERIALIZE_NUMPY
    | ormsgpack.OPT_NON_STR_KEYS
)


class SessionIdError(Exception):
    """
    Used to raise exceptions when a session ID is invalid
    """


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
    _global_instances: dict[int, dict[int, DistributedPaillier]] = {}
    _local_instances: dict[int, DistributedPaillier] = {}

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
        batch_size: int = 100,
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
        :param batch_size: How many $p$'s and $q$'s to generate at once (drastically
            reduces communication at the expense of potentially wasted computation)
        :raise ValueError: In case the number of parties $n$ and the corruption threshold $t$ do
            not satisfy that $n \geq 2*t + 1$
        :raise SessionIdError: In case the parties agree on a session ID that is already being used.
        :return: DistributedPaillier scheme containing a regular Paillier public key and a shared
            secret key.
        """
        (
            number_of_players,
            prime_length,
            prime_list,
            shamir_scheme_t,
            shamir_scheme_2t,
            shares,
        ) = cls.setup_input(pool, key_length, prime_threshold, corruption_threshold)
        index, party_indices, session_id = await cls.setup_protocol(pool)

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
            pool,
            prime_list,
            prime_length,
            party_indices,
            correct_param_biprime,
            shamir_scheme_t,
            shamir_scheme_2t,
            session_id,
            batch_size,
        )

        scheme = cls(
            public_key=public_key,
            secret_key=secret_key,
            precision=precision,
            pool=pool,
            index=index,
            party_indices=party_indices,
            session_id=session_id,
            distributed=distributed,
            corruption_threshold=corruption_threshold,
        )

        cls.__register_scheme(scheme, distributed)

        if key_length < 1024:
            warnings.warn(
                f"The key length={key_length} is lower than the advised minimum of 1024."
            )

        return scheme

    @classmethod
    def __register_scheme(cls, scheme: DistributedPaillier, distributed: bool) -> None:
        """
        Register the scheme such that the deserialization reuses the existing scheme and does not
        create a Paillier object.

        :param scheme: The scheme to register
        :param distributed: Whether the different parties are run on different python instances
        """
        # We need to distinguish the case where the parties share a python instance and where they
        # are run in different python instances. If the same python instance is used, then we need
        # to save a different DistributedPaillier instance for each party. If different python
        # instances are used, then we have exactly one DistributedPaillier instance in the python
        # instance for that session.
        if distributed:
            if scheme.session_id in cls._local_instances:
                raise SessionIdError(
                    "An already existing session ID is about to be overwritten. "
                    "This can only happen if multiple sessions are run within the same python "
                    "instance and one of those session has the same ID"
                )
            cls._local_instances[scheme.session_id] = scheme
        else:
            if scheme.index in cls._global_instances:
                if scheme.session_id in cls._global_instances[scheme.index]:
                    raise SessionIdError(
                        "An already existing session ID is about to be overwritten. "
                        "This can only happen if multiple sessions are run within the same python "
                        "instance and one of those session has the same ID"
                    )
                cls._global_instances[scheme.index][scheme.session_id] = scheme
            else:
                cls._global_instances[scheme.index] = {scheme.session_id: scheme}

    def __init__(
        self,
        public_key: PaillierPublicKey,
        secret_key: PaillierSharedKey,
        precision: int,
        pool: Pool,
        index: int,
        party_indices: dict[str, int],
        session_id: int,
        distributed: bool,
        corruption_threshold: int,
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
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param distributed: Boolean value indicating whether the protocol that generated the keys
            for this DistributedPaillier scheme was run in different Python instances (True) or in a
            single python instance (False)
        :param corruption_threshold: The corruption threshold used during the generation of the key
        :param kwargs: Any keyword arguments that are passed to the super __init__ function
        """
        super().__init__(
            public_key, cast(PaillierSecretKey, secret_key), precision, False, **kwargs
        )

        # these variables are necessary during decryption
        self.pool = pool
        self.index = index
        self.party_indices = party_indices
        self.session_id = session_id
        self.distributed = distributed
        self.corruption_threshold = corruption_threshold

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
        receivers: list[str] | None = None,
    ) -> paillier.Plaintext | None:
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
        receivers: list[str] | None = None,
    ) -> EncodedPlaintext[int] | None:
        """
        Function that starts a protocol between the parties involved to create local decryptions,
        send them to the other parties and combine them into full decryptions for each party.

        :param ciphertext: The ciphertext to be decrypted.
        :param receivers: An optional list specifying the names of the receivers, your own 'name'
            is "self". If none is provided it is sent to all parties in the pool.
        :return: The encoded plaintext corresponding to the ciphertext.
        """
        receivers_without_self: list[str] | None
        if receivers is not None:
            # If we are part of the receivers, we expect the other parties to send us partial
            # decryptions

            # We will broadcast our partial decryption to all receivers, but we do not need to send
            # anything to ourselves.
            if self_receive := "self" in receivers:
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
            other_partial_decryption_shares = cast(
                list[tuple[str, dict[str, Any]]],
                (await self.pool.recv_all(msg_id=message_id)),
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
        receivers: list[str] | None = None,
    ) -> list[paillier.Plaintext] | None:
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
        receivers: list[str] | None = None,
    ) -> list[EncodedPlaintext[int]] | None:
        """
        Function that starts a protocol between the parties involved to create local decryptions,
        send them to the other parties and combine them into full decryptions for each party.

        :param ciphertext_sequence: The sequence of ciphertext to be decrypted.
        :param receivers: An optional list specifying the names of the receivers, your own 'name'
            is "self". If None is provided it is sent to all parties.
        :return: The list of encoded plaintext corresponding to the ciphertext, or None if 'self' is not in the
            receivers list.
        """

        receivers_without_self: list[str] | None
        if receivers is not None:
            # If we are part of the receivers, we expect the other parties to send us partial
            # decryptions

            # We will broadcast our partial decryption to all receivers, but we do not need to send
            # anything to ourselves.
            if self_receive := "self" in receivers:
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
            shares_dict_per_decryption: list[dict[int, int]] = [
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
    ) -> tuple[int, int, list[int], Shamir, Shamir, Shares]:
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
        prime_list: list[int] = list(sympy.primerange(3, prime_threshold + 1))
        shamir_scheme_t = cls.__init_shamir_scheme(
            prime_length, number_of_players, corruption_threshold
        )
        shamir_scheme_2t = cls.__init_shamir_scheme(
            prime_length, number_of_players, corruption_threshold * 2
        )

        shares = Shares()

        return (
            number_of_players,
            prime_length,
            prime_list,
            shamir_scheme_t,
            shamir_scheme_2t,
            shares,
        )

    @classmethod
    async def setup_protocol(cls, pool: Pool) -> tuple[int, dict[str, int], int]:
        """
        Runs the indices protocol and sets own ID.

        :param pool: network of involved parties
        :return: This party's index, a dictionary with indices for the other parties, the session id
        """
        # start indices protocol
        party_indices, session_id = await cls.get_indices(pool)
        index = party_indices["self"]
        return index, party_indices, session_id

    @classmethod
    async def get_indices(cls, pool: Pool) -> tuple[dict[str, int], int]:
        """
        Function that initiates a protocol to determine IDs (indices) for each party

        :param pool: network of involved parties
        :return: dictionary from party name to index, where the entry "self" contains this party's
            index
        """
        success = False
        list_to_sort: list[tuple[str, int]] = []
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
        shamir_length = 2 * (prime_length + math.ceil(math.log2(number_of_players)))
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
        pool: Pool,
        prime_list: list[int],
        prime_length: int,
        party_indices: dict[str, int],
        correct_param_biprime: int,
        shamir_scheme_t: Shamir,
        shamir_scheme_2t: Shamir,
        session_id: int,
        batch_size: int = 1,
    ) -> tuple[PaillierPublicKey, PaillierSharedKey]:
        """
        Function to distributively generate a shared secret key and a corresponding public key

        :param stat_sec_shamir: security parameter for Shamir secret sharing over the integers
        :param number_of_players: number of parties involved in the protocol
        :param corruption_threshold: number of parties that are allowed to be corrupted
        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param index: index of this party
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param shamir_scheme_t: $t$-out-of-$n$ Shamir secret sharing scheme
        :param shamir_scheme_2t: $2t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param batch_size: How many $p$'s and $q$'s to generate at once (drastically
            reduces communication at the expense of potentially wasted computation)
        :return: regular Paillier public key and a shared secret key
        """
        secret_key = await cls.generate_secret_key(
            stat_sec_shamir,
            number_of_players,
            corruption_threshold,
            shares,
            index,
            pool,
            prime_list,
            prime_length,
            party_indices,
            correct_param_biprime,
            shamir_scheme_t,
            shamir_scheme_2t,
            session_id,
            batch_size,
        )
        modulus = secret_key.n
        public_key = PaillierPublicKey(modulus, modulus + 1)

        logger.info("Key generation complete")
        return public_key, secret_key

    @classmethod
    async def _generate_pq(
        cls,
        pool: Pool,
        index: int,
        prime_length: int,
        party_indices: dict[str, int],
        shamir_scheme_t: Shamir,
        shamir_scheme_2t: Shamir,
        session_id: int,
        batch_size: int = 1,
        msg_id: str = "",
    ) -> tuple[
        Batched[ShamirVariable],
        Batched[ShamirVariable],
        Batched[ShamirVariable],
        list[int],
        list[int],
    ]:
        """
        Generate secretively two random prime candidates $p$ and $q$.

        These primes must be tested for primality before using the modulus N=pq.

        The number q is picked such that q = 3 mod 4, as this is needed for the
        biprimality test.

        This method supports batching (set batch_size > 1) to generate and share
        multiple $p$'s and $q$'s in one go. This potentially speeds up the key
        generation as less communication is required. Setting the batch_size to
        high will mean that potentially more p and q pairs are generated than
        needed, resulting in wasted computation. The batch_size is a trade-off
        between wasted computation and reduced communication.

        :param pool: network of involved parties
        :param index: index of this party
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param shamir_scheme_t: $t$-out-of-$n$ Shamir secret sharing scheme
        :param shamir_scheme_2t: $2t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param batch_size: How many $p$'s and $q$'s to generate at once (drastically
            reduces communication at the expense of potentially wasted computation)
        :param msg_id: prefix used for the message id
        :return: sharings of $p$ and $q$
        """

        def x_j(x: str, j: int, shamir: Shamir) -> Batched[ShamirVariable]:
            r"""
            Helper function to generate a ShamirVariable with label $x_j$, owned
            by party $j$.

            :param x: label of the variable
            :param j: index of the party
            :return: Batched[ShamirVariable] with label $x_j$
            """
            return Batched(
                ShamirVariable(shamir=shamir, label=f"{x}_{j}", owner=j),
                batch_size=batch_size,
            )

        # We store all required Variables in a list to easily merge
        # communication for these Variables into the same message.
        group: list[Batched[ShamirVariable]] = []

        # Generate the local additive share of p and q, respectively p_i and q_i
        p_i = Batched(
            ShamirVariable(
                shamir=shamir_scheme_t, label="p_" + str(index), owner=index
            ),
            batch_size=batch_size,
        )
        p_i.set_plaintexts(
            [
                cls._generate_prime_candidate(index, prime_length)
                for _ in range(batch_size)
            ]
        )
        # Generate the local additive share of q, namely q_i
        q_i = Batched(
            ShamirVariable(
                shamir=shamir_scheme_t, label="q_" + str(index), owner=index
            ),
            batch_size=batch_size,
        )
        q_i.set_plaintexts(
            [
                cls._generate_prime_candidate(index, prime_length)
                for _ in range(batch_size)
            ]
        )
        # Generate a local additive share of 0, namely 0_i
        zero_i = Batched(
            ShamirVariable(
                shamir=shamir_scheme_2t, label="zero_" + str(index), owner=index
            ),
            batch_size=batch_size,
        )
        zero_i.set_plaintexts([0 for _ in range(batch_size)])

        group.append(p_i)
        group.append(q_i)
        group.append(zero_i)
        # Create variables to represent the p_i of all other parties and
        # store their shares (idem for q_i and 0_i)
        other_parties = [_ for _ in party_indices.values() if _ != index]
        group.extend([x_j("p", _, shamir_scheme_t) for _ in other_parties])
        group.extend([x_j("q", _, shamir_scheme_t) for _ in other_parties])
        group.extend([x_j("zero", _, shamir_scheme_2t) for _ in other_parties])

        # Create sharings of p_i to send to other parties
        p_i.share(index)
        q_i.share(index)
        zero_i.share(index)

        # Exchange shares of all p_i
        # We send over one share of our p_i to each party
        # And we receive one share per p_i of each other party
        shamir_msg_id = msg_id or f"distributed_keygen_session#{session_id}_shamir"
        await exchange_shares(group, index, pool, party_indices, msg_id=shamir_msg_id)

        # p = sum(p_i)
        p_i__s = [v for v in group if v.label.startswith("p_")]
        p = sum(p_i__s[1:], p_i__s[0])
        # q = sum(q_i)
        q_i__s = [v for v in group if v.label.startswith("q_")]
        q = sum(q_i__s[1:], q_i__s[0])
        # zero = sum(zero_i)
        zero_i__s = [v for v in group if v.label.startswith("zero_")]
        zero = sum(zero_i__s[1:], zero_i__s[0])

        # We also return our local additive share of P (p_i)
        p_additive = [p_i[_].get_plaintext() for _ in range(batch_size)]
        q_additive = [q_i[_].get_plaintext() for _ in range(batch_size)]

        return p, q, zero, p_additive, q_additive

    @classmethod
    def _generate_prime_candidate(cls, index: int, prime_length: int) -> int:
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
    def int_shamir_share_and_send(
        cls,
        content: str,
        shares: Shares,
        int_shamir_scheme: IntegerShamir,
        index: int,
        pool: Pool,
        party_indices: dict[str, int],
        msg_id: str | None = None,
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
            for v in asdict(shares)[content]["shares"].values()
        ]
        for i in range(1, len(integer_shares)):
            integer_shares[0] += integer_shares[i]
        return integer_shares[0]

    @classmethod
    async def gather_shares(
        cls,
        content: str,
        pool: Pool,
        shares: Shares,
        party_indices: dict[str, int],
        msg_id: str | None = None,
    ) -> None:
        r"""
        Gather all shares with label content

        :param content: string identifying a number
        :param pool: network of involved parties
        :param shares: dictionary keeping track of shares of different parties for certain numbers
        :param party_indices: mapping from party names to indices
        :param msg_id: Optional message id.
        :raise AttributeError: In case the given content is not any of the possible values
            for which we store shares ("p", "q", "n", "biprime", "lambda\_", "beta", "secret_key").
        """
        shares_from_other_parties = await pool.recv_all(msg_id=msg_id)
        for party, message in shares_from_other_parties:
            # Check if received content corresponds to the expected content
            msg_content = message["content"]
            err_msg = f"received a share for {msg_content}, but expected {content}"
            assert msg_content == content, err_msg

            # Check that the identifier 'content' exists in the Shares object
            try:
                value = getattr(shares, content)
            except AttributeError as e:
                err_msg = f"Don't know what to do with this content: {content}"
                raise AttributeError(err_msg) from e

            if isinstance(value, list):
                assert isinstance(
                    message["value"], list
                ), "The value {content} is stored as a list (to support batching) but the received message is not a list."

                for i, v in enumerate(value):
                    v.shares[party_indices[party]] = message["value"][i]
            else:
                value.shares[party_indices[party]] = message["value"]

    @classmethod
    async def __biprime_test_g_generation(
        cls,
        correct_param_biprime: int,
        index: int,
        candidate_n_list: list[int],
        party_indices: dict[str, int],
        pool: Pool,
        msg_id: str,
    ) -> list[list[int]]:
        r"""
        Function to generate the random $g$ values used for biprimality test of the entire batch of $N$ values.

        The $g$ is jointly picked at random. We pick more generators than needed
        to ensure sufficient values with a $\operatorname{JacobiSymbol}(g/N)=1$.
        In a later step we check if we can generate sufficient values.

        We can batch the joint picking of $g$ in a single communication round to optimize
        communication at the expesive of wasted computational resources.

        :param correct_param_biprime: correctness parameter that affects the
            certainty that the generated modulus is biprime
        :param pool: network of involved parties
        :param index: index of this party
        :param party_indices: mapping from party name to indices
        :param msg_id: Message id.
        :return: a list of jointly picked $g$ values
        """
        batch_g_size = correct_param_biprime * JACOBI_CORRECTION_FACTOR

        large_batch_of_gs = []
        for candidate_n in candidate_n_list:
            # The parties must agree on a random number g for each candidate n
            # Therefore every party picks a random number and sets it as its local
            # additive share of g
            batched_g_sharing = Batched(
                AdditiveVariable(label="biprime", modulus=candidate_n),
                batch_size=batch_g_size,
            )

            batched_g_sharing.set_share(
                index,
                [randint(0, candidate_n) for _ in range(batch_g_size)],
            )
            large_batch_of_gs.append(batched_g_sharing)

        # The parties exchange their additive shares of g
        await exchange_reconstruct(
            large_batch_of_gs, index, pool, party_indices, msg_id=f"{msg_id}_g"
        )
        # We reconstruct by adding the shares modulo N
        batched_g: list[list[int]] = [
            batched_g_sharing.reconstruct() for batched_g_sharing in large_batch_of_gs
        ]
        return batched_g

    @classmethod
    def __biprime_test_v_calculation(
        cls,
        g_values: list[int],
        index: int,
        modulus: int,
        p_i: int,
        q_i: int,
        correct_param_biprime: int,
    ) -> Batched[AdditiveVariable]:
        r"""
        Function to calculate the $v$ values for the biprimality test of each $N$.

        For the the biprimality test we calculate $v$ values. The $v$ values are based on the $g$ values generated in `biprime_test_g_generation`. $g$ values with a $\operatorname{JacobiSymbol}(g/N)!=1$ are skipped. We need at least `correct_param_biprime` values for the biprime test of $N$. The $v$ values are calculated for each $N$.

        :param g_values: The $g$ values generated by `biprime_test_g_generation`
        :param index: index of this party
        :param modulus: the modulus $N$
        :param p_i: The p share of the corresponding modulus $N$.
        :param q_i: The q share of the correcsponding modulus $N$.
        :param correct_param_biprime: correctness parameter that affects the
            certainty that the generated modulus is biprime
        :return: `correct_param_biprime` number of $v$ values.
        """
        v_values: list[int] = []
        N = modulus
        # Every party calculates their value of v_i where i is the index of the
        # party
        for g in g_values:
            # no need to compute more values than needed
            if len(v_values) == correct_param_biprime:
                break

            if sympy.jacobi_symbol(g, modulus) != 1:
                # We check if the Jacobi symbol of g and N is 1
                continue
            if index == 1:
                # The party with index 1 calculates v_1
                v = int(pow_mod(g, (N - p_i - q_i + 1) // 4, N))
            else:
                # The other parties calculate v_i
                v = int(pow_mod(g, (p_i + q_i) // 4, N))

            v_values.append(v)

        # Though we don't care for the sum of the v_i's, we use the
        # AdditiveVariable named 'v' to easily exchange the values of v_i
        batched_v_i = Batched(
            AdditiveVariable(label="v", modulus=modulus),
            batch_size=correct_param_biprime,
        )
        batched_v_i.set_share(index, v_values)
        return batched_v_i

    @classmethod
    def __biprime_test_with_v_i(
        cls,
        batched_v_i: Batched[AdditiveVariable],
        modulus: int,
        correct_param_biprime: int,
        party_indices: dict[str, int],
    ) -> bool:
        r"""
        Function to test for biprimality of $N$.

        To test the biprimality of $N$, we need to successfully perform a certain
        number of tests (set by 'correct_param_biprime'). All tests need to
        succeed successively, if any test fails we return False (early return).

        We can batch multiple tests in a single communication round to optimize
        communication at the expesive of wasted computational resources.

        The more tests we perform in a batch, the more likely it is that we can
        clear a biprime in a single batch. However, the more tests we perform in
        a batch, the more computational resources we waste because of early
        returns (e.g. Test 2 returns False).

        :param v_values: the v values calculated in `biprime_test_v_calculateion` for this $n$
        :param modulus: the modulus $N$
        :param correct_param_biprime: correctness parameter that affects the
            certainty that the generated modulus is biprime
        :param party_indices: mapping from party name to indices
        :return: true if the test succeeds and false if it fails
        """
        biprime_test_attempts = 0
        successful_biprime_tests = 0

        for v_i in batched_v_i.variables:
            biprime_test_attempts += 1

            # Test whether a primality check holds
            product = 1
            sharing = {i: v_i.get_share(i) for i in party_indices.values()}
            for key, value in sharing.items():
                if key != 1:
                    product *= value
            value1 = v_i.get_share(1)

            # The below test determines if N is "probably" the product of two primes (if the
            # statement is True). Otherwise, N is definitely not the product of two primes.
            success = ((value1 % modulus) == (product % modulus)) or (
                (value1 % modulus) == (-product % modulus)
            )

            if not success:
                logger.debug(
                    f"Biprime test failed! Took {biprime_test_attempts} attempts"
                )
                return False

            successful_biprime_tests += 1

            if successful_biprime_tests >= correct_param_biprime:
                logger.debug(
                    f"Biprime test succeeded! Took {biprime_test_attempts} attempts"
                )
                return True

        # not enough batched v_i available with jacobi symbol of 1
        return False

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
    def __small_prime_divisors_test(cls, prime_list: list[int], modulus: int) -> bool:
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
        index: int,
        pool: Pool,
        prime_list: list[int],
        party_indices: dict[str, int],
        prime_length: int,
        shamir_scheme_t: Shamir,
        shamir_scheme_2t: Shamir,
        correct_param_biprime: int,
        session_id: int,
        batch_size: int = 1,
    ) -> int:
        r"""
        Function that starts a protocol to generate candidates for $p$ and $q$ the multiplication of the two is then checked for biprimality to ensure it is a valid modulus. This is run until it succeeds.

        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param index: index of this party
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param party_indices: mapping from party names to indices
        :param prime_length: desired bit length of $p$ and $q$
        :param shamir_scheme_t: $t$-out-of-$n$ Shamir secret sharing scheme
        :param shamir_scheme_2t: $2t$-out-of-$n$ Shamir secret sharing scheme
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param batch_size: How many $p$'s and $q$'s to generate at once (drastically
            reduces communication at the expense of potentially wasted computation)
        :raises RuntimeError: thrown if the protocol is interrupted
        :return: modulus $N$
        """
        sp_err_counter = 0
        bip_err_counter = 0

        bip = False
        rounds = 0

        while not bip:
            rounds += 1

            # secreting sharings of p and q
            (
                prime_candidate_p,
                prime_candidate_q,
                zero,
                p_additive,
                q_additive,
            ) = await cls._generate_pq(
                pool,
                index,
                prime_length,
                party_indices,
                shamir_scheme_t,
                shamir_scheme_2t,
                session_id,
                batch_size=batch_size,
                msg_id=f"distributed_keygen_session#{session_id}_generate_pq_{rounds}",
            )

            candidate_n: Batched[ShamirVariable] = prime_candidate_p * prime_candidate_q

            # Add 0-share to fix distribution
            candidate_n += zero

            # Reconstruct n
            msg_id = f"distributed_keygen_session#{session_id}_n_{rounds}"
            await exchange_reconstruct(
                candidate_n, index, pool, party_indices, msg_id=msg_id
            )
            candidate_n_plaintext: list[int] = candidate_n.reconstruct()
            zipped_list = zip(
                candidate_n_plaintext, prime_candidate_q, p_additive, q_additive
            )
            candidate_n_small_prime_tested = [
                (n, prime_candidate_q, p_additive, q_additive)
                for (n, prime_candidate_q, p_additive, q_additive) in zipped_list
                if not cls.__small_prime_divisors_test(prime_list, n)
            ]
            sp_err_counter += len(candidate_n_plaintext) - len(
                candidate_n_small_prime_tested
            )

            if len(candidate_n_small_prime_tested) == 0:
                continue

            g_values = await cls.__biprime_test_g_generation(
                correct_param_biprime,
                index,
                [n for (n, _, _, _) in candidate_n_small_prime_tested],
                party_indices,
                pool,
                f"distributed_keygen_session#{session_id}_biprime_test_g_{rounds}",
            )

            candidate_n_small_prime_tests_with_g = [
                (gs,) + candidate
                for gs, candidate in zip(g_values, candidate_n_small_prime_tested)
            ]
            list_to_exchange_v_i = [
                cls.__biprime_test_v_calculation(
                    g_values,
                    index,
                    n,
                    p_additive,
                    q_additive,
                    correct_param_biprime,
                )
                for (
                    g_values,
                    n,
                    _,
                    p_additive,
                    q_additive,
                ) in candidate_n_small_prime_tests_with_g
            ]

            await exchange_reconstruct(
                list_to_exchange_v_i,
                index,
                pool,
                party_indices,
                msg_id=f"distributed_keygen_session#{session_id}_biprime_test_v_{rounds}_v",
            )

            for i, (n, prime_candidate_q_i, p_i_additive, q_i_additive) in enumerate(
                candidate_n_small_prime_tested
            ):
                # Once we found a modulus that is biprime, we will needs the
                # shares of p and q to generate the secret key later on
                shares.p = Shares.P(p_i_additive, prime_candidate_q_i.get_shares())
                shares.q = Shares.Q(q_i_additive, prime_candidate_q_i.get_shares())

                batched_v_i = list_to_exchange_v_i[i]
                bip = cls.__biprime_test_with_v_i(
                    batched_v_i, n, correct_param_biprime, party_indices
                )

                if not bip:
                    bip_err_counter += 1
                else:
                    logger.info(f"N = {n}")
                    logger.info(
                        f"Checked {sp_err_counter} primes for small prime divisors in {rounds} rounds"
                    )
                    logger.info(f"Checked {bip_err_counter} candidates for biprimality")
                    return n

        raise RuntimeError("Could not generate a valid modulus")

    @classmethod
    async def generate_secret_key(
        cls,
        stat_sec_shamir: int,
        number_of_players: int,
        corruption_threshold: int,
        shares: Shares,
        index: int,
        pool: Pool,
        prime_list: list[int],
        prime_length: int,
        party_indices: dict[str, int],
        correct_param_biprime: int,
        shamir_scheme_t: Shamir,
        shamir_scheme_2t: Shamir,
        session_id: int,
        batch_size: int,
    ) -> PaillierSharedKey:
        """
        Functions that generates the modulus and sets up the sharing of the private key

        :param stat_sec_shamir: security parameter for the Shamir secret sharing over the integers
        :param number_of_players: total number of participants in this session (including self)
        :param corruption_threshold: Maximum number of allowed corruptions
        :param shares: dictionary that keeps track of shares for parties for certain numbers
        :param index: index of this party
        :param pool: network of involved parties
        :param prime_list: list of prime numbers
        :param prime_length: desired bit length of $p$ and $q$
        :param party_indices: mapping from party names to indices
        :param correct_param_biprime: correctness parameter that affects the certainty that the
            generated $N$ is a product of two primes
        :param shamir_scheme_t: $t$-out-of-$n$ Shamir secret sharing scheme
        :param shamir_scheme_2t: $2t$-out-of-$n$ Shamir secret sharing scheme
        :param session_id: The unique session identifier belonging to the protocol that generated
            the keys for this DistributedPaillier scheme.
        :param batch_size: How many $p$'s and $q$'s to generate at once (drastically
            reduces communication at the expense of potentially wasted computation)
        :return: shared secret key
        """

        modulus = await cls.compute_modulus(
            shares,
            index,
            pool,
            prime_list,
            party_indices,
            prime_length,
            shamir_scheme_t,
            shamir_scheme_2t,
            correct_param_biprime,
            session_id,
            batch_size,
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

    class StoredDistributedPaillier(TypedDict):
        pub_key: PaillierPublicKey
        priv_key: PaillierSharedKey
        shares: Shares
        precision: int
        index: int
        party_indices: dict[str, int]
        corruption_threshold: int

    def store_private_key(self) -> bytes:
        """
        Serialize the entire key including the private key to bytes, such that it can be stored for
        later use. The key can be loaded using the function `load_private_key_from_bytes`.

        :return: byte object representing the key.
        :raise ImportError: When the 'tno.mpc.communication' module is not installed
        """
        if not COMMUNICATION_INSTALLED:
            raise ImportError(
                "Could not find the module 'tno.mpc.communication', which is needed for serialization"
            )

        object_to_serialize = {
            "pub_key": self.public_key,
            "priv_key": self.secret_key,
            "precision": self.precision,
            "index": self.index,
            "party_indices": self.party_indices,
            "corruption_threshold": self.corruption_threshold,
        }
        return Serialization.pack(
            object_to_serialize,
            msg_id="",
            use_pickle=False,
            option=DIST_KEY_STORAGE_PACK_OPTIONS,
        )

    @classmethod
    async def load_private_key_from_bytes(
        cls, obj_bytes: bytes, pool: Pool, distributed: bool
    ) -> DistributedPaillier:
        """
        Create a distributed paillier key from the bytes provided. The bytes must represent a distributed paillier key. The number of parties must be equal to the number of parties in a pool.

        :param obj_bytes: the bytes representing the key
        :param pool: The pool used for the communication
        :param distributed: Whether the different parties are run on different python instances
        :return: The distributed paillier key derived from the obj_bytes
        :raise ValueError: When the number of parties in the pool does not correspond to the number of parties expected by the key.
        :raise ImportError: When the 'tno.mpc.communication' module is not installed
        """
        if not COMMUNICATION_INSTALLED:
            raise ImportError(
                "Could not find the module 'tno.mpc.communication', which is needed for deserialization"
            )

        _, deserialized_dict = Serialization.unpack(
            obj_bytes, False, ormsgpack.OPT_NON_STR_KEYS
        )
        if isinstance(deserialized_dict, list):
            raise TypeError("Expected a dict not a list")
        deserialized = cast(
            DistributedPaillier.StoredDistributedPaillier, deserialized_dict
        )

        if len(deserialized["party_indices"]) != len(pool.pool_handlers) + 1:
            raise ValueError(
                f"The number of parties in the pool ({len(pool.pool_handlers)+1} does not correspond with the number of parties expected by the key({len(deserialized)}."
            )

        _, session_id = await DistributedPaillier.get_indices(pool)
        index = deserialized["party_indices"]["self"]
        dist_paillier = DistributedPaillier(
            deserialized["pub_key"],
            deserialized["priv_key"],
            deserialized["precision"],
            pool,
            index,
            deserialized["party_indices"],
            session_id,
            distributed,
            deserialized["corruption_threshold"],
        )
        cls.__register_scheme(dist_paillier, distributed)
        return dist_paillier

    class SerializedDistributedPaillier(Paillier.SerializedPaillier, TypedDict):
        """
        Serialized DistributedPaillier for use with the communication module.
        """

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
        origin: HTTPClient | None = ...,
        **kwargs: Any,
    ) -> DistributedPaillier: ...

    @overload
    @staticmethod
    def deserialize(
        obj: Paillier.SerializedPaillier,
        *,
        origin: HTTPClient | None = ...,
        **kwargs: Any,
    ) -> Paillier: ...

    @staticmethod
    def deserialize(
        obj: (
            DistributedPaillier.SerializedDistributedPaillier
            | Paillier.SerializedPaillier
        ),
        *,
        origin: HTTPClient | None = None,
        **kwargs: Any,
    ) -> DistributedPaillier | Paillier:
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
if COMMUNICATION_INSTALLED:
    try:
        Serialization.register_class(DistributedPaillier, check_annotations=False)
    except RepetitionError:
        pass
