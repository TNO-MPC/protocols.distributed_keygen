"""
Tests that can be run using pytest to test the distributed keygen functionality
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import pytest

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import PaillierCiphertext
from tno.mpc.encryption_schemes.paillier.paillier import Plaintext

from tno.mpc.protocols.distributed_keygen import DistributedPaillier


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_distributed_paillier_with_communication(
    distributed_schemes: tuple[DistributedPaillier, ...],
    plaintext: float | int,
) -> None:
    """
    Tests distributed encryption and decryption using communication

    :param distributed_schemes: a collection of schemes
    :param plaintext: plaintext to encrypt and decrypt
    """
    enc = {0: distributed_schemes[0].encrypt(plaintext)}
    distributed_schemes[0].pool.async_broadcast(enc[0], "encryption")
    assert not enc[0].fresh
    for iplayer in range(1, len(distributed_schemes)):
        enc[iplayer] = await distributed_schemes[iplayer].pool.recv(
            "local0", "encryption"
        )

    dec = await asyncio.gather(
        *[
            distributed_schemes[i].decrypt(enc[i])
            for i in range(len(distributed_schemes))
        ]
    )
    assert all(d == plaintext for d in dec)


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_distributed_paillier_serialization(
    distributed_schemes: tuple[DistributedPaillier, ...],
    plaintext: float | int,
) -> None:
    """
    Tests serialization of the distributed Paillier.

    :param distributed_schemes: a collection of schemes
    :param plaintext: plaintext to encrypt
    """
    enc = {0: distributed_schemes[0].encrypt(plaintext)}
    distributed_schemes[0].pool.async_broadcast(enc[0], "encryption")
    assert not enc[0].fresh
    distributed_schemes[0].pool.async_broadcast(distributed_schemes[0], "scheme")

    # check equality of received values
    for iplayer in range(1, len(distributed_schemes)):
        enc[iplayer] = await distributed_schemes[iplayer].pool.recv(
            "local0", "encryption"
        )
        d_scheme_recv = await distributed_schemes[iplayer].pool.recv("local0", "scheme")

        assert enc[0] == enc[iplayer]
        assert d_scheme_recv == distributed_schemes[iplayer] == distributed_schemes[0]


@pytest.mark.asyncio
async def test_distributed_paillier_exception(pool_http: tuple[Pool, ...]) -> None:
    """
    Tests raising of exception when corruption threshold is set incorrectly.

    :param pool_http: collection of communication pools
    """
    max_corruption_threshold = math.ceil(len(pool_http) / 2) - 1
    corruption_threshold = max_corruption_threshold + 1
    key_length = 64
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    with pytest.raises(ValueError):
        _distributed_schemes = await asyncio.gather(
            *[
                DistributedPaillier.from_security_parameter(
                    pool_http[i],
                    corruption_threshold,
                    key_length,
                    prime_threshold,
                    correct_param_biprime,
                    stat_sec_shamir,
                    distributed=False,
                )
                for i in range(len(pool_http))
            ]
        )


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt(
    distributed_schemes: tuple[DistributedPaillier, ...],
    plaintext: float | int,
) -> None:
    """
    Tests distributed encryption and decryption

    :param distributed_schemes: a collection of schemes
    :param plaintext: plaintext to encrypt and decrypt
    """
    enc = distributed_schemes[0].encrypt(plaintext)
    dec = await asyncio.gather(
        *[distributed_schemes[i].decrypt(enc) for i in range(len(distributed_schemes))]
    )
    assert all(d == plaintext for d in dec)


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt_parallel(
    distributed_schemes: tuple[DistributedPaillier, ...],
    plaintext: float | int,
) -> None:
    """
    Tests distributed encryption and decryption in parallel

    :param distributed_schemes: a collection of schemes
    :param plaintext: plaintext to encrypt and decrypt
    """
    encs = [distributed_schemes[0].encrypt(plaintext) for _ in range(3)]
    decs = await asyncio.gather(
        *[
            asyncio.gather(
                *[
                    distributed_schemes[i].decrypt(enc)
                    for i in range(len(distributed_schemes))
                ]
            )
            for enc in encs
        ]
    )
    assert all(all(d == plaintext for d in dec) for dec in decs)


@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt_sequence(
    distributed_schemes: tuple[DistributedPaillier, ...],
) -> None:
    """
    Tests distributed sequence decryption

    :param distributed_schemes: a collection of schemes
    """
    plaintexts = [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
    ciphertexts = []
    for plaintext in plaintexts:
        ciphertexts.append(distributed_schemes[0].encrypt(plaintext))

    decryptions = await asyncio.gather(
        *[
            distributed_schemes[i].decrypt_sequence(ciphertexts)
            for i in range(len(distributed_schemes))
        ]
    )

    for decryption_list in decryptions:
        assert decryption_list is not None
        for idx, decryption in enumerate(decryption_list):
            assert plaintexts[idx] == decryption


@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt_sequence_parallel(
    distributed_schemes: tuple[DistributedPaillier, ...],
) -> None:
    """
    Tests distributed sequence decryption when run in parallel

    :param distributed_schemes: a collection of schemes
    """
    plaintexts_list: list[list[float] | list[int]] = [
        [1, 2, 3],
        [-1, -2, -3],
        [1.5, 42.42424242, -1.5 - 42.42424242],
    ]
    ciphertexts_list = []
    for plaintext_list in plaintexts_list:
        ciphertexts_list.append(
            [distributed_schemes[0].encrypt(plaintext) for plaintext in plaintext_list]
        )

    async def safe_decrypt_sequence(
        distributed_scheme: DistributedPaillier,
        ciphertexts: Sequence[PaillierCiphertext],
    ) -> list[Plaintext]:
        decryption = await distributed_scheme.decrypt_sequence(ciphertexts)
        assert decryption is not None, "Decryption result should not be None"
        return decryption

    decryption_lists: list[list[list[Plaintext]]] = await asyncio.gather(
        *[
            asyncio.gather(
                *[
                    safe_decrypt_sequence(distributed_schemes[i], ciphertexts)
                    for i in range(len(distributed_schemes))
                ]
            )
            for ciphertexts in ciphertexts_list
        ]
    )

    for result_lists, correct_decryption_list in zip(decryption_lists, plaintexts_list):
        for decryption_list in result_lists:
            assert decryption_list == correct_decryption_list


@pytest.mark.parametrize(
    "receivers_id,result_indices",
    [
        (0, (0,)),
        (1, (0, 1)),
    ],
)
@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt_receivers(
    distributed_schemes: tuple[DistributedPaillier, ...],
    receivers_id: int,
    result_indices: tuple[int],
) -> None:
    """
    Tests distributed decryption revealing the results to a subset of receivers only.

    :param distributed_schemes: a collection of schemes
    :param receivers_id: parties to reveal the decryptions to
    :param result_indices: indices of the parties that should have received the decryptions
    :raises ValueError: if receivers_id is invalid
    """
    if receivers_id == 0:
        receiver0_list = [["local0"]] * len(distributed_schemes)
        receiver0_list[0] = ["self"]
        receivers = tuple(receiver0_list)
    elif receivers_id == 1:
        receivers01_list = [["local0", "local1"]] * len(distributed_schemes)
        receivers01_list[0] = ["self", "local1"]
        receivers01_list[1] = ["local0", "self"]
        receivers = tuple(receivers01_list)
    else:
        raise ValueError("Invalid receivers_id")

    enc = distributed_schemes[0].encrypt(42)
    dec = await asyncio.gather(
        *[
            distributed_schemes[i].decrypt(enc, receivers=receivers[i])
            for i in range(len(distributed_schemes))
        ]
    )
    for i in range(len(distributed_schemes)):
        if i in result_indices:
            assert dec[i] == 42
        else:
            assert dec[i] is None


@pytest.mark.parametrize(
    "collection_type",
    (
        dict,
        list,
        tuple,
    ),
)
@pytest.mark.asyncio
async def test_pool_broadcast_collection(
    distributed_schemes: tuple[DistributedPaillier, ...],
    collection_type: type[Any],
) -> None:
    """
    Test whether sending of collections of ciphertexts using the broadcast method works as expected.

    :param distributed_schemes: a collection of schemes
    :param collection_type: The type of collection that is to be communicated.
    """
    plaintexts = [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
    ciphertexts = map(distributed_schemes[0].encrypt, plaintexts)
    if collection_type == dict:
        collection: Any = {}
        for index, ciphertext in enumerate(ciphertexts):
            collection[str(index)] = ciphertext
    else:
        collection = collection_type(ciphertexts)

    distributed_schemes[0].pool.async_broadcast(collection, "ciphertext_collection")

    received_collections = await asyncio.gather(
        *[
            distributed_schemes[i].pool.recv("local0", "ciphertext_collection")
            for i in range(1, len(distributed_schemes))
        ]
    )

    for received_collection in received_collections:
        assert received_collection == collection
