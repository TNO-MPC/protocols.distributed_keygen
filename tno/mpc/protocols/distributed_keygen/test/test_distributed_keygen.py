"""
Tests that can be ran using pytest to test the distributed keygen functionality
"""

import asyncio
from typing import List, Tuple, Union, cast

import pytest

from tno.mpc.communication import Pool
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    fixture_pool_http_3p,
)

from tno.mpc.protocols.distributed_keygen import DistributedPaillier


@pytest.fixture(name="distributed_schemes")
@pytest.mark.asyncio
async def fixture_distributed_schemes(
    pool_http_3p: Tuple[Pool, Pool, Pool]
) -> Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier]:
    """
    Constructs schemes to use for distributed key generation.

    :param pool_http_3p: collection of (three) communication pools
    :return: a collection of (three) schemes
    """
    corruption_threshold = 1
    key_length = 64
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    distributed_schemes = tuple(
        await asyncio.gather(
            *[
                DistributedPaillier.from_security_parameter(
                    pool_http_3p[i],
                    corruption_threshold,
                    key_length,
                    prime_threshold,
                    correct_param_biprime,
                    stat_sec_shamir,
                    distributed=False,
                    precision=8,
                )
                for i in range(3)
            ]
        )
    )
    return cast(
        Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier],
        distributed_schemes,
    )


@pytest.mark.asyncio
async def test_distributed_paillier_init(
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ]
) -> None:
    """
    Tests initialization of distributed key generation (validate public keys)

    :param distributed_schemes: a collection of (three) schemes
    """
    assert (
        distributed_schemes[0].public_key
        == distributed_schemes[1].public_key
        == distributed_schemes[2].public_key
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
async def test_distributed_paillier_with_communication(
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    plaintext: Union[float, int],
) -> None:
    """
    Tests distributed encryption and decryption using communication

    :param distributed_schemes: a collection of (three) schemes
    :param plaintext: plaintext to encrypt and decrypt
    """
    enc = {0: distributed_schemes[0].encrypt(plaintext)}
    await distributed_schemes[0].pool.send("local1", enc[0])
    await distributed_schemes[0].pool.send("local2", enc[0])

    enc[1] = await distributed_schemes[1].pool.recv("local0")
    enc[2] = await distributed_schemes[2].pool.recv("local0")
    dec = await asyncio.gather(
        *[distributed_schemes[i].decrypt(enc[i]) for i in range(3)]
    )
    assert dec[0] == dec[1] == dec[2] == plaintext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
async def test_distributed_paillier_serialization(
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    plaintext: Union[float, int],
) -> None:
    """
    Tests serialization of the distributed Paillier.

    :param distributed_schemes: a collection of (three) schemes
    :param plaintext: plaintext to encrypt
    """
    enc = {0: distributed_schemes[0].encrypt(plaintext)}
    await distributed_schemes[0].pool.send("local1", enc[0])
    await distributed_schemes[0].pool.send("local2", enc[0])
    await distributed_schemes[0].pool.send("local1", distributed_schemes[0])
    await distributed_schemes[0].pool.send("local2", distributed_schemes[0])

    enc[1] = await distributed_schemes[1].pool.recv("local0")
    enc[2] = await distributed_schemes[2].pool.recv("local0")
    d_scheme_recv_1 = await distributed_schemes[1].pool.recv("local0")
    d_scheme_recv_2 = await distributed_schemes[2].pool.recv("local0")
    # check equality of received values
    assert enc[0] == enc[1] == enc[2]
    assert d_scheme_recv_1 == distributed_schemes[1] == distributed_schemes[0]
    assert d_scheme_recv_2 == distributed_schemes[2] == distributed_schemes[0]


@pytest.mark.asyncio
async def test_distributed_paillier_exception(
    pool_http_3p: Tuple[Pool, Pool, Pool]
) -> None:
    """
    Tests raising of exception when corruption threshold is set incorrectly.

    :param pool_http_3p: collection of (three) communication pools
    """
    corruption_threshold = 2
    key_length = 64
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    with pytest.raises(ValueError):
        _distributed_schemes = await asyncio.gather(
            *[
                DistributedPaillier.from_security_parameter(
                    pool_http_3p[i],
                    corruption_threshold,
                    key_length,
                    prime_threshold,
                    correct_param_biprime,
                    stat_sec_shamir,
                    distributed=False,
                )
                for i in range(3)
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
async def test_distributed_paillier_encrypt_decrypt(
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    plaintext: Union[float, int],
) -> None:
    """
    Tests distributed encryption and decryption

    :param distributed_schemes: a collection of (three) schemes
    :param plaintext: plaintext to encrypt and decrypt
    """
    enc = distributed_schemes[0].encrypt(plaintext)
    dec = await asyncio.gather(*[distributed_schemes[i].decrypt(enc) for i in range(3)])
    assert dec[0] == dec[1] == dec[2] == plaintext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receivers,result_indices",
    [
        ((["self"], ["local0"], ["local0"]), (0,)),
        ((["self", "local1"], ["local0", "self"], ["local0", "local1"]), (0, 1)),
    ],
)
async def test_distributed_paillier_encrypt_decrypt_receivers(
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    receivers: Tuple[List[str]],
    result_indices: Tuple[int],
) -> None:
    """
    Tests distributed decryption reveiling the results to a subset of receivers only.

    :param distributed_schemes: a collection of (three) schemes
    :param receivers: parties to reveal the decryptions to
    :param result_indices: indices of the parties that should have received the decryptions
    """
    enc = distributed_schemes[0].encrypt(42)
    dec = await asyncio.gather(
        *[distributed_schemes[i].decrypt(enc, receivers=receivers[i]) for i in range(3)]
    )
    for i in range(3):
        if i in result_indices:
            assert dec[i] == 42
        else:
            assert dec[i] is None
