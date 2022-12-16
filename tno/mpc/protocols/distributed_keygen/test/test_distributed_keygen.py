"""
Tests that can be run using pytest to test the distributed keygen functionality
"""

import asyncio
import math
from typing import Any, AsyncGenerator, List, Tuple, Type, Union

import pytest
import pytest_asyncio
from _pytest.fixtures import FixtureRequest

from tno.mpc.communication import Pool, Serialization
from tno.mpc.communication.test import event_loop  # pylint: disable=unused-import
from tno.mpc.communication.test.pool_fixtures_http import (  # pylint: disable=unused-import
    fixture_pool_http_3p,
    fixture_pool_http_4p,
    fixture_pool_http_5p,
)

from tno.mpc.protocols.distributed_keygen import DistributedPaillier


@pytest.fixture(
    name="pool_http",
    params=[3, 4, 5],
    ids=["3-party", "4-party", "5-party"],
    scope="module",
)
def fixture_pool_http(
    request: FixtureRequest,
    pool_http_3p: AsyncGenerator[Tuple[Pool, ...], None],
    pool_http_4p: AsyncGenerator[Tuple[Pool, ...], None],
    pool_http_5p: AsyncGenerator[Tuple[Pool, ...], None],
) -> AsyncGenerator[Tuple[Pool, ...], None]:
    """
    Creates a collection of 3, 4 and 5 communication pools

    :param pool_http_3p: Pool of 3 HTTP clients.
    :param pool_http_4p: Pool of 4 HTTP clients.
    :param pool_http_5p: Pool of 5 HTTP clients.
    :param request: A fixture request used to indirectly parametrize.
    :raise NotImplementedError: raised when based on the given param, no fixture can be created
    :return: a collection of communication pools
    """
    if request.param == 3:
        return pool_http_3p
    if request.param == 4:
        return pool_http_4p
    if request.param == 5:
        return pool_http_5p
    raise NotImplementedError("This has not been implemented")


@pytest_asyncio.fixture(
    name="distributed_schemes",
    params=list(range(2)),
    ids=["corruption_threshold " + str(i) for i in range(2)],
    scope="module",
)
async def fixture_distributed_schemes(
    pool_http: Tuple[Pool, ...],
    request: FixtureRequest,
) -> Tuple[DistributedPaillier, ...]:
    """
    Constructs schemes to use for distributed key generation.

    :param pool_http: collection of communication pools
    :param request: Fixture request
    :return: a collection of schemes
    """
    Serialization.register_class(
        DistributedPaillier, check_annotations=False, overwrite=True
    )
    corruption_threshold: int = request.param
    key_length = 64
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    distributed_schemes: Tuple[DistributedPaillier, ...] = tuple(
        await asyncio.gather(
            *[
                DistributedPaillier.from_security_parameter(
                    pool,
                    corruption_threshold,
                    key_length,
                    prime_threshold,
                    correct_param_biprime,
                    stat_sec_shamir,
                    distributed=False,
                    precision=8,
                )
                for pool in pool_http
            ]
        )
    )
    return distributed_schemes


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_distributed_paillier_with_communication(
    distributed_schemes: Tuple[DistributedPaillier, ...],
    plaintext: Union[float, int],
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
    plaintext: Union[float, int],
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
async def test_distributed_paillier_exception(pool_http: Tuple[Pool, ...]) -> None:
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
    plaintext: Union[float, int],
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
    plaintext: Union[float, int],
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
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
        for idx, decryption in enumerate(decryption_list):
            assert plaintexts[idx] == decryption


@pytest.mark.asyncio
async def test_distributed_paillier_encrypt_decrypt_sequence_parallel(
    distributed_schemes: Tuple[DistributedPaillier, ...],
) -> None:
    """
    Tests distributed sequence decryption when run in parallel

    :param distributed_schemes: a collection of schemes
    """
    plaintexts_list: List[Union[List[float], List[int]]] = [
        [1, 2, 3],
        [-1, -2, -3],
        [1.5, 42.42424242, -1.5 - 42.42424242],
    ]
    ciphertexts_list = []
    for plaintext_list in plaintexts_list:
        ciphertexts_list.append(
            [distributed_schemes[0].encrypt(plaintext) for plaintext in plaintext_list]
        )

    decryption_lists: List[List[Union[List[float], List[int]]]] = await asyncio.gather(
        *[
            asyncio.gather(
                *[
                    distributed_schemes[i].decrypt_sequence(ciphertexts)
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
    receivers_id: int,
    result_indices: Tuple[int],
) -> None:
    """
    Tests distributed decryption revealing the results to a subset of receivers only.

    :param distributed_schemes: a collection of schemes
    :param receivers_id: parties to reveal the decryptions to
    :param result_indices: indices of the parties that should have received the decryptions
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
    distributed_schemes: Tuple[DistributedPaillier, ...],
    collection_type: Type[Any],
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
