"""
Test fixtures
"""

from __future__ import annotations

import asyncio
import os.path
from pathlib import Path
from typing import Callable

import pytest
import pytest_asyncio
from _pytest.fixtures import FixtureRequest

from tno.mpc.communication import Pool, Serialization

from tno.mpc.protocols.distributed_keygen import DistributedPaillier


@pytest.fixture(
    name="pool_http",
    params=[3, 4, 5],
    ids=["3-party", "4-party", "5-party"],
    scope="module",
)
def fixture_pool_http(
    request: FixtureRequest,
    http_pool_group_factory: Callable[[int], tuple[Pool, ...]],
) -> tuple[Pool, ...]:
    """
    Creates a collection of 3, 4 and 5 communication pools

    :param http_pool_group_factory: Factory for generating a set of pools with configured
        connection of arbitrary size.
    :param request: A fixture request used to indirectly parametrize.
    :raise NotImplementedError: raised when based on the given param, no fixture can be created
    :return: a collection of communication pools
    """
    return http_pool_group_factory(request.param)


@pytest_asyncio.fixture(
    name="distributed_schemes_fresh",
    params=list(zip([0, 1], [1, 100])),
    ids=[
        "corruption_threshold " + str(t) + "_batch_" + str(b)
        for t, b in list(zip([0, 1], [1, 100]))
    ],
    scope="module",
)
async def fixture_distributed_schemes(
    pool_http: tuple[Pool, ...],
    request: FixtureRequest,
) -> tuple[DistributedPaillier, ...]:
    """
    Constructs schemes to use for distributed key generation.

    :param pool_http: collection of communication pools
    :param request: Fixture request
    :return: a collection of schemes
    """
    Serialization.register_class(
        DistributedPaillier, check_annotations=False, overwrite=True
    )
    corruption_threshold: int = request.param[0]
    batch_size: int = request.param[1]

    key_length = 64
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    distributed_schemes: tuple[DistributedPaillier, ...] = tuple(
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
                    batch_size=batch_size,
                )
                for pool in pool_http
            ]
        )
    )
    return distributed_schemes


@pytest_asyncio.fixture(
    name="distributed_schemes",
    params=list([0, 1]),
    ids=["corruption_threshold 0", "corruption_threshold 1"],
    scope="module",
)
async def fixture_distributed_schemes_from_file(
    pool_http: tuple[Pool, ...],
    request: FixtureRequest,
) -> tuple[DistributedPaillier, ...]:
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
    base_path = Path(f"{os.path.dirname(__file__)}/test_data")
    number_of_parties = len(pool_http)
    file_paths = [
        base_path.joinpath(
            f"distributed_key_threshold_{corruption_threshold}_{number_of_parties}parties_{index}.obj"
        )
        for index in range(number_of_parties)
    ]

    distributed_schemes: tuple[DistributedPaillier, ...] = tuple(
        await asyncio.gather(
            *[
                DistributedPaillier.load_private_key_from_bytes(
                    file_paths[index].read_bytes(), pool_http[index], False
                )
                for index in range(number_of_parties)
            ]
        )
    )
    return distributed_schemes
