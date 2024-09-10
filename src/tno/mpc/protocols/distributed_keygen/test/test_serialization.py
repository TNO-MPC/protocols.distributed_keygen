"""
This module tests the serialization of DistributedPaillier instances.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.shamir import IntegerShares, ShamirSecretSharingIntegers

from tno.mpc.protocols.distributed_keygen import DistributedPaillier, PaillierSharedKey


def test_serialization_paillier_shared_key() -> None:
    """
    Test to determine whether the secret key serialization works properly for Paillier shared keys.
    """
    orig_key = PaillierSharedKey(
        n=1,
        t=0,
        player_id=0,
        share=IntegerShares(ShamirSecretSharingIntegers(), {1: 1}, 1, 1),
        theta=1,
    )
    assert (
        PaillierSharedKey.deserialize(PaillierSharedKey.serialize(orig_key)) == orig_key
    )


@pytest.mark.parametrize(
    "plaintext", [1, 2, 3, -1, -2, -3, 1.5, 42.42424242, -1.5, -42.42424242]
)
@pytest.mark.asyncio
async def test_storing_and_loading_key(
    distributed_schemes: tuple[DistributedPaillier, ...],
    plaintext: float | int,
) -> None:
    """
    Test to see if we can store and load a key
    """
    number_of_schemes = len(distributed_schemes)
    pools: list[Pool] = [
        distributed_schemes[index].pool for index in range(number_of_schemes)
    ]
    schemes_as_bytes = [
        DistributedPaillier.store_private_key(distributed_schemes[index])
        for index in range(number_of_schemes)
    ]

    reconstructed_schemes = await asyncio.gather(
        *[
            DistributedPaillier.load_private_key_from_bytes(
                schemes_as_bytes[index], pools[index], False
            )
            for index in range(number_of_schemes)
        ]
    )

    enc = {0: reconstructed_schemes[0].encrypt(plaintext)}
    reconstructed_schemes[0].pool.async_broadcast(enc[0], "encryption")
    assert not enc[0].fresh
    for iplayer in range(1, number_of_schemes):
        enc[iplayer] = await reconstructed_schemes[iplayer].pool.recv(
            "local0", "encryption"
        )

    dec = await asyncio.gather(
        *[reconstructed_schemes[i].decrypt(enc[i]) for i in range(number_of_schemes)]
    )
    assert all(d == plaintext for d in dec)


# Comment out the skip to generate new key files to use for testing.
@pytest.mark.skip(reason="No need to generate key files for each run")
def test_store_key_to_file(
    distributed_schemes_fresh: tuple[DistributedPaillier, ...]
) -> None:
    """
    Test which generates different keys and store them to the file system. These files are also included in the package.

    :param distributed_schemes_fresh: The schemes to store to the file system.
    """
    base_path = Path(f"{os.path.dirname(__file__)}/test_data")
    for index, key in enumerate(distributed_schemes_fresh):
        with open(
            base_path.joinpath(
                f"distributed_key_threshold_{key.corruption_threshold}_{len(distributed_schemes_fresh)}parties_{index}.obj"
            ),
            "wb",
        ) as file:
            file.write(DistributedPaillier.store_private_key(key))
