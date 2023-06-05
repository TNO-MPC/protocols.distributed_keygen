"""
This module tests the serialization of DistributedPaillier instances.
"""

from tno.mpc.encryption_schemes.shamir import IntegerShares, ShamirSecretSharingIntegers

from tno.mpc.protocols.distributed_keygen import PaillierSharedKey


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
