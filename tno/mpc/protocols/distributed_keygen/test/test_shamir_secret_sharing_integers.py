"""
Tests for the regular Shamir Secret Sharing scheme functionality.
"""
import pytest
import sympy
from _pytest.fixtures import SubRequest

from tno.mpc.protocols.distributed_keygen.shamir_secret_sharing_integers import (
    ShamirSecretSharingIntegers,
)

moduli = [sympy.prime(_) for _ in range(13000, 13010)]  # at least 10657
polynomial_degree = [2, 2, 2, 3, 3, 3, 4, 4, 5, 5]
n_parties = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
secrets = list(range(0, 100, 10))


@pytest.fixture(
    name="shamir_scheme",
    params=[
        {"number_of_parties": n_parties[i], "polynomial_degree": polynomial_degree[i]}
        for i in range(10)
    ],
)
def fixture_shamir_scheme(request: SubRequest) -> ShamirSecretSharingIntegers:
    """
    Create Shamir Schemes using the parameters defined in the pre-amble.

    :param request: Request for a scheme.
    :return: ShamirSecretSharingIntegers scheme using one of the parameter sets.
    """
    return ShamirSecretSharingIntegers(max_int=1000, **request.param)


@pytest.mark.parametrize("secret", secrets)
def test_share_and_reconstruct_secret(
    shamir_scheme: ShamirSecretSharingIntegers, secret: int
) -> None:
    """
    Test the sharing and reconstructing of a secret using a shamir sharing scheme.

    :param shamir_scheme: Shamir sharing scheme to be used.
    :param secret: Secret to be shared and reconstructed.
    """
    sharing = shamir_scheme.share_secret(secret)
    assert sharing.reconstruct_secret() == secret


@pytest.mark.parametrize(
    "secret_1, secret_2",
    [(secrets[i], secrets[j]) for i in range(10) for j in range(10)],
)
def test_add(
    shamir_scheme: ShamirSecretSharingIntegers, secret_1: int, secret_2: int
) -> None:
    """
    Test the correct addition of two shared values using a shamir sharing scheme.

    :param shamir_scheme: The scheme with which both secrets are shared.
    :param secret_1: To be shared number.
    :param secret_2: To be shared number.
    """
    sharing_1 = shamir_scheme.share_secret(secret_1)
    sharing_2 = shamir_scheme.share_secret(secret_2)
    add_1 = sharing_1 + sharing_2
    add_2 = sharing_1 + sharing_2
    assert (
        secret_1 + secret_2 == add_1.reconstruct_secret() == add_2.reconstruct_secret()
    )


@pytest.mark.parametrize(
    "secret_1, secret_2",
    [(secrets[i], secrets[j]) for i in range(10) for j in range(10)],
)
def test_mul(
    shamir_scheme: ShamirSecretSharingIntegers, secret_1: int, secret_2: int
) -> None:
    """
    Test the correct multiplication of two shared values using a shamir sharing scheme.

    :param shamir_scheme: The scheme with which both secrets are shared.
    :param secret_1: To be shared number.
    :param secret_2: To be shared number.
    """
    sharing_1 = shamir_scheme.share_secret(secret_1)
    sharing_2 = shamir_scheme.share_secret(secret_2)
    mul_1 = sharing_1 * sharing_2
    mul_2 = sharing_1 * sharing_2
    assert (
        secret_1 * secret_2 == mul_1.reconstruct_secret() == mul_2.reconstruct_secret()
    )


@pytest.mark.parametrize(
    "secret_1, secret_2",
    [(secrets[i], secrets[j]) for i in range(10) for j in range(10)],
)
def test_rmul_scalar(
    shamir_scheme: ShamirSecretSharingIntegers, secret_1: int, secret_2: int
) -> None:
    """
    Test the correct multiplication of a scalar with a shared value.

    :param shamir_scheme: The scheme with which the secret is shared.
    :param secret_1: To be shared number.
    :param secret_2: Scalar for multiplication.
    """
    sharing_1 = shamir_scheme.share_secret(secret_1)
    mul = secret_2 * sharing_1
    assert secret_1 * secret_2 == mul.reconstruct_secret()
