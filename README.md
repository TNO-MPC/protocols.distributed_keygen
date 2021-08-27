# TNO MPC Lab - Protocols - Distributed Keygen

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.distributed_keygen is part of the TNO Python Toolbox.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the tno.mpc.protocols.distributed_keygen package can be found [here](https://docs.mpc.tno.nl/protocols/distributed_keygen/0.5.4).

## Install

Easily install the tno.mpc.protocols.distributed_keygen package using pip:
```console
$ python -m pip install tno.mpc.protocols.distributed_keygen
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.protocols.distributed_keygen[tests]'
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.protocols.distributed_keygen[gmpy]'
```

## Protocol description
A more elaborate protocol description can be found in [An implementation of the Paillier crypto system with threshold decryption without a trusted dealer](https://eprint.iacr.org/2019/1136.pdf).

## Usage

The distributed keygen module can be used by first creating a `Pool` 
from the `tno.mpc.communication` library. 

```python
from tno.mpc.communication.pool import Pool

pool = Pool(...) # initialize pool with ips etc
```

### Starting the protocol
After initializing a pool, you can use the class method `DistributedPaillier.from_security_parameter()` to create an instance of the `DistributedPaillier` class. The class method automatically starts the protocol between the parties inside the pool to jointly generate a public key and a shared secret key.

Under `Appendix` at the end of this README, you can find 3 files:
- `distributed_keygen_example_local.py`: this script runs the protocol in one python instance on different ports of the same machine.
- `distributed_keygen_example_distributed.py`: this script runs the protocol for one machine only and this script should be run on each machine.
- `run_protocol.sh`: this batch script takes one parameter, the number of parties, and starts `distributed_keygen_example_distributed.py` with the right arguments for each machine on `localhost`.

### After initialization

When a DistributedPaillier instance has been generated (either locally or distributedly), the public key can be used to encrypt messages and the shared secret key
can be used to distributively decrypt. Note that these methods are async methods, so they either
need to be run in an event loop or inside another async method using await.


```python
# The assumption here is that this code is placed inside an async method
ciphertext = distributed_scheme.encrypt(42)          # encryption of 42
await distributed_scheme.pool.send(..., ciphertext)  # send the ciphertext to another party
await distributed_scheme.pool.recv(...)              # receive message from other party

plaintext = await distributed_scheme.decrypt(ciphertext)   # execute decryption protocol given the received shares
plaintext = await distributed_scheme.decrypt(ciphertext, receivers=["self"])   # also execute decryption protocol given the received shares, but don't send your shares to other parties
```

There are a couple of parameters that need to be passed to the class method `DistributedPaillier.from_security_parameter()`. We list them here and provide information on how to choose the right values.
- `pool`: This pool should be initialised for each party (one pool per party). See the documentation for `tno.mpc.communication.pool` for more information.
- `corruption_threshold`: This is the `t` in `t-out-of-n` secret sharing. The secret sharing is used to distribute the secret key.  We require a dishonest minority, so we require for the
  number of parties in the pool and the corruption threshold that `number_of_parties >= 2 * corruption_threshold + 1`. The default value is `1`.
- `key_length`: This is the bit length of the biprime `N` used in the modulus of the scheme. The safety is similar to that of RSA, so typical values are `1024`, `2048` and `4096`. However, this comes at a performance cost. If you simply wish to play around with the code, we recommend using `128`, so the protocol will on average finish in under 1 minute. We stress that this is *NOT* safe and should never be done in production environments. The default value is `2048`.
- `prime_threshold`: This is an upper bound on the prime values that are checked before the expensive biprimality test is run. A higher value means that bad candidates are discarded faster. The default value is `2000`. 
- `correct_param_biprime`: This parameter determines the certainty level that the produced `N` is indeed the product of 2 primes. The value indicates the number of random values that are sampled and checked. The probability that a check passes, but `N` is not biprime is less than 0.5, so the probability that `N` is not biprime is less than `2**(-correct_param_biprime)`. The default value is `40`.
- `stat_sec_shamir`: security parameter for the shamir secret sharing over the integers. The higher this parameter, the larger the interval of random masking values will be and the smaller the statistical distance from uniform will be. The default value is `40`.
- `distributed`: This value determines how the resulting `DistributedPaillier` instance is stored. When the protocol is run within 1 python instance (such as in `distributed_keygen_example_local.py`), this value should be set to `False` and if each party uses their own python instance, this should be set to `True`. The default value is `True`.
- `precision`: This determines the fixed-point precision of the computations in the resulting encryption scheme. A precision of `n` gives `n` decimals behind the comma of precision.

## Appendix
`distributed_keygen_example_local.py`:
```python
import asyncio
from typing import List

from tno.mpc.communication import Pool

from tno.mpc.protocols.distributed_keygen import DistributedPaillier

corruption_threshold = 1  # corruption threshold
key_length = 128  # bit length of private key
prime_thresh = 2000  # threshold for primality check
correct_param_biprime = 40  # correctness parameter for biprimality test
stat_sec_shamir = (
    40  # statistical security parameter for secret sharing over the integers
)

PARTIES = 4  # number of parties that will be involved in the protocol, you can change this to any number you like


def setup_local_pool(server_port: int, ports: List[int]) -> Pool:
    pool = Pool()
    pool.add_http_server(server_port, "localhost")
    for client_port in (port for port in ports if port != server_port):
        pool.add_http_client(f"client{client_port}", "localhost", client_port)
    return pool


local_ports = [3000 + i for i in range(PARTIES)]
local_pools = [
    setup_local_pool(server_port, local_ports) for server_port in local_ports
]

loop = asyncio.get_event_loop()
async_coroutines = [
    DistributedPaillier.from_security_parameter(
        pool,
        corruption_threshold,
        key_length,
        prime_thresh,
        correct_param_biprime,
        stat_sec_shamir,
        distributed=False,
    )
    for pool in local_pools
]
print("Starting distributed key generation protocol.")
distributed_paillier_schemes = loop.run_until_complete(
    asyncio.gather(*async_coroutines)
)
print("The protocol has completed.")
```
`distributed_keygen_example_distributed.py`:
```python
import argparse
import asyncio
from typing import List, Tuple

from tno.mpc.communication import Pool

from tno.mpc.protocols.distributed_keygen import DistributedPaillier

corruption_threshold = 1  # corruption threshold
key_length = 128  # bit length of private key
prime_thresh = 2000  # threshold for primality check
correct_param_biprime = 40  # correctness parameter for biprimality test
stat_sec_shamir = (
    40  # statistical security parameter for secret sharing over the integers
)


def setup_local_pool(server_port: int, others: List[Tuple[str, int]]) -> Pool:
    pool = Pool()
    pool.add_http_server(server_port, "localhost")
    for client_ip, client_port in others:
        pool.add_http_client(
            f"client_{client_ip}_{client_port}", client_ip, client_port
        )
    return pool


# REGION EXAMPLE SETUP
# this region contains code that is used for the toy example, but can be deleted when the `others`
# variable underneath the region is set to the proper values.

parser = argparse.ArgumentParser(description="Set the parameters to run the protocol.")

parser.add_argument(
    "--party",
    type=int,
    help="Identifier for this party. This should be different for all scripts but should be in the "
    "set [0, ..., nr_of_parties - 1].",
)

parser.add_argument(
    "--nr_of_parties",
    type=int,
    help="Total number of parties involved. This should be the same for all scripts.",
)

args = parser.parse_args()
party_number = args.party
nr_of_parties = args.nr_of_parties

base_port = args.base_port
# ENDREGION

# Change this to the ips and server ports of the other machines
others = [
    ("localhost", base_port + i) for i in range(nr_of_parties) if i != party_number
]

# Change this to the port you want this machine to listen on (note that this should correspond
# to the port of this party in the scripts on the other machines)
server_port = base_port + party_number
pool = setup_local_pool(server_port, others)

loop = asyncio.get_event_loop()
protocol_coroutine = DistributedPaillier.from_security_parameter(
    pool,
    corruption_threshold,
    key_length,
    prime_thresh,
    correct_param_biprime,
    stat_sec_shamir,
    distributed=True,
)
distributed_paillier_scheme = loop.run_until_complete(protocol_coroutine)
```
`run_protocol.sh`:
```shell
#!/bin/bash
for ((PARTY=0;  PARTY < $1; PARTY++))
do
  echo "Initializing party $PARTY"
  python distributed_keygen_example_distributed.py --party $PARTY --nr_of_parties $1 &
  echo "Done"
done
wait
echo "The protocol has finished"
echo "Press any key to quit"
while [ true ] ; do
  read -t 3 -n 1
if [ $? = 0 ] ; then
  exit ;
else
  echo "waiting for the keypress"
fi
done
```

