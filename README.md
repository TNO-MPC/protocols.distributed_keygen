# TNO MPC Lab - Protocols - Distributed Keygen

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.distributed_keygen is part of the TNO Python Toolbox.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*  
*This implementation of cryptographic software has not been audited. Use at your own risk.*

## Documentation

Documentation of the tno.mpc.protocols.distributed_keygen package can be found [here](https://docs.mpc.tno.nl/protocols/distributed_keygen/3.1.4).

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


### After initialization

When a DistributedPaillier instance has been generated (either locally or distributedly), the public key can be used to encrypt messages and the shared secret key
can be used to distributively decrypt. Note that these methods are async methods, so they either
need to be run in an event loop or inside another async method using await.

In the following example we show how to use this library to make computations using a scheme that is distributed over 3
parties ("party1", "party2", and "party3"). We show the code for all 3 parties and assume that a `distributed_scheme`
has already been generated by the parties.

Note that in order to decrypt to ciphertext it must be known to all parties. Also, all parties must participate in the
decryption, even in the case that they do not receive any other shares or the result.

_Beware: When sending a ciphertext to more than one party, the method `pool.broadcast()` MUST be used. When using
`pool.send()` the parties will receive different ciphertexts due to intermediate re-randomization. For more details on
why this happens read the text below the examples._

```python
# Party 1

# The assumption here is that this code is placed inside an async method
ciphertext = distributed_scheme.encrypt(42)          # encryption of 42
await distributed_scheme.pool.send("party2", ciphertext, msg_id="step1")  # send the ciphertext to party 2

final_ciphertext = await distributed_scheme.recv("party3", msg_id="step3")  # receive the ciphertext from party 3

# all parties need to participate in the decryption protocol
plaintext = await distributed_scheme.decrypt(final_ciphertext)
assert plaintext == 426

# alternative decryption of which the shares (and result) are only obtained by party 2
# note: even though we do not receive the result, we are required to participate
await distributed_scheme.decrypt(final_ciphertext, receivers=["party2"])
```

```python
# Party 2

# The assumption here is that this code is placed inside an async method
ciphertext = await distributed_scheme.pool.recv("party1", msg_id="step1")  # receive the ciphertext from party 1

ciphertext += 100  # add 100 to the ciphertext (value is now 142)
await distributed_scheme.pool.send("party3", ciphertext, msg_id="step2")  # send the updated ciphertext to party 3

final_ciphertext = await distributed_scheme.recv("party3", msg_id="step3")  # recieve the ciphertext from party 3

# all parties need to participate in the decryption protocol
plaintext = await distributed_scheme.decrypt(final_ciphertext)
assert plaintext == 426

# alternative decryption of which the shares (and result) are only obtained by party 2
# note: even though we do not receive the result, we are required to participate
plaintext = await distributed_scheme.decrypt(final_ciphertext, receivers=["self"])
assert plaintext == 426
```

```python
# Party 3

# The assumption here is that this code is placed inside an async method
final_ciphertext = await distributed_scheme.pool.recv("party2", msg_id="step2")  # receive the ciphertext from party 1

final_ciphertext *= 3  # multiply the ciphertext by 3 (value is now 426)
# send the ciphertext to multiple parties (we cannot use `pool.send` now).
distributed_scheme.pool.broadcast(final_ciphertext, msg_id="step3", handler_names=["party1", "party2"])  # receivers=None does the same

# all parties need to participate in the decryption protocol
plaintext = await distributed_scheme.decrypt(final_ciphertext)
assert plaintext == 426

# alternative decryption of which the shares (and result) are only obtained by party 2
# note: even though we do not receive the result, we are required to participate
await distributed_scheme.decrypt(final_ciphertext, receivers=["party2"])
```

Running this example will show several warnings. The remainder of this documentation explains why the warnings are issued and how to get rid of them depending on the users' preferences.

## Fresh and unfresh ciphertexts

An encrypted message is called a ciphertext. A ciphertext in the current package has a property `is_fresh` that indicates whether this ciphertext has fresh randomness, in which case it can be communicated to another player securely. More specifically, a ciphertext `c` is fresh if another user, knowledgeable of all prior communication and all current ciphertexts marked as fresh, cannot deduce any more private information from learning `c`.

The package understands that the freshness of the result of a homomorphic operation depends on the freshness of the inputs, and that the homomorphic operation renders the inputs unfresh. For example, if `c1` and `c2` are fresh ciphertexts, then `c12 = c1 + c2` is marked as a fresh encryption (no rerandomization needed) of the sum of the two underlying plaintexts. After the operation, ciphertexts `c1` and `c2` are no longer fresh.

The fact that `c1` and `c2` were both fresh implies that, at some point, we randomized them. After the operation `c12 = c1 + c2`, only `c12` is fresh. This implies that one randomization was lost in the process. In particular, we wasted resources. An alternative approach was to have unfresh `c1` and `c2` then compute the unfresh result `c12` and only randomize that ciphertext. This time, no resources were wasted. The package issues a warning to inform the user this and similar efficiency opportunities.

The package integrates naturally with `tno.mpc.communication` and if that is used for communication, its serialization logic will ensure that all sent ciphertexts are fresh. A warning is issued if a ciphertext was randomized in the proces. A ciphertext is always marked as unfresh after it is serialized. Similarly, all received ciphertexts are considered unfresh.

## Tailor behavior to your needs

The crypto-neutral developer is facilitated by the package as follows: the package takes care of all bookkeeping, and the serialization used by `tno.mpc.communication` takes care of all randomization. The warnings can be [disabled](#warnings) for a smoother experience.

The eager crypto-youngster can improve their understanding and hone their skills by learning from the warnings that the package provides in a safe environment. The package is safe to use when combined with `tno.mpc.communication`. It remains to be safe while you transform your code from 'randomize-early' (fresh encryptions) to 'randomize-late' (unfresh encryptions, randomize before exposure). At that point you have optimized the efficiency of the library while ensuring that all exposed ciphertexts are fresh before they are serialized. In particular, you no longer rely on our serialization for (re)randomizing your ciphertexts.

Finally, the experienced cryptographer can turn off warnings / turn them into exceptions, or benefit from the `is_fresh` flag for own purposes (e.g. different serializer or communication).

### Warnings

By default, the `warnings` package prints only the first occurence of a warning for each location (module + line number) where the warning is issued. The user may easily [change this behaviour](https://docs.python.org/3/library/warnings.html#the-warnings-filter) to never see warnings:

```python
from tno.mpc.encryption_schemes.paillier import EncryptionSchemeWarning

warnings.simplefilter("ignore", EncryptionSchemeWarning)
```

Alternatively, the user may pass `"once"`, `"always"` or even `"error"`.

Finally, note that some operations issue two warnings, e.g. `c1-c2` issues a warning for computing `-c2` and a warning for computing `c1 + (-c2)`.

### Advanced usage

The basic usage in the example above can be improved upon by explicitly randomizing as late as possible, i.e. by
only randomizing non-fresh ciphertexts directly before they are communicated using the `randomize()` method.

## Appendix

*NOTE*: If you want to run `distributed_keygen_example_local.py` in a Jupyter Notebook, you will run into the issue that the event loop is already running upon calling `run_until_complete`.
In this case, you should add the following code to the top of the notebook:
```python
import nest_asyncio
nest_asyncio.apply()
```

distributed_keygen_example_local.py:
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
    pool.add_http_server(server_port)
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
distributed_keygen_example_distributed.py:
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
    pool.add_http_server(server_port)
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

parser.add_argument(
    "--base-port",
    type=int,
    default=8888,
    help="port first player used for communication, incremented for other players"
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
run_protocol.sh:
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
