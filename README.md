# TNO MPC Lab - Protocols - Distributed Keygen

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.distributed_keygen is part of the TNO Python Toolbox.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the tno.mpc.protocols.distributed_keygen package can be found [here](https://docs.mpc.tno.nl/protocols/distributed_keygen/0.5.1).

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

## Usage

The distributed keygen module can be used by first creating a `Pool` 
from the `tno.mpc.communication` library. 

```python
from tno.mpc.communication.pool import Pool

pool = Pool(...) # initialize pool with ips etc
```
You can then create an instance of the `DistributedPaillier` class and pass the pool to
this class. This also starts a protocol to generate indices, so all parties have an ID. Finally, 
this method also generates the shared secret key and accompanying public key.

```python
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

corruption_threshold = 1     # corruption threshold
key_length = 128             # bit length of private key
prime_thresh = 2000          # threshold for primality check
correct_param_biprime = 100  # correctness parameter for biprimality test 
stat_sec_shamir = 40         # statistical security parameter for secret sharing over the integers

distributed_scheme = await DistributedPaillier.from_security_parameter(
        pool,
        corruption_threshold,
        key_length,
        prime_thresh,
        correct_param_biprime,
        stat_sec_shamir,
)
```

Now the public key can be used to encrypt message, whereas the shared secret key
can be used to distributively decrypt.

```python
ciphertext = distributed_scheme.encrypt(42)          # encryption of 42
await distributed_scheme.pool.send(..., ciphertext)  # send the ciphertext to another party
await distributed_scheme.pool.recv(...)              # receive message from other party

plaintext = await distributed_scheme.decrypt(ciphertext)   # execute decryption protocol given the received shares
plaintext = await distributed_scheme.decrypt(ciphertext, receivers=["self"])   # also execute decryption protocol given the received shares, but don't send your shares to other parties
```
