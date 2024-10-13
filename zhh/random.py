"""
This provides a more friendly random number generator interface.
"""
import jax.random as _jax_random
import random as _sys_random


def getseed():
    return _sys_random.randint(0, 1 << 32)

class KeyManager:
    """
    A class used for managing random keys.

    ### Example

    >>> key = zr.KeyManager()
    >>> key.getkey()
    >>> key.getkey() # different as the first one
    """

    def __init__(self,):
        self.rng = _jax_random.PRNGKey(getseed())

    def getkey(self):
        self.rng, key = _jax_random.split(self.rng)
        return key

rng = KeyManager()

def seed(num:int):
    """Set the random seed.

    Notice this **CANNOT** influence the Key Managers you already created, so call it at the beginning.
    """
    _sys_random.seed(num)
    global rng
    rng = KeyManager()
    

def getkey():
    """Get the random key, only use it if you use a jax random function."""
    global rng
    return rng.getkey()

def next(rng):
    """Get the next random key."""
    return _jax_random.split(rng)[1]

def get_size_from(size):
    if isinstance(size, int):
        return (size,)
    if not isinstance(size[0], int):
        size = size[0]
    return size

def _default_key(rng):
    if rng is not None:
        if isinstance(rng, KeyManager):
            rng = rng.getkey()
        return rng
    return getkey()

def randn(*size, rng=None):
    """Generate a N(0,1) tensor, with shape `size`."""
    size = get_size_from(size)
    return _jax_random.normal(_default_key(rng), size)

def rand(*size, rng=None):
    """Generate a Uniform(0,1) tensor, with shape `size`."""
    size = get_size_from(size)
    return _jax_random.uniform(_default_key(rng), size)

def randn_like(x, rng=None):
    return randn(x.shape, rng=rng)

def rand_like(x, rng=None):
    return rand(x.shape, rng=rng)

def randint(low, high, size, rng=None):
    """Generate a random integer tensor, with shape `size`, and range [low, high)."""
    size = get_size_from(size)
    return _jax_random.randint(_default_key(rng), size, low, high)

def randperm(n, rng=None):
    """Generate a random permutation of 0,1,...,n-1."""
    return _jax_random.permutation(_default_key(rng), n)