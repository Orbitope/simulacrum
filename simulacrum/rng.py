"""Stateless counter-based RNG — the bit-identity contract between implementations.

THE CONTRACT
============
Every stochastic draw in an environment is a *pure function* of four words:

    bits = hash(episode_key, step, slot, index)

- ``episode_key``: derived once per episode from ``(instance_seed, episode_counter)``
  via :func:`episode_key`. Auto-reset increments the episode counter, so the
  batched implementation reseeds terminated instances without coordinating any
  stream state with the reference implementation.
- ``step``: the in-episode step index at which the draw occurs (0-based; draws
  made during reset use step 0 with a reserved slot, or their own slot).
- ``slot``: a small integer naming *which* random decision this is (e.g.
  "movement slip", "spawn position"). Slots MUST be enumerated in the
  environment's spec.md and defined as an IntEnum in the env package. Two
  different random decisions at the same step must use different slots.
- ``index``: disambiguates multiple draws of the same slot at the same step
  (e.g. per-card draws when shuffling). Defaults to 0.

Because there is no stream state, draw *order* is irrelevant: the batched
implementation may compute draws for ALL instances and discard the masked-out
ones — a discarded draw cannot corrupt parity with the reference. This is what
makes the ``torch.where``/masking idiom safe. It also makes a future JAX port
mechanical: JAX's RNG is already counter-based.

GUARANTEE: for equal ``(episode_key, step, slot, index)``, the scalar functions
(:func:`draw_uniform` etc., used by reference implementations) and the batched
functions (:func:`draw_uniform_torch` etc., used by batched implementations)
return bit-identical values. This is enforced by tests/test_rng.py, which
cross-checks the two backends over large random and structured inputs.

BACKENDS
========
- Scalar backend: plain Python ints masked to 64 bits. Exact by construction;
  reference envs stay dependency-free and readable.
- Batched backend: torch.int64 tensors. uint64 semantics are emulated with
  two's-complement wrapping arithmetic (add/mul wrap identically to uint64 in
  the low 64 bits) and logical right shifts emulated by masking off
  sign-extended high bits.

The mixer is splitmix64 (Steele et al.), applied as a sponge over the input
words. It is not cryptographic; it is statistically strong far beyond what
environment stochasticity needs, and small enough to re-implement identically
in any array framework.
"""

from __future__ import annotations

from typing import Union

import torch

_MASK64 = (1 << 64) - 1

# splitmix64 constants (Steele, Lea & Flood 2014).
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB

# Domain-separation tags so episode keys and draw streams can never collide.
_DOMAIN_EPISODE = 0x5EED5EED5EED5EED
_DOMAIN_DRAW = 0xD4A45EEDD4A45EED

# 1 / 2**53, for converting the top 53 bits to a float64 in [0, 1).
_INV_2_53 = 1.0 / (1 << 53)


# ---------------------------------------------------------------------------
# Scalar backend (reference implementations)
# ---------------------------------------------------------------------------


def _splitmix64(x: int) -> int:
    x = (x + _GOLDEN) & _MASK64
    x = ((x ^ (x >> 30)) * _MIX1) & _MASK64
    x = ((x ^ (x >> 27)) * _MIX2) & _MASK64
    return x ^ (x >> 31)


def _hash_words(*words: int) -> int:
    """Absorb words into a splitmix64 sponge. Returns 64 random-looking bits."""
    h = 0
    for w in words:
        h = _splitmix64(h ^ (w & _MASK64))
    return h


def episode_key(instance_seed: int, episode: int) -> int:
    """Derive the per-episode key for an instance.

    Episode 0 is the episode begun by the initial reset; each auto-reset (or
    explicit reset of the reference env within a differential run) increments
    ``episode``.
    """
    return _hash_words(_DOMAIN_EPISODE, instance_seed, episode)


def draw_bits(key: int, step: int, slot: int, index: int = 0) -> int:
    """64 uniform bits for this (key, step, slot, index). Returned as a
    non-negative Python int in [0, 2**64)."""
    return _hash_words(_DOMAIN_DRAW, key, step, slot, index)


def draw_uniform(key: int, step: int, slot: int, index: int = 0) -> float:
    """Uniform float64 in [0, 1). Uses the top 53 bits, so the conversion is
    exact and bit-identical across backends."""
    return (draw_bits(key, step, slot, index) >> 11) * _INV_2_53


def draw_randint(key: int, step: int, slot: int, n: int, index: int = 0) -> int:
    """Uniform int in [0, n). Uses the low 63 bits so the identical computation
    is representable in torch.int64; modulo bias is <= n / 2**63."""
    return (draw_bits(key, step, slot, index) >> 1) % n


def draw_bernoulli(key: int, step: int, slot: int, p: float, index: int = 0) -> bool:
    """True with probability p."""
    return draw_uniform(key, step, slot, index) < p


# ---------------------------------------------------------------------------
# Batched backend (torch.int64, uint64 semantics emulated)
# ---------------------------------------------------------------------------


def _i64(c: int) -> int:
    """Reinterpret a uint64 constant as its two's-complement int64 value."""
    return c - (1 << 64) if c >= (1 << 63) else c


def _lshr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (unsigned) right shift on int64: mask off sign-extended bits."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _splitmix64_torch(x: torch.Tensor) -> torch.Tensor:
    x = x + _i64(_GOLDEN)
    x = (x ^ _lshr(x, 30)) * _i64(_MIX1)
    x = (x ^ _lshr(x, 27)) * _i64(_MIX2)
    return x ^ _lshr(x, 31)


IntOrTensor = Union[int, torch.Tensor]


def _as_i64(w: IntOrTensor) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return w.to(torch.int64)
    return torch.tensor(_i64(w & _MASK64), dtype=torch.int64)


def _hash_words_torch(*words: IntOrTensor) -> torch.Tensor:
    """Batched sponge. Words may be int64 tensors (broadcastable) or Python
    ints; the result broadcasts across all tensor words."""
    h = torch.tensor(0, dtype=torch.int64)
    for w in words:
        h = _splitmix64_torch(h ^ _as_i64(w))
    return h


def episode_key_torch(instance_seeds: torch.Tensor, episodes: torch.Tensor) -> torch.Tensor:
    """Batched :func:`episode_key`: [N] int64 seeds x [N] int64 episode
    counters -> [N] int64 keys (bit-identical to the scalar version,
    reinterpreted as int64)."""
    return _hash_words_torch(_DOMAIN_EPISODE, instance_seeds, episodes)


def draw_bits_torch(keys: torch.Tensor, steps: IntOrTensor, slot: int,
                    index: IntOrTensor = 0) -> torch.Tensor:
    """Batched :func:`draw_bits`. Returns int64 tensor whose bits equal the
    scalar backend's uint64 result (values may print negative; that is the
    two's-complement reinterpretation, not a difference)."""
    return _hash_words_torch(_DOMAIN_DRAW, keys, steps, slot, index)


def draw_uniform_torch(keys: torch.Tensor, steps: IntOrTensor, slot: int,
                       index: IntOrTensor = 0) -> torch.Tensor:
    """Batched :func:`draw_uniform` -> float64 tensor, bit-identical."""
    return _lshr(draw_bits_torch(keys, steps, slot, index), 11).to(torch.float64) * _INV_2_53


def draw_randint_torch(keys: torch.Tensor, steps: IntOrTensor, slot: int, n: int,
                       index: IntOrTensor = 0) -> torch.Tensor:
    """Batched :func:`draw_randint` -> int64 tensor in [0, n), bit-identical."""
    return _lshr(draw_bits_torch(keys, steps, slot, index), 1) % n


def draw_bernoulli_torch(keys: torch.Tensor, steps: IntOrTensor, slot: int, p: float,
                         index: IntOrTensor = 0) -> torch.Tensor:
    """Batched :func:`draw_bernoulli` -> bool tensor, bit-identical."""
    return draw_uniform_torch(keys, steps, slot, index) < p


def uint64_to_int64(u: int) -> int:
    """Reinterpret a scalar-backend uint64 value as int64 (for comparing
    against the torch backend)."""
    return _i64(u & _MASK64)
