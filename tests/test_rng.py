"""Cross-backend bit-identity for the counter-based RNG.

This test is the foundation of every differential-testing guarantee in the
framework: if the scalar (reference) and torch (batched) backends ever disagree
on a single draw, bit-identity between implementations is unachievable.
"""

import random

import torch

from simulacrum import rng


def _scalar_bits_as_i64(key, step, slot, index):
    return rng.uint64_to_int64(rng.draw_bits(key, step, slot, index))


def test_bits_cross_backend_random():
    r = random.Random(0)
    n = 20_000
    keys = [r.getrandbits(64) for _ in range(n)]
    steps = [r.randrange(0, 10_000) for _ in range(n)]
    slots = [r.randrange(0, 64) for _ in range(n)]
    idxs = [r.randrange(0, 256) for _ in range(n)]

    scalar = torch.tensor(
        [_scalar_bits_as_i64(k, t, s, i) for k, t, s, i in zip(keys, steps, slots, idxs)],
        dtype=torch.int64,
    )
    keys_t = torch.tensor([rng._i64(k) for k in keys], dtype=torch.int64)
    # slot/index vary per element here, so hash with tensor slot/index words
    # (draw_bits_torch accepts tensors for steps/index; slots vary too in this
    # stress test, so go through the sponge directly):
    batched = rng._hash_words_torch(
        rng._DOMAIN_DRAW, keys_t, torch.tensor(steps), torch.tensor(slots), torch.tensor(idxs)
    )
    assert torch.equal(scalar, batched)


def test_bits_cross_backend_structured():
    # Edge-ish values: zeros, max uint64, small counters — the places where
    # sign-extension or wrapping bugs would hide.
    interesting = [0, 1, 2, (1 << 63) - 1, 1 << 63, (1 << 64) - 1, 0xDEADBEEF]
    for key in interesting:
        for step in [0, 1, 2, 1000]:
            for slot in [0, 1, 7]:
                for index in [0, 1, 255]:
                    scalar = _scalar_bits_as_i64(key, step, slot, index)
                    keys_t = torch.tensor([rng._i64(key)], dtype=torch.int64)
                    batched = rng.draw_bits_torch(keys_t, step, slot, index)
                    assert batched.item() == scalar, (key, step, slot, index)


def test_uniform_cross_backend_bitwise():
    r = random.Random(1)
    keys = [r.getrandbits(64) for _ in range(5_000)]
    scalar = torch.tensor([rng.draw_uniform(k, 3, 1) for k in keys], dtype=torch.float64)
    keys_t = torch.tensor([rng._i64(k) for k in keys], dtype=torch.int64)
    batched = rng.draw_uniform_torch(keys_t, 3, 1)
    # Bitwise, not approximate:
    assert torch.equal(scalar.view(torch.int64), batched.view(torch.int64))
    assert (batched >= 0).all() and (batched < 1).all()


def test_randint_cross_backend():
    r = random.Random(2)
    for n in [2, 3, 7, 100, 12345]:
        keys = [r.getrandbits(64) for _ in range(2_000)]
        scalar = torch.tensor([rng.draw_randint(k, 5, 2, n) for k in keys], dtype=torch.int64)
        keys_t = torch.tensor([rng._i64(k) for k in keys], dtype=torch.int64)
        batched = rng.draw_randint_torch(keys_t, 5, 2, n)
        assert torch.equal(scalar, batched)
        assert (batched >= 0).all() and (batched < n).all()


def test_bernoulli_cross_backend():
    r = random.Random(3)
    keys = [r.getrandbits(64) for _ in range(5_000)]
    keys_t = torch.tensor([rng._i64(k) for k in keys], dtype=torch.int64)
    for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
        scalar = torch.tensor([rng.draw_bernoulli(k, 0, 0, p) for k in keys])
        batched = rng.draw_bernoulli_torch(keys_t, 0, 0, p)
        assert torch.equal(scalar, batched)


def test_episode_key_cross_backend():
    r = random.Random(4)
    seeds = [r.getrandbits(64) for _ in range(2_000)]
    episodes = [r.randrange(0, 1000) for _ in range(2_000)]
    scalar = torch.tensor(
        [rng.uint64_to_int64(rng.episode_key(s, e)) for s, e in zip(seeds, episodes)],
        dtype=torch.int64,
    )
    batched = rng.episode_key_torch(
        torch.tensor([rng._i64(s) for s in seeds], dtype=torch.int64),
        torch.tensor(episodes, dtype=torch.int64),
    )
    assert torch.equal(scalar, batched)


def test_statistical_sanity():
    # Not a rigorous suite — just catches catastrophic mixer breakage.
    keys_t = rng.episode_key_torch(torch.arange(100_000), torch.zeros(100_000, dtype=torch.int64))
    u = rng.draw_uniform_torch(keys_t, 0, 0)
    assert abs(u.mean().item() - 0.5) < 0.01
    assert abs(u.var().item() - 1 / 12) < 0.01
    # Consecutive steps decorrelated:
    u2 = rng.draw_uniform_torch(keys_t, 1, 0)
    corr = torch.corrcoef(torch.stack([u, u2]))[0, 1].item()
    assert abs(corr) < 0.02


def test_distinct_slots_and_indices_decorrelate():
    keys_t = rng.episode_key_torch(torch.arange(1000), torch.zeros(1000, dtype=torch.int64))
    a = rng.draw_bits_torch(keys_t, 0, 0, 0)
    for other in [rng.draw_bits_torch(keys_t, 0, 1, 0),
                  rng.draw_bits_torch(keys_t, 0, 0, 1),
                  rng.draw_bits_torch(keys_t, 1, 0, 0)]:
        assert (a != other).all()
