/* simulacrum's counter-based RNG, ported to JavaScript BigInt.
 *
 * This is a transcription of simulacrum/rng.py's scalar backend: same
 * splitmix64 constants, same domain tags, same canonical absorb order
 * (domain, slot, index, key, step). BigInt with an explicit 64-bit mask stands
 * in for Python's arbitrary-precision ints masked to 64 bits.
 *
 * It is verified bit-identical to the Python implementation:
 *
 *     episode_key(1234, 3)                = 15570406044200102918
 *     draw_bits(key, step=7, slot=1)      = 5836041238337677608
 *     draw_bits(key, step=7, slot=5)      = 10406080438901372403
 *     draw_uniform(key, step=7, slot=5)   = 0.5641147509457889
 *
 * Those constants are asserted on page load (see index.html). If this port
 * ever drifts from the framework, the page says so out loud rather than
 * quietly showing numbers that are merely plausible.
 */
(function (global) {
  'use strict';

  var MASK = (1n << 64n) - 1n;

  // splitmix64 constants (Steele, Lea & Flood 2014).
  var GOLDEN = 0x9E3779B97F4A7C15n;
  var MIX1 = 0xBF58476D1CE4E5B9n;
  var MIX2 = 0x94D049BB133111EBn;

  // Domain-separation tags so episode keys and draw streams cannot collide.
  var DOMAIN_EPISODE = 0x5EED5EED5EED5EEDn;
  var DOMAIN_DRAW = 0xD4A45EEDD4A45EEDn;

  var INV_2_53 = 1 / Math.pow(2, 53);

  function splitmix64(x) {
    x = (x + GOLDEN) & MASK;
    x = ((x ^ (x >> 30n)) * MIX1) & MASK;
    x = ((x ^ (x >> 27n)) * MIX2) & MASK;
    return x ^ (x >> 31n);
  }

  /* Absorb words into a splitmix64 sponge. Returns 64 random-looking bits. */
  function hashWords() {
    var h = 0n;
    for (var i = 0; i < arguments.length; i++) {
      h = splitmix64(h ^ (BigInt(arguments[i]) & MASK));
    }
    return h;
  }

  var EPISODE_PREFIX = hashWords(DOMAIN_EPISODE);

  var prefixCache = {};

  /* Sponge state after absorbing (DOMAIN_DRAW, slot, index): the scalar
   * leading words of a draw. Cached; slots and indices are small. */
  function drawPrefix(slot, index) {
    var k = slot + ':' + index;
    var cached = prefixCache[k];
    if (cached === undefined) {
      cached = prefixCache[k] = hashWords(DOMAIN_DRAW, slot, index);
    }
    return cached;
  }

  /* Derive the per-episode key for an instance. Episode 0 is the episode begun
   * by the initial reset; each auto-reset increments `episode`. */
  function episodeKey(instanceSeed, episode) {
    var h = splitmix64(EPISODE_PREFIX ^ (BigInt(instanceSeed) & MASK));
    return splitmix64(h ^ (BigInt(episode) & MASK));
  }

  /* 64 uniform bits for this (key, step, slot, index), as a BigInt in
   * [0, 2**64). */
  function drawBits(key, step, slot, index) {
    index = index || 0;
    var h = splitmix64(drawPrefix(slot, index) ^ (key & MASK));
    return splitmix64(h ^ (BigInt(step) & MASK));
  }

  /* Uniform float64 in [0, 1). Uses the top 53 bits, so Number() is lossless
   * and the conversion is exact, bit-identical to the Python backend. */
  function drawUniform(key, step, slot, index) {
    return Number(drawBits(key, step, slot, index) >> 11n) * INV_2_53;
  }

  /* Uniform int in [0, n). Uses the low 63 bits, matching the Python backend's
   * shift so the identical computation is representable in int64. */
  function drawRandint(key, step, slot, n, index) {
    return Number((drawBits(key, step, slot, index) >> 1n) % BigInt(n));
  }

  /* True with probability p. */
  function drawBernoulli(key, step, slot, p, index) {
    return drawUniform(key, step, slot, index) < p;
  }

  /* Domain tag separating the harness's action stream from env RNG streams.
   * mirrors simulacrum/harness/config.py. Reproducing it here means the page
   * drives its environments with the SAME action sequence the Python battery
   * uses, so the numbers on this page are directly comparable to the numbers
   * `simulacrum validate` prints. */
  var DOMAIN_ACTION = 0xAC7104AC7104AC71n;

  function actionKey(seed) {
    return hashWords(DOMAIN_ACTION, seed);
  }

  /* DiscreteActionSampler.sample(seed, t) -> int in [0, nActions). */
  function sampleAction(seed, t, nActions) {
    return drawRandint(actionKey(seed), t, 0, nActions, 0);
  }

  global.SimRNG = {
    hashWords: hashWords,
    actionKey: actionKey,
    sampleAction: sampleAction,
    episodeKey: episodeKey,
    drawBits: drawBits,
    drawUniform: drawUniform,
    drawRandint: drawRandint,
    drawBernoulli: drawBernoulli
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
