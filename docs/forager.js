/* forager, implemented a third time, in JavaScript, from examples/forager/spec.md.
 *
 * Python has two implementations of this environment: a scalar reference and a
 * batched tensor one. This is a third, in a language with no int64 and no
 * tensors, following the same spec document. Every number this page shows is
 * produced by running it.
 *
 * float32 discipline: the spec makes energy and observation arithmetic
 * normatively float32, so every such operation is wrapped in Math.fround.
 * JavaScript numbers are float64, which is what the spec wants for rewards.
 *
 * The BUGS object at the bottom is the article's break-it lab: each flag is a
 * real mistake someone makes porting a scalar env to tensors, injected into the
 * "fast" side so the differential runner can catch it.
 */
(function (global) {
  'use strict';

  var RNG = global.SimRNG;

  // spec: Constants.
  var G = 8, K = 6, N_KINDS = 3, MAX_STEPS = 80, GUST_P = 0.15;
  var START_ENERGY = Math.fround(1.0);
  var STEP_COST = Math.fround(0.02);
  var GAINS = [Math.fround(0.1), Math.fround(0.15), Math.fround(0.3)];
  var REWARD_SCALE = 10.0, REWARD_STEP = -0.1;
  var ENERGY_MAX = 1.0 + K * GAINS[2];

  // spec: Actions. 0 north, 1 east, 2 south, 3 west.
  var DELTAS = [[0, 1], [1, 0], [0, -1], [-1, 0]];

  // spec: RNG slots.
  var S = { AGENT_X: 0, AGENT_Y: 1, BERRY_X: 2, BERRY_Y: 3, BERRY_KIND: 4, GUST: 5 };

  function noBugs() { return {}; }

  /* spec: Reset. All draws at step 0, each with its own slot. */
  function reset(seed, episode, bugs) {
    bugs = bugs || noBugs();
    var key = RNG.episodeKey(seed, episode);

    var pos = [RNG.drawRandint(key, 0, S.AGENT_X, G, 0),
               RNG.drawRandint(key, 0, S.AGENT_Y, G, 0)];

    var berries = [], kinds = [], k;
    for (k = 0; k < K; k++) {
      // spec: Reset. Berry k drawn at index = k.
      // BUG berryIndexZero: every berry drawn at index 0, so all K berries
      // land on the same cell. The spec's RNG-slot table warns about exactly
      // this: the index word must vary along the berry axis.
      var idx = bugs.berryIndexZero ? 0 : k;
      berries.push([RNG.drawRandint(key, 0, S.BERRY_X, G, idx),
                    RNG.drawRandint(key, 0, S.BERRY_Y, G, idx)]);
      kinds.push(RNG.drawRandint(key, 0, S.BERRY_KIND, N_KINDS, idx));
    }

    var alive = [];
    for (k = 0; k < K; k++) alive.push(true);

    return { key: key, seed: seed, episode: episode, pos: pos, berries: berries,
             kinds: kinds, alive: alive, energy: START_ENERGY, t: 0 };
  }

  /* spec: The transition, precisely (steps 1-7). */
  function step(st, action, bugs) {
    bugs = bugs || noBugs();

    // 1. Intended delta for this compass direction.
    var d = DELTAS[action], dx = d[0], dy = d[1];

    // 2. GUST is keyed on the PRE-move step counter.
    // BUG gustPostMove: keying on t + 1 instead. Every draw is still perfectly
    // uniform, so the env looks healthy and trains fine. It is just a
    // different environment than the reference one.
    var gustStep = bugs.gustPostMove ? st.t + 1 : st.t;
    var gust = RNG.drawBernoulli(st.key, gustStep, S.GUST, GUST_P, 0);

    var x = st.pos[0], y = st.pos[1];

    if (bugs.clampBeforeRotate) {
      // BUG clampBeforeRotate: spec step 3 says clamp AFTER the rotation.
      // Doing it first lets a rotated delta walk back off the edge.
      x = clamp(x + dx); y = clamp(y + dy);
      if (gust) { var t0 = dx; dx = dy; dy = -t0; x = x + dx; y = y + dy; }
    } else {
      if (gust) { var t1 = dx; dx = dy; dy = -t1; }
      // 3. Clamp after the rotation, per coordinate.
      // BUG noClamp: forgetting the bounds entirely. The agent walks straight
      // off the board and keeps going. This one the invariant sweep catches
      // before the differential test even matters.
      if (bugs.noClamp) { x = x + dx; y = y + dy; }
      else { x = clamp(x + dx); y = clamp(y + dy); }
    }

    // 4.
    var t = st.t + 1;

    // 5. spec: Collection. Live berry on the post-move cell; gains summed in
    // ascending k order.
    var alive = st.alive.slice(), gained = 0.0;
    for (var k = 0; k < K; k++) {
      if (alive[k] && st.berries[k][0] === x && st.berries[k][1] === y) {
        alive[k] = false;
        gained += GAINS[st.kinds[k]];
      }
    }

    // 6. spec: Energy. Float32, left to right.
    // BUG energyFloat64: skipping the float32 rounding. This is dtype drift:
    // the values stay plausible and only diverge in the low bits, which is
    // precisely what a bit-identity check is for.
    var energy;
    if (bugs.energyFloat64) {
      energy = (st.energy - STEP_COST) + gained;
    } else {
      energy = Math.fround(Math.fround(st.energy - STEP_COST) + Math.fround(gained));
    }

    // spec: Rewards. Float64, in this order.
    // BUG rewardBeforeCollect: computing the reward from the pre-collection
    // gain, which is always zero. Every step scores the same -0.1 whatever the
    // agent does, so there is no signal to learn from at all.
    var reward = bugs.rewardBeforeCollect
      ? REWARD_SCALE * 0.0 + REWARD_STEP
      : REWARD_SCALE * gained + REWARD_STEP;

    // 7. spec: Termination.
    // BUG terminateBeforeCollect: counting berries from the pre-collection
    // state, so the episode runs one extra step past the final pickup.
    var aliveCount = countAlive(bugs.terminateBeforeCollect ? st.alive : alive);
    var terminated = (aliveCount === 0) || (energy <= 0.0) || (t === MAX_STEPS);

    return {
      state: { key: st.key, seed: st.seed, episode: st.episode, pos: [x, y],
               berries: st.berries, kinds: st.kinds, alive: alive,
               energy: energy, t: t },
      reward: reward,
      terminated: terminated
    };
  }

  function clamp(v) { return v < 0 ? 0 : (v > G - 1 ? G - 1 : v); }

  function countAlive(alive) {
    var n = 0;
    for (var i = 0; i < alive.length; i++) if (alive[i]) n++;
    return n;
  }

  /* spec: Observations. Float32[5], every division computed in float32.
   * BUG obsFloat64: dividing in float64. The observation the agent sees is
   * off in the low bits on nearly every step. */
  function observe(st, bugs) {
    if (bugs && bugs.obsFloat64) {
      return [st.pos[0] / (G - 1), st.pos[1] / (G - 1), st.energy,
              countAlive(st.alive) / K, st.t / MAX_STEPS];
    }
    return [
      Math.fround(st.pos[0] / (G - 1)),
      Math.fround(st.pos[1] / (G - 1)),
      st.energy,
      Math.fround(countAlive(st.alive) / K),
      Math.fround(st.t / MAX_STEPS)
    ];
  }

  /* The serialized form the differential test compares (schema.json $defs/state). */
  function toJSON(st) {
    return {
      pos: st.pos.slice(),
      berries: st.berries.map(function (c) { return [c[0], c[1]]; }),
      kinds: st.kinds.slice(),
      alive: st.alive.slice(),
      energy: st.energy,
      t: st.t
    };
  }

  /* ---------------------------------------------------------------------
   * The differential runner, mirroring simulacrum/harness/differential.py:
   * reference and "fast" side by side under one action stream, stopping at
   * the first divergence with a field-level diff.
   * ------------------------------------------------------------------- */
  function runDifferential(seed, nSteps, bugs) {
    var ref = reset(seed, 0, noBugs());
    var fast = reset(seed, 0, bugs);
    var episode = 0;
    var trace = [];

    var div = compare(0, 'state', toJSON(ref), toJSON(fast));
    if (div) return { divergence: div, trace: trace, steps: 0 };

    for (var t = 0; t < nSteps; t++) {
      var action = RNG.sampleAction(seed, t, 4);
      var r = step(ref, action, noBugs());
      var f = step(fast, action, bugs);
      trace.push({ t: t, action: action, ref: r.state, fast: f.state });

      if (r.terminated !== f.terminated) {
        return { divergence: { step: t, kind: 'terminated', diffs: [], detail:
          'reference terminated=' + r.terminated + ', fast terminated=' + f.terminated },
          trace: trace, steps: t };
      }
      if (r.reward !== f.reward) {
        return { divergence: { step: t, kind: 'reward', diffs: [], detail:
          'reference reward=' + r.reward + ', fast reward=' + f.reward },
          trace: trace, steps: t };
      }
      div = compare(t, 'state', toJSON(r.state), toJSON(f.state));
      if (div) return { divergence: div, trace: trace, steps: t };

      var ro = observe(r.state, null), fo = observe(f.state, bugs);
      for (var i = 0; i < ro.length; i++) {
        if (ro[i] !== fo[i]) {
          return { divergence: { step: t, kind: 'obs', diffs: [], detail:
            'values differ: ref[' + i + ']=' + ro[i] + ' fast[' + i + ']=' + fo[i] },
            trace: trace, steps: t };
        }
      }

      ref = r.state; fast = f.state;
      if (r.terminated) {
        episode++;
        ref = reset(seed, episode, noBugs());
        fast = reset(seed, episode, bugs);
        div = compare(t, 'state', toJSON(ref), toJSON(fast));
        if (div) return { divergence: div, trace: trace, steps: t };
      }
    }
    return { divergence: null, trace: trace, steps: nSteps };
  }

  /* Field-level diff, mirroring simulacrum/harness/diffs.py. */
  function compare(t, kind, a, b) {
    var diffs = [];
    walk('', a, b, diffs);
    if (!diffs.length) return null;
    return { step: t, kind: kind, diffs: diffs, detail: '' };
  }

  function walk(path, a, b, out) {
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) {
        out.push({ path: path, ref: 'len ' + a.length, fast: 'len ' + b.length });
        return;
      }
      for (var i = 0; i < a.length; i++) walk(path + '[' + i + ']', a[i], b[i], out);
      return;
    }
    if (a && b && typeof a === 'object') {
      for (var k in a) if (Object.prototype.hasOwnProperty.call(a, k)) {
        walk(path ? path + '.' + k : k, a[k], b[k], out);
      }
      return;
    }
    if (a !== b) out.push({ path: path, ref: a, fast: b });
  }

  /* ---------------------------------------------------------------------
   * Batched runner. Models a [N, ...] tensor implementation stepping N
   * instances together, including the one bug a differential test at n=1
   * provably cannot see.
   *
   * broadcastLeak reproduces this mistake, which is a one-character slip:
   *
   *     alive_count = self.alive.sum(-1, keepdim=True)   # [N, 1]  <-- keepdim
   *     terminated  = (alive_count == 0) | (self.energy <= 0.0) | ...
   *
   * [N, 1] | [N] does not error. It broadcasts to [N, N], and a later
   * reduction collapses it back to a plausible-looking [N]. The result is that
   * instance i terminates as soon as ANY instance in the batch runs out of
   * energy. At n = 1 the outer product is 1x1, so the bug is a perfect no-op
   * and the differential test passes clean.
   * ------------------------------------------------------------------- */
  function rolloutBatch(seeds, nSteps, bugs) {
    bugs = bugs || noBugs();
    var n = seeds.length, i;
    var states = [], episodes = [], out = [];
    for (i = 0; i < n; i++) {
      states.push(reset(seeds[i], 0, bugs));
      episodes.push(0);
      out.push([]);
    }

    for (var t = 0; t < nSteps; t++) {
      var results = [];
      for (i = 0; i < n; i++) {
        results.push(step(states[i], RNG.sampleAction(seeds[i], t, 4), bugs));
      }

      if (bugs.broadcastLeak) {
        // (alive_count == 0) is the [N, 1] operand; the energy and step-cap
        // checks are the [N] operands. any(-1) over the accidental [N, N].
        var anyColumn = false;
        for (i = 0; i < n; i++) {
          var s = results[i].state;
          if (s.energy <= 0.0 || s.t === MAX_STEPS) { anyColumn = true; break; }
        }
        for (i = 0; i < n; i++) {
          results[i].terminated = (countAlive(results[i].state.alive) === 0) || anyColumn;
        }
      }

      for (i = 0; i < n; i++) {
        out[i].push({ t: t, state: results[i].state, reward: results[i].reward,
                      terminated: results[i].terminated });
        states[i] = results[i].state;
        if (results[i].terminated) {
          episodes[i]++;
          states[i] = reset(seeds[i], episodes[i], bugs);
        }
      }
    }
    return out;
  }

  /* One instance run alone, batched with n = 1. This is the comparison
   * test_batch_independence makes against the in-batch run. */
  function rolloutSolo(seed, nSteps, bugs) {
    return rolloutBatch([seed], nSteps, bugs)[0];
  }

  /* Search for a seed whose action stream actually reaches the state a bug
   * corrupts. Some bugs are only reachable from states random play rarely
   * visits, which is exactly why the battery sweeps K seeds rather than one. */
  function findCatchingSeed(bugs, nSteps, limit) {
    for (var s = 0; s < (limit || 400); s++) {
      if (runDifferential(s, nSteps || 300, bugs).divergence) return s;
    }
    return null;
  }

  global.SimForager = {
    G: G, K: K, N_KINDS: N_KINDS, MAX_STEPS: MAX_STEPS, GUST_P: GUST_P,
    START_ENERGY: START_ENERGY, STEP_COST: STEP_COST, GAINS: GAINS,
    ENERGY_MAX: ENERGY_MAX, DELTAS: DELTAS, Slots: S,
    reset: reset, step: step, observe: observe, toJSON: toJSON,
    countAlive: countAlive,
    runDifferential: runDifferential,
    findCatchingSeed: findCatchingSeed,
    rolloutBatch: rolloutBatch,
    rolloutSolo: rolloutSolo
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
