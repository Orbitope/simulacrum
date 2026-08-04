# Simulacrum — promo kit

**Article:** *Make Your RL Environment Up to 92× Faster Without Breaking It*
**Slot:** Week 3 — r/reinforcementlearning Tue 18 Aug · r/MachineLearning Thu 20 Aug (gated) · X Wed 19 Aug

## Links

| Where | URL | Status |
|---|---|---|
| Article | `https://orbitope.github.io/simulacrum/` | ✅ confirmed (linked from the repo README) |
| Repo | `https://github.com/orbitope/simulacrum` | goes in a **comment**, never the post body |

## The hook

`toywalk` goes **385,199 → 35,288,308 steps/s** — 92×. But the batched version is a *different
program*: every `if` becomes arithmetic that computes both outcomes and discards one, and a subtly
wrong simulator still produces a training curve that goes down. So you write the environment twice —
readable single-instance `reference.py` and batched `fast.py`, both from one `spec.md`, **never from
each other** — and a battery steps them side by side demanding bit-identical agreement. Ten tests;
six must pass before anything is training-eligible.

The honest detail worth leading with: **batching made toywalk 28× *slower* at N=1** (13,567 steps/s)
before it made it 92× faster at N=8192.

---

## Reddit — r/reinforcementlearning · Tue 18 Aug

**Title**

```text
Differential testing for vectorized envs: write the environment twice and demand bit-identical agreement
```

**Body**

```text
Batching my env took it from 385k to 35.3M steps/s, but the fast version is a different program —
every branch becomes arithmetic that computes both outcomes and throws one away — and nothing was
checking it still had the same dynamics. So I keep the readable single-instance implementation and
run a battery that steps both side by side and compares at zero tolerance (integer state, so there's
nothing to be approximately right about). Six of the ten tests have to pass before I'll train on it.
```

**Link:** the article.

**Image:** `img/reference-vs-batched.mp4` (GIF fallback `.gif`) — the reference and batched grids
stepped in lockstep, values matching. This sub wants the *how*, not the headline number; the
throughput chart is the second image if you post a gallery.

**Top-level comment**

```text
Code, the ten tests, and two worked examples (one deliberately boring, one shaped enough to show the
broadcasting mistakes): https://github.com/orbitope/simulacrum

The test I'd steal first is test_batch_independence — bugs that are invisible at n=1 and only show
up when instances start leaking into each other.
```

---

## Reddit — r/MachineLearning · Thu 20 Aug — ⚠️ gated

Post only if the account has real standing by then: age, karma, and ideally a few non-promotional
comments in the sub. If it's still thin, **skip it** — this project stands fine on
r/reinforcementlearning alone, and a removed post costs more than a skipped one.

**Title** (flair `[P]`)

```text
[P] Simulacrum: differential testing as a correctness guarantee for batched RL environments
```

**Body**

```text
Vectorizing an environment is a rewrite into a different programming style, and the usual result is
that you either keep the slow version you can reason about or take the speed and stop being able to
explain your results. Simulacrum's position is that you write both from a single spec and never from
each other, then prove them equivalent: shared counter-based RNG so both roll the same dice, and a
battery that compares state bit-for-bit across episode boundaries. If they agree, either you
implemented the spec correctly twice or you made the same misreading twice in two very different
styles. Speedups on the bundled examples run 13x to 92x.
```

Frame it as methodology, not a tool showcase — that's the difference between a `[P]` post that
survives and one that doesn't.

**Image:** `img/batch-independence-clean.png` + `img/batch-independence-bug.png` — the same heatmap
before and after the keepdim bug is injected.

---

## X — Wed 19 Aug

**Tweet 1** — attach `img/throughput-92x.png`

```text
Vectorizing my RL environment made it 92x faster.

It also quietly made it a different program. Every `if` becomes arithmetic that computes both
outcomes and throws one away.

A subtly wrong simulator still gives you a training curve that goes down.
```
`chars: 249/280`

**Tweet 2** — attach `img/reference-vs-batched.mp4`

```text
So you write it twice.

A readable single-instance reference, and the batched one. Both from a single spec, never from each
other. Shared counter-based RNG so they roll the same dice.

Then a battery steps them side by side and compares at zero tolerance.
```
`chars: 255/280`

**Tweet 3**

```text
Agreement isn't proof you were right — it's proof you'd have had to make the same misreading twice,
in two very different programming styles.

The readable one stays the thing you debug against. The fast one is provably the same environment.
```
`chars: 241/280`

**Tweet 4** — attach `img/throughput-92x.png` if you didn't use it on tweet 1

```text
Detail I liked: batching made toywalk 28x SLOWER at N=1 (13,567 steps/s vs 385,199) before it made
it 92x faster at N=8192.

All the vectorization overhead, none of the payoff, until the batch is big enough.

https://orbitope.github.io/simulacrum/
```
`chars: 232/280`

---

## LinkedIn — Wed 19 Aug

Short post + link, matching the format that already worked for you — not a long-form narrative.
Body stays link-free; post the link yourself as the first comment once it's up.

**Post**

```text
Vectorizing my RL environment made it up to 92x faster. It also quietly turned it into a different
program.

Every `if` in a single-instance simulator becomes arithmetic in a batched one — compute both
branches, mask off the one that didn't happen. That's an easy place to introduce a bug a training
curve won't visibly flag; it'll just train a bit worse and you'll blame the hyperparameters.

The fix: write the environment twice. A slow, readable single-instance reference and a fast batched
version, both derived from one spec, never from each other — sharing a counter-based RNG so they
roll identical dice. A differential test battery then steps both side by side and demands they agree
at zero tolerance.

Detail I liked: on the simplest example, batching made it 28x SLOWER at batch size 1 before it became
92x faster at batch size 8192. All the overhead, none of the payoff, until the batch is big enough to
amortize it.

Code and the ten-test battery in the comments.
```

**Hashtags:** `#ReinforcementLearning #MachineLearning #SoftwareTesting`
**Image:** `img/reference-vs-batched.mp4` or `img/throughput-92x.png`

**First comment**

```text
Write-up: https://orbitope.github.io/simulacrum/
Code: https://github.com/orbitope/simulacrum
```

---

## Asset index — `docs/promo/img/`

All captured from `docs/index.html` itself by `scripts/capture_promo.mjs`; nothing is redrawn.

| File | What it is |
|---|---|
| `throughput-92x.png` | 385,199 → 35,288,308 steps/s on a log scale, including the slower-at-N=1 bar |
| `reference-vs-batched.mp4` / `.gif` | reference and batched grids stepped in lockstep |
| `jump-to-divergence.mp4` / `.gif` | the moment an injected bug makes the two disagree |
| `batch-independence-clean.png` / `batch-independence-bug.png` | the heatmap before and after the keepdim bug |
| `dependency.png` | the framework's contract graph |
| `forager-board.png` | the shaped example env — 8×8 grid, `[K,2]` berry positions |

---

## Appendix — the five-week calendar

Warm-up **Wed 5 – Thu 6 Aug**: ordinary commenting, no links, in r/WebGames and r/puzzles. Keep
low-level commenting going in each week's target subs throughout — with a new account this matters
more than any single post.

| Week | Reddit #1 | Reddit #2 | X | LinkedIn |
|---|---|---|---|---|
| 1 | Thu 6 Aug — **Gridlocked** → r/WebGames | Sat 8 Aug — r/puzzles | — (already posted) | Wed 5 Aug |
| 2 | Tue 11 Aug — **Hex Truchet** → r/proceduralgeneration | Thu 13 Aug — r/tabletopgamedesign | Wed 12 Aug | Wed 12 Aug |
| 3 | Tue 18 Aug — **Simulacrum** → r/reinforcementlearning | Thu 20 Aug — r/MachineLearning `[P]` (gated) | Wed 19 Aug | Wed 19 Aug |
| 4 | Tue 25 Aug — **Pushman** → r/Unity3D | Thu 27 Aug — r/gamedev | Wed 26 Aug | Wed 26 Aug |
| 5 | Tue 1 Sep — **RLevator** → r/reinforcementlearning | Thu 3 Sep — r/MachineLearning `[P]` (gated) | Wed 2 Sep | Wed 2 Sep |

Reddit posts land Tuesday mornings US-Eastern; the second sub is staggered two days so two threads
are never live at once. X threads go Wednesday, a day behind Reddit, so a good comment can be folded
in. **r/MachineLearning is gated on account standing** — skip it if the account is still thin; both
RL projects stand fine on r/reinforcementlearning alone. r/algorithms is deliberately unused: best
topical fit for Gridlocked, but hostile to self-promotion from a new account. Revisit after week 5.

LinkedIn rides the same Wednesday slot as X — one extra post to draft per week, no new day added.
Body stays link-free on every LinkedIn post; the link goes in your own first comment once it's up,
same convention as Reddit. Week 1's LinkedIn post (Wed 5 Aug) is the exception that runs a day ahead
of the Reddit warm-up, since LinkedIn has no comment-karma ramp to respect.

Note the two r/reinforcementlearning posts (this one and RLevator) are two full weeks apart and on
genuinely different topics. Don't compress that gap.

Nothing here posts itself. Reddit and X both punish anything that reads as automated, and the
comment replies are most of the value.
