"""The differential runner — the keystone of the validation battery.

Runs the reference implementation (seed s) and batched instance i (seed s)
side by side under an identical action stream and asserts, at every step:

- serialized state identical (per schema.json, honoring declared x-atol),
  including the TERMINAL state on episode boundaries
  (``info["final_state_json"]``) and the post-auto-reset state;
- observations identical;
- rewards identical;
- termination flags identical.

On divergence it reports the first divergent step with a field-level diff and
dumps both trajectories for inspection/rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from simulacrum.harness.config import HarnessConfig
from simulacrum.harness.diffs import FieldDiff, _wildcard_indices, diff_states, format_diffs
from simulacrum.traj.writer import TrajectoryWriter


@dataclass
class Divergence:
    step: int
    kind: str                       # "state" | "terminal_state" | "obs" | "reward" | "terminated"
    diffs: list[FieldDiff] = field(default_factory=list)
    detail: str = ""
    ref_traj_path: Path | None = None
    fast_traj_path: Path | None = None

    def __str__(self) -> str:
        lines = [f"First divergence at step {self.step} ({self.kind}):"]
        if self.detail:
            lines.append(f"  {self.detail}")
        if self.diffs:
            lines.append(format_diffs(self.diffs))
        if self.ref_traj_path:
            lines.append(f"  reference trajectory: {self.ref_traj_path}")
            lines.append(f"  fast trajectory:      {self.fast_traj_path}")
        return "\n".join(lines)


@dataclass
class DifferentialResult:
    seed: int
    steps_run: int
    episodes_completed: int
    divergence: Divergence | None
    tolerance_fields_used: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.divergence is None


def _obs_equal(ref_obs, fast_obs_row: torch.Tensor, atol: float) -> tuple[bool, str]:
    ref_arr = np.asarray(ref_obs)
    fast_arr = fast_obs_row.detach().cpu().numpy()
    if ref_arr.shape != fast_arr.shape:
        return False, f"shape {ref_arr.shape} vs {fast_arr.shape}"
    if ref_arr.dtype != fast_arr.dtype:
        return False, f"dtype {ref_arr.dtype} vs {fast_arr.dtype} (dtype drift breaks bit-identity)"
    if atol > 0.0:
        if not np.allclose(ref_arr, fast_arr, atol=atol, rtol=0.0, equal_nan=True):
            return False, f"values differ beyond obs_atol={atol}: ref={ref_arr!r} fast={fast_arr!r}"
    elif not np.array_equal(ref_arr, fast_arr):
        return False, f"values differ: ref={ref_arr!r} fast={fast_arr!r}"
    return True, ""


def run_differential(cfg: HarnessConfig, seed: int, n_steps: int | None = None,
                     atol: dict[str, float] | None = None,
                     dump_dir: str | Path | None = None) -> DifferentialResult:
    """Run reference vs batched(n=1) for ``n_steps`` under the config's action
    stream, crossing episode boundaries. Returns at the first divergence."""
    n_steps = n_steps or cfg.n_steps
    atol = atol or {}

    ref = cfg.reference_factory()
    fast = cfg.batched_factory(n=1, debug=False)

    episode = 0
    state = ref.reset(seed, episode)
    fast.reset(torch.tensor([seed], dtype=torch.int64))

    ref_w = TrajectoryWriter(cfg.name, seed, ref.to_json(state), source="reference",
                             env_root=cfg.root)
    fast_w = TrajectoryWriter(cfg.name, seed, fast.slice_to_json(0), source="batched",
                              instance=0, env_root=cfg.root)
    tol_used: dict[str, float] = {}

    def finish(t: int, div: Divergence | None) -> DifferentialResult:
        if div is not None and dump_dir is not None:
            dump = Path(dump_dir)
            div.ref_traj_path = ref_w.write(dump / f"{cfg.name}_seed{seed}_reference.json")
            div.fast_traj_path = fast_w.write(dump / f"{cfg.name}_seed{seed}_batched.json")
        return DifferentialResult(seed, t, episode, div, tol_used)

    def compare_states(t: int, kind: str, ref_json: dict, fast_json: dict) -> Divergence | None:
        diffs = diff_states(ref_json, fast_json, atol)
        if diffs:
            return Divergence(t, kind, diffs)
        if atol:
            # Record which declared tolerances were actually exercised: fields
            # that fail a strict (no-tolerance) diff but passed above.
            for d in diff_states(ref_json, fast_json, {}):
                strict_path = _wildcard_indices(d.path)
                if strict_path in atol:
                    tol_used[strict_path] = atol[strict_path]
        return None

    # Initial states must already agree.
    div = compare_states(0, "state", ref.to_json(state), fast.slice_to_json(0))
    if div is not None:
        return finish(0, div)

    for t in range(n_steps):
        action = cfg.action_sampler.sample(seed, t)
        actions_t = cfg.action_sampler.to_tensor([action])

        state, r_reward, r_term, _ = ref.step(action)
        ref_terminal_json = ref.to_json(state)
        ref_w.record(t, episode, ref_terminal_json, action, r_reward, r_term)

        f_obs, f_reward, f_term, f_info = fast.step(actions_t)

        # Termination flags first — everything downstream depends on them.
        if bool(r_term) != bool(f_term[0]):
            return finish(t, Divergence(
                t, "terminated",
                detail=f"reference terminated={bool(r_term)}, fast terminated={bool(f_term[0])}"))

        # Rewards.
        fr = float(f_reward[0])
        reward_ok = (r_reward == fr) if cfg.reward_atol == 0.0 else (abs(r_reward - fr) <= cfg.reward_atol)
        if not reward_ok:
            return finish(t, Divergence(
                t, "reward", detail=f"reference reward={r_reward!r}, fast reward={fr!r}"
                                    f" (reward_atol={cfg.reward_atol})"))
        if cfg.reward_atol > 0.0 and r_reward != fr:
            tol_used["<reward>"] = cfg.reward_atol

        fast_json = fast.slice_to_json(0)  # post-step (post-auto-reset) state, once
        if r_term:
            # Terminal state: fast side exposes it via info (post-step state
            # tensors already hold the next episode).
            fast_terminal = f_info.get("final_state_json", {}).get(0)
            if fast_terminal is None:
                return finish(t, Divergence(
                    t, "terminal_state",
                    detail="fast env terminated but info['final_state_json'] missing index 0"))
            div = compare_states(t, "terminal_state", ref_terminal_json, fast_terminal)
            if div is not None:
                return finish(t, div)
            fast_w.record(t, episode, fast_terminal, action, fr, True)
            # Reference resets explicitly; batched already auto-reset.
            episode += 1
            state = ref.reset(seed, episode)
            ref_json_post = ref.to_json(state)
        else:
            fast_w.record(t, episode, fast_json, action, fr, False)
            ref_json_post = ref_terminal_json  # same state, already serialized

        # Post-step (or post-reset) states must agree.
        div = compare_states(t, "state", ref_json_post, fast_json)
        if div is not None:
            return finish(t, div)

        # Observations: reference observes its current (possibly fresh) state;
        # fast returned the first-obs-of-next-episode for terminated instances.
        ok, why = _obs_equal(ref.observe(state), f_obs[0], cfg.obs_atol)
        if not ok:
            return finish(t, Divergence(t, "obs", detail=why))
        if cfg.obs_atol > 0.0:
            tol_used["<obs>"] = cfg.obs_atol

    return finish(n_steps, None)
