#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Auto-routing agent that decides whether to:
  - Solve a KernelBench-style problem directly with KernelAgent
  - Or run the full Fuser pipeline (extract → dispatch → compose)

Decision is based on a lightweight static analysis of the problem file:
  - Parse the problem as Python AST, inspect Model.__init__/forward
  - Count presence of ops commonly hard to fuse (conv_transpose2d, attention, group_norm chains)
  - Detect control flow in forward (if/for/while)
  - Approximate operation chain length (number of sequential transformations)

Routing policy (conservative):
  - Route to Fuser if any of:
      * attention-like patterns (softmax over QK, multihead attention, einsum with bmm)
      * conv_transpose2d present
      * group_norm used together with conv/conv_transpose or long chains (>=4 steps)
      * explicit control flow in forward
  - Otherwise route to KernelAgent directly

If the chosen path fails, the agent can optionally fall back to the other path.

CLI:
  python -m Fuser.auto_agent --problem /abs/path/to/problem.py \
      [--ka-model gpt-5] [--extract-model gpt-5] [--dispatch-model o4-mini] [--compose-model o4-mini] \
      [--verify] [--no-fallback]

Returns a JSON summary to stdout and writes the generated kernel path (if available).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from Fuser.pipeline import run_pipeline

# Local imports (available inside repo)
from triton_kernel_agent import TritonKernelAgent
from triton_kernel_agent.platform_config import (
    get_platform_choices,
    get_platform,
)
from utils.providers.models import get_model_provider


# ------------------------
# Static complexity analyzer
# ------------------------

_HARD_OP_TOKENS = {
    # canonical names
    "conv_transpose2d",
    "multiheadattention",
    "scaled_dot_product_attention",
    "attention",
    # functions that often imply attention-like patterns
    "softmax",
    "einsum",
    # normalization that in practice splits/fuses separately
    "group_norm",
}

_CONV_OP_TOKENS = {"conv2d", "conv1d", "conv3d"}
_POOL_OP_TOKENS = {
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
}
_ACT_TOKENS = {"relu", "gelu", "tanh", "sigmoid", "silu", "mish", "leaky_relu"}


# ------------------------
# Router decision cache helpers
# ------------------------

_ROUTER_CACHE_PATH = Path.cwd() / ".fuse" / "router_cache.json"


def _ensure_dir(p: Path) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _file_sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _load_router_cache() -> Dict[str, Any]:
    try:
        if _ROUTER_CACHE_PATH.is_file():
            return json.loads(_ROUTER_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_router_cache(cache: Dict[str, Any]) -> None:
    try:
        _ensure_dir(_ROUTER_CACHE_PATH)
        _ROUTER_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def _dotted_name(n: ast.AST) -> str:
    parts: list[str] = []
    cur = n
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


@dataclass
class Complexity:
    has_control_flow: bool
    has_attention_like: bool
    has_conv_transpose: bool
    has_group_norm: bool
    has_conv: bool
    pool_ops: int
    act_ops: int
    chain_len_estimate: int
    raw_op_names: Dict[str, int]

    def route_to_fuser(self) -> bool:
        # Primary triggers
        if self.has_attention_like:
            # If it's a likely self-contained SDPA block (short chain, no gn/transpose/control)
            if (
                not self.has_control_flow
                and not self.has_group_norm
                and not self.has_conv_transpose
                and self.chain_len_estimate <= 3
            ):
                # Favor KernelAgent which has fused attention examples
                return False
            return True
        if self.has_conv_transpose:
            return True
        if self.has_control_flow:
            return True
        # GroupNorm with convs and a long chain tends to be multi-stage
        if self.has_group_norm and (self.has_conv or self.pool_ops > 0):
            return True
        # Long op chains are usually better via subgraphs
        if self.chain_len_estimate >= 4:
            return True
        # Otherwise, favor KernelAgent fast path
        return False


def analyze_problem_code(code: str) -> Complexity:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: plain text scan
        txt = code.lower()
        has_attention_like = any(tok in txt for tok in _HARD_OP_TOKENS)
        has_conv_transpose = "conv_transpose2d" in txt
        has_group_norm = "groupnorm" in txt or "group_norm" in txt
        has_conv = any(t in txt for t in _CONV_OP_TOKENS)
        pool_ops = sum(txt.count(t) for t in _POOL_OP_TOKENS)
        act_ops = sum(txt.count(t) for t in _ACT_TOKENS)
        # naive chain estimate: number of lines with "x =" followed by known tokens
        chain_len_estimate = 0
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith("x =") and any(
                t in s
                for t in (
                    list(_CONV_OP_TOKENS)
                    + list(_POOL_OP_TOKENS)
                    + list(_ACT_TOKENS)
                    + ["matmul", "bmm", "einsum"]
                )
            ):
                chain_len_estimate += 1
        return Complexity(
            has_control_flow=(" if " in txt or " for " in txt or " while " in txt),
            has_attention_like=has_attention_like,
            has_conv_transpose=has_conv_transpose,
            has_group_norm=has_group_norm,
            has_conv=has_conv,
            pool_ops=pool_ops,
            act_ops=act_ops,
            chain_len_estimate=chain_len_estimate,
            raw_op_names={},
        )

    # AST path: inspect Model.forward for ops and control flow
    has_control_flow = False
    raw_op_counts: Dict[str, int] = {}
    has_attention_like = False
    has_conv_transpose = False
    has_group_norm = False
    has_conv = False
    pool_ops = 0
    act_ops = 0
    chain_len_estimate = 0

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            nonlocal has_control_flow
            if node.name == "forward":
                for n in ast.walk(node):
                    if isinstance(n, (ast.If, ast.For, ast.While)):
                        has_control_flow = True
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> Any:
            nonlocal \
                has_attention_like, \
                has_conv_transpose, \
                has_group_norm, \
                has_conv, \
                pool_ops, \
                act_ops, \
                chain_len_estimate
            try:
                name = _dotted_name(node.func).lower()
            except Exception:
                name = ""
            if name:
                raw_op_counts[name] = raw_op_counts.get(name, 0) + 1
                base = name.split(".")[-1]
                if base in _HARD_OP_TOKENS:
                    has_attention_like = True
                if base == "conv_transpose2d":
                    has_conv_transpose = True
                if base in _CONV_OP_TOKENS:
                    has_conv = True
                if base == "group_norm":
                    has_group_norm = True
                if base in _POOL_OP_TOKENS:
                    pool_ops += 1
                if base in _ACT_TOKENS:
                    act_ops += 1
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> Any:
            nonlocal chain_len_estimate
            # Rough chain estimate: x = <Call(...)>
            try:
                is_x = any(
                    isinstance(t, ast.Name) and t.id == "x" for t in node.targets
                )
                is_call = isinstance(node.value, ast.Call)
                if is_x and is_call:
                    chain_len_estimate += 1
            except Exception:
                pass
            self.generic_visit(node)

    _Visitor().visit(tree)
    return Complexity(
        has_control_flow=has_control_flow,
        has_attention_like=has_attention_like,
        has_conv_transpose=has_conv_transpose,
        has_group_norm=has_group_norm,
        has_conv=has_conv,
        pool_ops=pool_ops,
        act_ops=act_ops,
        chain_len_estimate=chain_len_estimate,
        raw_op_names=raw_op_counts,
    )


# ------------------------
# Auto router agent
# ------------------------


@dataclass
class RouteResult:
    route: str  # "kernelagent" or "fuser"
    success: bool
    details: Dict[str, Any]
    kernel_code: Optional[str] = None


class AutoKernelRouter:
    def __init__(
        self,
        ka_model: Optional[str] = None,
        ka_num_workers: int = 4,
        ka_max_rounds: int = 10,
        ka_high_reasoning: bool = True,
        # Router LLM
        router_model: Optional[str] = "gpt-5",
        router_high_reasoning: bool = True,
        router_temperature: float = 0.2,
        router_max_tokens: int = 700,
        extract_model: str = "gpt-5",
        dispatch_model: str = "o4-mini",
        compose_model: str = "o4-mini",
        workers: int = 4,
        max_iters: int = 5,
        llm_timeout_s: int = 1200,
        run_timeout_s: int = 1200,
        compose_max_iters: int = 5,
        verify: bool = True,
        dispatch_jobs: int = 2,
        allow_fallback: bool = True,
        target_platform: Optional[str] = None,
    ) -> None:
        self.ka_model = ka_model
        self.ka_num_workers = ka_num_workers
        self.ka_max_rounds = ka_max_rounds
        self.ka_high_reasoning = ka_high_reasoning
        # Router
        self.router_model = router_model
        self.router_high_reasoning = router_high_reasoning
        self.router_temperature = router_temperature
        self.router_max_tokens = router_max_tokens
        self.extract_model = extract_model
        self.dispatch_model = dispatch_model
        self.compose_model = compose_model
        self.workers = workers
        self.max_iters = max_iters
        self.llm_timeout_s = llm_timeout_s
        self.run_timeout_s = run_timeout_s
        self.compose_max_iters = compose_max_iters
        self.verify = verify
        self.dispatch_jobs = dispatch_jobs
        self.allow_fallback = allow_fallback
        self.platform_config = get_platform(target_platform)

    def _solve_with_kernelagent(self, problem_code: str) -> RouteResult:
        agent = TritonKernelAgent(
            num_workers=self.ka_num_workers,
            max_rounds=self.ka_max_rounds,
            model_name=self.ka_model,
            high_reasoning_effort=self.ka_high_reasoning,
            target_platform=self.platform_config,
        )
        try:
            # Ensure exceptions in KernelAgent do not abort routing; return a structured failure
            try:
                res = agent.generate_kernel(
                    problem_description=problem_code, test_code=None
                )
            except BaseException as exc:
                return RouteResult(
                    route="kernelagent",
                    success=False,
                    details={
                        "stage": "kernelagent",
                        "error": str(exc),
                        "exception": exc.__class__.__name__,
                    },
                )
        finally:
            try:
                agent.cleanup()
            except Exception:
                pass
        if res.get("success"):
            return RouteResult(
                route="kernelagent",
                success=True,
                kernel_code=res.get("kernel_code"),
                details={
                    "worker_id": res.get("worker_id"),
                    "rounds": res.get("rounds"),
                    "session_dir": res.get("session_dir"),
                },
            )
        return RouteResult(
            route="kernelagent",
            success=False,
            details={
                "message": res.get("message"),
                "session_dir": res.get("session_dir"),
            },
        )

    def _solve_with_fuser(self, problem_path: Path) -> RouteResult:
        """Run the full fuser pipeline and convert any failure into a structured result.

        Important: downstream helpers (extract/compose) may raise SystemExit on
        parse or provider errors. We must catch BaseException here so the
        autorouter can still emit a final JSON summary and optionally fall back
        to the KernelAgent path.
        """
        try:
            res = run_pipeline(
                problem_path=problem_path,
                extract_model=self.extract_model,
                dispatch_model=self.dispatch_model,
                compose_model=self.compose_model,
                dispatch_jobs=self.dispatch_jobs,
                workers=self.workers,
                max_iters=self.max_iters,
                llm_timeout_s=self.llm_timeout_s,
                run_timeout_s=self.run_timeout_s,
                verify=self.verify,
                compose_max_iters=self.compose_max_iters,
                target_platform=self.platform_config.name,
            )
        except BaseException as exc:  # catch SystemExit and others
            # Return a structured failure so caller can decide on fallback
            return RouteResult(
                route="fuser",
                success=False,
                details={
                    "stage": "fuser_pipeline",
                    "error": str(exc),
                    "exception": exc.__class__.__name__,
                },
            )

        comp = res.get("composition", {}) or {}
        ok = bool(comp.get("verify_passed", not self.verify))
        kernel_code: Optional[str] = None
        try:
            composed_path = comp.get("composed_path")
            if composed_path and Path(composed_path).is_file():
                kernel_code = Path(composed_path).read_text(encoding="utf-8")
        except Exception:
            pass
        return RouteResult(
            route="fuser",
            success=ok,
            kernel_code=kernel_code,
            details=res,
        )

    def solve(self, problem_path: Path) -> RouteResult:
        code = problem_path.read_text(encoding="utf-8")
        cx = analyze_problem_code(code)

        # Default heuristic-only decision used STRICTLY as a last-resort tie-breaker
        heuristic_prefers_fuser = cx.route_to_fuser()

        # Cache lookup by content hash to avoid repeated router calls
        code_hash = _file_sha256_text(code)
        cache = _load_router_cache()
        cached = cache.get(code_hash)

        strategy: Optional[str] = None
        route_conf: Optional[float] = None
        route_cfg: Dict[str, Any] = {}

        if isinstance(cached, dict):
            strategy = (
                str(cached.get("route_strategy") or cached.get("route") or "") or None
            )
            route_conf = cached.get("confidence")
            route_cfg = cached.get("config") or {}

        if strategy is None:
            # Try LLM-driven decision
            try:
                strategy, route_conf, info = self._llm_decide_route(
                    problem_path, code, cx
                )
                # Persist in cache for future runs
                cache[code_hash] = info.get("parsed") or {
                    "route_strategy": strategy,
                    "confidence": route_conf,
                }
                _save_router_cache(cache)
            except Exception:
                # No provider or failure; fall back later
                pass

        # Normalize strategy
        valid_strategies = {
            "kernelagent",
            "fuser",
            "kernel_then_fuser",
            "fuser_then_kernel",
        }
        if not strategy or strategy not in valid_strategies:
            # Confidence too low or invalid JSON; resort to heuristic
            strategy = "fuser" if heuristic_prefers_fuser else "kernelagent"

        # Apply optional dynamic config from router
        if isinstance(route_cfg, dict):
            # KernelAgent tuning
            self.ka_max_rounds = int(route_cfg.get("ka_max_rounds", self.ka_max_rounds))
            self.ka_num_workers = int(
                route_cfg.get("ka_num_workers", self.ka_num_workers)
            )
            if route_cfg.get("ka_model"):
                self.ka_model = str(route_cfg.get("ka_model"))
            # Fuser tuning
            dj = route_cfg.get("fuser_dispatch_jobs")
            if dj is not None:
                self.dispatch_jobs = dj
            self.compose_max_iters = int(
                route_cfg.get("compose_max_iters", self.compose_max_iters)
            )
            if "fuser_verify" in route_cfg:
                self.verify = bool(route_cfg.get("fuser_verify"))
            # Model choices
            if isinstance(route_cfg.get("llm_models"), dict):
                mm = route_cfg["llm_models"]
                self.extract_model = str(mm.get("extract", self.extract_model))
                self.dispatch_model = str(mm.get("dispatch", self.dispatch_model))
                self.compose_model = str(mm.get("compose", self.compose_model))

        # Execute per strategy with symmetric fallback governed by allow_fallback
        if strategy == "kernelagent":
            ka_res = self._solve_with_kernelagent(code)
            if ka_res.success or not self.allow_fallback:
                return ka_res
            return self._solve_with_fuser(problem_path)
        elif strategy == "fuser":
            fuser_res = self._solve_with_fuser(problem_path)
            if fuser_res.success or not self.allow_fallback:
                return fuser_res
            return self._solve_with_kernelagent(code)
        elif strategy == "kernel_then_fuser":
            ka_res = self._solve_with_kernelagent(code)
            if ka_res.success or not self.allow_fallback:
                return ka_res
            return self._solve_with_fuser(problem_path)
        else:  # fuser_then_kernel
            fuser_res = self._solve_with_fuser(problem_path)
            if fuser_res.success or not self.allow_fallback:
                return fuser_res
            return self._solve_with_kernelagent(code)

    # -------- LLM decision helper --------
    def _llm_decide_route(
        self, problem_path: Path, code: str, cx: Complexity
    ) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        """Ask an LLM to choose a routing STRATEGY and optional budgets.

        The LLM must return JSON with keys:
          - route_strategy: 'kernelagent' | 'fuser' | 'kernel_then_fuser' | 'fuser_then_kernel'
          - confidence: float in [0,1]
          - rationale: string
          - expected_wall_time_s: number (optional)
          - expected_success_prob: number in [0,1] (optional)
          - config: optional object {ka_max_rounds, ka_num_workers, fuser_dispatch_jobs, compose_max_iters, fuser_verify, llm_models{extract,dispatch,compose,ka_model}}

        Returns (route_strategy, confidence, raw_info). May raise if provider unavailable.
        """
        if not self.router_model:
            raise RuntimeError("router_model not specified")
        provider = get_model_provider(self.router_model)
        # Build a compact feature JSON for the model
        pp = str(problem_path).lower()
        if ("kernelbench" in pp) and ("level3" in pp):
            lvl = "level3"
        elif ("kernelbench" in pp) and ("level2" in pp):
            lvl = "level2"
        elif ("kernelbench" in pp) and ("level1" in pp):
            lvl = "level1"
        else:
            lvl = "unknown"
        feats = {
            "has_control_flow": cx.has_control_flow,
            "has_attention_like": cx.has_attention_like,
            "has_conv_transpose": cx.has_conv_transpose,
            "has_group_norm": cx.has_group_norm,
            "has_conv": cx.has_conv,
            "pool_ops": cx.pool_ops,
            "act_ops": cx.act_ops,
            "chain_len_estimate": cx.chain_len_estimate,
            "raw_op_names_top": sorted(
                cx.raw_op_names.items(), key=lambda kv: kv[1], reverse=True
            )[:10],
            "path_hint": str(problem_path),
            "level_hint": lvl,
        }
        system = "Return exactly one JSON object. No extra text."
        user = (
            "You are an auto-router for KernelBench deciding the best strategy to minimize wall-clock time while maintaining correctness.\n"
            "You can choose one of four strategies: 'kernelagent', 'fuser', 'kernel_then_fuser', 'fuser_then_kernel'.\n\n"
            "Guidance:\n"
            "- Use 'kernelagent' for simple/linear op chains (pointwise, pooling, small convs), no attention/conv_transpose, no complex normalization chains, no control flow.\n"
            "- Use 'fuser' when complex patterns exist: conv_transpose2d, group_norm with conv/pool chains, explicit control flow, long chains (>=4), or multi-branch/residual merges.\n"
            "- Attention note: A standard scaled-dot-product attention block (Q,K,V + softmax + matmul) can be implemented as a single fused kernel using KernelAgent templates; prefer 'kernelagent' for such self-contained attention. If attention is embedded in a larger pipeline (QKV projections + residual + norms or cross-attention), prefer 'fuser'.\n"
            "- Use 'kernel_then_fuser' if it's likely trivial but uncertainty exists (balance time by trying KA first, then fall back).\n"
            "- Use 'fuser_then_kernel' if it's likely complex but some chance KA suffices (try fuser first).\n"
            "- Consider architectural cues: ResNet blocks (conv/bn/relu), UNet/decoder (conv_transpose2d), Transformers (QKV, softmax, matmul/einsum), MLPs, depthwise-separable convs, GroupNorm+activation chains, 1D/3D convs, RNN/LSTM ops.\n"
            "- Level hint (only if provided for KernelBench): level1 should strongly bias to 'kernelagent' unless complexity triggers; level2/3 more likely 'fuser'. If level_hint is 'unknown', ignore level-based bias.\n\n"
            "Examples:\n"
            "- If features show level1 + ops=['relu','max_pool2d'] + chain_len<=2 => route_strategy='kernelagent'.\n"
            "- If features show conv_transpose2d present (any level) => route_strategy='fuser'.\n"
            "- If features show a self-contained SDPA (Q,K,V + softmax + matmul) => route_strategy='kernelagent'.\n"
            "- If features show attention plus surrounding projections/residual/norm chains => route_strategy='fuser'.\n"
            "- If features show group_norm with conv and pooling, chain_len>=4 => route_strategy='fuser_then_kernel' with fuser first.\n"
            "- If uncertain on level2 small GEMM+ReLU, choose 'kernel_then_fuser' with small ka_max_rounds.\n\n"
            "Output strictly as a JSON object:\n"
            "{\n"
            "  'route_strategy': 'kernelagent'|'fuser'|'kernel_then_fuser'|'fuser_then_kernel',\n"
            "  'confidence': 0..1,\n"
            "  'rationale': str,\n"
            "  'expected_wall_time_s': number?,\n"
            "  'expected_success_prob': 0..1?,\n"
            "  'config': {\n"
            "    'ka_max_rounds': int?, 'ka_num_workers': int?, 'ka_model': str?,\n"
            "    'fuser_dispatch_jobs': int|'auto'?, 'compose_max_iters': int?, 'fuser_verify': bool?,\n"
            "    'llm_models': {'extract': str?, 'dispatch': str?, 'compose': str?, 'ka_model': str?}?\n"
            "  }?\n"
            "}\n\n"
            f"Features:\n```json\n{json.dumps(feats, indent=2)}\n```\n\n"
            "Problem code:\n```python\n" + code + "\n```\n"
        )
        kwargs: Dict[str, Any] = {
            "max_tokens": self.router_max_tokens,
            "temperature": self.router_temperature,
        }
        if self.router_high_reasoning:
            kwargs["high_reasoning_effort"] = True
        resp = provider.get_response(
            self.router_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        txt = resp.content or ""
        # Best-effort JSON parse
        route = None
        conf = None
        raw_info: Dict[str, Any] = {"raw": txt}
        try:
            # If model returned extra text, try to locate JSON object
            first = txt.find("{")
            last = txt.rfind("}")
            cand = (
                txt[first : last + 1]
                if first != -1 and last != -1 and last > first
                else txt
            )
            data = json.loads(cand)
            # Back-compat: accept 'route' or 'route_strategy'
            route = (
                str(data.get("route_strategy") or data.get("route") or "")
                .strip()
                .lower()
                or None
            )
            c = data.get("confidence")
            if isinstance(c, (int, float)):
                conf = float(c)
            raw_info["parsed"] = data
        except Exception:
            pass
        return route, conf, raw_info


# ------------------------
# CLI
# ------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Auto-router for KernelBench problems (KernelAgent vs Fuser)"
    )
    p.add_argument("--problem", required=True, help="Absolute path to the problem file")
    p.add_argument(
        "--ka-model",
        default=None,
        help="Model for KernelAgent (optional; uses env default if omitted)",
    )
    p.add_argument("--ka-workers", type=int, default=4)
    p.add_argument("--ka-rounds", type=int, default=10)
    p.add_argument("--no-ka-high-reasoning", action="store_true")
    p.add_argument("--router-model", default="gpt-5")
    p.add_argument("--no-router-high-reasoning", action="store_true")
    p.add_argument("--router-temp", type=float, default=0.2)
    p.add_argument("--router-max-tokens", type=int, default=700)
    p.add_argument("--extract-model", default="gpt-5")
    p.add_argument("--dispatch-model", default="o4-mini")
    p.add_argument("--compose-model", default="o4-mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--llm-timeout-s", type=int, default=1200)
    p.add_argument("--run-timeout-s", type=int, default=1200)
    p.add_argument("--compose-max-iters", type=int, default=5)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dispatch-jobs", type=int, default=2)
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform (default: cuda)",
    )
    args = p.parse_args(argv)

    # Load environment variables from .env file
    load_dotenv()

    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}", file=sys.stderr)
        return 2

    router = AutoKernelRouter(
        ka_model=args.ka_model,
        ka_num_workers=args.ka_workers,
        ka_max_rounds=args.ka_rounds,
        ka_high_reasoning=(not args.no_ka_high_reasoning),
        router_model=args.router_model,
        router_high_reasoning=(not args.no_router_high_reasoning),
        router_temperature=args.router_temp,
        router_max_tokens=args.router_max_tokens,
        extract_model=args.extract_model,
        dispatch_model=args.dispatch_model,
        compose_model=args.compose_model,
        workers=args.workers,
        max_iters=args.max_iters,
        llm_timeout_s=args.llm_timeout_s,
        run_timeout_s=args.run_timeout_s,
        compose_max_iters=args.compose_max_iters,
        verify=args.verify,
        dispatch_jobs=args.dispatch_jobs,
        allow_fallback=(not args.no_fallback),
        target_platform=args.target_platform,
    )

    try:
        res = router.solve(problem_path)
    except BaseException as exc:  # include SystemExit
        # Always emit a final JSON summary for coverage tooling
        print(
            json.dumps(
                {
                    "route": None,
                    "success": False,
                    "details": {
                        "stage": "autorouter",
                        "error": str(exc),
                        "exception": exc.__class__.__name__,
                    },
                }
            ),
            flush=True,
        )
        return 1

    out: Dict[str, Any] = {
        "route": res.route,
        "success": res.success,
        "details": res.details,
    }
    if res.kernel_code:
        out["kernel_code"] = res.kernel_code
    print(json.dumps(out, indent=2), flush=True)
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
