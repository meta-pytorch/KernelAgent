import logging
from pathlib import Path
from types import SimpleNamespace

from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationOrchestrator,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeProvider:
    def get_response(self, model_name: str, messages: list[dict[str, str]], **kwargs):
        return _FakeResponse(
            '{"verdict":"needs_changes","summary":"reviewed",'
            '"high_severity_findings":["occupancy risk"],'
            '"medium_severity_findings":[],"recommended_actions":["raise num_warps"],'
            '"prompt_addendum":"Revisit occupancy and launch geometry."}'
        )


def test_optimization_review_gate_runs_on_cadence(tmp_path: Path):
    prompt_manager = SimpleNamespace(
        render_adversarial_review_prompt=lambda **kwargs: "review prompt"
    )
    orchestrator = OptimizationOrchestrator(
        profiler=None,
        benchmarker=None,
        bottleneck_analyzer=None,
        verification_worker=None,
        prompt_manager=prompt_manager,
        provider=_FakeProvider(),
        model="gen-model",
        high_reasoning_effort=True,
        review_provider=_FakeProvider(),
        review_model="review-model",
        review_rounds=2,
        kernel_file=tmp_path / "kernel.py",
        gpu_specs=None,
        pytorch_baseline_time=None,
        artifact_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "output",
        logger=logging.getLogger("opt-review-test"),
        roofline_analyzer=SimpleNamespace(),
    )
    orchestrator.artifact_dir.mkdir(parents=True, exist_ok=True)

    review1 = orchestrator._maybe_request_review(
        completed_round_num=1,
        current_best_kernel="def kernel_function(): pass",
        problem_description="problem",
        test_code=["test"],
        evaluation_results={"time_ms": 1.0},
    )
    review2 = orchestrator._maybe_request_review(
        completed_round_num=2,
        current_best_kernel="def kernel_function(): pass",
        problem_description="problem",
        test_code=["test"],
        evaluation_results={"time_ms": 1.0},
    )

    assert review1 == ""
    assert "needs_changes" in review2
    assert (tmp_path / "artifacts" / "round002_review.txt").exists()
