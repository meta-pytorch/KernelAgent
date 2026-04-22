import multiprocessing as mp
from pathlib import Path

from triton_kernel_agent.worker import VerificationWorker


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_response(self, model_name: str, messages: list[dict[str, str]], **kwargs):
        if model_name == "review-model":
            return _FakeResponse(
                '{"verdict":"needs_changes","summary":"reviewed",'
                '"high_severity_findings":["masking risk"],'
                '"medium_severity_findings":[],"recommended_actions":["fix masks"],'
                '"prompt_addendum":"Audit mask handling before next attempt."}'
            )

        return _FakeResponse(
            '```python\n'
            'def kernel_function(*args, **kwargs):\n'
            '    return True\n'
            '```'
        )


def test_adversarial_review_feedback_is_injected(monkeypatch, tmp_path: Path):
    import triton_kernel_agent.worker as worker_module

    monkeypatch.setattr(
        worker_module,
        "get_model_provider",
        lambda model_name: _FakeProvider(model_name),
    )

    worker = VerificationWorker(
        worker_id=0,
        workdir=tmp_path / "work",
        log_dir=tmp_path / "logs",
        max_rounds=3,
        openai_model="gen-model",
        review_model="review-model",
        review_rounds=2,
    )

    verification_results = [
        (False, "", "round1 failure", None),
        (False, "", "round2 failure", None),
        (True, "PASS", "", None),
    ]
    refinement_feedback: list[str | None] = []

    def fake_single_verification_pass(kernel_code: str):
        return verification_results.pop(0)

    def fake_render_refinement_prompt(**kwargs):
        refinement_feedback.append(kwargs.get("review_feedback"))
        return "refine prompt"

    monkeypatch.setattr(worker, "_single_verification_pass", fake_single_verification_pass)
    monkeypatch.setattr(
        worker.prompt_manager,
        "render_kernel_refinement_prompt",
        fake_render_refinement_prompt,
    )
    monkeypatch.setattr(
        worker.prompt_manager,
        "render_adversarial_review_prompt",
        lambda **kwargs: "review prompt",
    )

    result = worker.run(
        kernel_code="def kernel_function(*args, **kwargs):\n    return False\n",
        test_code=["print('test')"],
        problem_description="Implement a kernel",
        success_event=mp.Event(),
    )

    assert result["success"] is True
    assert refinement_feedback[0] == ""
    assert "needs_changes" in refinement_feedback[1]
