from types import SimpleNamespace

from vllm_fl.worker.scheduler_fl import _mixed_step_token_budget


def req(computed, prompt=16384):
    return SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt)


def test_empty_and_pure_prefill_keep_default_budget():
    assert _mixed_step_token_budget([], 8192) == 8192
    assert _mixed_step_token_budget([req(0), req(4096)], 8192) == 8192


def test_pure_decode_keeps_default_budget():
    assert _mixed_step_token_budget([req(16384)], 8192) == 8192


def test_mixed_step_uses_smaller_budget():
    running = [req(4096), req(16384)]
    assert _mixed_step_token_budget(running, 8192) == 4096


def test_mixed_budget_never_exceeds_default():
    running = [req(4096), req(16384)]
    assert _mixed_step_token_budget(running, 2048) == 2048
