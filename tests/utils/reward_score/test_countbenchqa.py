from verl.utils.reward_score import countbenchqa


def _output(think: str, response: str, suffix: str = "") -> str:
    return f"{think}<|close|> think <|sep|><|open|> response <|sep|>{response}<|close|> response <|sep|>{suffix}"


def _result(think: str, response: str, ground_truth: str = "4") -> dict:
    return countbenchqa.compute_score(_output(think, response), ground_truth)


def test_correct_strict_response_gets_full_reward():
    result = _result("guess 9", "There are four. \\boxed{4}")
    assert result["score"] == 1.0
    assert result["predicted"] == "4"
    assert result["expected"] == "4"


def test_correct_relaxed_box_gets_accuracy_without_format_reward():
    assert _result("", "\\boxed(4)")["score"] == 0.9
    assert _result("", "\\boxed4")["score"] == 0.9


def test_answer_in_think_channel_is_never_rewarded():
    assert _result("I think \\boxed{4}", "\\boxed{5}")["score"] == 0.1
    assert countbenchqa.compute_score("I think \\boxed{4}", "4")["score"] == 0.0


def test_first_response_is_authoritative_over_trailing_generated_xtml():
    output = _output("", "\\boxed{5}", suffix="noise <|close|> think <|sep|>\\boxed{4}")
    assert countbenchqa.compute_score(output, "4")["score"] == 0.1


def test_invalid_ground_truth_is_not_rewarded():
    assert countbenchqa.compute_score(_output("", "\\boxed{4}"), "four")["score"] == 0.0


def test_integer_text_is_compared_canonically():
    result = _result("", "\\boxed{+0004}", ground_truth="0004")
    assert result["score"] == 1.0
    assert result["predicted"] == "4"
    assert result["expected"] == "4"


def test_arbitrary_precision_prediction_is_transfer_queue_serializable():
    from transfer_queue.utils.serial_utils import encode

    huge_integer = "9" * 5000
    result = _result("", f"\\boxed{{{huge_integer}}}")

    assert result["score"] == 0.1
    assert result["predicted"] == huge_integer
    # This is the exact encoder that raised in the 33-step long run.
    assert list(encode({"extra_fields": {"reward_extra_info": result}}))


def test_kimi_stop_conditions_preserve_existing_values():
    from verl.utils.model_adapters import add_kimi_k3_stop_conditions

    params = {"stop": ["existing"], "stop_token_ids": [7]}
    add_kimi_k3_stop_conditions(params, "kimi_k3", 163586)
    assert params == {
        "stop": ["existing", "<|close|> response <|sep|>"],
        "stop_token_ids": [7, 163586],
    }


def test_stop_conditions_ignore_other_models():
    from verl.utils.model_adapters import add_kimi_k3_stop_conditions

    params = {}
    add_kimi_k3_stop_conditions(params, "qwen3", None)
    assert params == {}


def test_kimi_ignore_eos_does_not_add_implicit_stop_conditions():
    from verl.utils.model_adapters import add_kimi_k3_stop_conditions

    params = {"ignore_eos": True, "stop": ["explicit"], "stop_token_ids": [7]}
    add_kimi_k3_stop_conditions(params, "kimi_k3", 163586)
    assert params == {"ignore_eos": True, "stop": ["explicit"], "stop_token_ids": [7]}
