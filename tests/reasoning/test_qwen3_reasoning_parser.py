# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "qwen3"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# ---- enable_thinking=True (default), <think> in model output ----

# With <think></think>, non-streaming
WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
# With <think></think>, streaming
WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}

# <think> in output, no </think> yet (incomplete reasoning)
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}
ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# ---- enable_thinking=True, <think> in prompt (generation prompt) ----

# <think> in prompt, not in model output
THINK_IN_PROMPT = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

# <think> in prompt, model only outputs reasoning (incomplete, no </think>)
THINK_IN_PROMPT_INCOMPLETE = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# <think> in prompt, streaming, model only outputs reasoning (incomplete)
# With enable_thinking=True, streaming defaults to reasoning
THINK_IN_PROMPT_INCOMPLETE_STREAM = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# <think> in prompt, model outputs reasoning then </think> with no content
THINK_IN_PROMPT_COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# ---- enable_thinking=False ----

# No think tags, non-streaming
WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}
# No think tags, streaming
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}

# enable_thinking=True cases (default)
THINKING_ENABLED_CASES = [
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
    # <think> in prompt (generation prompt) cases
    pytest.param(
        False,
        THINK_IN_PROMPT,
        id="think_in_prompt",
    ),
    pytest.param(
        True,
        THINK_IN_PROMPT,
        id="think_in_prompt_stream",
    ),
    pytest.param(
        False,
        THINK_IN_PROMPT_INCOMPLETE,
        id="think_in_prompt_incomplete",
    ),
    pytest.param(
        True,
        THINK_IN_PROMPT_INCOMPLETE_STREAM,
        id="think_in_prompt_incomplete_stream",
    ),
    pytest.param(
        False,
        THINK_IN_PROMPT_COMPLETE_REASONING,
        id="think_in_prompt_complete_reasoning",
    ),
    pytest.param(
        True,
        THINK_IN_PROMPT_COMPLETE_REASONING,
        id="think_in_prompt_complete_reasoning_stream",
    ),
]

# enable_thinking=False cases
THINKING_DISABLED_CASES = [
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", THINKING_ENABLED_CASES)
def test_reasoning_thinking_enabled(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


@pytest.mark.parametrize("streaming, param_dict", THINKING_DISABLED_CASES)
def test_reasoning_thinking_disabled(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]
