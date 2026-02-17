# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.

    When enable_thinking=True (default), <think> is added to the prompt as a
    generation prompt and may not appear in the model output. The streaming
    parser uses the enable_thinking flag to correctly classify tokens as
    reasoning vs content in this case.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._thinking_enabled = bool(chat_kwargs.get("enable_thinking", True))

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message during streaming.

        When enable_thinking=True and <think> is not in the model output
        (it's in the prompt as generation prompt), the base class would
        incorrectly treat reasoning tokens as content. This override
        uses the enable_thinking flag to correctly default to reasoning.

        When enable_thinking=False, falls back to the base class behavior
        which treats everything as content.
        """
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        # Only override when thinking is enabled and <think> is not in model
        # output (it's in the prompt as generation prompt).
        if (
            self._thinking_enabled
            and ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            else:
                return DeltaMessage(reasoning=delta_text)

        return ret

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        When enable_thinking=False, all output is treated as content.
        When enable_thinking=True:
        - If </think> is present, text before it is reasoning, after is content.
        - If </think> is absent (incomplete), all output is reasoning.
        - <think> may or may not be in the output (it could be in the prompt).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        if not self._thinking_enabled:
            return None, model_output

        # Strip <think> if present in model output (it may be in prompt instead).
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # If no </think>, the entire output is incomplete reasoning.
        if self.end_token not in model_output:
            return model_output, None

        # Split on </think>: reasoning before, content after.
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None
