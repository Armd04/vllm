# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposer for Speculative Cascades (arXiv:2405.19261).

Captures max draft probabilities at each step so the rejection sampler
can apply Chow's deferral rule: auto-accept tokens where the draft
model is sufficiently confident, skipping the expensive target-vs-draft
rejection check.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.draft_model import DraftModelProposer


class CascadeProposer(DraftModelProposer):
    """Proposer that captures max draft probabilities for cascade deferral."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self._max_draft_probs_list: list[torch.Tensor] = []
        self.max_draft_probs: torch.Tensor | None = None

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Override to capture max draft probability at each step."""
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        max_probs = probs.max(dim=-1).values  # [batch_size]
        self._max_draft_probs_list.append(max_probs)
        return logits.argmax(dim=-1)

    def propose(self, *args, **kwargs) -> torch.Tensor:
        self._max_draft_probs_list = []
        draft_token_ids = super().propose(*args, **kwargs)
        if self._max_draft_probs_list:
            if self.parallel_drafting and len(self._max_draft_probs_list) == 1:
                # Parallel drafting: _sample_from_logits was called once with
                # flattened logits of shape [batch_size * K, vocab_size],
                # producing max_probs of shape [batch_size * K].
                # Reshape to [batch_size, K] to match draft_token_ids layout.
                batch_size = draft_token_ids.shape[0]
                self.max_draft_probs = (
                    self._max_draft_probs_list[0]
                    .view(batch_size, self.num_speculative_tokens)
                )
            else:
                # Sequential drafting: K calls each produce [batch_size].
                # Stack [K entries of [batch_size]] -> [batch_size, K]
                self.max_draft_probs = torch.stack(
                    self._max_draft_probs_list, dim=1
                )
        else:
            self.max_draft_probs = None
        self._max_draft_probs_list = []
        return draft_token_ids
