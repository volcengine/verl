# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Tuple
import torch


class FusedEntropy(torch.autograd.Function):
    @staticmethod
    def entropy_fwd(
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Naive entropy function

        Args:
            hidden_states (torch.Tensor): [B, T, D]
            vocab_weights (torch.Tensor): [V, D]

        Returns:
            entropy (torch.Tensor): [B, T]
        """
        output_dtype = hidden_states.dtype
        logits = (hidden_states @ vocab_weights.t()) / temperature
        logits = logits.to(torch.float32)

        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

        return entropy.to(output_dtype)

    @staticmethod
    def entropy_bwd(
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function mirrors entroyp_fwd.
        Down to the location where casting of logits to fp32 is done.
        """

        logits = (hidden_states @ vocab_weights.t()) / temperature
        orig_logits_dtype = logits.dtype
        logits = logits.to(torch.float32)

        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

        # Safer to do this rather than log(pd)
        log_pd = torch.log_softmax(logits, dim=-1)
        grad_logits = pd * (log_pd + entropy.unsqueeze(-1)) * (-grad_output.unsqueeze(-1))
        grad_logits = grad_logits.to(orig_logits_dtype) / temperature

        grad_hidden_states = grad_logits @ vocab_weights
        grad_vocab_weights = (grad_logits.transpose(-1, -2) @ hidden_states).sum(0)

        return grad_hidden_states, grad_vocab_weights

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        entropy = torch.empty(
            B,
            T,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Process in chunks to save memory
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]

            chunk_entropy = FusedEntropy.entropy_fwd(
                chunk_hidden,
                vocab_weights,
                temperature=temperature,
            )

            entropy[:, chunk_start:chunk_end] = chunk_entropy

        # Save necessary tensors for backward
        ctx.save_for_backward(hidden_states, vocab_weights)
        ctx.chunk_size = chunk_size
        ctx.temperature = temperature

        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, vocab_weights = ctx.saved_tensors
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        B, T, H = hidden_states.shape
        V = vocab_weights.shape[0]

        # Initialize gradients
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_vocab_weights = torch.zeros_like(vocab_weights)

        # Process in chunks to save memory
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]
            chunk_grad_output = grad_output[:, chunk_start:chunk_end]

            h, v = FusedEntropy.entropy_bwd(
                grad_output=chunk_grad_output,
                hidden_states=chunk_hidden,
                vocab_weights=vocab_weights,
                temperature=temperature,
            )
            grad_hidden_states[:, chunk_start:chunk_end] += h
            grad_vocab_weights += v

        return (
            grad_hidden_states,  # hidden_states
            grad_vocab_weights,  # vocab_weights
            None,  # temperature
            None,  # chunk_size
        )


def fused_entropy(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Fuse the logits calculations with entropy calculation to save memory.

    Args:
        hidden_states (torch.Tensor): Last hidden states
        vocab_weights (torch.Tensor): lm_head weights
        chunk_size (int): Chunk size
    Returns:
        entropy (torch.Tensor): [B, T] shaped tensor representing token entropy.
    """
    return FusedEntropy.apply(
        hidden_states,
        vocab_weights,
        temperature,
        chunk_size,
    )


class FusedTokenLogProbs(torch.autograd.Function):
    @staticmethod
    def log_probs_fwd(
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Naive token log probs function

        Args:
            hidden_states (torch.FloatTensor): [B, T, D]
            vocab_weights (torch.FloatTensor): [V, D]
            input_ids (torch.LongTensor): [B, T]

        Returns:
            token_log_probs (torch.FloatTensor): [B, T]
        """
        output_dtype = hidden_states.dtype
        logits = torch.einsum("bth,vh->btv", hidden_states, vocab_weights) / temperature
        logits = logits.to(torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.to(output_dtype)

    @staticmethod
    def log_probs_bwd(
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bth,vh->btv", hidden_states, vocab_weights) / temperature
        orig_logits_dtype = logits.dtype
        logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=-1)

        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        grad_log_probs = one_hot_input - probs
        grad_logits = grad_output.to(torch.float32).unsqueeze(-1) * grad_log_probs
        grad_logits = grad_logits.to(orig_logits_dtype) / temperature

        grad_hidden_states = grad_logits @ vocab_weights
        grad_vocab_weights = (grad_logits.transpose(-1, -2) @ hidden_states).sum(0)

        return grad_hidden_states, grad_vocab_weights

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        B, T, H = hidden_states.shape
        V = vocab_weights.shape[0]

        # Initialize output tensor
        token_log_probs = torch.empty(
            B,
            T,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Process in chunks to save memory
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]
            chunk_ids = input_ids[:, chunk_start:chunk_end]

            chunk_token_log_probs = FusedTokenLogProbs.log_probs_fwd(
                hidden_states=chunk_hidden,
                vocab_weights=vocab_weights,
                input_ids=chunk_ids,
                temperature=temperature,
            )

            token_log_probs[:, chunk_start:chunk_end] = chunk_token_log_probs

        ctx.save_for_backward(input_ids, hidden_states, vocab_weights)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return token_log_probs

    @staticmethod
    def backward(ctx, grad_output):
        input_ids, hidden_states, vocab_weights = ctx.saved_tensors
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        B, T, H = hidden_states.shape
        V = vocab_weights.shape[0]

        # Initialize gradients
        grad_hidden = torch.zeros_like(hidden_states)
        grad_vocab = torch.zeros_like(vocab_weights)

        # Process in chunks to save memory
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            chunk_grad = grad_output[:, chunk_start:chunk_end]

            h, v = FusedTokenLogProbs.log_probs_bwd(
                grad_output=chunk_grad,
                hidden_states=chunk_hidden,
                vocab_weights=vocab_weights,
                input_ids=chunk_ids,
                temperature=temperature,
            )
            grad_hidden[:, chunk_start:chunk_end, :] += h
            grad_vocab += v

        return (
            grad_hidden,  # hidden_states
            grad_vocab,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
        )


def fused_log_probs(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Fuse the logits calculations with log probs calculation to save memory.

    Args:
        hidden_states (torch.Tensor): Last hidden states
        vocab_weights (torch.Tensor): lm_head weights
        input_ids (torch.Tensor): Token ids of log_probs
        chunk_size (int): Chunk size

    Returns:
        torch.Tensor: _description_
    """
    return FusedTokenLogProbs.apply(
        hidden_states,
        vocab_weights,
        input_ids,
        temperature,
        chunk_size,
    )
