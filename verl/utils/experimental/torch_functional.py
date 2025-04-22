from typing import Tuple
import torch


class FusedEntropy(torch.autograd.Function):
    @staticmethod
    def entropy_fn(
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
        logits = (hidden_states @ vocab_weights.t())
        logits.div_(temperature)
        logits = logits.to(torch.float32)
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        return entropy.to(output_dtype)

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

            chunk_entropy = FusedEntropy.entropy_fn(
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
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :].detach().requires_grad_(True)
            chunk_grad_output = grad_output[:, chunk_start:chunk_end]

            with torch.enable_grad():
                chunk_entropy = FusedEntropy.entropy_fn(
                    chunk_hidden,
                    vocab_weights,
                    temperature=temperature,
                )
                torch.autograd.backward(chunk_entropy, grad_tensors=chunk_grad_output, retain_graph=False)

            # Accumulate gradients
            grad_hidden_states[:, chunk_start:chunk_end, :] += chunk_hidden.grad
            grad_vocab_weights += vocab_weights.grad if vocab_weights.grad is not None else 0

            if chunk_hidden.grad is not None:
                chunk_hidden.grad.zero_()
            if vocab_weights.grad is not None:
                vocab_weights.grad.zero_()

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
    def log_probs_fn(
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
        logits = torch.einsum("bth,vh->btv", hidden_states, vocab_weights)
        logits.div_(temperature)
        log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.to(output_dtype)

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

            chunk_token_log_probs = FusedTokenLogProbs.log_probs_fn(
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
            chunk_hidden = hidden_states[:, chunk_start:chunk_end, :].detach().requires_grad_(True)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            chunk_grad = grad_output[:, chunk_start:chunk_end]

            with torch.enable_grad():
                chunk_token_log_probs = FusedTokenLogProbs.log_probs_fn(
                    hidden_states=chunk_hidden,
                    vocab_weights=vocab_weights,
                    input_ids=chunk_ids,
                    temperature=temperature,
                )

                # Compute gradients for this chunk
                torch.autograd.backward(
                    chunk_token_log_probs,
                    grad_tensors=chunk_grad,
                    retain_graph=False,
                )

            # Accumulate gradients
            grad_hidden[:, chunk_start:chunk_end, :] += chunk_hidden.grad
            grad_vocab += vocab_weights.grad if vocab_weights.grad is not None else 0

            # Clean up to save memory
            if chunk_hidden.grad is not None:
                chunk_hidden.grad.zero_()
            if vocab_weights.grad is not None:
                vocab_weights.grad.zero_()

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
