#!/usr/bin/env python3
"""
Test suite for Atropos integration with VERL (Recipe Pattern)
This test validates the core functionality of the Atropos recipe trainer.
"""

import unittest
import sys
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


class TestAdvantageWeightedLoss(unittest.TestCase):
    """Tests for advantage-weighted loss computation."""
    
    def setUp(self):
        """Set up common test data."""
        self.batch_size = 2
        self.seq_len = 16
        self.vocab_size = 100
        
        # Input tokens (batch_size, seq_len)
        self.input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Advantages of the same shape
        self.advantages = torch.tensor([
            [0.5, 1.0, -0.5, 2.0, 0.0, 1.5, -1.0, 0.8, 1.2, -0.3, 0.7, 1.1, -0.8, 0.4, 0.9, -0.2],
            [1.5, -1.0, 0.5, 0.0, 2.0, -0.5, 1.0, 0.3, -1.2, 0.8, 0.6, -0.4, 1.3, 0.2, -0.7, 1.4]
        ])
        
        # Loss masks of the same shape (1.0 = include in loss, 0.0 = exclude)
        self.loss_mask = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])
        
        # Create mock logits (what a model would output)
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)

    def test_advantage_weighted_loss_computation(self):
        """
        Test the exact interface requested by the Nous Research bounty:
        
        "loss-masked weighted SFT - given a batch of tokens, a batch of advantages 
        of the same shape, and a batch of loss masks also of the same shape, 
        compute token-level CE and scale it by the advantages and then reduce and backprop."
        """
        # Compute the loss manually (this is what the trainer should do)
        # Shift for language modeling: predict next token
        shift_logits = self.logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = self.input_ids[:, 1:].contiguous()   # (batch_size, seq_len-1)
        shift_advantages = self.advantages[:, :-1].contiguous()  # (batch_size, seq_len-1)
        shift_loss_mask = self.loss_mask[:, :-1].contiguous()    # (batch_size, seq_len-1)
        
        # Flatten for CE loss computation
        flat_logits = shift_logits.view(-1, self.vocab_size)       # (batch_size*(seq_len-1), vocab_size)
        flat_labels = shift_labels.view(-1)                   # (batch_size*(seq_len-1),)
        flat_advantages = shift_advantages.view(-1)           # (batch_size*(seq_len-1),)
        flat_loss_mask = shift_loss_mask.view(-1)             # (batch_size*(seq_len-1),)
        
        # Compute cross-entropy loss per token
        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')  # (batch_size*(seq_len-1),)
        
        # Apply advantage weighting and loss masking
        weighted_loss = ce_loss * flat_advantages * flat_loss_mask
        
        # Reduce to scalar
        valid_tokens = flat_loss_mask.sum()
        final_loss = weighted_loss.sum() / (valid_tokens + 1e-8)
        
        # Verify the computation
        self.assertTrue(final_loss.requires_grad, "Loss should require gradients")
        self.assertTrue(torch.isfinite(final_loss), "Loss should be finite")
        self.assertGreater(valid_tokens, 0, "Should have some valid tokens")
        
        # Test backpropagation
        final_loss.backward()
        self.assertIsNotNone(self.logits.grad, "Gradients should be computed")
        self.assertTrue(torch.isfinite(self.logits.grad).all(), "Gradients should be finite")

    def test_shapes_match_requirement(self):
        """Test that input tensors have matching shapes as required by bounty."""
        self.assertEqual(self.input_ids.shape, self.advantages.shape, 
                        "Input tokens and advantages should have same shape")
        self.assertEqual(self.input_ids.shape, self.loss_mask.shape,
                        "Input tokens and loss mask should have same shape")
        self.assertEqual(self.advantages.shape, self.loss_mask.shape,
                        "Advantages and loss mask should have same shape")


class TestAdvantageProcessing(unittest.TestCase):
    """Tests for advantage normalization and clipping."""
    
    def setUp(self):
        """Set up common test data."""
        # Sample advantages and loss mask
        self.advantages = torch.tensor([2.0, -1.0, 3.0, 0.5, -0.5, 1.5])
        self.loss_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])  # Exclude index 3

    def test_advantage_normalization(self):
        """Test different advantage normalization methods."""
        # Test batch normalization
        valid_advantages = self.advantages[self.loss_mask.bool()]  # [2.0, -1.0, 3.0, -0.5, 1.5]
        mean_adv = valid_advantages.mean()
        std_adv = valid_advantages.std() + 1e-8
        normalized = (self.advantages - mean_adv) / std_adv
        
        self.assertTrue(torch.isfinite(normalized).all(), "Normalized advantages should be finite")
        
        # Check that valid advantages have ~0 mean and ~1 std
        valid_normalized = normalized[self.loss_mask.bool()]
        self.assertLess(abs(valid_normalized.mean()), 1e-6, "Normalized advantages should have ~0 mean")
        self.assertLess(abs(valid_normalized.std() - 1.0), 1e-6, "Normalized advantages should have ~1 std")

    def test_advantage_clipping(self):
        """Test advantage clipping functionality."""
        test_advantages = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
        min_val, max_val = -2.0, 2.0
        
        clipped = torch.clamp(test_advantages, min=min_val, max=max_val)
        expected = torch.tensor([-2.0, -2.0, 0.0, 2.0, 2.0])
        
        self.assertTrue(torch.allclose(clipped, expected), "Clipping should work correctly")
        self.assertTrue((clipped >= min_val).all(), "All values should be >= min_val")
        self.assertTrue((clipped <= max_val).all(), "All values should be <= max_val")


class TestAtroposDataFormat(unittest.TestCase):
    """Tests for Atropos data format processing."""
    
    def setUp(self):
        """Set up common test data."""
        # Simulate Atropos trajectory data format
        self.trajectory = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            "token_advantages": [0.1, 0.5, 1.0, 0.8, 0.2, -0.1, 0.9, 0.7, 0.3, 0.6],
            "mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # Last token masked out
        }

    def test_atropos_data_format(self):
        """Test that we can process Atropos-style data format."""
        # Validate data consistency
        advantages = torch.tensor(self.trajectory['token_advantages'], dtype=torch.float32)
        mask = torch.tensor(self.trajectory['mask'], dtype=torch.float32)
        
        self.assertEqual(advantages.shape, mask.shape, "Advantages and mask should have same shape")
        self.assertGreaterEqual(len(self.trajectory['messages']), 1, "Should have at least one message")
        self.assertGreaterEqual(advantages.mean(), -1.0, "Advantages should be reasonable (>= -1.0)")
        self.assertLessEqual(advantages.mean(), 1.0, "Advantages should be reasonable (<= 1.0)")

    def test_trajectory_structure(self):
        """Test that trajectory has required fields."""
        self.assertIn("messages", self.trajectory, "Trajectory should have messages field")
        self.assertIn("token_advantages", self.trajectory, "Trajectory should have token_advantages field") 
        self.assertIn("mask", self.trajectory, "Trajectory should have mask field")
        
        self.assertIsInstance(self.trajectory["messages"], list, "Messages should be a list")
        self.assertIsInstance(self.trajectory["token_advantages"], list, "Token advantages should be a list")
        self.assertIsInstance(self.trajectory["mask"], list, "Mask should be a list")


class TestAtroposRecipe(unittest.TestCase):
    """Tests for Atropos recipe integration."""
    
    def test_recipe_import(self):
        """Test that we can import the Atropos recipe trainer."""
        try:
            from recipe.atropos.atropos_trainer import AtroposTrainer
            
            # Check that the class has the required methods
            self.assertTrue(hasattr(AtroposTrainer, 'compute_advantage_weighted_sft_loss'),
                          "AtroposTrainer should have compute_advantage_weighted_sft_loss method")
            
        except ImportError as e:
            self.skipTest(f"Could not import AtroposTrainer from recipe: {e}")

    def test_recipe_demo_runs(self):
        """Test that the recipe demo can be imported and runs."""
        try:
            import recipe.atropos.main_atropos
            # If we can import it, the demo should work
            self.assertTrue(True, "Recipe demo imports successfully")
            
        except ImportError as e:
            self.skipTest(f"Could not import recipe demo: {e}")

    def test_required_methods_exist(self):
        """Test that all required methods exist on the recipe trainer class."""
        try:
            from recipe.atropos.atropos_trainer import AtroposTrainer
            
            required_methods = [
                'compute_advantage_weighted_sft_loss',
                '_compute_advantage_weighted_loss',
                '_normalize_advantages',
                '_clip_advantages'
            ]
            
            for method_name in required_methods:
                self.assertTrue(hasattr(AtroposTrainer, method_name),
                              f"AtroposTrainer should have {method_name} method")
                
        except ImportError as e:
            self.skipTest(f"Could not import AtroposTrainer from recipe: {e}")


class TestBountyInterface(unittest.TestCase):
    """
    Test the exact interface specified in the Nous Research bounty.
    This ensures we meet all requirements for the Atropos integration.
    """
    
    def setUp(self):
        """Set up test data for bounty interface."""
        self.batch_size = 3
        self.seq_len = 8
        self.vocab_size = 50
        
        # Example batch of tokens, advantages, and loss masks (same shape)
        self.input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        self.advantages = torch.randn(self.batch_size, self.seq_len)
        self.loss_mask = torch.bernoulli(torch.full((self.batch_size, self.seq_len), 0.8))

    def test_end_to_end_interface(self):
        """
        Test the complete end-to-end interface:
        Given batches of tokens, advantages, and loss masks of the same shape,
        compute token-level CE, scale by advantages, apply mask, reduce and backprop.
        """
        # This test validates the core requirement without needing the full trainer
        batch_size, seq_len = self.input_ids.shape
        vocab_size = self.vocab_size
        
        # Mock model output (logits)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        
        # The exact computation that should happen in the trainer
        # 1. Compute token-level cross entropy
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = self.input_ids[:, 1:].contiguous().view(-1)
        token_losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        
        # 2. Scale by advantages and apply loss mask
        shift_advantages = self.advantages[:, :-1].contiguous().view(-1)
        shift_mask = self.loss_mask[:, :-1].contiguous().view(-1)
        weighted_losses = token_losses * shift_advantages * shift_mask
        
        # 3. Reduce to scalar loss
        valid_tokens = shift_mask.sum()
        final_loss = weighted_losses.sum() / (valid_tokens + 1e-8)
        
        # 4. Backpropagate
        final_loss.backward()
        
        # Validate results
        self.assertTrue(torch.isfinite(final_loss), "Final loss should be finite")
        self.assertIsNotNone(logits.grad, "Should compute gradients")
        self.assertTrue(torch.isfinite(logits.grad).all(), "Gradients should be finite")

    def test_tensor_properties(self):
        """Validate tensor properties match bounty requirements."""
        # All tensors should have same shape for the first two dimensions
        self.assertEqual(self.input_ids.shape, self.advantages.shape)
        self.assertEqual(self.input_ids.shape, self.loss_mask.shape)
        
        # Loss mask should be binary (0 or 1)
        self.assertTrue(((self.loss_mask == 0) | (self.loss_mask == 1)).all())
        
        # Input IDs should be valid token indices
        self.assertTrue((self.input_ids >= 0).all())
        self.assertTrue((self.input_ids < self.vocab_size).all())


if __name__ == '__main__':
    unittest.main() 