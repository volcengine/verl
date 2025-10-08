def compute_flowrl_objective_tis_clip(self,
                                        log_prob=None,
                                        ref_log_prob=None,
                                        old_log_prob=None,
                                        log_z=None,
                                        reward=None,
                                        response_mask=None,
                                        clip_ratio=None,
                                        rollout_log_probs=None):
      """
      FlowRL enhanced with both Clip-High and TIS.
      References:
      - Clip-High: https://arxiv.org/pdf/2503.14476
      - TIS: https://fengyao.notion.site/off-policy-rl
      """
      # log_z: (B, 1) → (B,)
      log_z = log_z.squeeze(-1)

      # Average token log-probs & rewards over valid positions
      avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
      avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
      seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

      # Trajectory Balance residual: logZ + logpf - β*R - logpref
      delta = log_z + avg_log_prob - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

      # Importance ratio from current vs old policy (geometric mean for numerical stability)
      log_w = verl_F.masked_mean(log_prob - old_log_prob, response_mask, axis=1)
      imp_w = torch.exp(log_w).detach()

      # PPO-style clipping with separate clip_low and clip_high (asymmetric clipping)
      clip_ratio_low = self.config.clip_ratio_low if hasattr(self.config, 'clip_ratio_low') else clip_ratio
      clip_ratio_high = self.config.clip_ratio_high if hasattr(self.config, 'clip_ratio_high') else clip_ratio
      imp_w = torch.clamp(imp_w, 1 - clip_ratio_low, 1 + clip_ratio_high)

      # Truncated Importance Sampling (TIS): w_TIS = min(π_old / π_rollout, C_TIS)
      w_tis = None
      if self.config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
          # Compute TIS weight using mean in log space to keep values in reasonable range
          # This computes: min((π_old / π_rollout)^(1/T), C_TIS) where T = sequence length
          # Equivalent to geometric mean of token-level ratios, avoiding numerical overflow
          log_w_tis = verl_F.masked_mean(old_log_prob - rollout_log_probs, response_mask, axis=1)  # (B,)
          w_tis = torch.exp(log_w_tis).detach()  # Geometric mean of π_old / π_rollout
          w_tis = torch.clamp(w_tis, max=self.config.tis_imp_ratio_cap)  # min(w_tis, C_TIS)
          imp_w = imp_w * w_tis

      # Loss: weighted squared residual
      weighted_losses = imp_w * (delta ** 2)
      avg_loss = torch.mean(weighted_losses)

      # Metrics
      loss_term_dict = {
          "actor/logpf": verl_F.masked_mean(log_prob, response_mask).detach().item(),
          "actor/logp_ref": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
          "actor/log_z": log_z.mean().detach().item(),
          "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
          "actor/final_loss": avg_loss.detach().item(),
          "actor/importance_weight": imp_w.mean().detach().item(),
          "actor/loss_variant": "tis_clip",
      }

      if w_tis is not None:
          loss_term_dict["actor/tis_weight"] = w_tis.mean().detach().item()

      return avg_loss, loss_term_dict