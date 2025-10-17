def compute_flowrl_objective_clip_only(self,
                                        log_prob=None,
                                        ref_log_prob=None,
                                        old_log_prob=None,
                                        log_z=None,
                                        reward=None,
                                        response_mask=None,
                                        clip_ratio=None,
                                        rollout_log_probs=None):
        
        """ FlowRL with clipped importance sampling (Clip-High) """

        # squeeze log_z to (B,)
        log_z = log_z.squeeze(-1)

        # Average token log-probs & rewards over valid positions
        avg_log_prob     = verl_F.masked_mean(log_prob,     response_mask, axis=1)
        avg_old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=1)
        avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
        seq_log_reward   = verl_F.masked_mean(reward,       response_mask, axis=1)

        # clip params
        eps_low  = self.config.clip_ratio_low  if hasattr(self.config, "clip_ratio_low")  else clip_ratio
        eps_high = self.config.clip_ratio_high if hasattr(self.config, "clip_ratio_high") else clip_ratio
        min_bound = avg_old_log_prob + math.log(1.0 - eps_low)
        max_bound = avg_old_log_prob + math.log(1.0 + eps_high)

        # Compute clip masks BEFORE clamping (for metrics)
        low_mask  = (avg_log_prob < min_bound)
        high_mask = (avg_log_prob > max_bound)

        # clamp (both sides)
        avg_log_prob = torch.clamp(avg_log_prob, min=min_bound, max=max_bound)

        # FlowRL residual: logZ + logpf - β*R - logpref
        delta = log_z + avg_log_prob - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

        # Importance ratio from current vs old policy (product of token ratios)
        log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
        imp_w_raw = torch.exp(log_w).detach()
        
        # Clamp importance weight to prevent extreme values (e.g., ~50)
        imp_w = torch.clamp(imp_w_raw, max=10.0)

        # Loss: weighted squared residual
        weighted_losses = imp_w * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)

        # Metrics
        loss_term_dict = {
            "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
            "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
            "actor/ref_log_prob": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
            "actor/final_loss": avg_loss.detach().item(),
            "actor/importance_weight_raw": imp_w_raw.mean().detach().item(),
            "actor/importance_weight": imp_w.mean().detach().item(),
            "actor/clip_rate_low":  low_mask.float().mean().detach().item(),
            "actor/clip_rate_high": high_mask.float().mean().detach().item()
        }

        return avg_loss, loss_term_dict

    def compute_flowrl_objective_with_gspo_selection(self,
                                                 log_prob=None,
                                                 ref_log_prob=None,
                                                 old_log_prob=None,
                                                 log_z=None,
                                                 reward=None,
                                                 response_mask=None,
                                                 clip_ratio=None,
                                                 rollout_log_probs=None):
    
        # ============ Step 1: GSPO clipping and selection (reuse GSPO code) ============
        
        # Get clip ratios from config
        clip_ratio_low = self.config.clip_ratio_low if hasattr(self.config, "clip_ratio_low") and self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if hasattr(self.config, "clip_ratio_high") and self.config.clip_ratio_high is not None else clip_ratio
        
        log_importance_ratio = log_prob - old_log_prob
        log_seq_importance_ratio = verl_F.masked_mean(log_importance_ratio, response_mask, axis=1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
        seq_importance_ratio = torch.exp(log_seq_importance_ratio)
        
        # Clipped ratio
        seq_importance_ratio_clipped = torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        
        seq_reward = verl_F.masked_mean(reward, response_mask, axis=1)
        pg_losses1 = -seq_reward * seq_importance_ratio
        pg_losses2 = -seq_reward * seq_importance_ratio_clipped
        
        # GSPO's two loss versions for selection
        use_clipped = torch.gt(pg_losses2, pg_losses1) 
    
        # ============ Step 2: Compute FlowRL with selected log_prob (reuse original function) ============
        log_z = log_z.squeeze(-1)
        
        # Average token log-probs & rewards over valid positions (from original function)
        avg_log_prob     = verl_F.masked_mean(log_prob,     response_mask, axis=1)
        avg_old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=1)
        avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
        seq_log_reward   = verl_F.masked_mean(reward,       response_mask, axis=1)

        # Compute clip bounds in log space
        min_bound = avg_old_log_prob + math.log(1.0 - clip_ratio_low)
        max_bound = avg_old_log_prob + math.log(1.0 + clip_ratio_high)

        # Apply clipping in log space
        low_mask = (avg_log_prob < min_bound)
        high_mask = (avg_log_prob > max_bound)
        avg_log_prob_clipped = torch.clamp(avg_log_prob, min=min_bound, max=max_bound)

        # Selectively apply clipping based on GSPO's decision
        avg_log_prob_final = torch.where(use_clipped, avg_log_prob_clipped, avg_log_prob)
                
        # FlowRL residual: logZ + logpf - β*R - logpref (from original function)
        delta = log_z + avg_log_prob_final - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

        log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
        imp_w_raw = torch.exp(log_w).detach()
        imp_w = torch.clamp(imp_w_raw, max=10.0)
        
        # Loss: weighted squared residual (from original function)
        weighted_losses = imp_w * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)
        
        # ============ Metrics ============
        batch_size = log_prob.size(0)
        actual_clipped_low = (low_mask & use_clipped).float().sum() 
        actual_clipped_high = (high_mask & use_clipped).float().sum() 
        total_clipped = (use_clipped).float().sum() 

        loss_term_dict = {
            # Log probs
            "actor/log_prob": avg_log_prob.mean().detach().item(),
            "actor/log_prob_final": avg_log_prob_final.mean().detach().item(),
            "actor/old_log_prob": avg_old_log_prob.mean().detach().item(),
            "actor/ref_log_prob": avg_ref_log_prob.mean().detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": seq_log_reward.mean().detach().item(),
            "actor/final_loss": avg_loss.detach().item(),
            # Clipping metrics
            "actor/gspo_clip_fraction": (total_clipped / batch_size).item(),  
            "actor/actual_clip_low": (actual_clipped_low / batch_size).item(),  
            "actor/actual_clip_high": (actual_clipped_high / batch_size).item(),
            "actor/importance_weight": imp_w.mean().detach().item(),
        }
        
        return avg_loss, loss_term_dict


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
        FlowRL enhanced with Clip-High (https://arxiv.org/pdf/2503.14476) and TIS (https://fengyao.notion.site/off-policy-rl)
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
        }

        if w_tis is not None:
            loss_term_dict["actor/tis_weight"] = w_tis.mean().detach().item()

        return avg_loss, loss_term_dict

    def compute_flowrl_objective_with_dapo_clip(self,
                                                 log_prob=None,
                                                 ref_log_prob=None,
                                                 old_log_prob=None,
                                                 log_z=None,
                                                 reward=None,
                                                 response_mask=None,
                                                 clip_ratio=None,
                                                 rollout_log_probs=None):
        """
        FlowRL with DAPO-style selective clipping.
        Uses DAPO/GSPO clipping logic to decide when to apply clipping to log_prob.
        Reference: https://arxiv.org/pdf/2507.18071 (GSPO paper)
        """

        # Get clip ratios from config
        clip_ratio_low = self.config.clip_ratio_low if hasattr(self.config, "clip_ratio_low") and self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if hasattr(self.config, "clip_ratio_high") and self.config.clip_ratio_high is not None else clip_ratio

        log_z = log_z.squeeze(-1)  # (B, 1) → (B,)

        # ============ Step 1: DAPO/GSPO-style clipping decision ============
        # Compute sequence-level importance ratio (geometric mean approach from GSPO)
        # seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
        negative_approx_kl = log_prob - old_log_prob
        # negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths  # (B,)

        # Combined ratio at token level (DAPO/GSPO hybrid ratio)
        # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
        # log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
        seq_importance_ratio = torch.exp(log_seq_importance_ratio)  # (B, T)

        # Clipped ratio
        seq_importance_ratio_clipped = torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)

        # Compute DAPO-style loss for selection
        pg_losses1 = -reward * seq_importance_ratio
        pg_losses2 = -reward * seq_importance_ratio_clipped

        # DAPO's selection: use clipped when clipped loss is higher
        use_clipped = torch.gt(pg_losses2, pg_losses1)  # (B, T)

        # ============ Step 2: Apply clipping to log_prob based on DAPO selection ============
        # Compute average log probs (sequence level)
        avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)  # (B,)
        avg_old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=1)
        avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
        seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

        # Compute clip bounds in log space
        min_bound = avg_old_log_prob + math.log(1.0 - clip_ratio_low)
        max_bound = avg_old_log_prob + math.log(1.0 + clip_ratio_high)

        # Clamp log_prob
        avg_log_prob_clipped = torch.clamp(avg_log_prob, min=min_bound, max=max_bound)

        # Selectively apply clipping: use majority voting across tokens for each sequence
        # If more than 50% of valid tokens in a sequence should be clipped, clip the entire sequence
        clip_fraction_per_seq = verl_F.masked_mean(use_clipped.float(), response_mask, axis=1)  # (B,)
        should_clip_seq = (clip_fraction_per_seq > 0.5)  # (B,)

        # Select clipped or unclipped log_prob based on DAPO decision
        avg_log_prob_final = torch.where(should_clip_seq, avg_log_prob_clipped, avg_log_prob)

        # ============ Step 3: Compute FlowRL loss ============
        # FlowRL residual: logZ + logpf - β*R - logpref
        delta = log_z + avg_log_prob_final - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

        # Importance ratio from current vs old policy
        log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
        imp_w_raw = torch.exp(log_w).detach()
        imp_w = torch.clamp(imp_w_raw, max=10.0)

        # Loss: weighted squared residual
        weighted_losses = imp_w * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)

        # ============ Step 4: Compute DAPO-style metrics ============
        batch_size = log_prob.size(0)
        num_clipped_seqs = should_clip_seq.float().sum()

        # Track clipping at boundaries
        low_mask = (avg_log_prob < min_bound)
        high_mask = (avg_log_prob > max_bound)

        loss_term_dict = {
            # Log probs
            "actor/log_prob": avg_log_prob.mean().detach().item(),
            "actor/log_prob_final": avg_log_prob_final.mean().detach().item(),
            "actor/old_log_prob": avg_old_log_prob.mean().detach().item(),
            "actor/ref_log_prob": avg_ref_log_prob.mean().detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": seq_log_reward.mean().detach().item(),
            "actor/final_loss": avg_loss.detach().item(),
            # DAPO-style clipping metrics
            "actor/dapo_clip_fraction": (num_clipped_seqs / batch_size).item(),
            "actor/dapo_clip_vote_mean": clip_fraction_per_seq.mean().detach().item(),
            "actor/clip_low_rate": (low_mask & should_clip_seq).float().sum().item() / batch_size,
            "actor/clip_high_rate": (high_mask & should_clip_seq).float().sum().item() / batch_size,
            # Importance weight
            "actor/importance_weight_raw": imp_w_raw.mean().detach().item(),
            "actor/importance_weight": imp_w.mean().detach().item(),
            # Ratio metrics (similar to DAPO paper)
            "actor/seq_importance_ratio": seq_importance_ratio.mean().detach().item(),
            "actor/kl_div": verl_F.masked_mean(-negative_approx_kl, response_mask).detach().item(),
        }

        return avg_loss, loss_term_dict