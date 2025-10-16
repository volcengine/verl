import pickle

import torch
from transformers import AutoTokenizer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


class MetricFunc(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decoupling_by_mask(self, tensor, mask):
        bsz, _ = tensor.shape
        return [torch.masked_select(tensor[i], mask[i].bool()) for i in range(bsz)]

    def get_binary_group_val(self, val, flag):
        val_w_flag = torch.masked_select(val, flag)
        val_wo_flag = torch.masked_select(val, ~flag)
        return val_w_flag, val_wo_flag

    def get_keword_count(self, response: str, keywords):
        count = 0
        for keyword in keywords:
            count += response.lower().count(keyword.lower())
        return count

    def get_self_reflection_count(self, responses):
        keywords = [
            "rethink",
            "recheck",
            "double check",
            "try again",
            "re-evaluate",
            "check again",
            "let's correct it",
            "verify this step",
            "let's think again",
            "let's check",
            "核实",
            "验证",
            "检查",
            "稍等",
        ]
        return torch.tensor(
            list(
                map(
                    lambda resp: float(self.get_keword_count(resp, keywords)), responses
                )
            )
        ).reshape(-1)

    def get_new_idea_count(self, responses):

        keywords = [
            "alternatively",
            "try another approach",
            "reframe",
            "reimagine",
            "reconsider",
            "re-envision",
            "改进",
            "重新思考",
            "重新构想",
            "新思路",
            " wait",
        ]
        return torch.tensor(
            list(
                map(
                    lambda resp: float(self.get_keword_count(resp, keywords)), responses
                )
            )
        ).reshape(-1)

    def parse_tags(self, data_source):
        fields = data_source.split("-")
        fields = [e.split(";") for e in fields]
        tags = fields[0]
        nodes = fields[0][::]
        for depth in range(1, len(fields)):
            childs = fields[depth]
            new_nodes = []
            for node in nodes:
                for child in childs:
                    tag = node + "/" + child
                    new_nodes.append(tag)
                    tags.append(tag)
            nodes = new_nodes
        return tags

    def get_tag_data(self, val, batch_tags, target_tag):
        mask = torch.tensor([target_tag in tags for tags in batch_tags]).bool()
        return torch.masked_select(val, mask)


    def __call__(self, batch_dict, batch):

        bsz, seq_len = batch["responses"].shape
        state_mask = batch["attention_mask"][:, :-seq_len]
        action_mask = batch["attention_mask"][:, -seq_len:]
        response_len = action_mask.sum(-1).float().reshape(-1)
        response_ids = self.decoupling_by_mask(batch["responses"], action_mask)
        responses = self.tokenizer.batch_decode(response_ids)

        prompts_ids = self.decoupling_by_mask(
            batch_dict["input_ids"], batch_dict["attention_mask"]
        )
        prompts = self.tokenizer.batch_decode(prompts_ids)
        prompt_to_data_source = dict(
            [prompt, data_source]
            for prompt, data_source in zip(prompts, batch_dict["data_source"])
        )
        aug_prompts_ids = self.decoupling_by_mask(batch["prompts"], state_mask)
        aug_prompts = self.tokenizer.batch_decode(aug_prompts_ids)

        new_idea_count = self.get_new_idea_count(responses)
        self_reflection_count = self.get_self_reflection_count(responses)
        advantages_mean = masked_mean(
            values=batch["advantages"], mask=action_mask, axis=-1
        ).reshape(-1)
        scores = (batch["token_level_scores"] * action_mask).sum(axis=-1).reshape(-1)

        batch_query_tags = []

        for prompt,resp_len in zip(aug_prompts,response_len):
            query_tags = self.parse_tags(prompt_to_data_source[prompt]) 
            truncated = (resp_len == seq_len)
            if not truncated :
                query_tags += [_tag + "_untruncated" for _tag in query_tags]
            batch_query_tags.append(query_tags)

        all_query_tags = list(set(sum(batch_query_tags, [])))

        # ------- response length -----
        w_pos_adv = advantages_mean > 0
        resp_len_w_pos_adv, resp_len_wo_pos_adv = self.get_binary_group_val(
            response_len, w_pos_adv
        )
        metric = {
            "response_length/pos_adv/mean": resp_len_w_pos_adv.mean().item(),
            "response_length/pos_adv/std": resp_len_w_pos_adv.std().item(),
            "response_length/pos_adv/min": (
                resp_len_w_pos_adv.min().item() if len(resp_len_w_pos_adv) > 0 else None
            ),
            "response_length/pos_adv/max": (
                resp_len_w_pos_adv.max().item() if len(resp_len_w_pos_adv) > 0 else None
            ),
            "response_length/neg_adv/mean": resp_len_wo_pos_adv.mean().item(),
            "response_length/neg_adv/std": resp_len_wo_pos_adv.std().item(),
            "response_length/neg_adv/min": (
                resp_len_wo_pos_adv.min().item()
                if len(resp_len_wo_pos_adv) > 0
                else None
            ),
            "response_length/neg_adv/max": (
                resp_len_wo_pos_adv.max().item()
                if len(resp_len_wo_pos_adv) > 0
                else None
            ),
            "response_length/pos_adv_minus_neg_adv/mean": (
                resp_len_w_pos_adv.mean() - resp_len_wo_pos_adv.mean()
            ).item(),
        }
        for query_tag in all_query_tags:
            tag_resp_len = self.get_tag_data(
                val=response_len, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"response_length/{query_tag}/mean"] = tag_resp_len.mean().item()
            metric[f"response_length/{query_tag}/std"] = tag_resp_len.std().item()
            metric[f"response_length/{query_tag}/min"] = (
                tag_resp_len.min().item() if len(tag_resp_len) > 0 else None
            )
            metric[f"response_length/{query_tag}/max"] = (
                tag_resp_len.max().item() if len(tag_resp_len) > 0 else None
            )

        # ------- self-reflection -----
        for query_tag in all_query_tags:
            tag_self_ref_count = self.get_tag_data(
                val=self_reflection_count,
                batch_tags=batch_query_tags,
                target_tag=query_tag,
            )
            tag_new_thou_count = self.get_tag_data(
                val=new_idea_count, batch_tags=batch_query_tags, target_tag=query_tag
            )
            tag_resp_len = self.get_tag_data(
                val=response_len, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"self-reflection/self-reflection/{query_tag}/count"] = (
                tag_self_ref_count.mean().item()
            )
            metric[f"self-reflection/self-reflection/{query_tag}/density"] = (
                (tag_self_ref_count / tag_resp_len * 1000).mean().item()
            )
            metric[f"self-reflection/new-thoughts/{query_tag}/count"] = (
                tag_new_thou_count.mean().item()
            )
            metric[f"self-reflection/new-thoughts/{query_tag}/density"] = (
                (tag_new_thou_count / tag_resp_len * 1000).mean().item()
            )

        # ----- exploration -----
        for query_tag in all_query_tags:
            tag_scores = self.get_tag_data(
                val=scores, batch_tags=batch_query_tags, target_tag=query_tag
            )
            tag_advantages_mean = self.get_tag_data(
                val=advantages_mean, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"exploration/scores/{query_tag}/mean"] = tag_scores.mean().item()
            metric[f"exploration/scores/{query_tag}/std"] = tag_scores.std().item()
            metric[f"exploration/advantages/{query_tag}/mean"] = (
                tag_advantages_mean.mean().item()
            )
            metric[f"exploration/advantages/{query_tag}/std"] = (
                tag_advantages_mean.std().item()
            )

        w_self_ref_flag = self_reflection_count > 0
        w_new_thou_flag = new_idea_count > 0

        adv_w_self_ref, adv_wo_self_ref = self.get_binary_group_val(
            advantages_mean, w_self_ref_flag
        )
        adv_w_new_thou, adv_wo_new_thou = self.get_binary_group_val(
            advantages_mean, w_new_thou_flag
        )
        metric["exploration/advantages/w_self_reflection/mean"] = (
            adv_w_self_ref.mean().item()
        )
        metric["exploration/advantages/w_self_reflection/std"] = (
            adv_w_self_ref.std().item()
        )
        metric["exploration/advantages/wo_self_reflection/mean"] = (
            adv_wo_self_ref.mean().item()
        )
        metric["exploration/advantages/wo_self_reflection/std"] = (
            adv_wo_self_ref.std().item()
        )

        metric["exploration/advantages/w_new_thoughts/mean"] = (
            adv_w_new_thou.mean().item()
        )
        metric["exploration/advantages/w_new_thoughts/std"] = (
            adv_w_new_thou.std().item()
        )
        metric["exploration/advantages/wo_new_thoughts/mean"] = (
            adv_wo_new_thou.mean().item()
        )
        metric["exploration/advantages/wo_new_thoughts/std"] = (
            adv_wo_new_thou.std().item()
        )

        metric["exploration/advantages/w_minus_wo_self_reflection/mean"] = (
            adv_w_self_ref.mean() - adv_wo_self_ref.mean()
        ).item()
        metric["exploration/advantages/w_minus_wo_new_thoughts/mean"] = (
            adv_w_new_thou.mean() - adv_wo_new_thou.mean()
        ).item()
        return metric

    def compute_metric_wo_advantages(self, batch_dict, batch):

        bsz, seq_len = batch["responses"].shape
        state_mask = batch["attention_mask"][:, :-seq_len]
        action_mask = batch["attention_mask"][:, -seq_len:]
        response_len = action_mask.sum(-1).float().reshape(-1)
        response_ids = self.decoupling_by_mask(batch["responses"], action_mask)
        responses = self.tokenizer.batch_decode(response_ids)

        prompts_ids = self.decoupling_by_mask(
            batch_dict["input_ids"], batch_dict["attention_mask"]
        )
        prompts = self.tokenizer.batch_decode(prompts_ids)
        prompt_to_data_source = dict(
            [prompt, data_source]
            for prompt, data_source in zip(prompts, batch_dict["data_source"])
        )
        aug_prompts_ids = self.decoupling_by_mask(batch["prompts"], state_mask)
        aug_prompts = self.tokenizer.batch_decode(aug_prompts_ids)

        new_idea_count = self.get_new_idea_count(responses)
        self_reflection_count = self.get_self_reflection_count(responses)
        scores = (batch["token_level_scores"] * action_mask).sum(axis=-1).reshape(-1)
        
        batch_query_tags = []

        for prompt,resp_len in zip(aug_prompts,response_len):
            query_tags = self.parse_tags(prompt_to_data_source[prompt]) 
            truncated = (resp_len == seq_len)
            if not truncated :
                query_tags += [_tag + "_untruncated" for _tag in query_tags]
            batch_query_tags.append(query_tags)
        
        all_query_tags = list(set(sum(batch_query_tags, [])))

        # ------- response length -----
        metric = {}
        for query_tag in all_query_tags:
            tag_resp_len = self.get_tag_data(
                val=response_len, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"response_length/{query_tag}/mean"] = tag_resp_len.mean().item()
            metric[f"response_length/{query_tag}/std"] = tag_resp_len.std().item()
            metric[f"response_length/{query_tag}/min"] = (
                tag_resp_len.min().item() if len(tag_resp_len) > 0 else None
            )
            metric[f"response_length/{query_tag}/max"] = (
                tag_resp_len.max().item() if len(tag_resp_len) > 0 else None
            )

        # ------- self-reflection -----
        for query_tag in all_query_tags:
            tag_self_ref_count = self.get_tag_data(
                val=self_reflection_count,
                batch_tags=batch_query_tags,
                target_tag=query_tag,
            )
            tag_new_thou_count = self.get_tag_data(
                val=new_idea_count, batch_tags=batch_query_tags, target_tag=query_tag
            )
            tag_resp_len = self.get_tag_data(
                val=response_len, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"self-reflection/self-reflection/{query_tag}/count"] = (
                tag_self_ref_count.mean().item()
            )
            metric[f"self-reflection/self-reflection/{query_tag}/density"] = (
                (tag_self_ref_count / tag_resp_len * 1000).mean().item()
            )
            metric[f"self-reflection/new-thoughts/{query_tag}/count"] = (
                tag_new_thou_count.mean().item()
            )
            metric[f"self-reflection/new-thoughts/{query_tag}/density"] = (
                (tag_new_thou_count / tag_resp_len * 1000).mean().item()
            )

        # ----- exploration -----
        for query_tag in all_query_tags:
            tag_scores = self.get_tag_data(
                val=scores, batch_tags=batch_query_tags, target_tag=query_tag
            )
            metric[f"exploration/scores/{query_tag}/mean"] = tag_scores.mean().item()
            metric[f"exploration/scores/{query_tag}/std"] = tag_scores.std().item()

        return metric
