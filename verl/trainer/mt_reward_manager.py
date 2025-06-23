from verl import DataProto
from verl.utils.reward_score import compute_spBLEU
import torch

class MTRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = config.algorithm.reward_type
        self.reward_metric = config.algorithm.reward_metric
        assert self.reward_type in ['discrete', 'continuous'], "reward_type must be discrete or continue"
        assert self.reward_metric in ['BLEU', 'Model', 'Merge'], "reward_metric must be BLEU or Model or Merge" 
        self.bleu_threshold = config.algorithm.bleu_threshold 
        self.comet_threshold = config.algorithm.comet_threshold
        self.scale_factor = config.algorithm.reward_continuous_scale
        self.check_think = config.algorithm.check_think

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            # if self.check_think:
            #     print("prompt: ", self.tokenizer.decode(prompt_ids))
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            
            if 'qe_rm' in data_item.batch.keys():
                metric_score = float(data_item.batch['qe_rm'])
            else:
                metric_score = None
                print("No model-based metric score found, use BLEU")
            reward_tensor[i, valid_response_length - 1] = metric_score

        return reward_tensor


class MTValidManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            # decode
            response = self.tokenizer.decode(data_item.batch['responses'][:valid_response_length])
            reference = data_item.non_tensor_batch['reference']
            score = compute_spBLEU([response], [reference])
            reward_tensor[i, valid_response_length - 1] = score

            if "valid_qe_metric" in data_item.batch.keys():
                print("valid_comet_free_metric: ", float(data_item.batch['valid_comet_free_metric']))
            print("="*80 + "\n")


        return reward_tensor

