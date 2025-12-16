# Running Guideline

## Step 1: Set up Hybrid Reward

**Prerequisite: 1 gpu needed**

Run `bash serve_rm.sh`  under `scripts/`

```
bash serve_rm.sh
```



**Parameter Explanation**

* `model_name`: the Quality Estimation (QE) model you would like to use. Could be set to `metricX` or `XComet`

* `base_model`: the base model you want to evaluate. The paths for the models are hard-coded in line 517-526 in `openrlhf/openrlhf/cli/serve_rm.py`.
* `port`: The port of the reward model on your machine.
* `max_len`: The maximum input sequence length.
* `rule`: whether to penalize  `\n` in the translation outputs. Set `True` will give the lowest reward if `\n`  be generated in the output.

* `lang_detect`: whether to turn on language detector or not. Set `True` to turn it on.

* `align`: whether to use word-alignment or not. Set `True` will turn it on.

* `masklid`: whether to mask the code-mixing part in the translation outputs. Set `True` will turn it on.
* `batch_size`: the batch size for the qe model to evaluate at one moment





## Step 2: Run RL

**Prerequisite: 4 or more gpus recommended**

Run `bash examples/scripts/train.sh `  under `openrlhf/`

```
bash examples/scripts/train.sh
```





**Parameter Explanation**

* `model`: The model you want to use. Please follow the `path_dict` in line 27-29

* `dataname`: The dataset you want to use. Please refer to the line 76 `prompt_data` for further info

* `size`: The model size you want to use. You can set whatever you want. It won't affect the final results and it will only affect the name appears on your checkpoint directory and wandb.

* `reward_name`: The reward name you want to use.  You can set whatever you want. It won't affect the final results and it will only affect the name appears on your checkpoint directory and wandb.