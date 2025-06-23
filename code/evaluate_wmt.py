# coding=utf-8
# Copyright 2024 Google LLC
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
"""Evaluates the predictions from a MetricX model."""

import collections
import dataclasses
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import code.models as models

from typing import Optional, Tuple, Union, List
import scipy
from scipy import stats
from mt_metrics_eval import data
from mt_metrics_eval import meta_info
from mt_metrics_eval import tasks
import torch
import numpy as np
import transformers
import datasets


@dataclasses.dataclass
class Arguments:
  wmt_year: int = dataclasses.field(
      default=24,
      metadata={"help": "The WMT year to evaluate."},
  )
  
  dtype: str = dataclasses.field(
      default="fp32",
      metadata={
          "help": "The data type to use for the model. "
                  "Supported types: 'fp16', 'bf16'."
      },
  )
  
  model_name: str = dataclasses.field(
      default="metricx",
      metadata={
          "help": "The name or path of the model to use for evaluation. "
                  "Supported models: 'metricx', 'XComet'."
      },
  )
  
  output_dir: str = dataclasses.field(
      metadata={"help": "The output directory with evaluation metrics."},
      default="/dev/null",
  )
  
  model_size: str = dataclasses.field(
      default="xl",
      metadata={
          "help": "The size of the model to use for evaluation. "
                  "Supported sizes: 'xxl', 'xl'."
      },
  )
  
def get_dataset(
    ds: List[dict], model_name: str, tokenizer, max_input_length: int, device, is_qe: bool
):
  """Gets the test dataset for prediction.

  If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
  If it is false, there must be "hypothesis" and "reference" fields.

  Args:
    input_file: The path to the jsonl input file.
    tokenizer: The tokenizer to use.
    max_input_length: The maximum input sequence length.
    device: The ID of the device to put the PyTorch tensors on.
    is_qe: Indicates whether the metric is a QE metric or not.

  Returns:
    The dataset.
  """

  def _make_input(example):
    if model_name == 'metricX':
      if is_qe:
        example["input"] = (
            "source: "
            + example["source"]
            + " candidate: "
            + example["hypothesis"]
        )
      else:
        example["input"] = (
            "source: "
            + example["source"]
            + " candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
    elif 'Comet' in model_name:
      src = example.pop("source", None)
      mt = example.pop("hypothesis", None)
      if src is None or mt is None:
        raise ValueError(
            "Input data must have 'source' and 'hypothesis' fields for Comet models."
        )
      if is_qe:
        example["input"] = {
          "src": src, 
          "mt": mt
        }
      else:
        ref = example.pop("reference", "")
        example["input"] = {
          "src": src, 
          "mt": mt, 
          "ref": ref
        }
    else:
      raise ValueError("Unsupported model name in Dataset Processing: {}".format(model_name))
    return example

  def _tokenize(example):
    return tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )

  def _remove_eos(example):
    example["input_ids"] = example["input_ids"][:-1]
    example["attention_mask"] = example["attention_mask"][:-1]
    return example
  
  if model_name == "metricX":
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
  elif "Comet" in model_name:
    ds = ds.map(_make_input)

  return ds

def load_tokenizer_and_model(metric_name: str, model_size: str, model_dtype:str, device: torch.device) -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
  """Loads the tokenizer and model for the specified metric."""
  tokenizer, model = None, None
  if metric_name == 'metricX':
    path_dict = {
      "xl": {
        "fp32": "google/metricx-24-hybrid-xl-v2p6",
        "bf16": "google/metricx-24-hybrid-xl-v2p6-bfloat16",
      },
      "xxl": {
        "fp32": "google/metricx-24-hybrid-xxl-v2p6",
        "bf16": "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
      }
    }
    path = path_dict[model_size][model_dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", cache_dir="/mnt/data1/yifengliu/model")
    model = models.MT5ForRegression.from_pretrained(
       path, torch_dtype="auto", device_map="auto", cache_dir="/mnt/data1/yifengliu/model"
    )
  elif 'Comet' in metric_name:
    path_dict = {
      "XComet": "/mnt/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt",
      "Comet-qe-da": "/mnt/data1/yifengliu/model/models--Unbabel--wmt20-comet-qe-da/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt",
    }
    model_path = path_dict.get(metric_name, None)
    from comet import download_model, load_from_checkpoint
    print(f"Loading model from {model_path}")
    model = load_from_checkpoint(model_path)
  else:
    raise ValueError(f"Unsupported metric name: {metric_name}")
  # model.to(device)
  # model.parallelize() 
  model.eval()
  return tokenizer, model

def get_predictions(
    metric_name: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    srcs: List[str],
    hyps: List[str],
    refs: List[str],
) -> np.ndarray:
  predictions = None
  if metric_name == 'metricX':
    ds = [{"source": src, "hypothesis": hyp, "reference": ref} for src, hyp, ref in zip(srcs, hyps, refs)]
    ds = datasets.Dataset.from_list(ds)
    ds = get_dataset(ds, metric_name, tokenizer, max_input_length=1536, device='cuda', is_qe=True)
    training_args = transformers.TrainingArguments(
      # output_dir="/dev/null",
      per_device_eval_batch_size=1,
      dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
    )
    predictions, a, b = trainer.predict(test_dataset=ds)
  elif 'Comet' in metric_name:
    ds = [{"source": src, "hypothesis": hyp, "reference": ref} for src, hyp, ref in zip(srcs, hyps, refs)]
    ds = datasets.Dataset.from_list(ds)
    ds = get_dataset(ds, metric_name, tokenizer, max_input_length=1536, device='cuda:0', is_qe=True)
    ds = list(ds)
    inputs = [data['input'] for data in ds]
    model_output = model.predict(inputs, batch_size=1, gpus=torch.cuda.device_count())
    predictions = model_output.scores
    predictions = np.array(predictions)
  return predictions

def NewMetric(
    metric_name: str,
    model_size: str,
    model_dtype: str,
    lp: str,
    domains: dict[str, list[list[int]]],
    docs: dict[str, list[int]],
    src: list[str],
    ref: list[str],
    hyps: dict[list[str]]
) -> dict[str, list[float]]:
  """
  Generate metric scores.

  Args:
    level: Level for which to produce scores, 'sys' or 'seg'.
    lp: Language pair, eg 'en-de'.
    domains: Map from domain name to [[beg, end+1], ...] segment position lists.
    docs: Map from doc name to [beg, end+1] segment positions.
    src: List of source segments.
    ref: List of reference segments.
    hyps: Map from MT system name to output segments for that system.

  Returns:
    Map from system name to scores, a list of segment-level scores if level is
    'seg', or a list containing a single score if level is 'sys'.
  """
  # Sample metric just computes a length match between each hypothesis and the
  # reference. It ignores lp, domains, docs, and source.

  del domains, docs

  seg_scores = {}
  sys_scores = {}
  tokenizer, model = load_tokenizer_and_model(metric_name, model_size, model_dtype, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  for sysname, hyp in hyps.items():
    print(f"Evaluating {metric_name}-{model_size}-{model_dtype} for system {sysname} on {lp}...")
    predictions = get_predictions(metric_name, model, tokenizer, src, hyp, ref)
    if metric_name == 'metricX':
      seg_scores[sysname] = -predictions
      sys_scores[sysname] = [-predictions.mean()]
    elif 'Comet' in metric_name:
      seg_scores[sysname] = predictions
      sys_scores[sysname] = [predictions.mean()]
  return seg_scores, sys_scores

def get_lps(wmt_year: int) -> list[str]:
  """Returns the list of language pairs for the given WMT year."""
  if wmt_year == 24:
    return ["en-de", "en-es", "ja-zh", "cs-uk", "en-cs", "en-hi", "en-is","en-ja", "en-ru", "en-uk", "en-zh"]
  elif wmt_year == 23:
    return ["cs-uk", "en-cs", "en-he", "en-ru", "en-zh", "ja-en", "uk-en", "de-en", "en-de", "en-ja", "en-uk", "he-en", "ru-en", "zh-en"]
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def get_meta_info(wmt_year: int) -> meta_info.MetaInfo:
  """Returns the meta info for the given WMT year."""
  if wmt_year == 24:
    return meta_info.WMT24
  elif wmt_year == 23:
    return meta_info.WMT23
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def get_tasks(wmt_year: int, lps: list[str], k: int = 0) -> Tuple[tasks.Task, dict]:
  """Returns the tasks and weights for the given WMT year and language pairs."""
  if wmt_year == 24:
    return tasks.WMT24(lps, k)
  elif wmt_year == 23:
    return tasks.WMT23(lps, k)
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def write_result(metric_name: str, model_size: str, lp: str, seg_scores: dict[str, list[float]], sys_scores: dict[str, list[float]], output_dir: str) -> None:
  output_dir = os.path.join(output_dir, metric_name + '-' + model_size)
  seg_file = os.path.join(output_dir, "seg", f"{lp}.jsonl")
  sys_file = os.path.join(output_dir, "sys", f"{lp}.jsonl")
  os.makedirs(os.path.dirname(seg_file), exist_ok=True)
  os.makedirs(os.path.dirname(sys_file), exist_ok=True)
  with open(seg_file, "w") as f:
    for sysname, scores in seg_scores.items():
      for score in scores:
        score = np.array(score, dtype=np.float16)
        score = score.astype(float).tolist()
        f.write(json.dumps({"system": sysname, "score": score}) + "\n")
  with open(sys_file, "w") as f:
    for sysname, scores in sys_scores.items():
      for score in scores:
        score = np.array(score, dtype=np.float16)
        score = score.astype(float).tolist()
        f.write(json.dumps({"system": sysname, "score": score}) + "\n")

def evaluate(lps: list[str], model_scores: dict[list[int]], sy_scores: dict[list[int]]):
  for lp in lps:
    model_score, sys_score = model_scores[lp], sy_scores[lp]
    mask = ~np.isnan(sys_score)
    pearsonr, _ = stats.pearsonr(
      model_score[mask],
      sys_score[mask],
    )
    kendalltau, _ = stats.kendalltau(
      model_score[mask],
      sys_score[mask],
    )
    spearmanr, _ = stats.spearmanr(
      model_score[mask],
      sys_score[mask],
    )
    print(f"Language pair: {lp}")
    print(f"Pearson correlation: {pearsonr:.4f}")
    print(f"Kendall tau correlation: {kendalltau:.4f}")
    print(f"Spearman correlation: {spearmanr:.4f}")
    print("=========================================")

def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  lps = get_lps(args.wmt_year)
  baseline_metainfo = get_meta_info(args.wmt_year)
  evs_dict = {(f'wmt{args.wmt_year}', lp): data.EvalSet(f'wmt{args.wmt_year}', lp, True) for lp in lps}
  model_scores, sy_scores = {}, {}
  # ESA
  # lps = ["en-cs", "en-hi", "en-is"]
  lps = ["cs-uk"]
  
  # MQM
  # lps = ["en-de", "en-es", "ja-zh"]
  
  for lp in lps:
    evs = evs_dict[(f'wmt{args.wmt_year}', lp)]
    # gold_scores = evs.Scores("seg", "mqm")
    gold_scores = evs.Scores("seg", "esa")
    for refname, ref in evs.all_refs.items():
      seg_scores, sys_scores = NewMetric(
          args.model_name, args.model_size, args.dtype, evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs)
      evs.AddMetric(args.model_name, {refname}, 'sys', sys_scores, replace=True)
      evs.AddMetric(args.model_name, {refname}, 'seg', seg_scores, replace=True)
    m_scores, s_scores = [], []
    keys = sorted(seg_scores.keys())
    m_value, s_value = [seg_scores[key] for key in keys], [gold_scores[key] for key in keys]
    for m, s in zip(m_value, s_value):
      if len(m) == 0 and len(s) == 0:
        continue
      m_scores.extend(m)
      s_scores.extend(s)
    model_scores[lp] = np.array(m_scores, dtype=np.float32)
    sy_scores[lp] = np.array(s_scores, dtype=np.float32)
    

  # Add new metric to the primary lists, so it will get picked up when tasks get
  # run with primary=True (avoiding having to evaluate all contrastive
  # submissions as well).
  for lp in lps:
    write_result(args.model_name, args.model_size, lp, seg_scores, sys_scores, args.output_dir)
  
  for evs in evs_dict.values():
    evs.SetPrimaryMetrics(evs.primary_metrics | {args.model_name})


  # Set k=1000 for a more realistic comparison
  tasks, wts = get_tasks(args.wmt_year, lps, k=0)

  # Takes about 3 minutes.
  new_results = tasks.Run(eval_set_dict=evs_dict)
  
  avg_corrs = new_results.AverageCorrs(wts)

  table = new_results.Table(
      metrics=list(avg_corrs),
      initial_column=avg_corrs,
      initial_column_header='avg-corr',
      attr_list=['lang', 'level', 'corr_fcn'],
      nicknames={'KendallWithTiesOpt': 'acc-t'},
      fmt='text',
      baselines_metainfo=baseline_metainfo)

  print(table)
  import code; code.interact(local=locals())
  evaluate(lps, model_scores, sy_scores)
  
  
  
  import code; code.interact(local=locals())

if __name__ == "__main__":
  main()