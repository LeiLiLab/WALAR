import re
import json
import transformers
import os
import sacrebleu
import tqdm 
import torch
from tqdm import *
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from comet import load_from_checkpoint, download_model
from transformers import AutoTokenizer, LlamaForCausalLM

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

def Prompt_template(query, src_language, trg_language):
    instruction = f'Translate the following sentences from {src_language} to {trg_language}.'
    prompt = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores101", force=True).score
    return result


def load_dataset(path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except:
                break
    return dataset

def load_flores(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def extract_boxed_number(answer):
    # Extract the number from a string in the form of \boxed{number}
    r = re.search(r"\\boxed\{(.*?)\}", answer)
    if r is not None:
        return str(r.group(1))
    return None

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs, batch_size=16, gpus=1)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

if __name__ == '__main__':
    # xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model_path = "/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
    # # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-XPlus-8B"
    # # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    # # src, tgt = "en", "zh"
    # model = LLM(model=model_path, max_model_len=2048, tensor_parallel_size=1, trust_remote_code=True, task="generate")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # sample = SamplingParams(n=1, temperature=0.0, top_k=-1, top_p=1, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id])
    # # params = BeamSearchParams(beam_width=5, max_tokens=256)
    # path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/ara.devtest"
    # dataset = load_flores(path)
    # prompts = []
    # src_lang, tgt_lang = "Arabic", "Azerbaijani"
    # tgt_dataset = load_flores("/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/azj.devtest")
    # for data in dataset:
    #     # sentence = f"""{data.strip()}\nTranslate the above sentence from {src_lang} to {tgt_lang}."""
    #     # sentence = f"""Translate the following sentences from {src_lang} to {tgt_lang}.\n### Input:\n{data}\n"""
    #     prompt = Prompt_template(data.strip(), src_lang, tgt_lang)
    #     # message = [
    #     #     {"role": "user", "content": sentence}
    #     # ]
    #     # prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    #     # prompts.append({"prompt": prompt})
    #     prompts.append(prompt)
    #     # break
    # outputs = model.generate(prompts, sample)
    # outputs = model.beam_search(prompts, params)
    
    
    src_lang, tgt_lang = "eng", "isl"
    data_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{src_lang}.devtest"
    tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
    dataset = load_flores(data_path)
    tgt_dataset = load_flores(tgt_path)
    # path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    path = "/mnt/gemini/data1/yifengliu/checkpoints/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
    # path = "/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
    # model = LlamaForCausalLM.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    device = "cuda:0"
    model.to(device)
    model.eval()
    # query = "أضاف قائلاً، \"لدينا الآن فئران تبلغ من العمر 4 أشهر، وكانت تعاني في السابق من مرض السكري، ولكنها لم تعد تعاني منه الآن."
    # query = ""
    # prompts = [Prompt_template(query, 'English', 'Icelandic') for query in dataset]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"{query}\nTranslate from English to Icelandic:"}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for query in dataset]
    
    # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    # generate_ids = model.generate(inputs.input_ids, max_length=200, do_sample=False, num_beams=1)
    batch_size = 8
    outputs_all = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            # tokenizer -> padding 到同长度 (batch)
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=True
            )
            # 把 tensors 移到 device
            inputs = inputs.to(device)

            # 生成（显式 greedy）
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask if "attention_mask" in inputs else None,
                max_length=256,
                do_sample=False,   # 禁用采样 -> greedy
                num_beams=4,        # beam=1 等价于 greedy
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # 解码
            input_length = inputs['input_ids'].shape[1]
            generate_ids = output_ids[:, input_length:]
            outputs = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            outputs_all.extend(outputs)
            # break
    # outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # pattern = r'### Response:+(.*)'
    # import code; code.interact(local=locals())
    # pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
    # outputs = [re.search(pattern, output, re.DOTALL).group(2).strip() for output in outputs_all]
    # import code; code.interact(local=locals())
    spbleu = get_spBLEU(outputs_all, tgt_dataset)
    # pattern = r'### Response:+(.*)'
    import code; code.interact(local=locals())