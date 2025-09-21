import re
import json
import transformers
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from comet import load_from_checkpoint, download_model

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
    xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_path = "/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
    # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-XPlus-8B"
    # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=2048)
    # src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=2048, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/tur.devtest"
    dataset = load_flores(path)
    prompts = []
    src_lang, tgt_lang = "English", "Turkish"
    for data in dataset:
        # src = "It also prevents psychological distress. \"Although social media has not yet been proven to cause depression, it is shown to intensify certain symptoms, such as social isolation and loneliness,\" Yassim added."
        # # 当时有一个国际贩毒集团总部设在牙买加，与南美有业务往来，据我所知，这个集团在交易中会将高价艺术品作为 抵押品
        # # tgt = "托尼·莫尔博士在南非夸祖鲁-纳塔尔省发现了这种广泛耐药结核病 (XDR-TB)。"
        # user_prompt = user_prompt.format(src=src, tgt=tgt)
        # sentence = f"""Translate the following sentences from {src_lang} to {tgt_lang}.\n### Input:\n{data}\n"""
        # sentence = Prompt_template(data.strip(), src_lang, tgt_lang)
        # sentence = f""""""
        sentence = f"""{data.strip()}\nTranslate the above sentence from {src_lang} to {tgt_lang}."""
        message = [
            {"role": "user", "content": sentence}
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt)
        break
    outputs = model.generate(prompts, sample)
    print(outputs[0].outputs[0].text)
    #     # output = outputs[0].outputs[0].text
    
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-mix-mid2-1m.jsonl"
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/3base_en-mix-mid2-1m.jsonl"
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-hin2-1m.jsonl"
    # dataset = load_dataset(path)
    # dataset = dataset[:1000]
    
    # model = load_from_checkpoint(xcomet_path)
    # inputs = [
    #     {
    #         "src": data['src'],
    #         "mt": data['ref'],
    #     }
    #     for data in dataset
    # ]
    # output = model.predict(inputs, batch_size=16, gpus=1)
    
    # scores, mean_score = output.scores, output.system_score
    
    # score = calculate_comet_score(srcs, refs, hyps, model_path=xcomet_path)
    # dataset = dataset[220000:]
    # with open(save_path, 'w') as f:
        # for data in dataset:
            # f.write(json.dumps(data, ensure_ascii=False) + "\n")
    import code; code.interact(local=locals())