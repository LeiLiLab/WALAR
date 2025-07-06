import json
import random
import pandas as pd

def load_dastset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    # Example usage
    path ="/mnt/gemini/data1/yifengliu/data/IndicMT/collated/punjabi.jsonl"
    save_path = "/mnt/gemini/data1/yifengliu/data/IndicMT/collated/punjabi2.jsonl"
    dataset = load_dastset(path)
    for data in dataset:
        data['src_lang'] = "English"
        data['tgt_lang'] = "Punjabi"
    with open(save_path, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + "\n")
    # import fasttext
    # lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    
    # lang_info = lang_detect_model.predict(tgts)
    # detect_rewards = [True if language[0].replace("__label__", "") == "zh" else False for language in lang_info[0]]
    # print("Percentage: {:.2f}%".format(detect_rewards.count(True) / len(detect_rewards) * 100))
    
    
    
    # Load the language ID model
    # file_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step140_hf/eng-fra.txt"
    # dataset = []
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[:-2]:
    #         dataset.append(json.loads(line))
    
    # tgts = [data['pred'] for data in dataset]
    
    
    
    # prediction = model.predict(sentence)
    

    import code; code.interact(local=locals())