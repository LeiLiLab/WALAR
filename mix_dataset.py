import os
import random
import json
import sys
import tqdm
from tqdm import *
from datasets import Dataset, concatenate_datasets
sys.path.append('/mnt/gemini/data1/yifengliu/qe-lr/code')
from utils import three2two, training_langs2, mm_dict, lang_dict
support_list = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def get_lang(lang_code):
    src_lang = mm_dict.get(lang_code, '')
    if len(src_lang) == 0:
        src_lang = lang_dict.get(lang_code, '')
    if src_lang == '':
        src_lang = lang_code.capitalize()
    return src_lang

def make_prompt(source, src, tgt, template_type='chat', tokenizer=None):
    if template_type == 'base':
        return f"{source}\nTranslate from {src} to {tgt}:"
    elif template_type == 'chat':
        return f"You are a helpful assistant. Translate this text from {src} to {tgt}:\n{source}"
    elif template_type == 'rl':
        return f"Translate this text from {src} to {tgt}:\n{source}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")

if __name__ == "__main__":
    # lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "ory", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav"]
    # lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", ]
    # lang_list = ["ltz", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "tur", "tam", "fin"]
    lang_list = training_langs2
    num_per_lang = 1000
    src_lang_list = ["en", "ar", "tr", "hi"]
    # src_lang = "tr"
    for src_lang in src_lang_list:
        meta_file_path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{src_lang}/ner-{src_lang}1m.jsonl"
        meta_dataset = load_dataset(meta_file_path)
        random.shuffle(meta_dataset)
        output_file = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/final_{src_lang}-mix-1m.jsonl"
        src_lang = get_lang(src_lang)
        # lang_list = ["isl", "ltz", "bel", "mkd", "srp", "slv", "ben", "guj", "mar", "npi", "pan", "urd", "hye", "ell", "lav", "lit", "fas", "jav", "kan", "mal", "tam", "tel", "fin", "hun", "heb"]
        final_lang_list = []
        for lang in lang_list:
            two_lang = three2two[lang]
            if two_lang in support_list:
                final_lang_list.append(lang)
        # import code; code.interact(local=locals())
        def make_map_fn(split, src_lang, tgt_lang):
            def process_fn(example, idx):
                data_source = example.get('data_source', 'unknown')
                # Dynamic source and target language field extraction
                source = example['src']
                # lg = src_lang + "-" + tgt_lang
                # Generate prefix
                # src_lang, tgt_lang = get_langs(args)
                # src_lang, tgt_lang = get_lang(src_lang), get_lang(tgt_lang)
                prompt = make_prompt(source, src_lang, tgt_lang, template_type="base")
                
                data = {
                    "data_source": data_source + "_" + f"{src_lang}-{tgt_lang}",
                    "lang_pair": f"{src_lang}-{tgt_lang}",
                    "src_text": source,
                    "input_key": prompt,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "translate",
                    "extra_info": {
                        'index': idx,
                    }
                }
                return data
            return process_fn
        
        # for lang in final_lang_list:
        all_datasets = []
        for i in tqdm(range(len(final_lang_list))):
            partial_dataset = meta_dataset[num_per_lang * i: num_per_lang * (i + 1)]
            partial_dataset = Dataset.from_list(partial_dataset)
            lang = final_lang_list[i]
            tgt_lang = get_lang(lang)
            train_dataset = partial_dataset.map(
                function=make_map_fn('train', src_lang, tgt_lang), 
                with_indices=True
            )
            all_datasets.append(train_dataset)
        final_dataset = concatenate_datasets(all_datasets)
        final_dataset = final_dataset.shuffle(seed=42)
        dir_name = os.path.dirname(output_file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)    
        final_dataset.to_json(output_file, lines=True)

        # Print dataset format

        print("Train dataset columns:")
        train_pdf = final_dataset.to_pandas()
        print(train_pdf.head())
        print(train_pdf['prompt'][0])
        
        print(f"Train dataset saved to: {output_file}")
    # num_dict = {k: 20000 for k in final_lang_list}
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_ar-mix-mid2-1m.jsonl"
    # new_dataset = []
    # index = 0
    # # for lang in final_lang_list:
    # for i in tqdm(range(len(final_lang_list))):
    #     lang = final_lang_list[i]
    #     file_path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_ar-{lang}-1m.jsonl"
    #     dataset = load_dataset(file_path)
    #     new_dataset.extend(dataset[index:index + num_dict[lang]])
    #     index += num_dict[lang]
    
    # random.shuffle(new_dataset)
    # with open(save_path, 'w') as f:
    #     for data in new_dataset:
    #         f.write(json.dumps(data) + "\n")