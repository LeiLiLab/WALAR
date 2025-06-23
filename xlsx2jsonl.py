import pandas as pd
import os
import json

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line))
    return dataset

def assign_segment(dataset):
    src = [data['src'] for data in dataset]
    src_set = set(src)
    src_set = {s: i for i, s in enumerate(src_set)}
    for data in dataset:
        source = data['src']
        data['segment_id'] = src_set[source]
    return dataset

if __name__ == '__main__':
    language_list = ['assamese', 'maithili', 'kannada', 'punjabi']
    excel_file = pd.ExcelFile('/mnt/data1/yifengliu/data/IndicMT/zero_shot/zero_shot_dataset.xlsx', engine="openpyxl")
    # language_list = ['assamese']
    for i, language in enumerate(language_list):
        path = f"/mnt/data1/yifengliu/data/IndicMT/zero_shot/{language}.jsonl"
        save_path = f"/mnt/data1/yifengliu/data/IndicMT/collated/{language}.jsonl"
        
        # df = pd.read_excel('/mnt/data1/yifengliu/data/IndicMT/zero_shot/zero_shot_dataset.xlsx', engine="openpyxl")
        print(f"Processing {excel_file.sheet_names[i]}...")
        df = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[i], engine="openpyxl")
        dataset1 = load_dataset(path)
        src, ref, mt, score_mqm, model = df['src'], df['ref'], df['mt'], df['score_mqm'], df['model']
        human_score = df['score']
        dataset2 = [{'src': src[i], 'ref': ref[i], 'translation': mt[i], 'score_mqm': score_mqm[i], 'model': model[i], 'human_score': human_score[i]} for i in range(len(src))]
        for data1, data2 in zip(dataset1, dataset2):
            if data1['src'] == data2['src'] and data1['ref'] == data2['ref'] and data1['translation'] == data2['translation']\
                and float(data1['mqm_norm_score']) == data2['score_mqm']:
                data1['model'] = data2['model']
                data1['full_score'] = data2['score_mqm'] * 25
                data1['human_score'] = data2['human_score'] * 25
        dataset1 = assign_segment(dataset1)
        
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(save_path, 'w') as f:
            for data in dataset1:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')