import spacy
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_dataset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except:
                break
    return dataset

# 加载英文模型（可以换成其他语言模型，如 zh_core_web_sm）
# nlp = spacy.load("xx_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# 获取NER的所有labels
labels = nlp.get_pipe("ner").labels
print(labels)
# doc = nlp("There were specific warnings from FIFPro for young players and the increased number of games they are being asked to take part in as their bodies continue to develop.")
# doc = nlp("إمارة أبوظبي هي إحدى إمارات دولة الإمارات العربية المتحدة السبع")
# import code; code.interact(local=locals())

lang = "hin"
# path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B/eng-{lang}.txt"
path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/3base_en-mix-mid2-1m.jsonl"

entity_list = []
dataset = load_dataset(path)[:10]
# for data in dataset:
for i in tqdm(range(len(dataset))):
    data = dataset[i]
    doc = nlp(data['src'])
    entity_list.append(len(doc.ents))
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
import code; code.interact(local=locals())
# lst = list(range(0, 50, 1))
# plt.hist(entity_list, bins=[temp for temp in lst], edgecolor='black')
# plt.xlabel('Ratio Range')
# plt.ylabel('Count')
# plt.title('Token Length Ratio Distribution (Detected/Original)')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/ratio_gap.png")
# import code; code.interact(local=locals())