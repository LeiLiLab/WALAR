import json
import os
import argparse

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line))
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collate afriMTE dataset")
    parser.add_argument("--input_dir", type=str, default="/mnt/data1/yifengliu/qe-lr/output/afriMTE/XComet",
                        help="Directory path containing the dataset files")
    args = parser.parse_args()
    dir_path = args.input_dir
    save_path = os.path.join(dir_path, "afriMTE.jsonl")
    new_dataset = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if 'afriMTE' in file:
                continue
            path = os.path.join(root, file)
            dataset = load_dataset(path)
            new_dataset.extend(dataset)
    
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data) + '\n')