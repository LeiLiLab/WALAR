import json
import os
import argparse
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

def read_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def combine_datasets(src_file, tgt_file):
    """Combine English and Chinese datasets into a JSONL file."""
    # Read both files
    src_lines = read_file(src_file)
    tgt_lines = read_file(tgt_file)
    
    # Check if both files have the same number of lines
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Number of lines don't match: English ({len(src_lines)}) vs Chinese ({len(tgt_lines)})")
    
    # Create output directory if it doesn't exist
    dataset = []
    for src, tgt in zip(src_lines, tgt_lines):
        dataset.append({'src': src, 'tgt': tgt})
    return dataset

def make_prompt(source, src, tgt, template_type='chat', tokenizer=None):
    language_map = {
        "en": "English",
        "zh_simpl": "Chinese",
        "zh_trad": "Chinese (Traditional)",
    }
    if template_type == 'base':
        return f"{source}\nTranslate from {language_map.get('en', 'English')} to {language_map.get('zh', 'Chinese')}:"
    elif template_type == 'chat':
        return f"You are a helpful assistant. Translate this text from {language_map.get('en', 'English')} to {language_map.get('zh', 'Chinese')}:\n{source}"
    elif template_type == 'rl':
        return f"Translate this text from {language_map.get('en', 'English')} to {language_map.get('zh', 'Chinese')}:\n{source}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")            

lang_dict = {
    "en": "eng",
    "zh_simpl": "zho_simpl",
    "zh_trad": "zho_trad",
}

def main():
    # File paths
    parser = argparse.ArgumentParser(description='Prepare translation dataset')
    parser.add_argument('--src', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt', type=str, default='zh', help='Target language code')
    parser.add_argument('--data_source', type=str, default='unknown', help='Data source name')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer')
    parser.add_argument('--template_type', type=str, choices=['base', 'chat', 'rl'], default='chat', help='Template type for prompts')
    parser.add_argument('--output_file', type=str, help='Number of training samples to use')
    args = parser.parse_args()

    base_dir = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev"
    src_file = os.path.join(base_dir, f"{lang_dict[args.src]}.dev")
    tgt_file = os.path.join(base_dir, f"{lang_dict[args.tgt]}.dev")
    
    # Combine datasets
    dataset = combine_datasets(src_file, tgt_file)

    def make_map_fn(split):
        def process_fn(example, idx):
            data_source = args.data_source
            # Dynamic source and target language field extraction
            source = example.pop('src', None)
            target = example.pop('tgt', None)
            lg = args.src + "-" + args.tgt
            # Generate prefix
            prompt = make_prompt(source, args.src, args.tgt, template_type=args.template_type)
            
            data = {
                "data_source": data_source + "_" + lg,
                "lang_pair": lg,
                "src_text": source,
                "tgt_text": target,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": target,
                },
                "ability": "translate",
                "extra_info": {
                    'index': idx,
                }
            }
            return data
        return process_fn
    dataset = Dataset.from_list(dataset)
    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    train_dataset.to_parquet(args.output_file)


if __name__ == "__main__":
    main()