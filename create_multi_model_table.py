#!/usr/bin/env python3
"""
Script to create a LaTeX table with spBLEU scores from multiple models in one run.
"""

import os
import re
import glob
from collections import defaultdict

def extract_spbleu_scores(model_dir):
    """Extract spBLEU scores from all translation files in a model directory."""
    scores = {}
    
    # Find all .txt files in the model directory
    pattern = os.path.join(model_dir, "*.txt")
    files = glob.glob(pattern)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        lang_pair = filename.replace('.txt', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Look for spBLEU score in the last few lines
            spbleu_score = None
            for line in reversed(lines[-10:]):  # Check last 10 lines
                if 'spBLEU:' in line:
                    match = re.search(r'spBLEU:\s*([\d.]+)', line)
                    if match:
                        spbleu_score = float(match.group(1))
                        break
            
            if spbleu_score is not None:
                scores[lang_pair] = spbleu_score
            else:
                print(f"Warning: No spBLEU score found in {filename}")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return scores

def organize_by_language_family(scores):
    """Organize scores by language family according to the table structure."""
    
    # Language family mappings based on the provided table
    language_families = {
        'Indo-European-Germanic': {
            'languages': ['isl', 'ltz'],
            'count': 2
        },
        'Indo-European-Slavic': {
            'languages': ['bel', 'ces', 'mkd', 'pol', 'srp', 'slk', 'slv', 'ukr'],
            'count': 8
        },
        'Indo-European-Indo-Aryan': {
            'languages': ['ben', 'guj', 'hin', 'mar', 'npi', 'pan', 'urd'],
            'count': 7
        },
        'Indo-European-Other': {
            'languages': ['hye', 'ell', 'cym', 'lav', 'lit', 'fas'],
            'count': 6
        },
        'Austronesian': {
            'languages': ['ceb', 'tgl', 'jav'],
            'count': 3
        },
        'Afro-Asiatic': {
            'languages': ['ara'],
            'count': 1
        },
        'Turkic': {
            'languages': ['azj', 'kaz', 'tur', 'uzb'],
            'count': 4
        },
        'Dravidian': {
            'languages': ['kan', 'mal', 'tam', 'tel'],
            'count': 4
        },
        'Sino-Tibetan': {
            'languages': ['mya'],
            'count': 1
        },
        'Other': {
            'languages': ['est', 'fin', 'hun', 'kat', 'heb', 'khm', 'kor', 'lao', 'tha'],
            'count': 9
        }
    }
    
    organized_scores = {}
    src_lang = "ara"
    for family_name, family_info in language_families.items():
        family_scores = {}
        for lang in family_info['languages']:
            # Look for eng-{lang} scores
            if src_lang != lang:
                eng_lang_key = f'{src_lang}-{lang}'
                if eng_lang_key in scores:
                    family_scores[lang] = scores[eng_lang_key]
                else:
                    print(f"Warning: No score found for {eng_lang_key}")
                    family_scores[lang] = 0.0
            else:
                family_info['count'] -= 1
        
        # Calculate average for the family
        if family_scores:
            avg_score = sum(family_scores.values()) / len(family_scores)
            family_scores['average'] = avg_score
        
        organized_scores[family_name] = {
            'scores': family_scores,
            'count': family_info['count']
        }
    
    return organized_scores

def create_multi_model_table(model_scores_dict):
    """Create a LaTeX table with scores from multiple models."""
    
    # Model display names mapping
    model_display_names = {
        'Qwen3-4B': 'Qwen3-4B',
        'Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
        'Llama-3.2-3B-Instruct': 'Llama3.2-3B-Instruct',
        'Qwen3-32B-AWQ': 'Qwen3-32B',
        'ChatGPT': 'ChatGPT',
        'GPT4': 'GPT4',
        'M2M-12B': 'M2M-12B',
        'NLLB-1.3B': 'NLLB-1.3B',
        'Google': 'Google'
    }
    
    # Get available models and their display names
    available_models = list(model_scores_dict.keys())
    display_names = [model_display_names.get(model, model) for model in available_models]
    
    # Get all language families from the first model (they should be the same)
    first_model = list(model_scores_dict.keys())[0]
    language_families = list(model_scores_dict[first_model].keys())
    
    latex_lines = []
    
    # Header
    header = " &                       &" + " & ".join(display_names) + "\\"
    latex_lines.append(header)
    latex_lines.append("    \\midrule")
    
    for family_name in language_families:
        family_data = model_scores_dict[first_model][family_name]
        count = family_data['count']
        languages = [lang for lang in family_data['scores'].keys() if lang != 'average']
        
        if count == 1:
            # Single language
            lang = languages[0]
            # latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}} ")
            row = f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}}\n    & {lang}"
            for model in available_models:
                if model in model_scores_dict and family_name in model_scores_dict[model]:
                    score = model_scores_dict[model][family_name]['scores'].get(lang, 0.0)
                    row += f"&{score:.2f}"
                else:
                    row += "&"
            row += "\\"
            latex_lines.append(row)
        else:
            # Multiple languages
            # latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}}")
            
            for i, lang in enumerate(languages):
                if i == 0:
                    row = f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}}\n    & {lang}"
                else:
                    row = f"     & {lang}"
                
                for model in available_models:
                    if model in model_scores_dict and family_name in model_scores_dict[model]:
                        score = model_scores_dict[model][family_name]['scores'].get(lang, 0.0)
                        row += f"&{score:.2f}"
                    else:
                        row += "&"
                row += "\\"
                latex_lines.append(row)
        
        # Add average line
        # latex_lines.append("    \\midrule")
        if count != 0:
            row = "    \\midrule    \\multicolumn{2}{c|}{Average}"
            for model in available_models:
                if model in model_scores_dict and family_name in model_scores_dict[model]:
                    avg_score = model_scores_dict[model][family_name]['scores'].get('average', 0.0)
                    row += f"&{avg_score:.2f}"
                else:
                    row += "&"
            row += "\\"
            latex_lines.append(row)
            latex_lines.append("    \\midrule")
    
    return "\\\n".join(latex_lines)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_multi_model_table.py <model1> <model2> ...")
        print("Example: python create_multi_model_table.py Qwen3-4B Llama-3.2-3B-Instruct Qwen3-30B-A3B-Instruct-2507")
        print("\nAvailable models in the flores directory:")
        base_dir = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores"
        if os.path.exists(base_dir):
            models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            for model in sorted(models):
                print(f"  - {model}")
        sys.exit(1)
    
    
    model_names = sys.argv[1:]
    base_dir = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores"
    
    model_scores_dict = {}
    
    print(f"Processing {len(model_names)} models...")
    
    for model_name in model_names:
        model_dir = os.path.join(base_dir, model_name)
        
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist, skipping...")
            continue
        
        print(f"\nExtracting spBLEU scores from {model_name}...")
        scores = extract_spbleu_scores(model_dir)
        print(f"Found {len(scores)} language pairs")
        
        print("Organizing by language family...")
        organized_scores = organize_by_language_family(scores)
        model_scores_dict[model_name] = organized_scores
    
    if not model_scores_dict:
        print("Error: No valid model directories found!")
        sys.exit(1)
    # import code; code.interact(local=locals())
    print(f"\nCreating multi-model LaTeX table with {len(model_scores_dict)} models...")
    latex_table = create_multi_model_table(model_scores_dict)
    
    print("\n" + "="*80)
    print("Multi-Model LaTeX Table:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    model_names_str = "_".join(model_names[:3])  # Use first 3 names for filename
    if len(model_names) > 3:
        model_names_str += f"_and_{len(model_names)-3}_more"
    output_file = f"/mnt/gemini/data1/yifengliu/qe-lr/multi_model_{model_names_str}_table.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"\nTable saved to: {output_file}")
    
    # Also create a summary
    print(f"\nSummary:")
    print(f"- Processed {len(model_scores_dict)} models: {', '.join(model_scores_dict.keys())}")
    print(f"- Language families: {len(list(model_scores_dict[list(model_scores_dict.keys())[0]].keys()))}")
    print(f"- Total language pairs processed: {sum(len(extract_spbleu_scores(os.path.join(base_dir, model))) for model in model_scores_dict.keys())}")

if __name__ == "__main__":
    main()
