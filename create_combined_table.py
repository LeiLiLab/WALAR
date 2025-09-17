#!/usr/bin/env python3
"""
Script to create a combined LaTeX table with spBLEU scores from multiple models.
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
    
    for family_name, family_info in language_families.items():
        family_scores = {}
        for lang in family_info['languages']:
            # Look for eng-{lang} scores
            eng_lang_key = f'eng-{lang}'
            if eng_lang_key in scores:
                family_scores[lang] = scores[eng_lang_key]
            else:
                print(f"Warning: No score found for {eng_lang_key}")
                family_scores[lang] = 0.0
        
        # Calculate average for the family
        if family_scores:
            avg_score = sum(family_scores.values()) / len(family_scores)
            family_scores['average'] = avg_score
        
        organized_scores[family_name] = {
            'scores': family_scores,
            'count': family_info['count']
        }
    
    return organized_scores

def create_combined_table(model_scores_dict):
    """Create a combined LaTeX table with scores from multiple models."""
    
    # Model order for the table - only include models we have data for
    available_models = list(model_scores_dict.keys())
    
    # Get all language families from the first model (they should be the same)
    first_model = list(model_scores_dict.keys())[0]
    language_families = list(model_scores_dict[first_model].keys())
    
    latex_lines = []
    
    # Header
    header = " &                       &" + " & ".join(available_models) + "\\\\"
    latex_lines.append(header)
    latex_lines.append("    \\midrule")
    
    for family_name in language_families:
        family_data = model_scores_dict[first_model][family_name]
        count = family_data['count']
        languages = [lang for lang in family_data['scores'].keys() if lang != 'average']
        
        if count == 1:
            # Single language
            lang = languages[0]
            latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}} ")
            row = f"     & {lang}"
            for model in available_models:
                if model in model_scores_dict and family_name in model_scores_dict[model]:
                    score = model_scores_dict[model][family_name]['scores'].get(lang, 0.0)
                    row += f"&{score:.2f}"
                else:
                    row += "&"
            row += "\\\\"
            latex_lines.append(row)
        else:
            # Multiple languages
            latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}}")
            
            for i, lang in enumerate(languages):
                if i == 0:
                    row = f"&{lang}"
                else:
                    row = f"     & {lang}"
                
                for model in available_models:
                    if model in model_scores_dict and family_name in model_scores_dict[model]:
                        score = model_scores_dict[model][family_name]['scores'].get(lang, 0.0)
                        row += f"&{score:.2f}"
                    else:
                        row += "&"
                row += "\\\\"
                latex_lines.append(row)
        
        # Add average line
        latex_lines.append("    \\midrule")
        row = "    \\multicolumn{2}{c|}{Average}"
        for model in available_models:
            if model in model_scores_dict and family_name in model_scores_dict[model]:
                avg_score = model_scores_dict[model][family_name]['scores'].get('average', 0.0)
                row += f"&{avg_score:.2f}"
            else:
                row += "&"
        row += "\\\\"
        latex_lines.append(row)
        latex_lines.append("    \\midrule")
    
    return "\\n".join(latex_lines)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_combined_table.py <model1> <model2> ...")
        print("Example: python create_combined_table.py Qwen3-4B Llama-3.2-3B-Instruct Qwen3-30B-A3B-Instruct-2507")
        sys.exit(1)
    
    model_names = sys.argv[1:]
    base_dir = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores"
    
    model_scores_dict = {}
    
    for model_name in model_names:
        model_dir = os.path.join(base_dir, model_name)
        
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist, skipping...")
            continue
        
        print(f"Extracting spBLEU scores from {model_dir}...")
        scores = extract_spbleu_scores(model_dir)
        print(f"Found {len(scores)} language pairs")
        
        print("Organizing by language family...")
        organized_scores = organize_by_language_family(scores)
        model_scores_dict[model_name] = organized_scores
    
    print("Creating combined LaTeX table...")
    latex_table = create_combined_table(model_scores_dict)
    
    print("\n" + "="*80)
    print("Combined LaTeX Table:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    output_file = "/mnt/gemini/data1/yifengliu/qe-lr/combined_spbleu_table.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"\nTable saved to: {output_file}")

if __name__ == "__main__":
    main()
