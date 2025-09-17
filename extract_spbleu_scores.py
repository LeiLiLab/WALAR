#!/usr/bin/env python3
"""
Script to extract spBLEU scores from translation files and format them as a LaTeX table.
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

def format_latex_table(organized_scores, model_name):
    """Format the organized scores as a LaTeX table."""
    
    # Model name mapping for display
    model_display_names = {
        'Qwen3-4B': 'Qwen3-4B',
        'Qwen3-30B': 'Qwen3-30B', 
        'Qwen3-32B': 'Qwen3-32B',
        'Llama-3.2-3B-Instruct': 'Llama3.2-3B-Instruct',
        'ChatGPT': 'ChatGPT',
        'GPT4': 'GPT4',
        'M2M-12B': 'M2M-12B',
        'NLLB-1.3B': 'NLLB-1.3B',
        'Google': 'Google'
    }
    
    display_name = model_display_names.get(model_name, model_name)
    
    latex_lines = []
    latex_lines.append(f" &                       &{display_name} & Llama3.2-3B-Instruct &Qwen3-30B & ChatGPT & GPT4 & M2M-12B & NLLB-1.3B & Google\\\\")
    latex_lines.append("    \\midrule")
    
    for family_name, family_data in organized_scores.items():
        scores = family_data['scores']
        count = family_data['count']
        
        if count == 1:
            # Single language
            lang = list(scores.keys())[0]
            if lang != 'average':
                score = scores[lang]
                latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}} ")
                latex_lines.append(f"     & {lang}&{score:.2f} & & & & & & & \\\\")
        else:
            # Multiple languages
            latex_lines.append(f"    \\multirow{{{count}}}{{*}}{{{family_name} ({count})}}")
            
            for i, (lang, score) in enumerate(scores.items()):
                if lang != 'average':
                    if i == 0:
                        latex_lines.append(f"&{lang} &{score:.2f} & & & & & & & \\\\")
                    else:
                        latex_lines.append(f"     & {lang}&{score:.2f} & & & & & & & \\\\")
        
        # Add average line
        if 'average' in scores:
            avg_score = scores['average']
            latex_lines.append("    \\midrule")
            latex_lines.append(f"    \\multicolumn{{2}}{{c|}}{{Average}} &{avg_score:.2f} & & & & & & & \\\\")
        
        latex_lines.append("    \\midrule")
    
    return "\\n".join(latex_lines)

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python extract_spbleu_scores.py <model_name>")
        print("Example: python extract_spbleu_scores.py Qwen3-4B")
        sys.exit(1)
    
    model_name = sys.argv[1]
    model_dir = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/{model_name}"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        sys.exit(1)
    
    print(f"Extracting spBLEU scores from {model_dir}...")
    scores = extract_spbleu_scores(model_dir)
    print(f"Found {len(scores)} language pairs")
    
    print("Organizing by language family...")
    organized_scores = organize_by_language_family(scores)
    
    print("Formatting as LaTeX table...")
    latex_table = format_latex_table(organized_scores, model_name)
    
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    output_file = f"/mnt/gemini/data1/yifengliu/qe-lr/{model_name}_spbleu_table.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"\nTable saved to: {output_file}")

if __name__ == "__main__":
    main()



