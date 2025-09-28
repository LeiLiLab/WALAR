#!/bin/bash

# Language Detection Script
# This script provides language detection functionality for multiple language files
# Similar to flores.sh but focused on language detection

# Configuration
MODEL_PATH="/mnt/gemini/data1/yifengliu/model/lid.176.bin"

# Language pairs configuration (similar to flores.sh)
source_language_list=(
    "eng"
    "tur"
    "ara"
    "hin"
    # "ben"
    # "guj"
    # "mar"
    # "npi"
    # "pan"
    # "urd"
    # "hye"
    # "ell"
    # "lav"
    # "lit"
    # "fas"
    # "cym"
    # "ceb"
    # "tgl"
    # "jav"
    # "ara"
    # "azj"
    # "tur"
    # "uzb"
    # "kan"
    # "mal"
    # "tam"
    # "tel"
    # "est"
    # "fin"
    # "hun"
    # "kat"
    # "heb"
    # "khm"
    # "kor"
    # "tha"
)

target_language_list=(
    # "ara" "ces" "dan" "deu" "spa" "fin" "fra" "hrv" "hun" "ind" "ita" "jpn" "kor" "msa" "nld" "nob" "pol" "por" "ron" "rus" "swe" "tha" "tur" "ukr" "vie" "zho_simpl"
    "isl" 
    "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
    "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
    "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
    "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"

    # "ben"
    # "guj"
    # "hin"
    # "mar"
    # "npi"
    # "pan"
    # "urd"
    # "hye"
    # "ell"
    # "lav"
    # "lit"
    # "fas"
    # "cym"
    # "ceb"
    # "tgl"
    # "jav"
    # "ara"
    # "azj"
    # "tur"
    # "uzb"
    # "kan"
    # "mal"
    # "tam"
    # "tel"
    # "est"
    # "fin"
    # "hun"
    # "kat"
    # "heb"
    # "khm"
    # "kor"
    # "tha"
)

# Input directory for flores files
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct"
INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"

# Convert language lists to comma-separated strings
SOURCE_LANGUAGES=$(IFS=','; echo "${source_language_list[*]}")
TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")

if [ -n "$SOURCE_LANGUAGES" ] && [ -n "$TARGET_LANGUAGES" ]; then
    # Multiple language pairs mode
    echo "Running language detection for multiple language pairs"
    echo "Source languages: $SOURCE_LANGUAGES"
    echo "Target languages: $TARGET_LANGUAGES"
    
    # Change to project directory
    cd /mnt/gemini/data1/yifengliu/qe-lr
    
    # Run the language detection for all language pairs
    python code/lang_detect.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES"
        
else
    echo "Error: SOURCE_LANGUAGES and TARGET_LANGUAGES must be set"
    echo "Usage examples:"
    echo "  Single language pair: LANG_PAIR=\"eng-hin\" ./lang_detect.sh"
    echo "  Multiple pairs: SOURCE_LANGUAGES=\"eng,deu\" TARGET_LANGUAGES=\"hin,ben,tam\" ./lang_detect.sh"
    exit 1
fi

