import json
import os
from utils import training_langs, training_langs2, llamax_langs, qwen_support, high_langs

def print_result(file_path):
    bleu_score, xcomet_score = None, None
    metricx_score = None
    lang_acc_rate = None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'spBLEU' in line: 
                bleu_score = line.strip().split()[-1]
            if 'XComet' in line:
                xcomet_score = line.strip().split()[-1]
            if "MetricX" in line:
                metricx_score = line.strip().split()[-1]
            if "Lang_Error_Rate" in line:
                lang_acc_rate = 100 - float(line.strip().split()[-1].split("%")[0])
    lang_pair = file_path.split('/')[-1].replace('.txt', '')
    if xcomet_score and metricx_score and lang_acc_rate:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}\t{metricx_score}\t{lang_acc_rate}")
    elif xcomet_score and metricx_score:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}\t{metricx_score}")
    elif xcomet_score:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}")
    else:
        print(f"{lang_pair}:\t{bleu_score}\t")
    return {lang_pair: (bleu_score, xcomet_score, metricx_score, lang_acc_rate)}
    
def check_result(file_path):
    bleu_score, xcomet_score, metricx_score = None, None, None
    lang_error_rate = None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'spBLEU' in line: 
                if bleu_score is not None:
                    print(f"spBLEU: {file_path}")
                    return
                bleu_score = line.strip().split()[-1]
            if 'XComet' in line:
                if xcomet_score is not None:
                    print(f"XComet: {file_path}")
                    return
                xcomet_score = line.strip().split()[-1]
            if "MetricX" in line:
                if metricx_score is not None:
                    print(f"MetricX: {file_path}")
                    return
                metricx_score = line.strip().split()[-1]
            if "Lang_Error_Rate" in line:
                if lang_error_rate is not None:
                    print(f"Lang_Error_Rate: {file_path}")
                    return
                lang_error_rate = line.strip().split()[-1].split("%")[0]
    return

if __name__ == "__main__":
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Final-Qwen3-4B-final_mix-160k-1M-bsz128/global_step1240_hf"
    
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-XPlus-8B"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/LLaMAX3-8B-Alpaca"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Final-Qwen3-4B-final_mix-160k-1M-bsz128/global_step100_hf"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Final-Llama3.2-3B-final_mix-160k-1M-bsz128/global_step480_hf"
    dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128/global_step400_hf"
    
    
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Generalization2-Qwen3-4B-final_tr-mix-1m-1M-bsz128"
    # Walk through the directory and print all file paths
    whole_dict = {}
    xcomet_not_support_list = ["ltz", "ceb"]
    # src_langs_i_care = training_langs2
    # tgt_langs_i_care = ["eng"]
    src_langs_i_care = ["eng"]
    tgt_langs_i_care = training_langs2
    # tgt_langs_i_care = ["ara", "ces", "dan", "deu", "spa", "fin", "fra", "hrv", "hun", "ind", "ita", "jpn", "kor", "msa", "nld", "nob", "pol", "por", "ron", "rus", "swe", "tha", "tur", "ukr", "vie", "zho_simpl"]
    # tgt_langs_i_care = high_langs
    # tgt_langs_i_care = ["hun", "vie", "spa", "ces", "fra", "deu", "rus", "ben", "srp", "kor", "jpn", "ara", "tha", "swh", "zho_simpl", "tel"]
    # tgt_langs_i_care = ["hun", "spa", "ces", "fra", "deu", "rus", "ben", "srp", "kor", "jpn", "ara", "tha", "zho_simpl", "tel"]
    # tgt_langs_i_care = qwen_support
    # tgt_langs_i_care = llamax_langs
    
    # tgt_langs_i_care = ["ltz", "mkd","pol","srp","slk","slv","ben","guj","hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "ara", "tur", "tam", "fin"]
    # tgt_langs_i_care = [tgt for tgt in tgt_langs_i_care if tgt != src_lang and tgt != "lao" and tgt != "khm" and tgt != "mya"]
    # import code; code.interact(local=locals())
    print(f"Direction\tspBLEU\tXComet\tMetricX\tLang_Acc_Rate")
    for root, dirs, files in os.walk(dir_path):
        for src in src_langs_i_care:
            for tgt in tgt_langs_i_care:
                # import code; code.interact(local=locals())
                if src != tgt:
                    file_path = os.path.join(root, f"{src}-{tgt}.txt")
                    check_result(file_path)
                    result = print_result(file_path)
                    if src in xcomet_not_support_list or tgt in xcomet_not_support_list:
                        result = {f"{src}-{tgt}": (result[f"{src}-{tgt}"][0], None, result[f"{src}-{tgt}"][2], result[f"{src}-{tgt}"][3])}
                    whole_dict.update(result)
                    
                    # import code; code.interact(local=locals())

    # print([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None])
    # print(len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None]))
    # import code; code.interact(local=locals())
    # print("Average spBLEU: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][0]) for tgt in tgt_langs_i_care])/len(tgt_langs_i_care))
    # print("Average XComet: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][1]) for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None and tgt not in xcomet_not_support_list])/len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None and tgt not in xcomet_not_support_list]))
    print("Average spBLEU: ", sum([float(value[0]) for value in whole_dict.values()]) / len(whole_dict))
    print("Average XComet: ", sum([float(value[1]) for value in whole_dict.values() if value[1] is not None])/len([value for value in whole_dict.values() if value[1] is not None]))
    print("Average MetricX: ", sum([float(value[2]) for value in whole_dict.values() if value[2] is not None])/len([value for value in whole_dict.values() if value[2] is not None]))
    print("Average Lang_Acc_Rate: ", sum([float(value[3]) for value in whole_dict.values() if value[3] is not None])/len([value for value in whole_dict.values() if value[3] is not None]))
    # print("Average MetricX: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][2]) for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][2] is not None])/len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][2] is not None]))
    
    # print the average here
    # import code; code.interact(local=locals())
    # Below is only for copying purpose
    
    # list_of_strings = []
    # for tgt in tgt_langs_i_care:
    #     lang_pair = f"{src_lang}-{tgt}"
    #     if whole_dict[lang_pair][1] is not None and whole_dict[lang_pair][2] is not None:
    #         list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f} / {float(whole_dict[lang_pair][1])*100:.2f} / {-float(whole_dict[lang_pair][2]):.2f}")
    #     elif whole_dict[lang_pair][1] is not None:
    #         list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f} / {float(whole_dict[lang_pair][1])*100:.2f}")
    #     else:
    #         list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f}")
        
        
    # temp_file = "output_for_sheets.txt"
    # with open(temp_file, "w") as f:
    #     f.write("\t".join(list_of_strings))
    