# from FlagEmbedding import BGEM3FlagModel

# model_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
# model = BGEM3FlagModel(model_path,  use_fp16=True) 

# # sentences_1 = ["Because the  dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features.",]
# # sentences_2 = ["由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。",
# #                "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"]

# # sentences_1 = ["Dr. Ehud Ur,professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."]
# # sentences_2 = ["加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。",
# #                "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段。"]

# sentences_1 = ['On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.']
# sentences_2 = ["周一，瑞典学院诺贝尔文学委员会常务秘书萨拉·丹尼尔斯在瑞典广播电台的一档节目中向公众宣布，委员会因无法直接联系到鲍勃·迪伦，通知他获得了 2016 年诺贝尔文学奖，已经放弃了与他联系的尝试。",
#                '周一，瑞典斯德哥尔摩诺贝尔文学奖评审委员会秘书萨拉·达尼乌斯在瑞典广播公司Sveriges Radio的一档节目中公开宣布，由于无法直接联系到鲍勃·迪伦本人，评审委员会决定放弃继续尝试联系他，以确认他是否确实获得了2016年诺贝尔文学奖。此前，评审委员会曾多次尝试通过各种方式联系迪伦，但均未成功。']
# # sentences_1 = ["\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added."]
# # sentences_2 = ["现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。\n\n中文翻译如下：\n\n现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。",
# #               "现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。"]
# sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

# print(model.compute_score(sentence_pairs, 
#                           max_passage_length=128, # a smaller max length leads to a lower latency
#                           weights_for_different_modes=[0.4, 0.2, 0.4])) # weights_for_different_modes(w) is used to do weighted sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score

# from sentence_transformers import SentenceTransformer

# model_path = "/mnt/gemini/data1/yifengliu/model/LaBSE"
# # sentence_1 = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# # sentence_2 = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# # sentence_2 = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"

# # sentence_1 = "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."
# # sentence_2 = "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。"
# # sentence_2 = "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段。",

# # sentence_1 = "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added."
# # sentence_2 = "现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。\n\n中文翻译如下：\n\n现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。",
# # # sentence_2 = "现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。"

# sentence_1 = 'On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.'
# # sentence_2 = "周一，瑞典学院诺贝尔文学委员会常务秘书萨拉·丹尼尔斯在瑞典广播电台的一档节目中向公众宣布，委员会因无法直接联系到鲍勃·迪伦，通知他获得了 2016 年诺贝尔文学奖，已经放弃了与他联系的尝试。"
# sentence_2 = '周一，瑞典斯德哥尔摩诺贝尔文学奖评审委员会秘书萨拉·达尼乌斯在瑞典广播公司Sveriges Radio的一档节目中公开宣布，由于无法直接联系到鲍勃·迪伦本人，评审委员会决定放弃继续尝试联系他，以确认他是否确实获得了2016年诺贝尔文学奖。此前，评审委员会曾多次尝试通过各种方式联系迪伦，但均未成功。'

# model = SentenceTransformer(model_path)
# embeddings1 = model.encode(sentence_1)
# embeddings2 = model.encode(sentence_2)
# similarity = embeddings1 @ embeddings2.T
# print(f"Similarity: {similarity}")


# from simalign import SentenceAligner

# # making an instance of our model.
# # You can specify the embedding model and all alignment settings in the constructor.
# myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

# # The source and target sentences should be tokenized to words.
# src = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# tgt =  "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# tgt = ""
# src_sentence = src.split()
# trg_sentence = [c for c in tgt]
# # src_sentence = ["This", "is", "a", "test", "."]
# # trg_sentence = ["Das", "ist", "ein", "Test", "."]
# # The output is a dictionary with different matching methods.
# # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
# alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

# def convert_to_word(src, tgt, align_list):
#     for src_idx, tgt_idx in align_list:
#         src_word = src[src_idx]
#         tgt_word = tgt[tgt_idx]
#         print(f"{src_word} -> {tgt_word}")

# for matching_method in alignments:
#     print(matching_method, ":", alignments[matching_method])


import torch
import transformers
import itertools

def align_score(srcs, tgts, model, tokenizer):
  align_score_list = []
  for src, tgt in zip(srcs, tgts):
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    print(token_src)
    print(token_tgt)
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
      sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
      sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
      out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
      out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

      dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

      softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
      softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

      softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
      align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    align_percent = len(align_words) / len(token_tgt)
    align_score_list.append(align_percent)
  return align_score_list

model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

src = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# tgt = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
tgt = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# tgt = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"
src = tokenizer.tokenize(src)
src = " ".join(src)
tgt = tokenizer.tokenize(tgt)
tgt = " ".join(tgt)

align_score_list = align_score([src], [tgt], model, tokenizer)
print(f"Align Score: {align_score_list}")
# pre-processing
# sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
# token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
# print(token_src)
# print(token_tgt)
# wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
# ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
# sub2word_map_src = []
# for i, word_list in enumerate(token_src):
#   sub2word_map_src += [i for x in word_list]
# sub2word_map_tgt = []
# for i, word_list in enumerate(token_tgt):
#   sub2word_map_tgt += [i for x in word_list]

# # alignment
# align_layer = 8
# threshold = 1e-3
# model.eval()
# with torch.no_grad():
#   out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
#   out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

#   dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

#   softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
#   softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

#   softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

# align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
# align_words = set()
# for i, j in align_subwords:
#   align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

# # printing
# class color:
#    PURPLE = '\033[95m'
#    CYAN = '\033[96m'
#    DARKCYAN = '\033[36m'
#    BLUE = '\033[94m'
#    GREEN = '\033[92m'
#    YELLOW = '\033[93m'
#    RED = '\033[91m'
#    BOLD = '\033[1m'
#    UNDERLINE = '\033[4m'
#    END = '\033[0m'

# for i, j in sorted(align_words):
#   print(f'{color.BOLD}{color.BLUE}{sent_src[i]}{color.END}==={color.BOLD}{color.RED}{sent_tgt[j]}{color.END}')
  

# Expected output:
# mwmf (Match): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# inter (ArgMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# itermax (IterMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# import math
# from laser_encoders.models import SentenceEncoder
# from pathlib import Path
# import os
# base_path = "/mnt/gemini/data1/yifengliu/model/LASER"
# en = "eng_Latn"
# zh = "zho_Hans"
# en_encoder = SentenceEncoder(model_path=os.path.join(base_path, en, 'laser2.pt'), spm_model=Path(os.path.join(base_path, en, 'laser2.spm')), spm_vocab=os.path.join(base_path, en, 'laser2.cvocab'))
# zh_encoder = SentenceEncoder(model_path=os.path.join(base_path, zh, 'laser2.pt'), spm_model=Path(os.path.join(base_path, zh, 'laser2.spm')), spm_vocab=os.path.join(base_path, zh, 'laser2.cvocab'))
# # en_embeddings = en_encoder("Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.")
# # zh_embeddings = zh_encoder("加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段。")
# # zh_embeddings2 = zh_encoder("加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。")

# # en_embeddings = en_encoder("Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features.")
# # zh_embeddings = zh_encoder("由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。")
# # zh_embeddings2 = zh_encoder("恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。")

# en_embeddings = en_encoder("On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.")
# zh_embeddings = zh_encoder("周一，瑞典斯德哥尔摩诺贝尔文学奖评审委员会秘书萨拉·达尼乌斯在瑞典广播公司Sveriges Radio的一档节目中公开宣布，由于无法直接联系到鲍勃·迪伦本人，评审委员会决定放弃继续尝试联系他，以确认他是否确实获得了2016年诺贝尔文学奖。此前，评审委员会曾多次尝试通过各种方式联系迪伦，但均未成功。")
# zh_embeddings2 = zh_encoder("周一，瑞典学院诺贝尔文学委员会常务秘书萨拉·丹尼尔斯在瑞典广播电台的一档节目中向公众宣布，委员会因无法直接联系到鲍勃·迪伦，通知他获得了 2016 年诺贝尔文学奖，已经放弃了与他联系的尝试。")
# similarity1 = en_embeddings @ zh_embeddings.T
# similarity2 = en_embeddings @ zh_embeddings2.T
# similarity3 = zh_embeddings @ zh_embeddings2.T
# print(f"Similarity: {float(similarity1), float(similarity2), float(similarity3)}")
import code; code.interact(local=locals())