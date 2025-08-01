import transformers
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507"
    sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768, tensor_parallel_size=2, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    system_prompt = "You are a professional translation evaluator."
    user_prompt = """Your task is to assess whether a translation segment successfully conveys the semantic content of the original sentence according to the following criteria:
Key Information Recognition: Identify whether the key information in the source (e.g., proper nouns, keywords, terminologies, or sentence structures) is present in the translation.


Correctness Assessment: Determine whether the translation accurately conveys the source sentence's intention, without misinterpretation or contextual errors.


Expressiveness Assessment: Evaluate whether the translation is fluent, clear, and intuitive to human readers. It should avoid unnecessary verbosity, ambiguous phrases, or awkward grammar.
Given a source sentence and its translation, please first analyze the translation and finally answer "Yes" if the translation meets all three criteria and answer "No" otherwise.

Source sentence: {src}
Translation: {tgt}
"""
    src = "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."
    tgt = "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。"
    # tgt = "托尼·莫尔博士在南非夸祖鲁-纳塔尔省发现了这种广泛耐药结核病 (XDR-TB)。"
    user_prompt = user_prompt.format(src=src, tgt=tgt)
    # sentence = f"""{src}\nTranslate from Chinese to English:\n"""
    # prompt = f"{sentence}\nTranslate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:"
    message = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    print(f"Prompt: {prompt}")
    output = model.generate(prompt, sample)
    print("==================")
    print(output[0].outputs[0].text)
    import code; code.interact(local=locals())