import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

if __name__ == '__main__':
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen2.5-1.5B"
    sample = SamplingParams(n=1, temperature=1, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentence = "If Mr. Trump again wins the presidency, he might order that the federal cases brought by special counsel Jack Smith be dropped by the Justice Department, or even pardon himself to avoid trial."
    prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    print(f"Prompt: {prompt}")
    output = model.generate(prompt, sample)
    print("==================")
    print(output[0].outputs[0].text)