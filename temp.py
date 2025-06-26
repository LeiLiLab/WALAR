import json
import random
import pandas as pd
from transforemers import AutoTokenizer
from vLLM import LLM, SamplingParams

def load_dastset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    # Example usage
    # file_path = "/mnt/gemini/data1/yifengliu/qe-lr/result/wmt24/XComet-xl/seg/cs-uk.jsonl"
    # dataset = load_dastset(file_path)
    
    import fasttext

    # Load the language ID model
    model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")

    src = "Hello, how are you?"
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LLM(model_path)
    samples = SamplingParams(n=1, temperature=0.0)
    message = [
        {'user': src}
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = model.generate(prompt, samples=samples)
    # prediction = model.predict(sentence)
    import code; code.interact(local=locals())
    language_code = prediction[0][0].replace("__label__", "")  # e.g., 'fr'
    confidence = prediction[1][0]

    print(f"Language: {language_code}, Confidence: {confidence:.2f}")

    import code; code.interact(local=locals())