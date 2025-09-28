import json
import fasttext

def load_dataset(path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except:
                break
    return dataset


if __name__ == "__main__":
    model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct/hin-npi.txt"
    dataset = load_dataset(path)
    preds = [data['pred'] for data in dataset]
    preds = [pred.replace("\n", " ") for pred in preds]
    # tgt = "Kamar wasu ƙwararrun, yana da shakkun cewa ko za a iya warkar da ciwon sukari, inda ya bayyana cewa sakamakon bincike ba shi da alaƙa da mutanen da tuni su ke da nau’in ciwon sukari na 1."
    lang_info = model.predict(preds)
    lang_labels = [label[0].replace("__label__", "") for label in lang_info[0]]
    
    import code; code.interact(local=locals())