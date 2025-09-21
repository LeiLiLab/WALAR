import fasttext

if __name__ == "__main__":
    model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    tgt = "Kamar wasu ƙwararrun, yana da shakkun cewa ko za a iya warkar da ciwon sukari, inda ya bayyana cewa sakamakon bincike ba shi da alaƙa da mutanen da tuni su ke da nau’in ciwon sukari na 1."
    lang_info = model.predict(tgt)
    # import code; code.interact(local=locals())