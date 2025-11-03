import json
from camel_tools.diacritization import Diacritizer

dia = Diacritizer.pretrained()  # load model
with open("sentences_split_preserve_chars_newline.json", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"] if isinstance(data, dict) else data
sentences_tashkeel = [dia.diacritize(s) for s in sentences]

output = {"sentences": sentences_tashkeel} if isinstance(data, dict) else sentences_tashkeel
with open("sentences_split_preserve_chars_newline_tashkeel.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
