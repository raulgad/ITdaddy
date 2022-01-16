import os
import torch
import textwrap
from transformers import GPT2Tokenizer

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

# Init the model
try:
    model = torch.load(os.path.dirname(__file__) + "/itdaddy_en_model.pth", map_location=torch.device('cpu'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
except Exception as e:
    print(f'Loading model error: {e.args}')

def generate(prompt: str, len_gen=40, temperature=0.7):
    generated = tokenizer.encode(prompt, return_tensors='pt')
    out = model.generate(input_ids=generated, max_length=len_gen, temperature=temperature,
        num_beams=5, do_sample=True, top_k=10, top_p=0.95,
        no_repeat_ngram_size=2, num_return_sequences=1,
        ).cpu().numpy()
    # Decode and preprocess text
    for out_ in out:
        wraped = textwrap.fill(tokenizer.decode(out_), 120)
        wraped = wraped.replace("  ", " ")
        if '.' in wraped:
            arr = wraped.split('.')
            arr.pop()
            prompt  = '.'.join(arr) + '.'
        else:
            prompt = wraped + '.'
    return prompt
