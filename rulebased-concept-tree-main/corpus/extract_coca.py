import os
from tqdm import tqdm


OUTPUT_DIR = 'corpus/eng_standard/coca'

with open('corpus/coca.txt', 'r') as f:
    texts = f.readlines()

os.makedirs(OUTPUT_DIR, exist_ok=True)
for text in tqdm(texts):
    name = text.split()[0][2:]
    with open(os.path.join(OUTPUT_DIR, f'coca_{name}.txt'), 'w') as f:
        f.write(text[10:])