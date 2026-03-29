import csv
import requests
import pdfplumber
import io
import os
from tqdm import tqdm

INPUT_FILE = '/Users/mac/Desktop/vscode/gromov/corpus/micusp_papers_non_native.csv'
OUTPUT_DIR = 'corpus/eng_non_standard/micusp'

with open(INPUT_FILE, 'r') as f:
    reader = csv.reader(f)
    paper_ids = [row[0] for row in reader]
    paper_ids = paper_ids[1:]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for paper_id in tqdm(paper_ids):
    url = f'https://elicorpora.info/static/search/pdf/{paper_id}.pdf'
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}       
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Ошибка скачивания {paper_id}: {e}')

    text = ''
    try:
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + ' '
    except Exception as e:
        print(f'Ошибка чтения {paper_id}: {e}')
        continue

    txt_path = f'{OUTPUT_DIR}/{paper_id}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f'Готово: {paper_id}')


