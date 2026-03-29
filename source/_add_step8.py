"""
Скрипт добавляет ячейки шагов 8.1 и 8.2 в ноутбук 04_neural_architecture.ipynb.
Запускать из корня проекта: python source/_add_step8.py
"""
import json

with open('source/04_neural_architecture.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_md(source):
    lines = source.split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    }

def make_code(source):
    lines = source.split('\n')
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    }

# ============================================================
# Ячейка A (markdown): заголовок шага 8
# ============================================================
cell_a = make_md("""## Шаг 8. Прототип нейросети на PyTorch

### 8.1. Подготовка обучающих данных

На этом шаге мы превращаем сырые леммы из базы данных в готовые обучающие данные для LSTM-генератора. Работа идёт в несколько этапов:

1. Загружаем леммы из таблицы `classla_tokens` (результат подплана 2)
2. Фильтруем слишком редкие и слишком частые леммы (min_df=2, max_df=90%)
3. Строим BM25-матрицу (98 документов x N отфильтрованных лемм)
4. Применяем SVD-разложение (k=50) и получаем k-мерные эмбеддинги для каждой леммы
5. Сохраняем эмбеддинги в FAISS-индекс (для быстрого поиска ближайших/дальних соседей) и в pickle
6. Строим vocabulary (словарь лемма → числовой индекс) со спец-токенами
7. Нарезаем тексты на перекрывающиеся фрагменты (sliding window)
8. Оборачиваем всё в PyTorch Dataset и DataLoader

После этого шага у нас будет всё необходимое для обучения нейросети.""")

# ============================================================
# Ячейка B (code): BM25 + SVD + FAISS
# ============================================================
cell_b = make_code("""import sqlite3
import numpy as np
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import faiss
import pickle
import json as json_mod
import os
import time

# путь к базе данных (ноутбук лежит в source/, база — в datasets/)
db_path = '../datasets/nlp_data.db'
out_dir = '../datasets'

# подключаемся к базе с лемматизированными текстами
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# загружаем все леммы, сгруппированные по text_id
# фильтруем знаки препинания (PUNCT), числа (NUM) и символы (SYM) —
# они не несут смысла для языковой модели
cursor.execute(\"\"\"
    SELECT text_id, lemma
    FROM classla_tokens
    WHERE upos NOT IN ('PUNCT', 'NUM', 'SYM')
    ORDER BY text_id, sentence_id, token_id
\"\"\")
rows = cursor.fetchall()

# собираем леммы каждого текста в отдельный список
docs_lemmas = defaultdict(list)
for text_id, lemma in rows:
    # приводим к нижнему регистру для единообразия
    docs_lemmas[text_id].append(lemma.lower())

# сортируем text_id для воспроизводимости
text_ids = sorted(docs_lemmas.keys())
# tokenized_corpus — список списков лемм, по одному списку на текст
tokenized_corpus = [docs_lemmas[tid] for tid in text_ids]

print(f"Загружено текстов: {len(text_ids)}")
print(f"Пример (текст {text_ids[0]}, первые 15 лемм): {tokenized_corpus[0][:15]}")

# считаем document frequency (DF) — в скольких текстах встречается каждая лемма
doc_freq = Counter()
for doc_lemmas in tokenized_corpus:
    # set() убирает дубликаты внутри одного текста
    for lemma in set(doc_lemmas):
        doc_freq[lemma] += 1

n_docs = len(text_ids)
# min_df=2: лемма должна встретиться хотя бы в 2 текстах (иначе она слишком редкая)
min_df = 2
# max_df=0.9: лемма не должна встречаться более чем в 90% текстов (иначе она стоп-слово)
max_df_ratio = 0.9
max_df = int(max_df_ratio * n_docs)  # 88 текстов

# фильтруем леммы по DF
filtered_vocab = sorted([
    lemma for lemma, freq in doc_freq.items()
    if min_df <= freq <= max_df
])

print(f"\\nФильтрация лемм:")
print(f"  min_df={min_df} (хотя бы в {min_df} текстах)")
print(f"  max_df={max_df} ({max_df_ratio*100:.0f}% от {n_docs} текстов)")
print(f"  до фильтрации: {len(doc_freq)} уникальных лемм")
print(f"  после фильтрации: {len(filtered_vocab)} лемм")

# маппинг лемма → позиция в отфильтрованном словаре
lemma2idx_bm25 = {lemma: i for i, lemma in enumerate(filtered_vocab)}

# создаём BM25-объект из всего корпуса
# BM25Okapi автоматически вычисляет IDF, среднюю длину документа и параметры k1, b
bm25 = BM25Okapi(tokenized_corpus)

# строим BM25-матрицу W[doc_i, word_j] = BM25 score слова j в документе i
# для каждого слова из словаря вычисляем его BM25-скор против всех 98 документов
n_words = len(filtered_vocab)
print(f"\\nСтроим BM25-матрицу {n_docs} x {n_words}...")
t0 = time.time()

W = np.zeros((n_docs, n_words), dtype=np.float32)
for j, lemma in enumerate(filtered_vocab):
    if j % 5000 == 0 and j > 0:
        print(f"  обработано {j}/{n_words} лемм...")
    # get_scores([lemma]) возвращает массив из 98 скоров — по одному на документ
    scores = bm25.get_scores([lemma])
    W[:, j] = scores

elapsed = time.time() - t0
print(f"BM25-матрица готова за {elapsed:.1f} сек")
print(f"  shape: {W.shape}")
print(f"  ненулевых: {np.count_nonzero(W)} из {W.size} ({100*np.count_nonzero(W)/W.size:.1f}%)")
print(f"  min={W.min():.4f}, max={W.max():.4f}, mean={W.mean():.6f}")

# SVD-разложение BM25-матрицы
# k=50 — стартовый ранг (ограничен сверху: k < min(98, n_words))
k = 50
print(f"\\nSVD-разложение (k={k})...")

# преобразуем в sparse-матрицу для эффективности svds
W_sparse = csr_matrix(W)
u, sigma, vt = svds(W_sparse, k=k)

# svds возвращает сингулярные значения в произвольном порядке —
# сортируем по убыванию (от самого важного измерения к наименее важному)
desc_order = np.flip(np.argsort(sigma))
u = u[:, desc_order]       # U: (98, k) — эмбеддинги документов
sigma = sigma[desc_order]  # Sigma: (k,) — "важность" каждого измерения
vt = vt[desc_order]        # V^T: (k, n_words) — эмбеддинги слов (по строкам)

print(f"  U: {u.shape} — эмбеддинги документов")
print(f"  Sigma: {sigma.shape} — сингулярные значения")
print(f"  V^T: {vt.shape} — эмбеддинги слов")
print(f"  Топ-10 сингулярных значений: {sigma[:10].round(2)}")

# матрица эмбеддингов слов: (diag(Sigma) @ V^T).T
# каждая строка i — это k-мерный вектор для леммы filtered_vocab[i]
# умножение на Sigma взвешивает компоненты по их важности
word_embeddings = (np.diag(sigma) @ vt).T  # shape: (n_words, k)
print(f"\\nМатрица эмбеддингов слов: {word_embeddings.shape}")
print(f"Примеры (первые 5 слов, первые 5 компонент):")
for i in range(min(5, n_words)):
    print(f"  {filtered_vocab[i]:20s} {word_embeddings[i, :5].round(4)}")

# нормализуем векторы для FAISS
# IndexFlatIP (Inner Product) на нормализованных векторах = cosine similarity
norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1  # защита от деления на ноль
word_embeddings_normed = (word_embeddings / norms).astype(np.float32)

# создаём FAISS-индекс для быстрого поиска ближайших соседей
index = faiss.IndexFlatIP(k)
index.add(word_embeddings_normed)
print(f"\\nFAISS-индекс: {index.ntotal} векторов, размерность {k}")

# сохраняем FAISS-индекс на диск
faiss_path = os.path.join(out_dir, 'embeddings.faiss')
faiss.write_index(index, faiss_path)

# сохраняем маппинг lemma <-> faiss_id в таблицу embedding_index
cursor.execute("DROP TABLE IF EXISTS embedding_index")
cursor.execute(\"\"\"
    CREATE TABLE embedding_index (
        lemma TEXT PRIMARY KEY,
        faiss_id INTEGER
    )
\"\"\")
for i, lemma in enumerate(filtered_vocab):
    cursor.execute("INSERT INTO embedding_index (lemma, faiss_id) VALUES (?, ?)", (lemma, i))
conn.commit()

# сохраняем pickle-копию словаря {лемма: вектор} (ненормализованный)
svd_dict = {lemma: word_embeddings[i] for i, lemma in enumerate(filtered_vocab)}
pkl_path = os.path.join(out_dir, 'svd_embeddings_mk.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(svd_dict, f)

# сохраняем документ-эмбеддинги (U) и сингулярные значения (Sigma)
np.save(os.path.join(out_dir, 'document_embeddings.npy'), u)
np.save(os.path.join(out_dir, 'svd_sigma.npy'), sigma)

# сохраняем BM25-матрицу для воспроизводимости
np.save(os.path.join(out_dir, 'bm25_matrix.npy'), W)
# сохраняем список отфильтрованных лемм (порядок столбцов BM25-матрицы)
with open(os.path.join(out_dir, 'filtered_vocab.json'), 'w', encoding='utf-8') as f:
    json_mod.dump(filtered_vocab, f, ensure_ascii=False)

print(f"\\nСохранено:")
print(f"  {faiss_path} — FAISS-индекс ({index.ntotal} векторов)")
print(f"  {pkl_path} — pickle словарь")
print(f"  {out_dir}/document_embeddings.npy — U {u.shape}")
print(f"  {out_dir}/svd_sigma.npy — Sigma {sigma.shape}")
print(f"  {out_dir}/bm25_matrix.npy — BM25 матрица {W.shape}")
print(f"  {out_dir}/filtered_vocab.json — {len(filtered_vocab)} лемм")
print(f"  nlp_data.db:embedding_index — {len(filtered_vocab)} записей")

# тест: поиск ближайших соседей через FAISS
# выбираем несколько частых македонских слов для проверки
test_words = [w for w in ['човек', 'жена', 'дете', 'град', 'вода', 'рака', 'ден', 'ноќ'] if w in lemma2idx_bm25]
if test_words:
    print(f"\\nТест FAISS — 5 ближайших соседей:")
    for word in test_words[:4]:
        idx = lemma2idx_bm25[word]
        vec = word_embeddings_normed[idx:idx+1]
        # ищем 6 соседей (первый — само слово)
        distances, indices = index.search(vec, 6)
        neighbors = [
            (filtered_vocab[int(i)], f"{d:.3f}")
            for d, i in zip(distances[0], indices[0])
            if int(i) != idx
        ][:5]
        print(f"  {word} → {neighbors}")
else:
    print("\\nТестовые слова не найдены в словаре")

conn.close()""")

# ============================================================
# Ячейка C (markdown): пояснение к BM25 и SVD
# ============================================================
cell_c = make_md("""### 8.1.1. Что мы получили: BM25-матрица и SVD-эмбеддинги

**BM25-матрица** — это таблица размером 98 (текстов) на N (отфильтрованных лемм), где каждая ячейка показывает, насколько важна конкретная лемма для конкретного текста. BM25 учитывает три вещи: как часто слово встречается в тексте (term frequency), насколько оно редкое в корпусе (inverse document frequency), и длину текста (короткий текст с тем же числом вхождений получает больший скор).

**SVD-разложение** сжимает эту большую матрицу в три маленьких: U (98 × 50) — "координаты" каждого текста в 50-мерном пространстве, Sigma (50) — "важность" каждого из 50 измерений, V^T (50 × N) — "координаты" каждого слова в тех же 50 измерениях. Умножая Sigma на V^T, мы получаем 50-мерный эмбеддинг для каждой леммы — компактное числовое представление смысла слова.

**FAISS-индекс** — библиотека от Meta для быстрого поиска похожих векторов. Мы нормализовали SVD-эмбеддинги и загрузили в IndexFlatIP (поиск по скалярному произведению = cosine similarity для нормализованных векторов). Теперь за доли секунды можно найти семантически близкие слова (для анализа) или максимально далёкие (для создания юмористических сочетаний — semantic incongruity).""")

# ============================================================
# Ячейка D (code): Vocabulary + Sliding Window + Dataset + DataLoader
# ============================================================
cell_d = make_code("""import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import json as json_mod
import numpy as np
import sqlite3
import pickle
import os

# пути к данным
db_path = '../datasets/nlp_data.db'
out_dir = '../datasets'

# загружаем отфильтрованный словарь лемм (из предыдущей ячейки или с диска)
vocab_path = os.path.join(out_dir, 'filtered_vocab.json')
with open(vocab_path, 'r', encoding='utf-8') as f:
    filtered_vocab = json_mod.load(f)

# загружаем SVD-эмбеддинги из pickle
pkl_path = os.path.join(out_dir, 'svd_embeddings_mk.pkl')
with open(pkl_path, 'rb') as f:
    svd_dict = pickle.load(f)

# размерность эмбеддингов (k из SVD)
embed_dim = next(iter(svd_dict.values())).shape[0]
print(f"SVD-эмбеддинги загружены: {len(svd_dict)} слов, размерность {embed_dim}")

# множество слов, для которых есть SVD-эмбеддинги
svd_words = set(svd_dict.keys())

# строим vocabulary: маппинг лемма <-> числовой индекс
# спец-токены занимают первые 4 позиции
PAD_IDX = 0   # padding — заполнитель для выравнивания длины последовательностей
BOS_IDX = 1   # beginning of sequence — начало текста
EOS_IDX = 2   # end of sequence — конец текста
UNK_IDX = 3   # unknown — неизвестное слово (нет SVD-эмбеддинга)

# word2idx: лемма → числовой индекс
word2idx = {'<PAD>': PAD_IDX, '<BOS>': BOS_IDX, '<EOS>': EOS_IDX, '<UNK>': UNK_IDX}
# добавляем все леммы из SVD-словаря
for lemma in filtered_vocab:
    word2idx[lemma] = len(word2idx)

# idx2word: числовой индекс → лемма (обратный маппинг)
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)
print(f"Vocabulary: {vocab_size} слов (включая 4 спец-токена)")

# сохраняем vocabulary на диск
vocab_save_path = os.path.join(out_dir, 'vocabulary_mk.json')
with open(vocab_save_path, 'w', encoding='utf-8') as f:
    json_mod.dump(word2idx, f, ensure_ascii=False)
print(f"Vocabulary сохранён: {vocab_save_path}")

# загружаем леммы из базы для построения обучающих последовательностей
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute(\"\"\"
    SELECT text_id, lemma
    FROM classla_tokens
    WHERE upos NOT IN ('PUNCT', 'NUM', 'SYM')
    ORDER BY text_id, sentence_id, token_id
\"\"\")
rows = cursor.fetchall()
conn.close()

# группируем леммы по текстам
from collections import defaultdict
docs_lemmas = defaultdict(list)
for text_id, lemma in rows:
    docs_lemmas[text_id].append(lemma.lower())

text_ids = sorted(docs_lemmas.keys())

# конвертируем каждый текст в последовательность индексов
# слова без SVD-эмбеддинга заменяются на <UNK>
sequences_by_text = []
unk_count = 0
total_count = 0
for tid in text_ids:
    seq = []
    for lemma in docs_lemmas[tid]:
        total_count += 1
        if lemma in word2idx:
            seq.append(word2idx[lemma])
        else:
            seq.append(UNK_IDX)
            unk_count += 1
    sequences_by_text.append(seq)

print(f"\\nКонвертация лемм в индексы:")
print(f"  всего токенов: {total_count}")
print(f"  <UNK> замен: {unk_count} ({100*unk_count/total_count:.1f}%)")

# sliding window: нарезаем каждый текст на перекрывающиеся фрагменты
# seq_len=64: длина фрагмента (сколько токенов в одном примере)
# step=32: шаг сдвига окна (перекрытие 50%)
# перекрытие помогает: (а) увеличить число обучающих примеров,
# (б) каждый токен появляется в разных контекстах
seq_len = 64
step = 32

all_fragments = []  # список пар (input, target)

for seq in sequences_by_text:
    # добавляем <BOS> в начало и <EOS> в конец текста
    full_seq = [BOS_IDX] + seq + [EOS_IDX]

    # скользящее окно
    for start in range(0, len(full_seq) - seq_len, step):
        fragment = full_seq[start : start + seq_len + 1]  # +1 для target
        # input = все токены кроме последнего
        inp = fragment[:-1]
        # target = все токены кроме первого (сдвиг на 1 вправо)
        tgt = fragment[1:]
        all_fragments.append((inp, tgt))

    # если текст короче seq_len, берём его целиком
    if len(full_seq) <= seq_len + 1:
        inp = full_seq[:-1]
        tgt = full_seq[1:]
        all_fragments.append((inp, tgt))

print(f"\\nSliding window (seq_len={seq_len}, step={step}):")
print(f"  всего фрагментов: {len(all_fragments)}")

# train/val split: 90% train, 10% val
np.random.seed(42)
indices = np.random.permutation(len(all_fragments))
split_idx = int(0.9 * len(indices))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_fragments = [all_fragments[i] for i in train_indices]
val_fragments = [all_fragments[i] for i in val_indices]
print(f"  train: {len(train_fragments)}, val: {len(val_fragments)}")

# PyTorch Dataset — обёртка, которая позволяет PyTorch перебирать данные
class MacedonianTextDataset(Dataset):
    # fragments — список пар (input, target), каждая пара — список индексов
    def __init__(self, fragments):
        self.fragments = fragments

    def __len__(self):
        # сколько всего примеров в датасете
        return len(self.fragments)

    def __getitem__(self, idx):
        # возвращаем один пример как пару тензоров
        inp, tgt = self.fragments[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# collate_fn — функция для сборки батча
# фрагменты в батче могут быть разной длины (короткие тексты),
# поэтому дополняем короткие до максимальной длины нулями (<PAD>)
def collate_fn(batch):
    inputs, targets = zip(*batch)
    # находим максимальную длину в батче
    max_len = max(inp.size(0) for inp in inputs)
    # дополняем нулями (PAD_IDX=0) до max_len
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt
    return padded_inputs, padded_targets

# создаём Dataset и DataLoader
batch_size = 32
train_dataset = MacedonianTextDataset(train_fragments)
val_dataset = MacedonianTextDataset(val_fragments)

# shuffle=True для train — перемешиваем примеры каждую эпоху
# shuffle=False для val — оценка должна быть воспроизводимой
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"\\nDataLoader:")
print(f"  batch_size: {batch_size}")
print(f"  train батчей: {len(train_loader)}")
print(f"  val батчей: {len(val_loader)}")

# строим embedding matrix — матрицу весов для nn.Embedding
# каждая строка i — это SVD-эмбеддинг для слова с индексом i
embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

# спец-токены: <PAD> — нулевой вектор (не несёт информации),
# <BOS>, <EOS>, <UNK> — маленькие случайные векторы
np.random.seed(42)
embedding_matrix[BOS_IDX] = np.random.normal(0, 0.01, embed_dim)
embedding_matrix[EOS_IDX] = np.random.normal(0, 0.01, embed_dim)
embedding_matrix[UNK_IDX] = np.random.normal(0, 0.01, embed_dim)

# заполняем SVD-эмбеддингами для всех слов из словаря
filled_count = 0
for word, idx in word2idx.items():
    if word in svd_dict:
        embedding_matrix[idx] = svd_dict[word]
        filled_count += 1

print(f"\\nEmbedding matrix: {embedding_matrix.shape}")
print(f"  заполнено SVD-эмбеддингами: {filled_count} из {vocab_size}")

# оборачиваем в nn.Embedding с freeze=True —
# веса не будут обновляться при обучении (frozen embedding layer)
# padding_idx=0 — вектор для <PAD> всегда остаётся нулевым
embedding_layer = nn.Embedding.from_pretrained(
    torch.tensor(embedding_matrix),
    freeze=True,
    padding_idx=PAD_IDX
)
print(f"  nn.Embedding: freeze=True, padding_idx={PAD_IDX}")

# проверяем один батч
sample_input, sample_target = next(iter(train_loader))
print(f"\\nПример батча:")
print(f"  input shape: {sample_input.shape}")   # (batch_size, seq_len)
print(f"  target shape: {sample_target.shape}")  # (batch_size, seq_len)

# прогоняем через embedding layer
sample_embedded = embedding_layer(sample_input)
print(f"  embedded shape: {sample_embedded.shape}")  # (batch_size, seq_len, embed_dim)

# декодируем первые 15 токенов первого примера обратно в леммы
first_seq = sample_input[0].tolist()
decoded = [idx2word.get(idx, '?') for idx in first_seq[:15]]
print(f"\\nПервые 15 токенов первого примера:")
print(f"  индексы: {first_seq[:15]}")
print(f"  леммы:   {decoded}")

# сохраняем embedding matrix для будущего использования
emb_matrix_path = os.path.join(out_dir, 'embedding_matrix.npy')
np.save(emb_matrix_path, embedding_matrix)
print(f"\\nEmbedding matrix сохранена: {emb_matrix_path}")

# итоговая сводка
print(f"\\n{'='*60}")
print(f"Подготовка данных завершена:")
print(f"  vocab_size = {vocab_size}")
print(f"  embed_dim = {embed_dim} (SVD k)")
print(f"  seq_len = {seq_len}")
print(f"  train примеров: {len(train_fragments)}")
print(f"  val примеров: {len(val_fragments)}")
print(f"  train батчей: {len(train_loader)}")
print(f"  val батчей: {len(val_loader)}")
print(f"  embedding: frozen SVD-эмбеддинги")
print(f"{'='*60}")""")

# ============================================================
# Ячейка E (markdown): пояснение к подготовке данных
# ============================================================
cell_e = make_md("""### 8.1.2. Что такое sliding window, Dataset и embedding matrix

**Sliding window (скользящее окно)** — приём для увеличения числа обучающих примеров. Мы берём текст, ставим "окно" длиной 64 токена в начало, вырезаем фрагмент, затем сдвигаем окно на 32 токена вперёд и вырезаем следующий. Перекрытие в 50% нужно, чтобы каждый токен побывал в разных позициях: в начале фрагмента, в середине и в конце. Из каждого фрагмента делаем пару: input (все токены кроме последнего) и target (все токены кроме первого) — модель учится предсказывать следующий токен по предыдущим.

**Dataset и DataLoader** — стандартные инструменты PyTorch для работы с данными. Dataset хранит все примеры и умеет отдавать по одному. DataLoader группирует примеры в батчи (по 32 штуки), перемешивает их перед каждой эпохой и подготавливает для подачи в нейросеть. Функция collate_fn дополняет короткие фрагменты нулями (padding) до длины самого длинного в батче — так все примеры в батче имеют одинаковую длину.

**Embedding matrix (frozen)** — матрица размером vocab_size x 50, где каждая строка — это SVD-эмбеддинг слова. Параметр freeze=True означает, что эти веса не меняются при обучении: модель использует их "как есть", без подстройки. Это важно, потому что у нас всего 98 текстов — если разрешить эмбеддингам обучаться, они переобучатся (запомнят обучающие данные вместо общих закономерностей). Frozen-эмбеддинги работают как "внешние знания" о семантике слов, извлечённые из всего корпуса через BM25 + SVD.""")

# ============================================================
# Добавляем все 5 ячеек
# ============================================================
nb['cells'].extend([cell_a, cell_b, cell_c, cell_d, cell_e])

with open('source/04_neural_architecture.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Done. Total cells: {len(nb["cells"])}')
